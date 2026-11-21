"""Independently replay four-cell CSV arithmetic and paired native-model contrasts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--reference-audit', type=Path, required=True)
    p.add_argument('--baseline-id', default='R1')
    args = p.parse_args()
    out = args.input
    read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    plan = read(out/'PLAN.json'); complete = read(out/'COMPLETE.json')
    assert complete['status'] == 'COMPLETE' and complete['optimizer_steps'] == 0
    collection = read(out/'COLLECTION_MANIFEST.json')
    for name, meta in collection['files'].items():
        assert digest(out/name) == meta['sha256'], name
    assert collection['temporary_weights_removed']
    for name in ['scripts/analysis/measure_source_grounding.py', 'src/methods/original/clrr.py',
                 'src/methods/common/helpers.py', 'src/methods/grounded_interaction/v1/model.py']:
        assert digest(ROOT/name) == plan['source_sha256'][name]
    # Engine orchestration has evolved; verify the only callable used by this
    # frozen probe separately, and preserve the exact remote engine as source.
    node = next(n for n in ast.parse((ROOT/'src/experiment_runner/engine.py').read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'configure_languages')
    # AST serialization differs between Python versions. Parse both exact source
    # files with this interpreter, after checking the captured file's SHA256.
    used_engine=out/'source/src/experiment_runner/engine.py'
    assert digest(used_engine)==plan['source_sha256']['src/experiment_runner/engine.py']
    used_node=next(n for n in ast.parse(used_engine.read_text(encoding='utf-8')).body
                  if isinstance(n,ast.FunctionDef) and n.name=='configure_languages')
    assert ast.dump(node,include_attributes=False)==ast.dump(used_node,include_attributes=False)
    pair = pd.read_csv(out/'NEGATIVE_PAIRS.csv')
    assert len(pair) == plan['rows'] and pair.valid_negative.all()
    # All 576 rows have a valid pair here; independent bootstrap uses equal-size
    # block means instead of the measurement worker's sums/counts reduction.
    batch = plan['batch_size']; assert len(pair) % batch == 0
    blocks = len(pair)//batch
    assert pair.block.tolist() == np.repeat(np.arange(blocks), batch).tolist()
    rng = np.random.default_rng(plan['bootstrap']['seed'])
    resamples = rng.integers(0, blocks, (plan['bootstrap']['repetitions'], blocks))
    def ci(values):
        means = np.asarray(values).reshape(blocks,batch,-1).mean(1)
        return np.quantile(means[resamples].mean(1), [.025,.975], axis=0).T
    frames = {}; checks = {}; native = {}
    for ident in complete['ids']:
        result = read(out/ident/'RESULT.json'); csv = out/ident/'per_sentence.csv'
        assert digest(csv) == result['csv_sha256']
        assert digest(out/'NEGATIVE_PAIRS.csv') == result['pairs_sha256']
        assert result['selection'] == plan['selections'][ident]
        assert result['selection'] == read(args.reference_audit/ident/'RESULT.json')['selection']
        assert result['parameter_versions_unchanged'] and result['gradients_none'] and result['optimizer_steps']==0
        assert not any(result['strict_loading'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        df = pd.read_csv(csv)
        pd.testing.assert_frame_equal(df[list(pair.columns)],pair)
        calc = dict(g_on=df.on_mismatch-df.on_match, g_off=df.off_mismatch-df.off_match,
                    route_matched_gain=df.off_match-df.on_match,
                    route_mismatched_gain=df.off_mismatch-df.on_mismatch)
        calc['interaction'] = calc['g_on']-calc['g_off']
        for key, values in calc.items():
            np.testing.assert_allclose(df[key],values,atol=1e-12,rtol=0)
        keys = list(calc); intervals=ci(df[keys].values)
        for key, bounds in zip(keys, intervals):
            metric=result['summary']['metrics_valid_pairs'][key]
            np.testing.assert_allclose(metric['mean'],df[key].mean(),atol=1e-12,rtol=0)
            np.testing.assert_allclose(metric['descriptive_block_ci95'],bounds,atol=1e-12,rtol=0)
        for cell in ['on_match','on_mismatch','off_match','off_mismatch']:
            np.testing.assert_allclose(df[cell],df['nll_'+cell]/df.tokens,atol=2e-6,rtol=0)
            np.testing.assert_allclose(df[cell].mean(),result['summary']['all_rows_mean_CE'][cell],atol=1e-12,rtol=0)
        previous=pd.read_csv(args.reference_audit/ident/'per_sentence.csv')
        replay={}
        for old,new in [('all-on','on_match'),('all-off','off_match')]:
            error=float(np.max(np.abs(previous[previous.condition==old].sort_values('row').ce.values-df[new].values)))
            assert error < 1e-4
            replay[new]=error
        frames[ident]=df
        branch='off' if ident==args.baseline_id else 'on'
        native[ident]=dict(branch=branch, matched_CE=float(df[branch+'_match'].mean()),
                          source_gap=float(df['g_'+branch].mean()))
        checks[ident]=dict(status='PASS',rows=len(df),valid_pairs=int(df.valid_negative.sum()),
            reductions_and_CI_replayed=True, prior_CE_max_errors=replay)
    comparisons={}
    base=frames[args.baseline_id]
    for ident,df in frames.items():
        if ident==args.baseline_id: continue
        for name,values in dict(native_matched_CE=df.on_match-base.off_match,
                                native_source_gap=df.g_on-base.g_off).items():
            comparisons[ident+'-'+args.baseline_id+'/'+name]=dict(mean=float(values.mean()),
                descriptive_paired_block_ci95=ci(values.values)[0].tolist())
    result=dict(status='PASS',checks=checks,native=native,paired_native_comparisons=comparisons,
        optimizer_steps=0,test_evaluated=False,bootstrap_unit='negative-pair block, descriptive fixed-checkpoint only',
        source_sha256=digest(Path(__file__)),gpu_after=collection['gpu_after'])
    (out/'LOCAL_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
