"""Replay frozen intervention metrics and summarize per-layer perturbation norms."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostics', type=Path, required=True)
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--upload-prefix', help='Publish small verified diagnostics in the existing public checkpoint repo')
    args = p.parse_args()
    root = args.diagnostics
    assert (root / 'COMPLETE.json').exists(), 'All three seeds must finish'
    plan = json.loads((root / 'PLAN.json').read_text())
    import hashlib
    assert hashlib.sha256(args.references.read_bytes()).hexdigest() == plan['validation_sha256']
    rows = pd.read_csv(args.references, dtype=str, keep_default_na=False)
    metrics = {'bleu': BLEU(tokenize='zh'), 'chrf++': CHRF(word_order=2), 'chrf0': CHRF(word_order=0)}
    results, norms = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder / 'RESULT.json').exists():
            continue
        r = json.loads((folder / 'RESULT.json').read_text())
        assert r['status'] == 'PASS' and r['all_on_replay'] and r['parameter_versions_unchanged']
        assert r['optimizer_steps'] == 0
        frame = pd.read_csv(folder / 'interventions.csv', dtype=str, keep_default_na=False)
        for condition, scores in r['scores'].items():
            part = frame[frame.condition == condition].copy()
            part['row'] = part.row.astype(int)
            part = part.sort_values('row')
            assert part.row.tolist() == list(range(len(rows)))
            for name, metric in metrics.items():
                replay = metric.corpus_score(part.prediction.str.strip().tolist(), [rows.target.str.strip().tolist()]).score
                if name in scores:
                    assert abs(replay - scores[name]) < 1e-8
                else:
                    assert name == 'chrf0'
                    scores[name] = replay
            ce = part.nll_sum.astype(float).sum() / part.tokens.astype(int).sum()
            assert abs(ce - scores['token_ce']) < 1e-7
            results.append(dict(seed=r['seed'], id=r['id'], condition=condition, **scores,
                delta_bleu=scores['bleu']-r['scores']['all-on']['bleu'],
                delta_chrf0=scores['chrf0']-r['scores']['all-on']['chrf0'],
                delta_token_ce=scores['token_ce']-r['scores']['all-on']['token_ce']))
        for layer in sorted({x['layer'] for x in r['residual_norms']}):
            subset = [x for x in r['residual_norms'] if x['layer'] == layer]
            correction = np.asarray([v for x in subset for v in x['correction_over_current']])
            residual = np.asarray([v for x in subset for v in x['residual_over_current']])
            assert len(correction) == len(residual) == len(rows)
            # The v1 probe reports A(d)/u before alpha, but alpha*(b+A(d))/u after alpha.
            norms.append(dict(seed=r['seed'], layer=layer, rows=len(correction),
                correction_unscaled_mean=float(correction.mean()),
                applied_correction_mean=float(.1*correction.mean()),
                applied_correction_p95=float(.1*np.quantile(correction,.95)),
                applied_residual_mean=float(residual.mean()), applied_residual_p95=float(np.quantile(residual,.95))))
    assert len(results) == 12 and {x['seed'] for x in results} == {42,43,44}
    pd.DataFrame(results).to_csv(root / 'intervention_summary.csv', index=False)
    pd.DataFrame(norms).to_csv(root / 'layer_norms.csv', index=False)
    summary = dict(status='VERIFIED', conditions=results, layer_norms=norms,
                   optimizer_steps=0, metric_replays=24, additional_chrf0_scores=12, scope=plan['scope'])
    (root / 'GPU_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines=['# Frozen C0 validation sensitivity', '', '## Material Passport', '',
        'All 3 seeds, 576 validation rows each, FP32 SDPA, beam1, no optimizer steps. All-on predictions match the frozen checkpoints.',
        'Interventions are inference changes on co-adapted weights; they do not predict retraining improvements.', '',
        '| Seed | Intervention | BLEU | Delta BLEU | Delta chrF0 | Delta token CE |',
        '|---|---|---:|---:|---:|---:|']
    for x in results:
        lines.append(f"| {x['seed']} | {x['condition']} | {x['bleu']:.4f} | {x['delta_bleu']:+.4f} | {x['delta_chrf0']:+.4f} | {x['delta_token_ce']:+.6f} |")
    lines += ['', 'Per-layer norms in layer_norms.csv; applied correction includes alpha=.1. No significance tests or test-set selection added.',
              'A small perturbation norm does not prove the module has no training effect; a large norm does not prove harmfulness.']
    (root / 'GPU_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))
    if args.upload_prefix:
        import os
        import re
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
        from huggingface_hub import HfApi
        from experiment_runner.integrity import verify_remote_files
        assert re.fullmatch(r'analysis/[a-zA-Z0-9_./-]+', args.upload_prefix)
        assert '..' not in args.upload_prefix.split('/')
        api = HfApi(token=os.environ['HF_TOKEN'])
        repo = 'FiveC/amis-rewire-checkpoints'
        assert not api.model_info(repo).private
        files = {}
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.csv', '.md') and not any(x in path.parts for x in ('model', 'restore')):
                assert path.stat().st_size < 10_000_000
                files[args.upload_prefix + '/gpu/' + path.relative_to(root).as_posix()] = path
        for name in ('CPU_REPORT.md', 'trajectory.csv', 'selection.csv', 'surface_metrics.csv', 'length_strata.csv', 'BLEU_COMPONENTS.csv'):
            path = root.parent / name
            files[args.upload_prefix + '/' + name] = path
        for remote, local in files.items():
            api.upload_file(repo_id=repo, path_in_repo=remote, path_or_fileobj=str(local))
        receipt = verify_remote_files(api, repo, files)
        (root / 'HF_UPLOAD.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print('DIAGNOSTICS_UPLOAD_VERIFIED', receipt['revision'], flush=True)


if __name__ == '__main__':
    main()
