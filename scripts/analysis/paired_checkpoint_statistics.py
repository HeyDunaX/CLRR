"""Paired corpus bootstrap of frozen prediction CSVs, with one explicit Holm family."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest, _compute_p_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-dir', required=True)
    parser.add_argument('--references', required=True)
    parser.add_argument('--design', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('test', 'validation'), default='test')
    parser.add_argument('--upload-prefix', help='Publish and verify these small reports in the canonical public repo')
    args = parser.parse_args()
    folder, out = Path(args.predictions_dir), Path(args.output)
    design = json.loads(Path(args.design).read_text(encoding='utf-8-sig'))
    if 'post_training_test' in design:
        design = design['post_training_test']['design']
    raw = pd.read_csv(args.references, dtype=str, keep_default_na=False)
    refs, n = [raw.target.str.strip().tolist()], len(raw)
    count, seed = design['draws'], design['seed']
    indices = np.random.default_rng(seed).choice(n, size=(count, n), replace=True)
    metrics = {'BLEU': BLEU(tokenize=design['bleu_tokenizer']),
               'chrF++': CHRF(word_order=2), 'chrF0': CHRF(word_order=0)}
    systems, observed, draws, provenance = {}, {}, {}, {}
    contrasts = design.get('contrasts', [])
    ids = sorted({ident for pair in design['pairs'] for ident in pair} |
                 {ident for item in contrasts for ident in item['coefficients']})
    for ident in ids:
        run = folder / 'runs' / ident
        prediction_file = run / f'{args.split}_predictions.csv'
        frame = pd.read_csv(prediction_file, dtype=str, keep_default_na=False)
        assert frame[['source', 'target']].equals(raw[['source', 'target']]), ident
        if args.split == 'test':
            complete = json.loads((run / 'COMPLETE.json').read_text())
            assert complete['status'] == 'COMPLETE_VERIFIED' and complete['test_backup']['verified']
            provenance[ident] = {'selection': complete['selection'], 'test_revision': complete['test_backup']['revision']}
        else:
            complete = json.loads((run / 'VAL_COMPLETE.json').read_text())
            assert complete['status'] == 'VALIDATION_COMPLETE_VERIFIED' and complete['backup']['verified']
            meta = json.loads((run / 'validation_metrics.json').read_text())
            assert meta['raw_reference_sha256'] == hashlib.sha256(Path(args.references).read_bytes()).hexdigest()
            assert meta['beams'] == 1 and meta['generation_precision'] == 'fp32'
            provenance[ident] = {'archive_sha256': complete['archive_sha256'], 'validation_revision': complete['backup']['revision']}
        systems[ident] = frame.prediction.str.strip().tolist()
        provenance[ident]['prediction_sha256'] = hashlib.sha256(prediction_file.read_bytes()).hexdigest()
        for name, metric in metrics.items():
            score = metric.corpus_score(systems[ident], refs).score
            key = {'BLEU': 'bleu', 'chrF++': 'chrf++', 'chrF0': 'chrf0'}[name]
            assert abs(score - complete['scores'][key]) < 1e-8
            observed[ident, name] = score
            stats = np.asarray(metric._extract_corpus_statistics(systems[ident], refs), dtype=np.float32)
            boot = np.empty(count)
            for start in range(0, count, 128):
                sums = stats[indices[start:start + 128]].sum(axis=1)
                boot[start:start + len(sums)] = [metric._compute_score_from_stats(s).score for s in sums]
            draws[ident, name] = boot
        print('BOOTSTRAP_SCORES', ident, flush=True)
    os.environ['SACREBLEU_SEED'] = str(seed)
    tests = []
    for control, candidate in design['pairs']:
        _, native = PairedTest(named_systems=[(control, systems[control]), (candidate, systems[candidate])],
                               metrics=metrics, references=refs, test_type='bs', n_samples=count, n_jobs=1)()
        for name, metric in metrics.items():
            delta = observed[candidate, name] - observed[control, name]
            diffs = draws[candidate, name] - draws[control, name]
            absolute = np.abs(diffs)
            p = float(_compute_p_value(absolute - absolute.mean(), abs(delta)))
            native_name = metric.corpus_score(systems[control], refs).name
            assert abs(p - native[native_name][1].p_value) < 1e-12
            tests.append({'control': control, 'candidate': candidate, 'metric': name,
                          'delta': delta, 'ci95': np.quantile(diffs, [.025, .975]).tolist(), 'p_raw': p})
        print('NATIVE_P_MATCH', control, candidate, flush=True)
    for contrast in contrasts:
        coefficients = contrast['coefficients']
        assert abs(sum(coefficients.values())) < 1e-12, 'Contrast coefficients must sum to zero'
        assert any(w != 0 for w in coefficients.values())
        for name in metrics:
            delta = sum(w * observed[ident, name] for ident, w in coefficients.items())
            diffs = sum(w * draws[ident, name] for ident, w in coefficients.items())
            # A general linear contrast has no native SacreBLEU paired-test equivalent.
            # Center the signed bootstrap contrast at its observed value under H0=0.
            p = float((np.count_nonzero(np.abs(diffs - delta) >= abs(delta)) + 1) / (count + 1))
            tests.append({'control': 'linear_contrast_zero', 'candidate': contrast['name'],
                          'coefficients': coefficients, 'metric': name, 'delta': delta,
                          'ci95': np.quantile(diffs, [.025, .975]).tolist(), 'p_raw': p,
                          'p_method': 'two-sided null-centered signed bootstrap contrast; approximate'})
    running = 0.
    for rank, index in enumerate(np.argsort([row['p_raw'] for row in tests])):
        running = max(running, min(1., (len(tests) - rank) * tests[index]['p_raw']))
        tests[index].update(p_holm=running, significant_holm=running < .05)
    assert len(tests) == (len(design['pairs']) + len(contrasts)) * len(metrics)
    report = {'status': 'VERIFIED_NATIVE_PAIRS_AND_BOOTSTRAP_CONTRASTS' if contrasts else 'VERIFIED_NATIVE_P_VALUES',
              'split': args.split, 'rows': n, 'draws': count, 'seed': seed,
              'sacrebleu_version': sacrebleu.__version__, 'holm_family_size': len(tests),
              'reference_sha256': hashlib.sha256(Path(args.references).read_bytes()).hexdigest(),
              'design_sha256': hashlib.sha256(Path(args.design).read_bytes()).hexdigest(),
              'scores': {ident: {name: observed[ident, name] for name in metrics} for ident in ids},
              'tests': tests, 'provenance': provenance,
              'contrast_method': 'Two-sided null-centered signed bootstrap with +1 correction; no native equivalent',
              'scope': design.get('scope', 'Exploratory sentence uncertainty conditional on one frozen training seed; excludes training randomness.')}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'paired_bootstrap.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(tests).to_csv(out / 'paired_bootstrap.csv', index=False)
    lines = [f'# Frozen RTX checkpoint {args.split} statistics', '',
             f'{n} {args.split} sentences; {count} paired resamples, RNG {seed}; Holm across all {len(tests)} tests.', '',
             '| Control | Candidate | Metric | Delta | 95% delta CI | p raw | p Holm | Significant |',
             '|---|---|---|---:|---|---:|---:|---|']
    for row in tests:
        low, high = row['ci95']
        lines.append(f"| {row['control']} | {row['candidate']} | {row['metric']} | {row['delta']:+.4f} | [{low:+.4f}, {high:+.4f}] | {row['p_raw']:.6f} | {row['p_holm']:.6f} | {row['significant_holm']} |")
    lines += ['', report['scope'], '', 'Native SacreBLEU p-values independently matched for every pairwise test.',
              'Method: https://github.com/mjpost/sacrebleu#paired-significance-tests .']
    if contrasts:
        lines += ['', 'Linear contrast p-values use an approximate two-sided null-centered signed bootstrap.',
                  'They are not native SacreBLEU paired p-values. All contrasts share resampled sentence indices.',
                  'Percentile CIs are marginal, not simultaneous confidence intervals.']
    (out / 'PAIRED_BOOTSTRAP.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if args.upload_prefix:
        import re
        from huggingface_hub import HfApi
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
        from experiment_runner.integrity import verify_remote_files
        assert re.fullmatch(r'analysis/[a-zA-Z0-9_./-]+', args.upload_prefix)
        assert '..' not in args.upload_prefix.split('/')
        repo = 'FiveC/amis-rewire-checkpoints'
        api = HfApi(token=os.environ['HF_TOKEN'])
        assert not api.model_info(repo).private
        files = {args.upload_prefix + '/' + f.name: f for f in
                 [out / 'paired_bootstrap.json', out / 'paired_bootstrap.csv',
                  out / 'PAIRED_BOOTSTRAP.md']}
        if args.split == 'test':
            files[args.upload_prefix + '/selection_lock.json'] = folder / 'selection_lock.json'
        files[args.upload_prefix + '/statistics_design.json'] = Path(args.design)
        for remote, local in files.items():
            api.upload_file(repo_id=repo, path_in_repo=remote, path_or_fileobj=str(local))
        receipt = verify_remote_files(api, repo, files)
        (out / 'STATISTICS_HF_UPLOAD.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print('STATISTICS_UPLOAD_VERIFIED', receipt['revision'], flush=True)


if __name__ == '__main__':
    main()
