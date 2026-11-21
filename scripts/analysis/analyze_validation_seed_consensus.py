"""Describe cross-seed validation overlap; never label semantic errors from metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grounding', type=Path, help='Optional frozen seed42 source-gap audit')
    args = parser.parse_args()
    plan = json.loads((args.inputs / 'PLAN.json').read_text(encoding='utf-8'))
    refs = pd.read_csv(args.references, dtype=str, keep_default_na=False)
    metrics = {'bleu': BLEU(tokenize='zh'), 'chrf++': CHRF(word_order=2),
               'chrf0': CHRF(word_order=0)}
    rows = refs.copy()
    rows.insert(0, 'row', range(len(rows)))
    provenance, corpus, predictions = {}, [], {}
    for run in plan['runs']:
        ident, seed, arm = run['id'], run['seed'], run['arm']
        assert seed in (42, 43, 44) and arm in ('Baseline', 'C0')
        folder = args.inputs / 'inputs' / ident
        names = ('validation_predictions.csv', 'validation_metrics.json', 'run_metadata.json')
        hashes = {name: sha(folder / name) for name in names}
        expected = plan['provenance'][ident]['file_sha256']
        assert all(hashes[name] == expected[name] for name in names), ident
        frame = pd.read_csv(folder / names[0], dtype=str, keep_default_na=False)
        assert frame[['source', 'target']].equals(refs[['source', 'target']]), ident
        receipt = json.loads((folder / names[1]).read_text(encoding='utf-8'))
        assert receipt['raw_reference_sha256'] == sha(args.references)
        assert receipt['beams'] == 1 and receipt['generation_precision'] == 'fp32'
        assert receipt['protocol']['version'] == sacrebleu.__version__
        hyp, target = frame.prediction.str.strip().tolist(), frame.target.str.strip().tolist()
        scores = {k: metric.corpus_score(hyp, [target]).score for k, metric in metrics.items()}
        assert all(abs(scores[k] - receipt['scores'][k]) < 1e-7 for k in scores), ident
        corpus.append(dict(seed=seed, arm=arm, **scores))
        key = f'{arm}_{seed}'
        predictions[key] = hyp
        rows[key + '_prediction'] = hyp
        rows[key + '_chrf0'] = [metrics['chrf0'].sentence_score(h, [t]).score for h, t in zip(hyp, target)]
        provenance[ident] = dict(selection=run['selection'], file_sha256=hashes)
    assert len(corpus) == 6
    n = len(rows)
    # Fixed-size bottom quartiles with row-index tie breaking, descriptive only.
    for seed in (42, 43, 44):
        ids = sorted(range(n), key=lambda i: (rows.loc[i, f'Baseline_{seed}_chrf0'], i))[:n // 4]
        rows[f'baseline_bottom_quartile_{seed}'] = rows.row.isin(ids)
        rows[f'C0_delta_{seed}'] = rows[f'C0_{seed}_chrf0'] - rows[f'Baseline_{seed}_chrf0']
    flags = [f'baseline_bottom_quartile_{s}' for s in (42, 43, 44)]
    rows['bottom_quartile_seeds'] = rows[flags].sum(axis=1)
    rows['baseline_mean_sentence_chrf0'] = rows[[f'Baseline_{s}_chrf0' for s in (42, 43, 44)]].mean(axis=1)
    rows['C0_mean_delta_sentence_chrf0'] = rows[[f'C0_delta_{s}' for s in (42, 43, 44)]].mean(axis=1)
    rows['baseline_all_same_prediction'] = [len({predictions[f'Baseline_{s}'][i] for s in (42, 43, 44)}) == 1 for i in range(n)]
    rows['C0_gain_all_seeds'] = (rows[[f'C0_delta_{s}' for s in (42, 43, 44)]] > 0).all(axis=1)
    rows['C0_loss_all_seeds'] = (rows[[f'C0_delta_{s}' for s in (42, 43, 44)]] < 0).all(axis=1)
    grounding_provenance = None
    if args.grounding:
        manifest = json.loads((args.grounding / 'COLLECTION_MANIFEST.json').read_text(encoding='utf-8'))
        for name in ('R1/RESULT.json', 'R1/per_sentence.csv'):
            assert sha(args.grounding / name) == manifest['files'][name]['sha256']
        old = json.loads((args.grounding / 'R1/RESULT.json').read_text(encoding='utf-8'))
        assert old['selection'] == provenance['R1']['selection']
        frame = pd.read_csv(args.grounding / 'R1/per_sentence.csv')
        assert frame.row.tolist() == rows.row.tolist() and frame.valid_negative.all()
        assert ((frame.off_mismatch - frame.off_match - frame.g_off).abs() < 1e-10).all()
        rows['baseline_s42_native_source_gap'] = frame.g_off
        grounding_provenance = dict(selection=old['selection'], csv_sha256=sha(args.grounding / 'R1/per_sentence.csv'))
    groups = {}
    for name, mask in [('all', rows.row >= 0), ('bottom_all_three', rows.bottom_quartile_seeds == 3),
                       ('bottom_at_least_two', rows.bottom_quartile_seeds >= 2)]:
        group = rows[mask]
        groups[name] = dict(rows=len(group), identical_baseline_predictions=int(group.baseline_all_same_prediction.sum()),
                           C0_gain_all_seeds=int(group.C0_gain_all_seeds.sum()),
                           C0_loss_all_seeds=int(group.C0_loss_all_seeds.sum()),
                           C0_mean_delta_sentence_chrf0=float(group.C0_mean_delta_sentence_chrf0.mean()))
        if grounding_provenance:
            gap = group.baseline_s42_native_source_gap
            groups[name]['baseline_s42_positive_native_source_gap'] = int((gap > 0).sum())
            groups[name]['baseline_s42_mean_native_source_gap'] = float(gap.mean())
    result = dict(status='VERIFIED_CPU_VALIDATION_ONLY', reference_sha256=sha(args.references),
                  script_sha256=sha(Path(__file__)), provenance=provenance, corpus_scores=corpus,
                  groups=groups, bottom_quartile_size=n // 4, optimizer_steps=0,
                  grounding_provenance=grounding_provenance,
                  interpretation='Exploratory reference-relative overlap, not human error labels. Baseline-selected strata favor regression to the mean; no causal C0 repair claim or significance test.')
    args.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output / 'ROWS.csv', index=False)
    rows[rows.bottom_quartile_seeds == 3].sort_values(['baseline_mean_sentence_chrf0', 'row']).head(20).to_csv(args.output / 'REVIEW_EXAMPLES.csv', index=False)
    (args.output / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(corpus_scores=corpus, groups=groups), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
