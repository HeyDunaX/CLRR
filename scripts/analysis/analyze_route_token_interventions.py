"""Independently replay route-token CSVs and descriptive sentence bootstrap."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--consensus', type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.root / 'results/PLAN.json')
    complete = read(args.root / 'results/COMPLETE.json')
    assert complete['status'] == 'COMPLETE' and complete['optimizer_steps'] == 0
    baseline = pd.read_csv(args.previous / 'R1/per_token.csv').set_index(['row', 'position'])
    assert sha(args.previous / 'R1/per_token.csv') == read(args.previous / 'R1/RESULT.json')['file_sha256']['per_token.csv']
    consensus = pd.read_csv(args.consensus / 'ROWS.csv')
    assert consensus.row.tolist() == list(range(plan['rows']))
    hard = set(consensus.loc[consensus.bottom_quartile_seeds == 3, 'row'])
    draws = np.random.default_rng(42).integers(0, plan['rows'], size=(10000, plan['rows']))
    summaries, provenance = [], {}
    correctness = {'baseline': baseline.correct[baseline.category.str.startswith('content')]}
    for ident in ('P1', 'R2'):
        folder = args.root / 'results' / ident
        receipt = read(folder / 'RESULT.json')
        assert receipt['selection'] == plan['previous']['selections'][ident]
        assert sha(folder / 'TOKENS.csv') == receipt['token_csv_sha256']
        frame = pd.read_csv(folder / 'TOKENS.csv')
        native = frame[frame.condition == 'native'].set_index(['row', 'position'])
        assert native.index.equals(baseline.index)
        old = pd.read_csv(args.previous / ident / 'per_token.csv').set_index(['row', 'position'])
        assert sha(args.previous / ident / 'per_token.csv') == read(args.previous / ident / 'RESULT.json')['file_sha256']['per_token.csv']
        assert native.correct.equals(old.correct)
        correctness[ident] = native.correct[native.category.str.startswith('content')]
        assert all((native[k] - old[k]).abs().max() < 1e-4 for k in ('nll', 'margin'))
        provenance[ident] = receipt['token_csv_sha256']
        for condition in plan['conditions']:
            part = frame[frame.condition == condition].set_index(['row', 'position'])
            assert part.index.equals(native.index)
            for subset in ('content', 'consensus_low_content', 'eos'):
                mask = native.category.eq('eos') if subset == 'eos' else native.category.str.startswith('content')
                if subset == 'consensus_low_content':
                    mask &= native.index.get_level_values('row').isin(hard)
                a, b = native[mask], part[mask]
                repairs = int((~a.correct & b.correct).sum())
                damages = int((a.correct & ~b.correct).sum())
                margin = float((b.margin - a.margin).mean())
                nll = float((b.nll - a.nll).mean())
                if subset in ('content', 'eos'):
                    expected = receipt['summary'][condition + '_' + subset]
                    assert repairs == expected['repairs'] and damages == expected['damages']
                    assert abs(margin - expected['mean_margin_delta']) < 1e-10
                    assert abs(nll - expected['mean_nll_delta']) < 1e-10
                net = (b.correct.astype(int) - a.correct.astype(int)).groupby(level='row').sum().reindex(range(plan['rows']), fill_value=0).to_numpy()
                counts = a.groupby(level='row').size().reindex(range(plan['rows']), fill_value=0).to_numpy()
                numerator, denominator = net[draws].sum(1), counts[draws].sum(1)
                valid = denominator > 0
                ci = np.percentile(100 * numerator[valid] / denominator[valid], [2.5, 97.5]).tolist()
                summaries.append(dict(id=ident, condition=condition, subset=subset, tokens=len(a),
                     repairs=repairs, damages=damages, net_correct=repairs-damages,
                     accuracy_delta_pp=100 * (repairs-damages)/len(a), ci95_pp=ci,
                     mean_margin_delta=margin, mean_nll_delta=nll))
    decisions = pd.DataFrame(correctness)
    patterns = decisions.groupby(list(decisions.columns)).size().reset_index(name='tokens')
    assert int(patterns.tokens.sum()) == len(decisions)
    union = decisions.any(axis=1)
    complementarity = dict(patterns=patterns.to_dict('records'), content_tokens=len(decisions),
        baseline_correct=int(decisions.baseline.sum()), reference_oracle_union_correct=int(union.sum()),
        baseline_wrong_rescued_by_at_least_one_route_model=int((~decisions.baseline & union).sum()),
        all_three_wrong=int((~union).sum()),
        caution='Gold-reference oracle is not an executable router or generated translation score; experts share one seed.')
    failed = baseline.loc[decisions.index[~union]]
    complementarity['all_wrong_baseline_gold_rank_bins'] = {
        '<=5': int((failed['rank'] <= 5).sum()),
        '6-50': int(((failed['rank'] > 5) & (failed['rank'] <= 50)).sum()),
        '51-500': int(((failed['rank'] > 50) & (failed['rank'] <= 500)).sum()),
        '>500': int((failed['rank'] > 500).sum())}
    assert sum(complementarity['all_wrong_baseline_gold_rank_bins'].values()) == len(failed)
    result = dict(status='INDEPENDENT_REPLAY_PASS', rows=plan['rows'], summaries=summaries,
                  complementarity=complementarity,
                  input_sha256=provenance, draws=10000, seed=42, resampling_unit='validation sentence',
                  interpretation='Exploratory unadjusted fixed-checkpoint intervals, not training-seed uncertainty or confirmatory significance.',
                  test_access=False, optimizer_steps=0)
    (args.root / 'ANALYSIS.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    lines = ['# Route-token validation interventions', '',
             'CPU independent replay PASS. Fixed checkpoints, one seed, gold prefixes; no training or test.', '',
             '| Model | Edges disabled | Repairs | Damages | Net correct | Accuracy delta (pp), descriptive CI95% | Mean margin delta |',
             '|---|---|---:|---:|---:|---|---:|']
    for record in summaries:
        if record['subset'] != 'content' or record['condition'] == 'native':
            continue
        lo, hi = record['ci95_pp']
        lines.append(f"| {record['id']} | {record['condition']} | {record['repairs']} | {record['damages']} | {record['net_correct']} | {record['accuracy_delta_pp']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {record['mean_margin_delta']:+.6f} |")
    lines += ['', 'Repairs/damages compare the intervention against native routing of the SAME trained weights.',
              'They do not compare separately trained models, identify source-token utility, or measure generated translation quality.',
              'The 10,000 draws resample the original 576 sentences, not new examples. Intervals are unadjusted exploratory summaries.',
              'Full row/category summaries and consensus-low exploratory slice are in ANALYSIS.json. Native replay and input SHA checks passed.',
              'No checkpoint, protocol, baseline score or test result was changed.']
    lines += ['', '## Complementarity ceiling, gold-reference dependent', '',
              f"Baseline correct: {complementarity['baseline_correct']}/{len(decisions)}. A gold-reference oracle choosing among baseline/CLRR/Full would correct {int(union.sum())}/{len(decisions)}.",
              f"Only {complementarity['baseline_wrong_rescued_by_at_least_one_route_model']} baseline-wrong positions are correct in at least one route model; {complementarity['all_three_wrong']} positions are wrong in all three.",
              'This does not supply a deployable selection rule. It does not upper-bound new training, other experts, or sentence adequacy. Never select experts using validation gold at inference.']
    lines += ['', 'Baseline gold-rank bins within the 2,402 shared wrong positions: ' +
              json.dumps(complementarity['all_wrong_baseline_gold_rank_bins']) + '.',
              'Strict rank counts ties as in the historical token audit. This describes gold-prefix decisions, not adequacy labels or an inference-time selector.']
    (args.root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
