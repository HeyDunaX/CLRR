"""Summarize locked V5 diagnostics; candidates are selected on validation only."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import csv
import json
from pathlib import Path
import numpy as np
from sacrebleu.metrics import BLEU, CHRF

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs_rebuttal/amis_routing_diagnostics_20261004_v5'
SEEDS = [42, 43, 44]
METRICS = ['bleu', 'chrf++', 'chrf0']


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def describe(vals):
    return {'mean': float(np.mean(vals)), 'sample_std': float(np.std(vals, ddof=1)),
            'seed_values': dict(zip(map(str, SEEDS), map(float, vals))), 'positive_seeds': sum(v > 0 for v in vals)}


def main():
    audit = json.loads((OUT / 'independent_audit.json').read_text())
    assert audit['core_status'] == 'VERIFIED42'
    assert not audit['errors'] and audit['baseline_archive_audit']['status'] == 'VERIFIED'
    rows = json.loads((OUT / 'summary.json').read_text())
    baseline = json.loads((OUT / 'baseline_archive_audit_v3/summary.json').read_text())
    rows += [{'family': 'baseline', 'seed': r['seed'], 'id': r['id'], 'condition': 'val_main_b1',
              'scores': r['conditions']['validation']['scores']} for r in baseline]
    lookup = {(r['family'], r['seed'], r['condition']): r for r in rows}
    report = {'scope': 'Exploratory frozen-checkpoint diagnostics. Old test remains unchanged; no train-time alpha claims.',
              'family_scores': {}, 'validation_interventions': {}, 'full_minus_tuned_by_decode': {}}
    for family in ['original_full_detached', 'tuned_lsr', 'matched_lsr', 'baseline']:
        for mode in (['val_main_b1', 'val_main_b4', 'test_main_b1'] if family in ['original_full_detached', 'tuned_lsr'] else ['val_main_b1']):
            report['family_scores'][family + '/' + mode] = {
                m: describe([lookup[family, seed, mode]['scores'][m] for seed in SEEDS]) for m in METRICS}
    for mode in ['val_main_b1', 'val_main_b4', 'test_main_b1']:
        report['full_minus_tuned_by_decode'][mode] = {m: describe([
            lookup['original_full_detached', s, mode]['scores'][m] - lookup['tuned_lsr', s, mode]['scores'][m]
            for s in SEEDS]) for m in METRICS}
    old = read(ROOT / 'outputs_rebuttal/amis_final_evaluation_20261004/test_scores_all_runs.csv')
    old_lookup = {(r['family'], int(r['seed'])): r for r in old}
    report['full_minus_tuned_by_decode']['test_main_b4_existing'] = {
        name: describe([float(old_lookup['original_full_detached', s][column]) - float(old_lookup['tuned_lsr', s][column])
                       for s in SEEDS]) for name, column in [('bleu', 'BLEU'), ('chrf++', 'chrF++'), ('chrf0', 'chrF0')]}
    report['ranking_sensitivity'] = {}
    for title, a, b in [('validation_beam4_minus_beam1', 'val_main_b4', 'val_main_b1'),
                        ('test_beam4_minus_beam1', 'test_main_b4_existing', 'test_main_b1'),
                        ('test_minus_validation_beam1', 'test_main_b1', 'val_main_b1'),
                        ('test_minus_validation_beam4', 'test_main_b4_existing', 'val_main_b4')]:
        report['ranking_sensitivity'][title] = {m: describe([
            report['full_minus_tuned_by_decode'][a][m]['seed_values'][str(s)]
            - report['full_minus_tuned_by_decode'][b][m]['seed_values'][str(s)] for s in SEEDS]) for m in METRICS}
    full = 'original_full_detached'
    modes = [r['condition'] for r in rows if r['family'] == full and r['seed'] == 42 and r['condition'].startswith(('val_alpha_', 'val_no_'))]
    for mode in modes:
        group = {m: describe([lookup[full, s, mode]['scores'][m] - lookup[full, s, 'val_main_b1']['scores'][m]
                             for s in SEEDS]) for m in METRICS}
        group['teacher_ce_fixed32_delta'] = describe([lookup[full, s, mode]['teacher_ce_fixed32'] - lookup[full, s, 'val_main_b1']['teacher_ce_fixed32'] for s in SEEDS])
        group['changed_rows'] = {}
        for s in SEEDS:
            entry = lookup[full, s, mode]
            candidate = read(OUT / 'runs' / entry['id'] / mode / 'predictions.csv')
            main = read(OUT / 'runs' / entry['id'] / 'val_main_b1/predictions.csv')
            group['changed_rows'][str(s)] = sum(a['prediction'] != b['prediction'] for a, b in zip(candidate, main))
        report['validation_interventions'][mode] = group
    # Plan's no-harm guard is at mean level; additionally report all paired seeds.
    eligible = [mode for mode in modes if mode.startswith('val_alpha_')
                and all(report['validation_interventions'][mode][m]['mean'] >= 0 for m in METRICS)]
    best = max(eligible, key=lambda mode: report['validation_interventions'][mode]['chrf++']['mean']) if eligible else 'val_main_b1'
    report['validation_only_alpha_choice'] = {'condition': best, 'train_alpha': .1,
        'inference_alpha': lookup[full, 42, best]['inference_alpha'],
        'eligible_conditions': eligible, 'confirmed_val_beam4': best == 'val_main_b1',
        'candidate_test_search_performed': False, 'selected_setting_has_existing_frozen_test_evaluation': best == 'val_main_b1',
        'rule': 'Highest mean chrF++ among alpha candidates with mean BLEU/chrF0/chrF++ no harm versus alpha .1; original is eligible.'}
    # Descriptive paired corpus intervals, conditional on the three frozen seeds.
    # All 7 interventions are retained. These intervals are unadjusted, do not
    # establish seed variability or correct optimism from selecting the winner.
    refs = [r['target'].strip() for r in read(ROOT / 'data_processed/amis_mandarin/validation.csv')]
    indices = np.random.default_rng(20261004).integers(0, len(refs), size=(5000, len(refs)))
    metrics = {'bleu': BLEU(tokenize='zh'), 'chrf++': CHRF(word_order=2), 'chrf0': CHRF(word_order=0)}
    boot = {}
    for s in SEEDS:
        for mode in ['val_main_b1'] + modes:
            entry = lookup[full, s, mode]
            preds = [r['prediction'] for r in read(OUT / 'runs' / entry['id'] / mode / 'predictions.csv')]
            for name, metric in metrics.items():
                stats = np.asarray(metric._extract_corpus_statistics(preds, [refs]), dtype=np.float64)
                assert abs(metric._compute_score_from_stats(stats.sum(0)).score - entry['scores'][name]) < 1e-8
                draws = np.empty(5000)
                for start in range(0, 5000, 128):
                    sums = stats[indices[start:start + 128]].sum(1)
                    draws[start:start + len(sums)] = [metric._compute_score_from_stats(v).score for v in sums]
                boot[s, mode, name] = draws
            print('BOOTSTRAPPED', s, mode, flush=True)
    report['validation_intervention_bootstrap'] = {'samples': 5000, 'seed': 20261004,
        'resampling': 'same validation sentence indices in both conditions and all three seeds; corpus scores averaged across fixed seeds',
        'limitations': 'Unadjusted descriptive intervals across 21 comparisons; not training-seed uncertainty, not evidence for a selected winner.', 'comparisons': {}}
    for mode in modes:
        report['validation_intervention_bootstrap']['comparisons'][mode] = {}
        for m in METRICS:
            delta = np.mean([boot[s, mode, m] - boot[s, 'val_main_b1', m] for s in SEEDS], axis=0)
            low, high = np.percentile(delta, [2.5, 97.5])
            report['validation_intervention_bootstrap']['comparisons'][mode][m] = {
                'mean_delta': report['validation_interventions'][mode][m]['mean'], 'ci95_low': float(low), 'ci95_high': float(high)}
    (OUT / 'diagnostic_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['validation_only_alpha_choice', 'full_minus_tuned_by_decode']}, indent=2))


if __name__ == '__main__':
    main()
