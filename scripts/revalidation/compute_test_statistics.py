"""Compute final test checkpoint evaluation statistics, paired deltas, bootstrap CIs, and Holm-corrected p-values.

Complies strictly with Section 11 of docs/operations/CONFIGURATION_SEARCH_HANDOFF.md.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest, _compute_p_value

ROOT = Path(__file__).resolve().parents[2]
FINAL_OUT = ROOT / 'outputs_rebuttal/amis_final_evaluation_20261004'
CONF_OUT = ROOT / 'outputs_rebuttal/amis_confirmation_20261003'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def holm_bonferroni(values):
    """Computes Holm-Bonferroni adjusted p-values."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    adjusted = np.zeros(n)
    running = 0.0
    for rank, idx in enumerate(np.argsort(values)):
        multiplier = n - rank
        adj = min(1.0, multiplier * values[idx])
        running = max(running, adj)
        adjusted[idx] = running
    return adjusted


def main():
    test_csv = ROOT / 'data_processed/amis_mandarin/test.csv'
    test_rows = read_csv(test_csv)
    assert len(test_rows) == 575, f"Expected 575 test rows, got {len(test_rows)}"
    references = [r['target'].strip() for r in test_rows]

    # Metrics
    bleu_metric = BLEU(tokenize='zh')
    chrf_pp_metric = CHRF(word_order=2)
    chrf0_metric = CHRF(word_order=0)
    metrics = {
        'BLEU': bleu_metric,
        'chrF++': chrf_pp_metric,
        'chrF0': chrf0_metric,
    }

    # Model definitions
    # 6 families:
    # 1. baseline (C seeds 42, 43, 44)
    # 2. original_full_detached (Replica 42, C 43, C 44)
    # 3. matched_lsr (Replica 42, C 43, C 44)
    # 4. tuned_lsr (New 42, 43, 44)
    # 5. full_source_attached (New 42, 43, 44)
    # 6. clrr_only_detached (C 42, 43, 44)
    model_configs = [
        # Baseline
        {
            'family': 'baseline',
            'display_name': 'Baseline (mBART)',
            'seed': 42,
            'path': CONF_OUT / 'runs/mbart-amis-baseline-seed42/test_predictions.csv',
            'provenance': 'Phase C confirmation (reused)',
        },
        {
            'family': 'baseline',
            'display_name': 'Baseline (mBART)',
            'seed': 43,
            'path': CONF_OUT / 'runs/mbart-amis-baseline-seed43/test_predictions.csv',
            'provenance': 'Phase C confirmation (reused)',
        },
        {
            'family': 'baseline',
            'display_name': 'Baseline (mBART)',
            'seed': 44,
            'path': CONF_OUT / 'runs/mbart-amis-baseline-seed44/test_predictions.csv',
            'provenance': 'Phase C confirmation (reused)',
        },

        # Original Full Detached (Selected Full)
        {
            'family': 'original_full_detached',
            'display_name': 'Full CLRR Detached (alpha=0.10, lambda=0.10)',
            'seed': 42,
            'path': FINAL_OUT / 'runs/mbart-amis-full_a010_l010-seed42-replica/test_predictions.csv',
            'provenance': 'New verified replica evaluation',
        },
        {
            'family': 'original_full_detached',
            'display_name': 'Full CLRR Detached (alpha=0.10, lambda=0.10)',
            'seed': 43,
            'path': CONF_OUT / 'runs/mbart-amis-full-seed43/test_predictions.csv',
            'provenance': 'Phase C confirmation (reused)',
        },
        {
            'family': 'original_full_detached',
            'display_name': 'Full CLRR Detached (alpha=0.10, lambda=0.10)',
            'seed': 44,
            'path': CONF_OUT / 'runs/mbart-amis-full-seed44/test_predictions.csv',
            'provenance': 'Phase C confirmation (reused)',
        },

        # Matched LSR (lambda=0.10)
        {
            'family': 'matched_lsr',
            'display_name': 'Matched LSR (lambda=0.10)',
            'seed': 42,
            'path': FINAL_OUT / 'runs/mbart-amis-lsr_l010-seed42/test_predictions.csv',
            'provenance': 'New verified evaluation',
        },
        {
            'family': 'matched_lsr',
            'display_name': 'Matched LSR (lambda=0.10)',
            'seed': 43,
            'path': CONF_OUT / 'runs/mbart-amis-lsr-seed43/test_predictions.csv',
            'provenance': 'Phase C confirmation (reused)',
        },
        {
            'family': 'matched_lsr',
            'display_name': 'Matched LSR (lambda=0.10)',
            'seed': 44,
            'path': CONF_OUT / 'runs/mbart-amis-lsr-seed44/test_predictions.csv',
            'provenance': 'Phase C confirmation (reused)',
        },

        # Tuned LSR (lambda=0.03)
        {
            'family': 'tuned_lsr',
            'display_name': 'Tuned LSR (lambda=0.03)',
            'seed': 42,
            'path': FINAL_OUT / 'runs/mbart-amis-lsr_l003-seed42/test_predictions.csv',
            'provenance': 'New verified evaluation (from search)',
        },
        {
            'family': 'tuned_lsr',
            'display_name': 'Tuned LSR (lambda=0.03)',
            'seed': 43,
            'path': FINAL_OUT / 'runs/mbart-amis-lsr_l003-seed43/test_predictions.csv',
            'provenance': 'New verified evaluation (from confirmation)',
        },
        {
            'family': 'tuned_lsr',
            'display_name': 'Tuned LSR (lambda=0.03)',
            'seed': 44,
            'path': FINAL_OUT / 'runs/mbart-amis-lsr_l003-seed44/test_predictions.csv',
            'provenance': 'New verified evaluation (from confirmation)',
        },

        # Full Source Attached (alpha=0.10, lambda=0.10, detach=False)
        {
            'family': 'full_source_attached',
            'display_name': 'Full Attached Control (alpha=0.10, lambda=0.10, detach=False)',
            'seed': 42,
            'path': FINAL_OUT / 'runs/mbart-amis-full-attached-a010-l010-seed42/test_predictions.csv',
            'provenance': 'Phase C2 newly trained attached control',
        },
        {
            'family': 'full_source_attached',
            'display_name': 'Full Attached Control (alpha=0.10, lambda=0.10, detach=False)',
            'seed': 43,
            'path': FINAL_OUT / 'runs/mbart-amis-full-attached-a010-l010-seed43/test_predictions.csv',
            'provenance': 'Phase C2 newly trained attached control',
        },
        {
            'family': 'full_source_attached',
            'display_name': 'Full Attached Control (alpha=0.10, lambda=0.10, detach=False)',
            'seed': 44,
            'path': FINAL_OUT / 'runs/mbart-amis-full-attached-a010-l010-seed44/test_predictions.csv',
            'provenance': 'Phase C2 newly trained attached control',
        },

        # Detached CLRR-only (alpha=0.10, lambda=0.00)
        {
            'family': 'clrr_only_detached',
            'display_name': 'CLRR-only Detached (alpha=0.10, lambda=0.00)',
            'seed': 42,
            'path': CONF_OUT / 'runs/mbart-amis-clrr-seed42/test_predictions.csv',
            'provenance': 'Phase C confirmation (reused)',
        },
        {
            'family': 'clrr_only_detached',
            'display_name': 'CLRR-only Detached (alpha=0.10, lambda=0.00)',
            'seed': 43,
            'path': CONF_OUT / 'runs/mbart-amis-clrr-seed43/test_predictions.csv',
            'provenance': 'Phase C confirmation (reused)',
        },
        {
            'family': 'clrr_only_detached',
            'display_name': 'CLRR-only Detached (alpha=0.10, lambda=0.00)',
            'seed': 44,
            'path': CONF_OUT / 'runs/mbart-amis-clrr-seed44/test_predictions.csv',
            'provenance': 'Phase C confirmation (reused)',
        },
    ]

    all_scores = []
    systems = {}
    observed = {}
    boot = {}

    # Set up 10,000 bootstrap resamples with seed 42 (matching canonical benchmark)
    n_resamples = 10000
    indices = np.random.default_rng(42).integers(0, len(test_rows), size=(n_resamples, len(test_rows)))

    print(f"Extracting scores and corpus statistics for {len(model_configs)} models...")
    for cfg in model_configs:
        path = cfg['path']
        assert path.exists(), f"File missing: {path}"
        preds_data = read_csv(path)
        assert len(preds_data) == 575, f"Row count mismatch {len(preds_data)} in {path}"
        
        # Verify alignment
        for i, (pred_row, test_row) in enumerate(zip(preds_data, test_rows)):
            assert pred_row['source'].strip() == test_row['source'].strip(), f"Source mismatch at {i} in {path}"
            assert pred_row['target'].strip() == test_row['target'].strip(), f"Target mismatch at {i} in {path}"

        predictions = [r['prediction'].strip() for r in preds_data]
        key = (cfg['family'], cfg['seed'])
        systems[key] = predictions

        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

        score_row = {
            'family': cfg['family'],
            'display_name': cfg['display_name'],
            'seed': cfg['seed'],
            'provenance': cfg['provenance'],
            'prediction_file': str(path.relative_to(ROOT)).replace('\\', '/'),
            'prediction_sha256': sha256,
        }

        for m_name, m_func in metrics.items():
            corpus_score = m_func.corpus_score(predictions, [references]).score
            score_row[m_name] = corpus_score
            observed[(cfg['family'], cfg['seed'], m_name)] = corpus_score

            # Corpus-level bootstrap draws via pre-extracted statistics
            stats = np.array(m_func._extract_corpus_statistics(predictions, [references]), dtype='float32')
            draws = np.empty(n_resamples)
            chunk_size = 128
            for start in range(0, n_resamples, chunk_size):
                sums = stats[indices[start:start + chunk_size]].sum(axis=1)
                draws[start:start + len(sums)] = [m_func._compute_score_from_stats(x).score for x in sums]
            boot[(cfg['family'], cfg['seed'], m_name)] = draws

        all_scores.append(score_row)
        print(f"  {cfg['family']} (seed {cfg['seed']}): BLEU={score_row['BLEU']:.4f}, chrF++={score_row['chrF++']:.4f}, chrF0={score_row['chrF0']:.4f}")

    # 1. Save all individual run scores
    write_csv(FINAL_OUT / 'test_scores_all_runs.csv', all_scores)

    # 2. Compute Family Summaries (Mean ± Std, Min, Max)
    family_order = [
        'baseline',
        'original_full_detached',
        'matched_lsr',
        'tuned_lsr',
        'full_source_attached',
        'clrr_only_detached',
    ]
    family_summaries = []
    for fam in family_order:
        fam_cfgs = [c for c in model_configs if c['family'] == fam]
        disp_name = fam_cfgs[0]['display_name']
        for m_name in ['BLEU', 'chrF++', 'chrF0']:
            vals = np.array([observed[(fam, s, m_name)] for s in [42, 43, 44]])
            family_summaries.append({
                'family': fam,
                'display_name': disp_name,
                'metric': m_name,
                'seed42': vals[0],
                'seed43': vals[1],
                'seed44': vals[2],
                'mean': float(vals.mean()),
                'std_across_seeds': float(vals.std(ddof=1)),
                'min': float(vals.min()),
                'max': float(vals.max()),
                'n_seeds': 3,
            })
    write_csv(FINAL_OUT / 'test_family_summary.csv', family_summaries)

    # 3. Compute Paired Deltas across Seeds
    # Challenger is always Selected Full: original_full_detached
    comparisons = [
        ('baseline', 'Baseline (mBART)'),
        ('matched_lsr', 'Matched LSR (lambda=0.10)'),
        ('tuned_lsr', 'Tuned LSR (lambda=0.03)'),
        ('full_source_attached', 'Full Attached Control (alpha=0.10, lambda=0.10)'),
        ('clrr_only_detached', 'CLRR-only Detached (alpha=0.10, lambda=0.00)'),
    ]

    paired_deltas = []
    for ctrl, ctrl_disp in comparisons:
        for m_name in ['BLEU', 'chrF++', 'chrF0']:
            vals = [
                observed[('original_full_detached', s, m_name)] - observed[(ctrl, s, m_name)]
                for s in [42, 43, 44]
            ]
            paired_deltas.append({
                'challenger': 'original_full_detached',
                'control': ctrl,
                'control_display_name': ctrl_disp,
                'metric': m_name,
                'seed42_delta': vals[0],
                'seed43_delta': vals[1],
                'seed44_delta': vals[2],
                'mean_delta': float(np.mean(vals)),
                'std_paired_delta': float(np.std(vals, ddof=1)),
                'positive_seeds': sum(v > 0 for v in vals),
                'n_seeds': 3,
            })
    write_csv(FINAL_OUT / 'test_paired_deltas.csv', paired_deltas)

    # 4. Bootstrap Significance Testing
    # Primary Family (Holm 18):
    # 3 controls: baseline, matched_lsr, tuned_lsr
    # x 3 seeds (42, 43, 44) x 2 metrics (BLEU, chrF++) = 18 tests
    primary_controls = ['baseline', 'matched_lsr', 'tuned_lsr']
    primary_metrics = ['BLEU', 'chrF++']

    primary_tests = []
    for s in [42, 43, 44]:
        for ctrl in primary_controls:
            for m_name in primary_metrics:
                delta = observed[('original_full_detached', s, m_name)] - observed[(ctrl, s, m_name)]
                draws = boot[('original_full_detached', s, m_name)] - boot[(ctrl, s, m_name)]
                ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
                absolute = np.abs(draws)
                # Centered absolute-difference bootstrap p-value
                p_val = float(_compute_p_value(absolute - absolute.mean(), abs(delta)))
                primary_tests.append({
                    'seed': s,
                    'challenger': 'original_full_detached',
                    'control': ctrl,
                    'metric': m_name,
                    'delta': delta,
                    'ci_low': float(ci_low),
                    'ci_high': float(ci_high),
                    'p_value': p_val,
                })

    assert len(primary_tests) == 18, f"Expected 18 tests, got {len(primary_tests)}"
    primary_p_adj = holm_bonferroni([t['p_value'] for t in primary_tests])
    for t, p_adj in zip(primary_tests, primary_p_adj):
        t['p_holm18'] = float(p_adj)
        t['significant_holm18_05'] = (p_adj < 0.05)
    write_csv(FINAL_OUT / 'test_paired_bootstrap18.csv', primary_tests)

    # Secondary Family (Holm 30):
    # 5 controls: baseline, matched_lsr, tuned_lsr, full_source_attached, clrr_only_detached
    # x 3 seeds (42, 43, 44) x 2 metrics (BLEU, chrF++) = 30 tests
    secondary_controls = ['baseline', 'matched_lsr', 'tuned_lsr', 'full_source_attached', 'clrr_only_detached']
    secondary_tests = []
    for s in [42, 43, 44]:
        for ctrl in secondary_controls:
            for m_name in primary_metrics:
                delta = observed[('original_full_detached', s, m_name)] - observed[(ctrl, s, m_name)]
                draws = boot[('original_full_detached', s, m_name)] - boot[(ctrl, s, m_name)]
                ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
                absolute = np.abs(draws)
                p_val = float(_compute_p_value(absolute - absolute.mean(), abs(delta)))
                secondary_tests.append({
                    'seed': s,
                    'challenger': 'original_full_detached',
                    'control': ctrl,
                    'metric': m_name,
                    'delta': delta,
                    'ci_low': float(ci_low),
                    'ci_high': float(ci_high),
                    'p_value': p_val,
                })

    assert len(secondary_tests) == 30, f"Expected 30 tests, got {len(secondary_tests)}"
    secondary_p_adj = holm_bonferroni([t['p_value'] for t in secondary_tests])
    for t, p_adj in zip(secondary_tests, secondary_p_adj):
        t['p_holm30'] = float(p_adj)
        t['significant_holm30_05'] = (p_adj < 0.05)
    write_csv(FINAL_OUT / 'test_paired_bootstrap30.csv', secondary_tests)

    # 5. Native SacreBLEU Verification Check
    os.environ['SACREBLEU_SEED'] = '42'
    _, native = PairedTest(
        named_systems=[
            ('Baseline', systems[('baseline', 42)]),
            ('Full', systems[('original_full_detached', 42)]),
        ],
        metrics={'BLEU': bleu_metric},
        references=[references],
        test_type='bs',
        n_samples=n_resamples,
        n_jobs=1,
    )()
    native_result = next(v for k, v in native.items() if k != 'System')[1]
    ref_test = next(t for t in primary_tests if t['seed'] == 42 and t['control'] == 'baseline' and t['metric'] == 'BLEU')
    print(f"Native SacreBLEU p-value: {native_result.p_value}, Computed p-value: {ref_test['p_value']}")
    assert abs(native_result.p_value - ref_test['p_value']) < 1e-12, "SacreBLEU reference p-value check failed!"
    print("SacreBLEU reference check passed perfectly.")

    # 6. Build Final Test Decision / Provenance JSON
    full_vs_base_bleu = next(r for r in paired_deltas if r['control'] == 'baseline' and r['metric'] == 'BLEU')
    full_vs_matched_bleu = next(r for r in paired_deltas if r['control'] == 'matched_lsr' and r['metric'] == 'BLEU')
    full_vs_tuned_bleu = next(r for r in paired_deltas if r['control'] == 'tuned_lsr' and r['metric'] == 'BLEU')
    full_vs_attached_bleu = next(r for r in paired_deltas if r['control'] == 'full_source_attached' and r['metric'] == 'BLEU')

    full_vs_base_chrf = next(r for r in paired_deltas if r['control'] == 'baseline' and r['metric'] == 'chrF++')
    full_vs_matched_chrf = next(r for r in paired_deltas if r['control'] == 'matched_lsr' and r['metric'] == 'chrF++')
    full_vs_tuned_chrf = next(r for r in paired_deltas if r['control'] == 'tuned_lsr' and r['metric'] == 'chrF++')
    full_vs_attached_chrf = next(r for r in paired_deltas if r['control'] == 'full_source_attached' and r['metric'] == 'chrF++')

    decision = {
        'status': 'EVALUATION_AND_ANALYSIS_COMPLETE',
        'suite': 'amis_final_evaluation_20261004',
        'selected_architecture': {
            'method': 'original_full_detached',
            'alpha': 0.10,
            'lambda': 0.10,
            'source_detach': True,
            'target_lsr_detach': True,
            'rationale': 'Locked on validation: Source-attached control suffered representation degradation on seed 42 (-0.55 chrF++) and lower 3-seed validation mean (14.45 vs 14.60 chrF++). Under prespecified gate 11.4, original detached CLRR is retained.',
        },
        'test_matrix_summary': {
            'families_evaluated': 6,
            'models_total': 18,
            'seeds_per_family': 3,
            'test_rows': 575,
            'precision': 'FP32',
            'beam_size': 4,
            'tf32': False,
        },
        'key_paired_deltas': {
            'full_vs_baseline': {
                'BLEU_mean_delta': full_vs_base_bleu['mean_delta'],
                'BLEU_positive_seeds': f"{full_vs_base_bleu['positive_seeds']}/3",
                'chrFpp_mean_delta': full_vs_base_chrf['mean_delta'],
                'chrFpp_positive_seeds': f"{full_vs_base_chrf['positive_seeds']}/3",
            },
            'full_vs_matched_lsr': {
                'BLEU_mean_delta': full_vs_matched_bleu['mean_delta'],
                'BLEU_positive_seeds': f"{full_vs_matched_bleu['positive_seeds']}/3",
                'chrFpp_mean_delta': full_vs_matched_chrf['mean_delta'],
                'chrFpp_positive_seeds': f"{full_vs_matched_chrf['positive_seeds']}/3",
            },
            'full_vs_tuned_lsr': {
                'BLEU_mean_delta': full_vs_tuned_bleu['mean_delta'],
                'BLEU_positive_seeds': f"{full_vs_tuned_bleu['positive_seeds']}/3",
                'chrFpp_mean_delta': full_vs_tuned_chrf['mean_delta'],
                'chrFpp_positive_seeds': f"{full_vs_tuned_chrf['positive_seeds']}/3",
            },
            'full_vs_attached_control': {
                'BLEU_mean_delta': full_vs_attached_bleu['mean_delta'],
                'BLEU_positive_seeds': f"{full_vs_attached_bleu['positive_seeds']}/3",
                'chrFpp_mean_delta': full_vs_attached_chrf['mean_delta'],
                'chrFpp_positive_seeds': f"{full_vs_attached_chrf['positive_seeds']}/3",
            },
        },
        'statistical_testing': {
            'primary_family_tests': 18,
            'secondary_family_tests': 30,
            'bootstrap_resamples': n_resamples,
            'bootstrap_seed': 42,
            'sacrebleu_p_verified': True,
        },
        'limits_and_integrity': [
            'Zero test snooping: Selected architecture was strictly locked before test evaluation was initiated.',
            'Replica seed 42 was evaluated using freshly verified Colab weights with exact archive and prediction hashes.',
            'Sentence bootstrap measures sample uncertainty within test set, not between-seed training variance.',
            'Claims are bounded strictly to the 3 prespecified training seeds.',
        ],
    }

    decision_path = FINAL_OUT / 'final_test_decision.json'
    decision_path.write_text(json.dumps(decision, indent=2))
    print(f"Saved decision to {decision_path}")
    print("\n--- TEST STATISTICAL ANALYSIS COMPLETE ---")


if __name__ == '__main__':
    main()
