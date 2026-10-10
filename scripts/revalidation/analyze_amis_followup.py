"""Calculate the prespecified secondary 30 tests, bootstrap CIs, Holm-Bonferroni, and sensitivity diagnostics across 21 models.

Implements Section 14.4 (F2) of docs/operations/CONFIGURATION_SEARCH_HANDOFF.md.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest, _compute_p_value

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP_DIR = ROOT / 'outputs_rebuttal/amis_followup_20261004_v2'
FINAL_DIR = ROOT / 'outputs_rebuttal/amis_final_evaluation_20261004'
CONF_DIR = ROOT / 'outputs_rebuttal/amis_confirmation_20261003'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def holm_bonferroni(values):
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
    assert len(test_rows) == 575
    references = [r['target'].strip() for r in test_rows]

    # Metrics
    bleu_metric = BLEU(tokenize='zh')
    chrf_pp_metric = CHRF(word_order=2)
    chrf0_metric = CHRF(word_order=0)

    # Prediction file registry for all 21 models
    # 7 families x 3 seeds
    model_paths = {
        # 1. Baseline
        ('baseline', 42): CONF_DIR / 'runs/mbart-amis-baseline-seed42/test_predictions.csv',
        ('baseline', 43): CONF_DIR / 'runs/mbart-amis-baseline-seed43/test_predictions.csv',
        ('baseline', 44): CONF_DIR / 'runs/mbart-amis-baseline-seed44/test_predictions.csv',

        # 2. CLRR-only detached
        ('clrr_only_detached', 42): CONF_DIR / 'runs/mbart-amis-clrr-seed42/test_predictions.csv',
        ('clrr_only_detached', 43): CONF_DIR / 'runs/mbart-amis-clrr-seed43/test_predictions.csv',
        ('clrr_only_detached', 44): CONF_DIR / 'runs/mbart-amis-clrr-seed44/test_predictions.csv',

        # 3. Matched LSR (lambda=0.10)
        ('matched_lsr', 42): FINAL_DIR / 'runs/mbart-amis-lsr_l010-seed42/test_predictions.csv',
        ('matched_lsr', 43): CONF_DIR / 'runs/mbart-amis-lsr-seed43/test_predictions.csv',
        ('matched_lsr', 44): CONF_DIR / 'runs/mbart-amis-lsr-seed44/test_predictions.csv',

        # 4. Tuned LSR (lambda=0.03)
        ('tuned_lsr', 42): FINAL_DIR / 'runs/mbart-amis-lsr_l003-seed42/test_predictions.csv',
        ('tuned_lsr', 43): FINAL_DIR / 'runs/mbart-amis-lsr_l003-seed43/test_predictions.csv',
        ('tuned_lsr', 44): FINAL_DIR / 'runs/mbart-amis-lsr_l003-seed44/test_predictions.csv',

        # 5. Full detached (alpha=0.10, lambda=0.10)
        ('original_full_detached', 42): FINAL_DIR / 'runs/mbart-amis-full_a010_l010-seed42-replica/test_predictions.csv',
        ('original_full_detached', 43): CONF_DIR / 'runs/mbart-amis-full-seed43/test_predictions.csv',
        ('original_full_detached', 44): CONF_DIR / 'runs/mbart-amis-full-seed44/test_predictions.csv',

        # 6. Full source-attached (alpha=0.10, lambda=0.10, detach=False)
        ('full_source_attached', 42): FINAL_DIR / 'runs/mbart-amis-full-attached-a010-l010-seed42/test_predictions.csv',
        ('full_source_attached', 43): FINAL_DIR / 'runs/mbart-amis-full-attached-a010-l010-seed43/test_predictions.csv',
        ('full_source_attached', 44): FINAL_DIR / 'runs/mbart-amis-full-attached-a010-l010-seed44/test_predictions.csv',

        # 7. Full detached (alpha=0.10, lambda=0.03)
        ('full_detached_l003', 42): FOLLOWUP_DIR / 'supplemental_test/runs/mbart-amis-full_a010_l003-seed42/test_predictions.csv',
        ('full_detached_l003', 43): FOLLOWUP_DIR / 'supplemental_test/runs/mbart-amis-full_a010_l003-seed43/test_predictions.csv',
        ('full_detached_l003', 44): FOLLOWUP_DIR / 'supplemental_test/runs/mbart-amis-full_a010_l003-seed44/test_predictions.csv',
    }

    # Verify all files exist
    missing_paths = [p for p in model_paths.values() if not p.exists()]
    if missing_paths:
        print(f"Cannot run F2 yet, missing {len(missing_paths)} prediction files:")
        for mp in missing_paths:
            print(" ", mp)
        return

    systems = {}
    observed = {}
    boot = {}

    n_resamples = 10000
    indices = np.random.default_rng(42).integers(0, len(test_rows), size=(n_resamples, len(test_rows)))

    print("Auditing all 21 models and computing corpus bootstrap statistics...")
    for (fam, seed), path in model_paths.items():
        preds_data = read_csv(path)
        assert len(preds_data) == 575
        predictions = [r['prediction'].strip() for r in preds_data]
        systems[(fam, seed)] = predictions

        for m_name, m_func in [('BLEU', bleu_metric), ('chrF++', chrf_pp_metric), ('chrF0', chrf0_metric)]:
            score = m_func.corpus_score(predictions, [references]).score
            observed[(fam, seed, m_name)] = score

            if m_name in ['BLEU', 'chrF++']:
                stats = np.array(m_func._extract_corpus_statistics(predictions, [references]), dtype='float32')
                draws = np.empty(n_resamples)
                chunk_size = 128
                for start in range(0, n_resamples, chunk_size):
                    sums = stats[indices[start:start + chunk_size]].sum(axis=1)
                    draws[start:start + len(sums)] = [m_func._compute_score_from_stats(x).score for x in sums]
                boot[(fam, seed, m_name)] = draws

    # The 5 prespecified secondary comparisons:
    # 1. CLRR-only detached vs Baseline
    # 2. Full detached (alpha=0.10, lambda=0.10) vs CLRR-only detached
    # 3. Full detached (alpha=0.10, lambda=0.10) vs Full attached (alpha=0.10, lambda=0.10)
    # 4. Full detached (alpha=0.10, lambda=0.03) vs LSR-only (lambda=0.03)
    # 5. Full detached (alpha=0.10, lambda=0.10) vs Full detached (alpha=0.10, lambda=0.03)
    pairs = [
        ('clrr_only_detached', 'baseline', 'CLRR-only detached vs Baseline'),
        ('original_full_detached', 'clrr_only_detached', 'Full detached (lambda=0.10) vs CLRR-only detached'),
        ('original_full_detached', 'full_source_attached', 'Full detached (lambda=0.10) vs Full attached (lambda=0.10)'),
        ('full_detached_l003', 'tuned_lsr', 'Full detached (lambda=0.03) vs Tuned LSR (lambda=0.03)'),
        ('original_full_detached', 'full_detached_l003', 'Full detached (lambda=0.10) vs Full detached (lambda=0.03)'),
    ]

    tests = []
    for s in [42, 43, 44]:
        for ch, ctrl, desc in pairs:
            for m_name in ['BLEU', 'chrF++']:
                delta = observed[(ch, s, m_name)] - observed[(ctrl, s, m_name)]
                draws = boot[(ch, s, m_name)] - boot[(ctrl, s, m_name)]
                low, high = np.percentile(draws, [2.5, 97.5])
                absolute = np.abs(draws)
                p_val = float(_compute_p_value(absolute - absolute.mean(), abs(delta)))
                tests.append({
                    'seed': s,
                    'challenger': ch,
                    'control': ctrl,
                    'comparison_name': desc,
                    'metric': m_name,
                    'delta': delta,
                    'ci_low_unadjusted': float(low),
                    'ci_high_unadjusted': float(high),
                    'p_value': p_val,
                })

    assert len(tests) == 30, f"Expected 30 tests, got {len(tests)}"
    p_adj = holm_bonferroni([t['p_value'] for t in tests])
    for t, adj in zip(tests, p_adj):
        t['p_holm30'] = float(adj)
        t['significant_holm30_05'] = (adj < 0.05)

    out_p = FOLLOWUP_DIR / 'statistics/secondary_prespecified_holm30.csv'
    write_csv(out_p, tests)
    print(f"Saved 30 tests to {out_p}")

    # Paired deltas summary across seeds for these 5 pairs
    paired_summary = []
    for ch, ctrl, desc in pairs:
        for m_name in ['BLEU', 'chrF++', 'chrF0']:
            deltas = [observed[(ch, s, m_name)] - observed[(ctrl, s, m_name)] for s in [42, 43, 44]]
            paired_summary.append({
                'challenger': ch,
                'control': ctrl,
                'comparison': desc,
                'metric': m_name,
                'seed42': deltas[0],
                'seed43': deltas[1],
                'seed44': deltas[2],
                'mean_delta': float(np.mean(deltas)),
                'std_paired_delta': float(np.std(deltas, ddof=1)),
                'positive_seeds': sum(d > 0 for d in deltas),
                'n_seeds': 3,
            })
    write_csv(FOLLOWUP_DIR / 'statistics/secondary_paired_deltas.csv', paired_summary)

    # Sensitivity diagnostic: Leave-one-out and row 170
    metric_chrf = CHRF(word_order=2)
    ref_word_bigrams = [i for i, stat in enumerate(metric_chrf._extract_corpus_statistics(references, [references])) if stat[-2] > 0]
    sensitivity = {
        'reference_word_bigram_rows': ref_word_bigrams,
        'leave_one_out_row170': {},
    }
    for s in [42, 43, 44]:
        for ch, ctrl, desc in pairs:
            ch_hyps = systems[(ch, s)]
            ctrl_hyps = systems[(ctrl, s)]
            for excl in [None, 170]:
                idx_filter = [i for i in range(len(references)) if i != excl]
                r_sub = [references[i] for i in idx_filter]
                ch_sub = [ch_hyps[i] for i in idx_filter]
                ctrl_sub = [ctrl_hyps[i] for i in idx_filter]
                score_ch = sacrebleu.corpus_chrf(ch_sub, [r_sub], word_order=2).score
                score_ctrl = sacrebleu.corpus_chrf(ctrl_sub, [r_sub], word_order=2).score
                sensitivity['leave_one_out_row170'][f"{ch}_vs_{ctrl}_seed{s}_excl_{excl}"] = {
                    'delta_chrf++': score_ch - score_ctrl,
                    'challenger_chrf++': score_ch,
                    'control_chrf++': score_ctrl,
                }

    save_json(FOLLOWUP_DIR / 'statistics/sensitivity_diagnostic.json', sensitivity)
    print("F2 analysis and sensitivity diagnostics complete!")


if __name__ == '__main__':
    main()
