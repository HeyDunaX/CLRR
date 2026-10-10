"""Independent CPU audit; preserve follow-up inputs and write a separate receipt."""
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs_rebuttal/amis_followup_20261004_v2'

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

test = read(ROOT / 'data_processed/amis_mandarin/test.csv')
val = read(ROOT / 'data_processed/amis_mandarin/validation.csv')
refs = [r['target'] for r in test]
records = read(ROOT / 'outputs_rebuttal/amis_final_evaluation_20261004/test_scores_all_runs.csv')
for seed in (42, 43, 44):
    folder = OUT / 'supplemental_test/runs' / f'mbart-amis-full_a010_l003-seed{seed}'
    metrics = json.loads((folder / 'test_metrics.json').read_text())
    records.append({'family': 'full_detached_l003', 'seed': seed,
                    'prediction_file': str((folder / 'test_predictions.csv').relative_to(ROOT)),
                    'prediction_sha256': metrics['prediction_sha256'],
                    'BLEU': metrics['test_bleu'], 'chrF++': metrics['test_chrf++'],
                    'chrF0': metrics['test_chrf0']})
systems, scores, score_errors = {}, {}, []
for row in records:
    path = ROOT / row['prediction_file']
    predictions = read(path)
    assert len(predictions) == len(test) == 575
    assert sha(path) == row['prediction_sha256']
    assert all(p['source'] == t['source'] and p['target'] == t['target'] for p, t in zip(predictions, test))
    assert [int(p['index']) for p in predictions] == list(range(575))
    key = row['family'], int(row['seed'])
    hyps = [p['prediction'] for p in predictions]
    systems[key] = hyps
    result = {'BLEU': sacrebleu.corpus_bleu(hyps, [refs], tokenize='zh').score,
              'chrF++': sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score,
              'chrF0': sacrebleu.corpus_chrf(hyps, [refs], word_order=0).score}
    error = max(abs(result[m] - float(row[m])) for m in result)
    assert error < 1e-8
    score_errors.append(error)
    scores[key] = result

table = read(OUT / 'statistics/secondary_prespecified_holm30.csv')
pairs = {('clrr_only_detached', 'baseline'),
         ('original_full_detached', 'clrr_only_detached'),
         ('original_full_detached', 'full_source_attached'),
         ('full_detached_l003', 'tuned_lsr'),
         ('original_full_detached', 'full_detached_l003')}
assert len(table) == 30
assert {(r['challenger'], r['control']) for r in table} == pairs
assert len({(r['challenger'], r['control'], r['metric'], r['seed']) for r in table}) == 30
pv = [float(r['p_value']) for r in table]
adjusted = [0.] * len(pv)
running = 0.
for rank, idx in enumerate(sorted(range(len(pv)), key=lambda i: pv[i])):
    running = max(running, min(1., (len(pv) - rank) * pv[idx]))
    adjusted[idx] = running
for r, adj in zip(table, adjusted):
    assert abs(adj - float(r['p_holm30'])) < 1e-12
    delta = scores[r['challenger'], int(r['seed'])][r['metric']] - scores[r['control'], int(r['seed'])][r['metric']]
    assert abs(delta - float(r['delta'])) < 1e-8

os.environ['SACREBLEU_SEED'] = '42'
_, native = PairedTest(named_systems=[('control', systems['tuned_lsr', 42]),
                                     ('full003', systems['full_detached_l003', 42])],
                       metrics={'BLEU': BLEU(tokenize='zh'), 'chrF++': CHRF(word_order=2)},
                       references=[refs], test_type='bs', n_samples=10000, n_jobs=1)()
native_checks = {}
for metric in ('BLEU', 'chrF++'):
    native_key = metric if metric in native else next(k for k in native if k.startswith('chrF'))
    native_p = native[native_key][1].p_value
    row = next(r for r in table if r['seed'] == '42' and r['challenger'] == 'full_detached_l003' and r['metric'] == metric)
    error = abs(native_p - float(row['p_value']))
    assert error < 1e-12, (metric, native_p, row)
    native_checks[metric] = {'native_p': native_p, 'difference': error}

fixed = json.loads((ROOT / 'outputs_rebuttal/amis_confirmation_20261003/trajectory_summary.json').read_text())['fixed_validation_indices']
probe_checks = []
for folder in sorted((OUT / 'probes/runs').iterdir()):
    probe = json.loads((folder / 'probe_results.json').read_text())
    assert [r['indices'] for r in probe['gradient_batches']] == fixed
    assert len(probe['gradient_batches']) == 8
    for phase in ('source', 'target'):
        sp = probe[phase + '_spectrum']
        assert sp['rows'] == 576 and sp['maximum_centered_rank'] == 575
        assert sp['numerical_rank'] <= 575
    rs = read(folder / 'residual_per_example.csv')
    assert len(rs) == 576 * 12 * 2
    assert len({(r['phase'], int(r['row_index']), int(r['layer'])) for r in rs}) == len(rs)
    phases = {}
    for phase in ('source', 'target'):
        values = [float(r['residual_to_current_rms']) for r in rs if r['phase'] == phase and int(r['layer']) >= 3]
        cosines = [float(r['skip_current_cosine']) for r in rs if r['phase'] == phase and int(r['layer']) >= 3 and r['skip_current_cosine']]
        phases[phase] = {'mean': mean(values), 'max': max(values),
                         'skip_current_cosine_mean': mean(cosines) if cosines else None,
                         'skip_current_cosine_min': min(cosines) if cosines else None}
    probe_checks.append({'id': probe['id'], 'family': probe['family'],
                         'source_rank_singular': probe['source_spectrum']['effective_rank_singular_values'],
                         'source_rank_energy': probe['source_spectrum']['effective_rank_covariance_energy'],
                         'residual_ratios': phases})
assert len(probe_checks) == 21

route_files = []
for folder in sorted((OUT / 'route_off_validation/runs').iterdir()):
    on = read(folder / 'val_predictions_route_on.csv')
    off = read(folder / 'val_predictions_route_off.csv')
    assert len(on) == len(off) == 576
    assert all(p['source'] == r['source'] and p['target'] == r['target'] for rows in (on, off) for p, r in zip(rows, val))
    route_files.append({'id': folder.name, 'changed': sum(a['prediction'] != b['prediction'] for a, b in zip(on, off)),
                        'valid_intervention': False, 'reason': 'on/off wrappers share base_model; on hooks remain active during off'})

inventory = json.loads((OUT / 'validation_inventory_21.json').read_text())
summary = {
    'test_sets_verified': len(records), 'max_score_error': max(score_errors),
    'paired_deltas_and_holm30_verified': True,
    'native_p_check_full003_vs_tuned_seed42': native_checks,
    'significant_holm30_05': sum(p < .05 for p in adjusted),
    'probe_files_internal_consistency_count': len(probe_checks),
    'probe_reproduction_status': 'GPU measurements not independently rerun; missing raw spectra/representations and runtime/loading provenance',
    'probe_checks': probe_checks, 'route_measurements': route_files,
    'full003_minus_tuned_mean': {m: mean(scores['full_detached_l003', s][m] - scores['tuned_lsr', s][m] for s in (42, 43, 44)) for m in ('BLEU','chrF++','chrF0')},
    'audit_scope': 'CPU score/hash/order/statistics audit, source review, and separate CPU demonstration of F4 hook contamination. Original artifacts preserved.',
    'worker_source_sha256': sha(ROOT / 'scripts/revalidation/run_amis_followup_worker.py'),
    'inventory_review': {'validation_predictions_marked_present': sum(bool(m.get('validation_predictions_exists')) for m in inventory),
                         'missing_archive_sha': [m['id'] for m in inventory if not m.get('archive_sha256')],
                         'missing_revision_count': sum(not m.get('revision') for m in inventory)},
}
(OUT / 'independent_review_20261004.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ('probe_checks','route_measurements')}, indent=2))
