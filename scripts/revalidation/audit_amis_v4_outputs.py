"""Independent CPU audit of downloaded V4 artifacts; never edits raw outputs."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '4'
import numpy as np
import sacrebleu

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs_rebuttal/amis_followup_20261004_v4'
sys.path.insert(0, str(ROOT / 'scripts/revalidation'))
from amis_v4_measurements import spectrum


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def score(preds, refs):
    return {'bleu': sacrebleu.corpus_bleu(preds, [refs], tokenize='zh').score,
            'chrf++': sacrebleu.corpus_chrf(preds, [refs], word_order=2).score,
            'chrf0': sacrebleu.corpus_chrf(preds, [refs], word_order=0).score}


def main():
    runtime = load(BASE / 'runtime.json')
    assert runtime['src_lang'] == 'tl_XX' and runtime['tgt_lang'] == 'zh_CN'
    assert runtime['source_prefix'] == 250047 and runtime['target_prefix'] == 250025
    assert runtime['tokenizer_revision'] == 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30'
    assert runtime['torch'].startswith('2.6.0') and runtime['transformers'] == '4.57.6'
    assert not runtime['tf32_matmul'] and not runtime['tf32_cudnn']
    for name, sha in runtime['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
    validation = rows(ROOT / 'data_processed/amis_mandarin/validation.csv')
    refs = [r['target'].strip() for r in validation]
    inv = {m['id']: m for m in load(ROOT / 'outputs_rebuttal/amis_followup_20261004_v2/validation_inventory_21.json')}
    fixed = load(ROOT / 'outputs_rebuttal/amis_confirmation_20261003/trajectory_summary.json')['fixed_validation_indices']
    route_checks = []
    for path in sorted((BASE / 'route_control').glob('*/route_record.json')):
        rec = load(path)
        csv_path = path.parent / 'route_predictions_comparison.csv'
        actual = rows(csv_path)
        assert len(actual) == len(validation) == 576
        assert all(int(a['index']) == i and a['source'] == b['source'] and a['target'] == b['target'].strip()
                   for i, (a, b) in enumerate(zip(actual, validation)))
        old = rows(ROOT / 'outputs_rebuttal/amis_followup_20261004_v2/route_off_validation/runs' / rec['id'] / 'val_predictions_route_on.csv')
        on = [a['prediction_route_on'].strip() for a in actual]
        off = [a['prediction_route_off'].strip() for a in actual]
        assert on == [a['prediction'].strip() for a in old]
        computed = {s: score(predictions, refs) for s, predictions in [('route_on', on), ('route_off', off)]}
        error = max(abs(computed[s][m] - rec[s][m]) for s in computed for m in computed[s])
        assert error < 1e-8
        assert rec['fixed_validation_indices'] == fixed
        assert rec['sentences_changed'] == sum(a != b for a, b in zip(on, off))
        assert rec['fixed_32_diagnostics']['alpha0_vs_clean_raw_max'] < 1e-6
        route_checks.append({'id': rec['id'], 'seed': rec['seed'], 'scores': computed,
                             'delta_bleu': computed['route_on']['bleu'] - computed['route_off']['bleu'],
                             'changed': rec['sentences_changed'], 'score_max_error': error})
    probe_checks = []
    for path in sorted((BASE / 'probes/runs').glob('*/probe_results.json')):
        rec = load(path)
        archive = path.parent / 'representations.npz'
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == rec['representations_sha256']
        receipt = load(BASE / 'loading_receipts' / (rec['id'] + '.json'))
        assert receipt['verified_sha256'] == inv[rec['id']]['archive_sha256']
        assert receipt['download_revision'] == inv[rec['id']]['revision']
        residual_rows = rows(path.parent / 'residuals.csv')
        assert len(residual_rows) == (11520 if rec['alpha'] > 0 else 0)
        if residual_rows:
            for phase in ['source', 'target']:
                for layer in range(3, 13):
                    indices = [int(r['index']) for r in residual_rows if r['phase'] == phase and int(r['layer']) == layer]
                    assert sorted(indices) == list(range(576))
            for r in residual_rows:
                expected = float(r['residual_rms']) / max(float(r['current_rms']), 1e-12)
                assert abs(expected - float(r['residual_to_current_rms'])) < 1e-6
        with np.load(archive) as arrays:
            assert len(arrays.files) == 26
            for name in arrays.files:
                assert arrays[name].shape == (576, 1024) and np.isfinite(arrays[name]).all()
            # Recompute both endpoint sites from raw matrices, not reported spectra.
            endpoint = {}
            for site in [12, 'final']:
                for phase in ['source', 'target']:
                    result = spectrum(arrays[f'{phase}_{site}'])
                    saved = rec['layer_spectra_and_alignment'][f'layer_{site}'][f'{phase}_full_spectrum']
                    for key in ['effective_rank_singular_values', 'effective_rank_covariance_energy', 'participation_ratio', 'top1_variance_fraction']:
                        assert abs(result[key] - saved[key]) < 1e-6
                    endpoint[f'{phase}_{site}_rank'] = result['effective_rank_singular_values']
        assert len(rec['gradient_batches']) == 8
        names = [g['name'] for g in rec['parameter_groups']]
        assert len(names) == len(set(names))
        assert sum(g['numel'] for g in rec['parameter_groups']) == 610879488
        for i, batch in enumerate(rec['gradient_batches']):
            assert batch['indices'] == fixed[i] and batch['encoder_requires_grad'] == [True, False]
            groups = batch['groups']; full = groups['all_parameters']
            assert groups['decoder']['ce_gradient_norm'] > 0
            for key in ['ce_norm_squared', 'lsr_norm_squared', 'dot_product']:
                total = sum(g[key] for name, g in groups.items() if name != 'all_parameters')
                assert abs(total - full[key]) < 1e-6 * max(1, abs(full[key]))
            for group in groups.values():
                a, b = group['ce_gradient_norm'], group['lsr_gradient_norm']
                assert abs(a ** 2 - group['ce_norm_squared']) < 1e-6 * max(1, a ** 2)
                assert abs(b ** 2 - group['lsr_norm_squared']) < 1e-6 * max(1, b ** 2)
                if a > 0 and b > 0:
                    assert abs(group['gradient_cosine'] - group['dot_product'] / (a * b)) < 1e-8
                else:
                    assert group['gradient_cosine'] is None
        probe_checks.append({'id': rec['id'], 'family': rec['family'], 'seed': rec['seed'], **endpoint,
                             'global_gradients': rec['average_gradient_by_layer_group']['all_parameters']})
        print('AUDITED', len(probe_checks), rec['id'], flush=True)
    complete = (BASE / 'COMPLETE_V4.json').exists() and len(route_checks) == 3 and len(probe_checks) == 21
    audit = {'status': 'VERIFIED' if complete else 'PARTIAL', 'scope': 'CPU rescore, exact replay, raw endpoint SVD, gradient sums and archive receipts',
             'runtime_and_code_hashes': 'PASS', 'strict_loading': 'Successful executed worker asserts zero missing/unexpected/mismatched/error keys',
             'route_checks': route_checks, 'probe_checks': probe_checks, 'route_count': len(route_checks), 'probe_count': len(probe_checks)}
    (BASE / 'independent_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': audit['status'], 'routes': len(route_checks), 'probes': len(probe_checks)}))


if __name__ == '__main__':
    main()
