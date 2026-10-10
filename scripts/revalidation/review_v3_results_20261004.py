"""Local audit only: preserve GPU outputs and write a separate review receipt."""
import ast
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import sacrebleu
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs_rebuttal/amis_followup_20261004_v2'
WORKER = ROOT / 'scripts/revalidation/run_amis_followup_v3_worker.py'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def metrics(predictions, refs):
    return {
        'bleu': sacrebleu.corpus_bleu(predictions, [refs], tokenize='zh').score,
        'chrf++': sacrebleu.corpus_chrf(predictions, [refs], word_order=2).score,
        'chrf0': sacrebleu.corpus_chrf(predictions, [refs], word_order=0).score,
    }


def main():
    rows = read_csv(ROOT / 'data_processed/amis_mandarin/validation.csv')
    refs = [r['target'].strip() for r in rows]
    inventory = load(BASE / 'validation_inventory_21.json')
    route = load(BASE / 'amis_route_control_corrected_20261004_v3/route_control_summary.json')
    review = {'scope': 'local CPU recomputation and source inspection; no checkpoint load or GPU rerun',
              'sacrebleu_version': sacrebleu.__version__, 'validation_rows': len(rows),
              'inventory': {'models': len(inventory)}}
    for field in ['archive_sha256', 'archive_bytes', 'revision', 'eval_bleu', 'eval_chrf++', 'eval_loss']:
        review['inventory'][field + '_present'] = sum(m.get(field) is not None and m.get(field) != '' for m in inventory)
    review['inventory']['validation_predictions_marked_present'] = sum(bool(m.get('validation_predictions_exists')) for m in inventory)
    review['route_checks'] = []
    for model_id, record in route.items():
        path = BASE / 'amis_route_control_corrected_20261004_v3' / model_id / 'route_predictions_comparison.csv'
        actual = read_csv(path)
        assert len(actual) == len(rows) == 576
        assert all(int(a['index']) == i and a['source'] == b['source'] and a['target'] == b['target'].strip()
                   for i, (a, b) in enumerate(zip(actual, rows)))
        on = [a['prediction_route_on'].strip() for a in actual]
        off = [a['prediction_route_off'].strip() for a in actual]
        old = read_csv(BASE / 'route_off_validation/runs' / model_id / 'val_predictions_route_on.csv')
        assert len(old) == len(rows)
        assert all(int(a['index']) == i and a['source'] == b['source'] and a['target'] == b['target'].strip()
                   for i, (a, b) in enumerate(zip(old, rows)))
        old_preds = [a['prediction'].strip() for a in old]
        computed = {'route_on': metrics(on, refs), 'route_off': metrics(off, refs)}
        error = max(abs(computed[side][metric] - record[side][metric]) for side in computed for metric in computed[side])
        changed = sum(a != b for a, b in zip(on, off))
        assert error < 1e-8 and changed == record['sentences_changed']
        previous = metrics(old_preds, refs)
        review['route_checks'].append({'id': model_id, 'seed': record['seed'], 'scores': computed,
                                      'native_score_max_error': error, 'on_off_changed': changed,
                                      'v2_valid_on_scores': previous,
                                      'v3_on_vs_v2_on_changed': sum(a != b for a, b in zip(on, old_preds)),
                                      'v3_on_minus_v2_on_bleu': computed['route_on']['bleu'] - previous['bleu'],
                                      'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    probes = load(BASE / 'probes_v3/probe_summary_v3.json')
    batches = [b for m in probes.values() for b in m['gradient_batches']]
    review['probes'] = {'models': len(probes), 'gradient_batches': len(batches),
                        'decoder_ce_norm_zero_batches': sum(b['groups']['decoder']['ce_gradient_norm'] == 0 for b in batches),
                        'models_with_zero_decoder_ce': [m['id'] for m in probes.values()
                                                       if any(b['groups']['decoder']['ce_gradient_norm'] == 0 for b in m['gradient_batches'])],
                        'numpy_representation_files': len(list((BASE / 'probes_v3').rglob('*.npz'))) + len(list((BASE / 'probes_v3').rglob('*.npy')))}
    source = WORKER.read_text(encoding='utf-8')
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_spectrum_metrics')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<spectrum function audit>', 'exec'), namespace)
    matrix = np.array([[3., 0.], [-3., 0.], [0., 1.], [0., -1.]])
    spectrum = namespace['compute_spectrum_metrics'](matrix)
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular ** 2 / (singular ** 2).sum()
    review['rank_formula_check'] = {'worker_covariance_effective_rank_field': spectrum['effective_rank_covariance_energy'],
                                  'actual_participation_ratio': float(1 / (energy ** 2).sum()),
                                  'entropy_effective_rank_covariance_energy': float(np.exp(-(energy * np.log(energy)).sum()))}
    review['worker_sha256'] = hashlib.sha256(WORKER.read_bytes()).hexdigest()
    with zipfile.ZipFile(ROOT / '.cache/frozen_sources/amis_followup_v3_bundle.zip') as bundle:
        name = next(n for n in bundle.namelist() if n.endswith(WORKER.name))
        packed = bundle.read(name)
        review['bundled_worker_sha256'] = hashlib.sha256(packed).hexdigest()
        review['bundled_worker_language_configuration_present'] = any(s in packed.decode() for s in ['src_lang=', 'tgt_lang=', '.src_lang =', '.tgt_lang ='])
    snapshot = Path('C:/Users/Lenovo/.cache/huggingface/hub/models--facebook--mbart-large-50-many-to-many-mmt/snapshots/e30b6cb8eb0d43a0b73cab73c7676b9863223a30')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    review['local_tokenizer_reproduction'] = {'class': type(tokenizer).__name__, 'default_src_lang': tokenizer.src_lang,
                                            'default_tgt_lang': tokenizer.tgt_lang}
    try:
        tokenizer(text_target=['target'])
        review['local_tokenizer_reproduction']['unconfigured_target_encoding'] = 'succeeded'
    except Exception as exc:
        review['local_tokenizer_reproduction']['unconfigured_target_encoding'] = type(exc).__name__ + ': ' + str(exc)
    configured = AutoTokenizer.from_pretrained(snapshot, src_lang='tl_XX', tgt_lang='zh_CN', local_files_only=True)
    review['local_tokenizer_reproduction']['configured_source_prefix'] = configured('source')['input_ids'][0]
    review['local_tokenizer_reproduction']['configured_target_prefix'] = configured(text_target='target')['input_ids'][0]
    review['conclusion'] = ('V3 CSV scores reproduce, but route-on fails previous validation replay; actual remote tokenizer/runtime provenance is missing. '
                            'Do not promote V3 route percentages or rank patterns as matched-protocol mechanism evidence. '
                            'Gradient accumulator uses intersection of non-None CE/LSR gradients; baseline and CLRR-only controls exclude decoder CE (48 batches). '
                            'Covariance rank field is participation ratio, not entropy effective rank. Previous verified test quality remains unchanged.')
    receipt = BASE / 'v3_independent_review_20261004.json'
    receipt.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(review, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
