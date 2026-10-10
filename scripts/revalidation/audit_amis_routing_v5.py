"""Independent CPU audit of collected V5 artifacts, with fail-closed status."""
import csv
import hashlib
import json
import math
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs_rebuttal/amis_routing_diagnostics_20261004_v5'
ALL = set(range(3, 13))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def alignment(rows, gold, context):
    require(len(rows) == len(gold), context + ': row count')
    for i, (row, ref) in enumerate(zip(rows, gold)):
        require(int(row['index']) == i and row['source'] == ref['source']
                and row['target'] == ref['target'].strip(), context + f': raw alignment row {i}')


def expected_conditions(entry, plan):
    alpha = entry['alpha']
    active = ALL if alpha else set()
    result = {'val_main_b1': ('validation', 1, alpha, active)}
    if entry['family'] == 'original_full_detached':
        for value in plan['alpha_values']:
            if value != .1:
                result[f'val_alpha_{round(value * 1000):04d}_b1'] = ('validation', 1, value, ALL)
        for group, layers in plan['disabled_groups'].items():
            result[f'val_no_{group}_b1'] = ('validation', 1, .1, ALL - set(layers))
    if entry['family'] in ['original_full_detached', 'tuned_lsr']:
        result['val_main_b4'] = ('validation', 4, alpha, active)
        result['test_main_b1'] = ('test', 1, alpha, active)
    return result


def audit(report):
    plan = load(OUT / 'locked_plan_local.json')
    require(load(OUT / 'locked_plan.json') == plan, 'locked remote/local plan differ')
    require(plan['models'] == 12 and plan['full_generations'] == 45 and plan['training_runs'] == 0, 'plan totals')
    require(plan['alpha_values'] == [.05, .075, .1, .125, .15], 'predeclared alpha grid')
    require(plan['disabled_groups'] == {'early': [3, 4, 5], 'middle': [6, 7, 8], 'late': [9, 10, 11, 12]}, 'predeclared groups')
    payload = load(OUT / 'payload_receipt.json')
    runtime = load(OUT / 'runtime.json')
    require(runtime['plan_sha256'] == payload['plan_sha256'], 'payload/runtime plan SHA')
    for split in ['validation', 'test']:
        require(runtime[split + '_sha256'] == sha(ROOT / f'data_processed/amis_mandarin/{split}.csv'), split + ' data SHA')
    required_sources = ['scripts/revalidation/run_amis_routing_v5_worker.py', 'scripts/revalidation/diagnostic_routing.py',
                        'scripts/revalidation/amis_v4_measurements.py', 'src/amis_rewire/modeling.py', 'src/amis_rewire/evaluation.py']
    hashes = {key.replace('\\', '/'): value for key, value in runtime['source_hashes'].items()}
    require(set(required_sources) <= hashes.keys(), 'runtime missing required source hashes')
    for source, expected in hashes.items():
        require(sha(ROOT / source) == expected, 'source SHA: ' + source)
    require(runtime['src_lang'] == 'tl_XX' and runtime['tgt_lang'] == 'zh_CN'
            and runtime['source_prefix'] == 250047 and runtime['target_prefix'] == 250025, 'language/prefix')
    require(runtime['tokenizer_revision'] == 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30', 'tokenizer pinned revision')
    require(runtime['tf32_matmul'] is False and runtime['tf32_cudnn'] is False, 'TF32 disabled')
    require(runtime['sacrebleu'] == sacrebleu.__version__, 'SacreBLEU runtime/local version')
    inventory = load(ROOT / 'outputs_rebuttal/amis_followup_20261004_v2/validation_inventory_21.json')
    selected = [e for e in inventory if e['family'] in plan['families']]
    require(len(selected) == 12 and len({e['id'] for e in selected}) == 12, 'inventory selected models')
    require(all(sorted(e['seed'] for e in selected if e['family'] == f) == [42, 43, 44] for f in plan['families']), 'family seed coverage')
    selected = [e for e in selected if e['family'] != 'baseline']
    require(len(selected) == 9, 'nine core models')
    gold = {s: read(ROOT / f'data_processed/amis_mandarin/{s}.csv') for s in ['validation', 'test']}
    require(len(gold['validation']) == 576 and len(gold['test']) == 575, 'raw split lengths')
    designs = {e['id']: expected_conditions(e, plan) for e in selected}
    require(sum(map(len, designs.values())) == 42, 'expected core design count')
    paths = list((OUT / 'runs').glob('*/*/record.json'))
    csvs = list((OUT / 'runs').glob('*/*/predictions.csv'))
    expected_keys = {(model, key) for model, conditions in designs.items() for key in conditions}
    require({(p.parent.parent.name, p.parent.name) for p in paths} == expected_keys and len(paths) == 42, '42 exact core record conditions')
    require({(p.parent.parent.name, p.parent.name) for p in csvs} == expected_keys and len(csvs) == 42, '42 exact core CSV conditions')
    summary = load(OUT / 'summary.json')
    require(len(summary) == 42 and len({(r['id'], r['condition']) for r in summary}) == 42, 'core summary count/uniqueness')
    summary_map = {(r['id'], r['condition']): r for r in summary}
    test_scores = read(ROOT / 'outputs_rebuttal/amis_final_evaluation_20261004/test_scores_all_runs.csv')
    sentinel_ids = set()
    for entry in selected:
        model_id = entry['id']
        receipt = load(OUT / 'loading_receipts' / (model_id + '.json'))
        require(all(receipt[k] == v for k, v in entry.items()), 'checkpoint inventory receipt: ' + model_id)
        require(receipt['download_revision'] == entry['revision'] and receipt['verified_sha256'] == entry['archive_sha256']
                and receipt['verified_bytes'] == entry['archive_bytes'], 'pinned loading verification: ' + model_id)
        if entry['family'] in ['original_full_detached', 'tuned_lsr']:
            sentinel_ids.add(model_id)
            sentinel = load(OUT / 'sentinels' / (model_id + '.json'))
            require(sentinel['status'] == 'PASS' and sentinel['split'] == 'test' and sentinel['beams'] == 4 and sentinel['rows'] == 8, 'sentinel ' + model_id)
            original = [r for r in test_scores if r['family'] == entry['family'] and int(r['seed']) == entry['seed']]
            require(len(original) == 1, 'sentinel original lookup')
            original = original[0]
            require(sha(ROOT / original['prediction_file']) == original['prediction_sha256']
                    == sentinel['gold_sha256'] == plan['gold_test_sha256'][model_id], 'sentinel gold SHA')
            alignment(read(ROOT / original['prediction_file']), gold['test'], 'sentinel gold ' + model_id)
        for key, (split, beams, alpha, active) in designs[model_id].items():
            folder = OUT / 'runs' / model_id / key
            record = load(folder / 'record.json')
            require(record == summary_map[(model_id, key)], 'summary/record mismatch')
            require(record['id'] == model_id and record['family'] == entry['family'] and record['seed'] == entry['seed'], 'identity ' + key)
            require(record['split'] == split and record['beams'] == beams and record['train_alpha'] == entry['alpha']
                    and record['inference_alpha'] == alpha and record['active_routing_layers'] == sorted(active), 'design ' + key)
            if split == 'test':
                require(key == 'test_main_b1' and alpha == entry['alpha'], 'frozen test guard')
            rows = read(folder / 'predictions.csv')
            alignment(rows, gold[split], model_id + '/' + key)
            require(record['rows'] == len(rows) and sha(folder / 'predictions.csv') == record['prediction_sha256'], 'prediction count/SHA')
            predictions, refs = [r['prediction'] for r in rows], [r['target'] for r in rows]
            scores = {'bleu': sacrebleu.corpus_bleu(predictions, [refs], tokenize='zh').score,
                      'chrf++': sacrebleu.corpus_chrf(predictions, [refs], word_order=2).score,
                      'chrf0': sacrebleu.corpus_chrf(predictions, [refs], word_order=0).score}
            require(all(math.isfinite(record['scores'][k]) and abs(v - record['scores'][k]) < 1e-8 for k, v in scores.items()), 'native score mismatch ' + key)
            if split == 'validation' and beams == 1:
                require(math.isfinite(record['teacher_ce_fixed32']) and record['teacher_ce_fixed32'] >= 0, 'fixed32 CE')
            if key == 'val_main_b1':
                require(abs(scores['bleu'] - entry['eval_bleu']) < 1e-6 and abs(scores['chrf++'] - entry['eval_chrf++']) < 1e-6, 'inventory metrics ' + model_id)
                for metric, inventory_metric in [('bleu', 'eval_bleu'), ('chrf++', 'eval_chrf++')]:
                    require(abs(record['inventory_metric_delta'][metric] - (scores[metric] - entry[inventory_metric])) < 1e-8, 'inventory recorded delta')
                if entry['family'] == 'original_full_detached':
                    v4 = ROOT / 'outputs_rebuttal/amis_followup_20261004_v4/route_control' / model_id / 'route_predictions_comparison.csv'
                    baseline = read(v4)
                    alignment(baseline, gold['validation'], 'V4 reference')
                    require(predictions == [r['prediction_route_on'] for r in baseline], 'Full full576 replay ' + model_id)
                    require(record['validation_replay'] is True, 'replay flag')
                else:
                    require(record['validation_replay'] is False, 'unexpected replay flag')
            report['conditions_checked'].append({'id': model_id, 'condition': key, 'rows': len(rows), 'prediction_sha256': record['prediction_sha256'], 'recomputed_scores': scores})
    require({p.stem for p in (OUT / 'sentinels').glob('*.json')} == sentinel_ids and len(sentinel_ids) == 6, 'six exact sentinels')
    failure = load(OUT / 'FAILED.json')
    require(failure['type'] == 'AssertionError' and 'mbart-amis-baseline-seed42' in failure['message'], 'original baseline42 failure retained')
    require(not (OUT / 'COMPLETE.json').exists(), 'original45 completion must not be fabricated')
    report.update(status='INVENTORY_PROTOCOL_DISAGREEMENT', core_status='VERIFIED42', models=9, conditions=42, sentinels=6, full_validation_replays=3,
                  inventory_validation_agreements=9, original_worker_failure=failure,
                  original_baseline42_best_step='UNSUPPORTED: inventory hardcodes216; recovered metrics explicitly state original selected step unavailable.',
                  test_settings='frozen predeclared main only',
                  sentinel_evidence='Worker PASS receipts and original gold SHA/alignment; sentinel predictions are not separately stored.')


def audit_baselines(report):
    directory = OUT / 'baseline_archive_audit_v3'
    initial_failure = load(OUT / 'baseline_archive_audit/FAILED.json')
    require(initial_failure['type'] == 'ValueError' and 'too many values to unpack' in initial_failure['message'], 'initial baseline audit lookup failure retained')
    retry_failure = load(OUT / 'baseline_archive_audit_v2/FAILED.json')
    require(retry_failure['type'] == 'FileExistsError', 'v2 stale extraction failure retained')
    runtime = load(directory / 'runtime.json')
    for split in ['validation', 'test']:
        require(runtime[split + '_sha256'] == sha(ROOT / f'data_processed/amis_mandarin/{split}.csv'), 'baseline data SHA ' + split)
    require(runtime['src_lang'] == 'tl_XX' and runtime['tgt_lang'] == 'zh_CN'
            and runtime['source_prefix'] == 250047 and runtime['target_prefix'] == 250025, 'baseline language prefixes')
    require(runtime['tokenizer_revision'] == 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30'
            and runtime['tf32_matmul'] is False and runtime['tf32_cudnn'] is False, 'baseline revision/TF32')
    for source, expected in runtime['source_hashes'].items():
        require(sha(ROOT / source.replace('\\', '/')) == expected, 'baseline source SHA ' + source)
    inventory = load(ROOT / 'outputs_rebuttal/amis_followup_20261004_v2/validation_inventory_21.json')
    entries = [e for e in inventory if e['family'] == 'baseline']
    require(len(entries) == 3 and sorted(e['seed'] for e in entries) == [42, 43, 44], 'baseline coverage')
    gold = {s: read(ROOT / f'data_processed/amis_mandarin/{s}.csv') for s in ['validation', 'test']}
    old_scores = read(ROOT / 'outputs_rebuttal/amis_final_evaluation_20261004/test_scores_all_runs.csv')
    summary = load(directory / 'summary.json')
    require(len(summary) == 3 and {r['id'] for r in summary} == {e['id'] for e in entries}, 'baseline summary')
    details = []
    for entry in entries:
        folder = directory / entry['id']
        record = load(folder / 'record.json')
        require(record in summary and record['id'] == entry['id'] and record['seed'] == entry['seed'], 'baseline record identity')
        require(set(record['conditions']) == {'validation', 'test'}, 'baseline6 condition design')
        receipt = load(directory / 'loading_receipts' / (entry['id'] + '.json'))
        require(all(receipt[k] == v for k, v in entry.items()), 'baseline inventory receipt')
        require(receipt['verified_sha256'] == entry['archive_sha256'] and receipt['verified_bytes'] == entry['archive_bytes']
                and receipt['download_revision'] == entry['revision'], 'baseline pinned receipt')
        load(folder / 'checkpoint_metadata.json')
        for split, beams in [('validation', 1), ('test', 4)]:
            item = record['conditions'][split]
            path = folder / (split + '_predictions.csv')
            rows = read(path)
            alignment(rows, gold[split], 'baseline ' + split)
            require(item['rows'] == len(rows) and item['beams'] == beams and item['prediction_sha256'] == sha(path), 'baseline row/beam/hash')
            preds, refs = [r['prediction'] for r in rows], [r['target'] for r in rows]
            scores = {'bleu': sacrebleu.corpus_bleu(preds, [refs], tokenize='zh').score,
                      'chrf++': sacrebleu.corpus_chrf(preds, [refs], word_order=2).score,
                      'chrf0': sacrebleu.corpus_chrf(preds, [refs], word_order=0).score}
            require(all(abs(scores[k] - item['scores'][k]) < 1e-8 for k in scores), 'baseline native scores')
            if split == 'validation':
                require(item['repeat16_exact'] is True, 'baseline deterministic repeat receipt')
                for metric, old_key in [('bleu', 'eval_bleu'), ('chrf++', 'eval_chrf++')]:
                    require(abs(item['inventory_delta'][metric] - (scores[metric] - entry[old_key])) < 1e-8, 'baseline inventory delta')
            else:
                old, = [r for r in old_scores if r['family'] == 'baseline' and int(r['seed']) == entry['seed']]
                old_path = ROOT / old['prediction_file']
                require(sha(old_path) == old['prediction_sha256'] == item['gold_sha256'], 'baseline original test SHA')
                original = read(old_path)
                alignment(original, gold['test'], 'baseline old test')
                differences = [i for i, (a, b) in enumerate(zip(rows, original)) if a['prediction'] != b['prediction']]
                require(differences == item['different_predictions'], 'baseline exact difference indices')
                require(not differences, 'baseline archived weights must replay all575 oldtest predictions')
                for metric, col in [('bleu', 'BLEU'), ('chrf++', 'chrF++'), ('chrf0', 'chrF0')]:
                    require(abs(scores[metric] - float(old[col])) < 1e-8 and abs(item['old_test_metric_delta'][metric]) < 1e-8, 'baseline oldtest scores')
            details.append({'id': entry['id'], 'split': split, 'beams': beams, 'rows': len(rows), 'recomputed_scores': scores,
                            'prediction_sha256': sha(path), 'inventory_protocol': 'New explicit valbeam1 measurement; historical inventory untouched.'})
    require(len(list(directory.glob('*/validation_predictions.csv'))) == 3 and len(list(directory.glob('*/test_predictions.csv'))) == 3, 'baseline6 CSV counts')
    complete = load(directory / 'COMPLETE.json')
    require(complete['status'] == 'BASELINE_ARCHIVE_AUDIT_COMPLETE' and complete['models'] == 3
            and complete['full_generations'] == 6 and complete['original_failure_retained'] is True, 'baseline completion')
    require(not (directory / 'FAILED.json').exists(), 'baseline audit failure')
    report['baseline_archive_audit'] = {'status': 'VERIFIED', 'models': 3, 'conditions': 6, 'all575_test_replay_each_seed': True,
                                       'artifact_directory': 'baseline_archive_audit_v3', 'initial_lookup_failure_retained': initial_failure,
                                       'v2_stale_extraction_failure_retained': retry_failure,
                                       'conditions_checked': details, 'optional_baseline42_valbeam4': 'NOT_COLLECTED'}
    extension = OUT / 'baseline42_beam_confirmation'
    if extension.exists():
        audit_beam_confirmation(report, extension, entries)


def audit_beam_confirmation(report, directory, entries):
    entry, = [e for e in entries if e['seed'] == 42]
    runtime = load(directory / 'runtime.json')
    require(runtime['validation_sha256'] == sha(ROOT / 'data_processed/amis_mandarin/validation.csv'), 'beam confirmation data SHA')
    require(runtime['src_lang'] == 'tl_XX' and runtime['tgt_lang'] == 'zh_CN' and runtime['source_prefix'] == 250047
            and runtime['target_prefix'] == 250025 and runtime['tokenizer_revision'] == 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30', 'beam confirmation tokenizer')
    require(runtime['tf32_matmul'] is False and runtime['tf32_cudnn'] is False, 'beam confirmation TF32')
    for source, expected in runtime['source_hashes'].items():
        require(sha(ROOT / source.replace('\\', '/')) == expected, 'beam confirmation source SHA ' + source)
    receipt = load(directory / 'loading_receipts' / (entry['id'] + '.json'))
    require(all(receipt[k] == v for k, v in entry.items()) and receipt['verified_sha256'] == entry['archive_sha256']
            and receipt['verified_bytes'] == entry['archive_bytes'] and receipt['download_revision'] == entry['revision'], 'beam confirmation pinned checkpoint')
    record = load(directory / 'record.json')
    path = directory / 'validation_predictions_beam4.csv'
    rows = read(path)
    alignment(rows, read(ROOT / 'data_processed/amis_mandarin/validation.csv'), 'beam confirmation')
    require(record['id'] == entry['id'] and record['rows'] == 576 and record['beams'] == 4
            and record['prediction_sha256'] == sha(path), 'beam confirmation identity/hash')
    preds, refs = [r['prediction'] for r in rows], [r['target'] for r in rows]
    scores = {'bleu': sacrebleu.corpus_bleu(preds, [refs], tokenize='zh').score,
              'chrf++': sacrebleu.corpus_chrf(preds, [refs], word_order=2).score,
              'chrf0': sacrebleu.corpus_chrf(preds, [refs], word_order=0).score}
    require(all(abs(scores[k] - record['scores'][k]) < 1e-8 for k in scores), 'beam confirmation native scores')
    require(record['saved_training_args_selected_fields']['generation_num_beams'] == 4, 'saved training args beam4')
    for metric, old_key in [('bleu', 'eval_bleu'), ('chrf++', 'eval_chrf++')]:
        require(abs(scores[metric] - entry[old_key]) < 1e-6
                and abs(record['inventory_delta'][metric] - (scores[metric] - entry[old_key])) < 1e-8, 'beam4 inventory reproduction')
    require(load(directory / 'COMPLETE.json')['status'] == 'COMPLETE', 'beam confirmation completion')
    report['baseline_archive_audit']['optional_baseline42_valbeam4'] = {'status': 'VERIFIED_BEAM_PROTOCOL_MISMATCH',
        'rows': 576, 'recomputed_scores': scores, 'saved_training_args_selected_fields': record['saved_training_args_selected_fields'],
        'historical_inventory_validation_protocol': 'beam4, established by metric reproduction and saved arguments',
        'new_validation_control_protocol': 'beam1 from separate archive audit v3', 'best_step': 'unknown;216 unsupported'}


def main():
    report = {'status': 'NOT_VERIFIED', 'conditions_checked': [], 'audit_source_sha256': sha(Path(__file__)), 'errors': []}
    try:
        audit(report)
        audit_baselines(report)
    except Exception as exc:
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc)})
        report['status'] = 'NOT_VERIFIED'
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'independent_audit.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'conditions_checked': len(report['conditions_checked']), 'errors': report['errors'], 'artifact': str(path)}, indent=2))
    if report['errors'] or report.get('core_status') != 'VERIFIED42' or report.get('baseline_archive_audit', {}).get('status') != 'VERIFIED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
