"""Locked validation interventions and frozen-control beam diagnostics, no training."""
import csv
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from amis_rewire.evaluation import generation_precision
from amis_rewire.modeling import _find_stack_layers
from amis_v4_measurements import digest, load_checkpoint, strict_model, runtime
from diagnostic_routing import DiagnosticRouting

OUT = ROOT / 'outputs_rebuttal/amis_routing_diagnostics_20261004_v5'
ALL = frozenset(range(3, 13))


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def scores(preds, refs):
    return {'bleu': sacrebleu.corpus_bleu(preds, [refs], tokenize='zh').score,
            'chrf++': sacrebleu.corpus_chrf(preds, [refs], word_order=2).score,
            'chrf0': sacrebleu.corpus_chrf(preds, [refs], word_order=0).score}


def gold_rows(path, data, expected_hash):
    assert digest(path) == expected_hash, f'Gold SHA mismatch: {path}'
    rows = read(path)
    assert len(rows) == len(data)
    assert all(int(r['index']) == i and r['source'] == d['source']
               and r['target'] == d['target'].strip() for i, (r, d) in enumerate(zip(rows, data)))
    return rows


def teacher_ce(model, tokenizer, rows, indices):
    total = tokens = 0
    for batch in indices:
        chunk = [rows[i] for i in batch]
        inputs = tokenizer([r['source'] for r in chunk], padding=True, truncation=True,
                           max_length=256, pad_to_multiple_of=8, return_tensors='pt')
        labels = tokenizer(text_target=[r['target'] for r in chunk], padding=True,
                           truncation=True, max_length=256, return_tensors='pt')['input_ids'].cuda()
        labels[labels == tokenizer.pad_token_id] = -100
        inputs = {k: v.cuda() for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
        with torch.no_grad(), generation_precision(model, 'fp32'):
            output = model(**inputs, labels=labels)
        count = int(labels.ne(-100).sum())
        total += float(output.loss) * count
        tokens += count
        del output
    return total / tokens


def generate(model, tokenizer, rows, beams, expected=None, progress_key=''):
    preds = []
    for start in range(0, len(rows), 8):
        chunk = rows[start:start + 8]
        inputs = tokenizer([r['source'] for r in chunk], padding=True, truncation=True,
                           max_length=256, pad_to_multiple_of=8, return_tensors='pt')
        inputs = {k: v.cuda() for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
        with torch.no_grad(), generation_precision(model, 'fp32'):
            ids = model.generate(**inputs, num_beams=beams, max_length=256, forced_bos_token_id=250025)
        decoded = [p.strip() for p in tokenizer.batch_decode(ids, skip_special_tokens=True)]
        if expected is not None:
            assert decoded == expected[start:start + len(decoded)], f'Replay mismatch {progress_key} rows {start}'
        preds.extend(decoded)
        if start % 128 == 0:
            save(OUT / 'status.json', {'condition': progress_key, 'rows_done': len(preds), 'rows_total': len(rows)})
    return preds


def condition(model, tokenizer, entry, key, split, beams, alpha, active, data, fixed, expected=None):
    if split == 'test':
        assert key == 'test_main_b1' and beams == 1 and alpha == entry['alpha']
        assert frozenset(active) == (ALL if entry['alpha'] else frozenset())
    folder = OUT / 'runs' / entry['id'] / key
    assert not folder.exists(), f'Do not silently reuse condition: {folder}'
    folder.mkdir(parents=True)
    if isinstance(model, DiagnosticRouting):
        model.strength = alpha
        model.active_layers = frozenset(active)
    print('START', entry['id'], key, flush=True)
    started = time.time()
    rows = data[split]
    preds = generate(model, tokenizer, rows, beams, expected, entry['id'] + '/' + key)
    record = {'id': entry['id'], 'family': entry['family'], 'seed': entry['seed'],
              'condition': key, 'split': split, 'beams': beams, 'train_alpha': entry['alpha'],
              'inference_alpha': alpha, 'active_routing_layers': sorted(active),
              'scores': scores(preds, [r['target'].strip() for r in rows]), 'rows': len(rows),
              'validation_replay': expected is not None, 'seconds': time.time() - started}
    if split == 'validation' and beams == 1:
        record['teacher_ce_fixed32'] = teacher_ce(model, tokenizer, rows, fixed)
    if key == 'val_main_b1':
        record['inventory_metric_delta'] = {'bleu': record['scores']['bleu'] - entry['eval_bleu'],
            'chrf++': record['scores']['chrf++'] - entry['eval_chrf++']}
        assert all(abs(v) < 1e-6 for v in record['inventory_metric_delta'].values()), record
    path = folder / 'predictions.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'source', 'target', 'prediction'])
        writer.writeheader()
        writer.writerows({'index': i, 'source': r['source'], 'target': r['target'].strip(), 'prediction': p}
                         for i, (r, p) in enumerate(zip(rows, preds)))
    record['prediction_sha256'] = digest(path)
    save(folder / 'record.json', record)
    print('DONE', entry['id'], key, json.dumps(record['scores']), flush=True)
    return record


def main():
    started = time.time()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    plan = json.loads((ROOT / 'diagnostic_plan.json').read_text())
    save(OUT / 'locked_plan.json', plan)
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt',
                    revision='e30b6cb8eb0d43a0b73cab73c7676b9863223a30', src_lang='tl_XX', tgt_lang='zh_CN')
    metadata = runtime(tokenizer, ROOT)
    metadata['sacrebleu'] = sacrebleu.__version__
    metadata['test_sha256'] = digest(ROOT / 'data_processed/amis_mandarin/test.csv')
    metadata['plan_sha256'] = digest(ROOT / 'diagnostic_plan.json')
    save(OUT / 'runtime.json', metadata)
    data = {split: read(ROOT / 'data_processed/amis_mandarin' / (split + '.csv')) for split in ['validation', 'test']}
    assert len(data['validation']) == 576 and len(data['test']) == 575
    fixed = json.loads((ROOT / 'trajectory_summary.json').read_text())['fixed_validation_indices']
    inventory = json.loads((ROOT / 'validation_inventory_21.json').read_text())
    selected = [m for m in inventory if m['family'] in plan['families']]
    # Start with all three Fulls; alpha/group decisions only use validation.
    selected.sort(key=lambda m: (plan['families'].index(m['family']), m['seed']))
    assert len(selected) == 12
    results = []
    for entry in selected:
        extract = ROOT / 'scratch/extract' / entry['id']
        ckpt = load_checkpoint(entry, extract, OUT / 'loading_receipts')
        raw = strict_model(ckpt).eval()
        model = DiagnosticRouting(raw).eval() if entry['alpha'] > 0 else raw
        conditions = [('val_main_b1', 'validation', 1, .1 if entry['alpha'] else 0, ALL if entry['alpha'] else [])]
        if entry['family'] == 'original_full_detached':
            expected_rows = gold_rows(ROOT / 'gold_validation' / (entry['id'] + '.csv'),
                data['validation'], plan['gold_validation_sha256'][entry['id']])
            expected = [r['prediction'] for r in expected_rows]
            for alpha in plan['alpha_values']:
                if alpha != .1:
                    conditions.append((f'val_alpha_{round(alpha * 1000):04d}_b1', 'validation', 1, alpha, ALL))
            for group, layers in plan['disabled_groups'].items():
                conditions.append((f'val_no_{group}_b1', 'validation', 1, .1, ALL - frozenset(layers)))
        else:
            expected = None
        if entry['family'] in ['original_full_detached', 'tuned_lsr']:
            # Eight test beam4 rows are a replay guard, not parameter search.
            gold = gold_rows(ROOT / 'gold_test' / (entry['id'] + '.csv'), data['test'],
                plan['gold_test_sha256'][entry['id']])
            if isinstance(model, DiagnosticRouting):
                model.strength, model.active_layers = .1, ALL
            generate(model, tokenizer, data['test'][:8], 4, [r['prediction'] for r in gold[:8]], entry['id'] + '/test_b4_sentinel')
            save(OUT / 'sentinels' / (entry['id'] + '.json'), {'status': 'PASS', 'split': 'test',
                 'beams': 4, 'rows': 8, 'gold_sha256': digest(ROOT / 'gold_test' / (entry['id'] + '.csv'))})
            conditions += [('val_main_b4', 'validation', 4, entry['alpha'], ALL if entry['alpha'] else []),
                           ('test_main_b1', 'test', 1, entry['alpha'], ALL if entry['alpha'] else [])]
        for key, split, beams, alpha, active in conditions:
            results.append(condition(model, tokenizer, entry, key, split, beams, alpha, active, data, fixed,
                                     expected if key == 'val_main_b1' else None))
        if isinstance(model, DiagnosticRouting):
            for handle in model._handles:
                handle.remove()
            model._handles.clear()
        del model, raw
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(extract)
        save(OUT / 'summary.json', results)
    assert len(results) == 45
    save(OUT / 'COMPLETE.json', {'status': 'GPU_COMPLETE', 'conditions': len(results), 'models': len(selected),
                              'training_runs': 0, 'elapsed_seconds': time.time() - started})
    print('V5 GPU COMPLETE', len(results), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(OUT / 'FAILED.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
