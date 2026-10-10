"""Explicit recovery audit after the V5 baseline42 inventory guard stopped.

Keep original FAILED and all earlier records. Measure all baseline archives
separately, and compare complete test predictions instead of guessing provenance.
"""
import gc
import json
import shutil
import sys
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from amis_v4_measurements import load_checkpoint, strict_model, runtime, digest
from run_amis_routing_v5_worker import read, save, generate, scores

OUT = ROOT / 'outputs_rebuttal/amis_routing_diagnostics_20261004_v5'
AUDIT = OUT / 'baseline_archive_audit_v3'


def main():
    failure = json.loads((OUT / 'FAILED.json').read_text())
    assert failure['type'] == 'AssertionError' and 'mbart-amis-baseline-seed42' in failure['message']
    assert not AUDIT.exists(), 'No duplicate recovery worker'
    AUDIT.mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt',
        revision='e30b6cb8eb0d43a0b73cab73c7676b9863223a30', src_lang='tl_XX', tgt_lang='zh_CN')
    info = runtime(tokenizer, ROOT)
    info['test_sha256'] = digest(ROOT / 'data_processed/amis_mandarin/test.csv')
    save(AUDIT / 'runtime.json', info)
    inventory = json.loads((ROOT / 'validation_inventory_21.json').read_text())
    data = {s: read(ROOT / 'data_processed/amis_mandarin' / (s + '.csv')) for s in ['validation', 'test']}
    tests = [r for r in read(ROOT / 'baseline_gold_test_scores.csv') if r['family'] == 'baseline']
    results = []
    for entry in sorted([e for e in inventory if e['family'] == 'baseline'], key=lambda e: e['seed']):
        folder = AUDIT / entry['id']
        folder.mkdir()
        extract = ROOT / 'scratch/baseline_recovery_extract_v3' / entry['id']
        ckpt = load_checkpoint(entry, extract, AUDIT / 'loading_receipts')
        model = strict_model(ckpt).eval()
        files = {str(p.relative_to(ckpt)): digest(p) for p in ckpt.rglob('*') if p.is_file() and p.suffix in ['.json', '.txt']}
        metadata = {}
        for name in ['config.json', 'trainer_state.json', 'generation_config.json']:
            p = ckpt / name
            if p.exists():
                metadata[name] = json.loads(p.read_text())
        save(folder / 'checkpoint_metadata.json', {'checkpoint_relative_path': str(ckpt.relative_to(extract)), 'file_hashes': files, 'metadata': metadata})
        record = {'id': entry['id'], 'seed': entry['seed'], 'conditions': {}}
        for split, beams in [('validation', 1), ('test', 4)]:
            print('BASELINE_AUDIT_START', entry['id'], split, flush=True)
            rows = data[split]
            preds = generate(model, tokenizer, rows, beams, progress_key=entry['id'] + '/archive_audit_' + split)
            item = {'scores': scores(preds, [r['target'].strip() for r in rows]), 'rows': len(rows), 'beams': beams}
            if split == 'validation':
                generate(model, tokenizer, rows[:16], beams, preds[:16], entry['id'] + '/repeat16')
                item['repeat16_exact'] = True
                item['inventory_delta'] = {'bleu': item['scores']['bleu'] - entry['eval_bleu'], 'chrf++': item['scores']['chrf++'] - entry['eval_chrf++']}
            else:
                gold = read(ROOT / 'baseline_gold_test' / (entry['id'] + '.csv'))
                old, = [r for r in tests if int(r['seed']) == entry['seed']]
                assert digest(ROOT / 'baseline_gold_test' / (entry['id'] + '.csv')) == old['prediction_sha256']
                assert len(gold) == len(rows)
                assert all(int(r['index']) == i and r['source'] == d['source'] and r['target'] == d['target'].strip()
                           for i, (r, d) in enumerate(zip(gold, rows)))
                item['gold_sha256'] = old['prediction_sha256']
                item['different_predictions'] = [i for i, (p, r) in enumerate(zip(preds, gold)) if p != r['prediction']]
                item['old_test_metric_delta'] = {m: item['scores'][m] - float(old[col]) for m, col in [('bleu', 'BLEU'), ('chrf++', 'chrF++'), ('chrf0', 'chrF0')]}
            import csv
            path = folder / (split + '_predictions.csv')
            with path.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['index', 'source', 'target', 'prediction'])
                writer.writeheader()
                writer.writerows({'index': i, 'source': r['source'], 'target': r['target'].strip(), 'prediction': p}
                                  for i, (r, p) in enumerate(zip(rows, preds)))
            item['prediction_sha256'] = digest(path)
            record['conditions'][split] = item
            print('BASELINE_AUDIT_DONE', entry['id'], split, json.dumps(item['scores']), flush=True)
        save(folder / 'record.json', record)
        results.append(record)
        save(AUDIT / 'summary.json', results)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(extract)
    save(AUDIT / 'COMPLETE.json', {'status': 'BASELINE_ARCHIVE_AUDIT_COMPLETE', 'models': 3, 'full_generations': 6,
                                 'elapsed_seconds': time.time() - started, 'original_failure_retained': True})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(AUDIT / 'FAILED.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
