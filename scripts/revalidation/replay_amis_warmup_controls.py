"""Replay frozen controls under the locked runtime before starting new training."""
import argparse
import csv
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from transformers import AutoTokenizer
from amis_rewire.modeling import CrossLayerResidualRewire
from amis_rewire.scientific_records import digest, save
from amis_rewire.evaluation import generation_precision
from amis_v4_measurements import load_checkpoint, strict_model
from amis_warmup_protocol import REVISION, preflight


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def predictions(model, tokenizer, rows):
    result = []
    model.eval()
    for start in range(0, len(rows), 8):
        inputs = tokenizer([r['source'] for r in rows[start:start + 8]], padding=True,
                           truncation=True, max_length=256, pad_to_multiple_of=8, return_tensors='pt')
        inputs = {k: v.cuda() for k, v in inputs.items() if k in ('input_ids', 'attention_mask')}
        with torch.no_grad(), generation_precision(model, 'fp32'):
            ids = model.generate(**inputs, num_beams=1, max_length=256, forced_bos_token_id=250025)
        result.extend(p.strip() for p in tokenizer.batch_decode(ids, skip_special_tokens=True))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', choices=('machine1', 'machine2'), required=True)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()
    out = Path(args.out_dir)
    preflight(out)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt',
                    revision=REVISION, src_lang='tl_XX', tgt_lang='zh_CN')
    assert tokenizer('source')['input_ids'][0] == 250047
    assert tokenizer(text_target='target')['input_ids'][0] == 250025
    rows = read(ROOT / 'data_processed/amis_mandarin/validation.csv')
    inventory = json.loads((ROOT / 'warmup_controls_v2.json').read_text())
    families = ('original_full_detached', 'tuned_lsr') if args.worker == 'machine1' else ('clrr_only_detached',)
    results = []
    for entry in inventory:
        if entry['family'] not in families:
            continue
        print('REPLAY_CONTROL', entry['id'], flush=True)
        destination = out / 'control_extract' / entry['id']
        # Stale incomplete extraction is an error, not a silent reuse.
        path = load_checkpoint(entry, destination, out / 'control_loading_receipts')
        base = strict_model(path)
        model = CrossLayerResidualRewire(base, distance=2, strength=entry['alpha'], stack='encoder') if entry['alpha'] else base
        preds = predictions(model, tokenizer, rows)
        exact = None
        if entry.get('gold_file'):
            gold_path = ROOT / entry['gold_file']
            assert digest(gold_path) == entry['gold_sha256']
            gold = read(gold_path)
            assert len(gold) == len(rows) and all(g['source'] == r['source'] and g['target'] == r['target']
                for g, r in zip(gold, rows))
            assert preds == [r['prediction'].strip() for r in gold], f'Control replay mismatch: {entry["id"]}'
            exact = True
        import sacrebleu
        scores = {'bleu': sacrebleu.corpus_bleu(preds, [[r['target'] for r in rows]], tokenize='zh').score,
                  'chrf++': sacrebleu.corpus_chrf(preds, [[r['target'] for r in rows]], word_order=2).score,
                  'chrf0': sacrebleu.corpus_chrf(preds, [[r['target'] for r in rows]], word_order=0).score}
        folder = out / 'controls' / entry['id']
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'predictions.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['index', 'source', 'target', 'prediction'])
            writer.writeheader()
            writer.writerows({'index': i, **r, 'prediction': pred} for i, (r, pred) in enumerate(zip(rows, preds)))
        record = {'id': entry['id'], 'family': entry['family'], 'seed': entry['seed'],
                  'archive_sha256': entry['archive_sha256'], 'exact_v5_replay': exact,
                  'validation_sha256': digest(ROOT / 'data_processed/amis_mandarin/validation.csv'),
                  'beams': 1, 'precision': 'fp32', 'tf32': False, 'scores': scores}
        save(folder / 'receipt.json', record)
        results.append(record)
        if isinstance(model, CrossLayerResidualRewire):
            for h in model._handles:
                h.remove()
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
        # Loading/hash and exact replay receipt are durable; temporary unpack only.
        import shutil
        shutil.rmtree(destination)
    assert len(results) == (6 if args.worker == 'machine1' else 3)
    save(out / f'CONTROLS_PASS_{args.worker}.json', {'status': 'PASS', 'results': results,
          'limit': 'Exact decoding replay validates inference reuse; original training provenance is checked separately.'})


if __name__ == '__main__':
    main()
