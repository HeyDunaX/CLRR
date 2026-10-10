"""Confirm recovered baseline42 validation beam metadata without changing results."""
import json
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from amis_v4_measurements import load_checkpoint, strict_model, runtime, digest
from run_amis_routing_v5_worker import read, save, generate, scores

OUT = ROOT / 'outputs_rebuttal/amis_routing_diagnostics_20261004_v5'
DEST = OUT / 'baseline42_beam_confirmation'
assert (OUT / 'baseline_archive_audit_v3/COMPLETE.json').exists()
assert not DEST.exists()
DEST.mkdir()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
entry, = [e for e in json.loads((ROOT / 'validation_inventory_21.json').read_text())
           if e['family'] == 'baseline' and e['seed'] == 42]
ckpt = load_checkpoint(entry, ROOT / 'scratch/baseline42_beam_extract', DEST / 'loading_receipts')
args = torch.load(ckpt / 'training_args.bin', map_location='cpu', weights_only=False)
args_info = {k: getattr(args, k, None) for k in ['generation_num_beams', 'generation_max_length']}
del args
model = strict_model(ckpt).eval()
tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt',
    revision='e30b6cb8eb0d43a0b73cab73c7676b9863223a30', src_lang='tl_XX', tgt_lang='zh_CN')
save(DEST / 'runtime.json', runtime(tokenizer, ROOT))
rows = read(ROOT / 'data_processed/amis_mandarin/validation.csv')
preds = generate(model, tokenizer, rows, 4, progress_key=entry['id'] + '/confirm_validation_b4')
import csv
path = DEST / 'validation_predictions_beam4.csv'
with path.open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['index', 'source', 'target', 'prediction'])
    w.writeheader()
    w.writerows({'index': i, 'source': r['source'], 'target': r['target'].strip(), 'prediction': p}
                for i, (r, p) in enumerate(zip(rows, preds)))
values = scores(preds, [r['target'].strip() for r in rows])
save(DEST / 'record.json', {'id': entry['id'], 'rows': len(rows), 'beams': 4,
    'saved_training_args_selected_fields': args_info, 'scores': values, 'prediction_sha256': digest(path),
    'inventory_delta': {'bleu': values['bleu'] - entry['eval_bleu'], 'chrf++': values['chrf++'] - entry['eval_chrf++']},
    'historical_best_step': 'unknown; inventory step216 not certified'})
print('BASELINE42_BEAM4', json.dumps(values), 'SAVED_ARGS', json.dumps(args_info), flush=True)
save(DEST / 'COMPLETE.json', {'status': 'COMPLETE'})
