"""Run the audited V4 measurements on preserved local warmup trajectory weights."""
import argparse
import gc
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from transformers import AutoTokenizer
from amis_rewire.modeling import CrossLayerResidualRewire
from amis_rewire.scientific_records import digest, save
from amis_v4_measurements import strict_model, runtime
from amis_warmup_protocol import REVISION
from replay_amis_warmup_controls import predictions, read
import run_amis_followup_v4_worker as measurements


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    args = p.parse_args()
    folder = Path(args.run_dir)
    metadata = json.loads((folder / 'metrics.json').read_text())
    cfg = metadata['configuration']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt', revision=REVISION,
                                              src_lang='tl_XX', tgt_lang='zh_CN')
    rows = read(ROOT / 'data_processed/amis_mandarin/validation.csv')
    measurements.OUT_BASE = folder / 'trajectory_probes'
    measurements.FIXED_BATCHES = json.loads((ROOT / 'warmup_fixed_batches_v2.json').read_text())['fixed_validation_indices']
    assert len(measurements.FIXED_BATCHES) == 8 and all(len(b) == 4 for b in measurements.FIXED_BATCHES)
    save(measurements.OUT_BASE / 'runtime.json', runtime(tokenizer, ROOT))
    save(measurements.OUT_BASE / 'fixed_batches.json', {'indices': measurements.FIXED_BATCHES,
         'validation_sha256': digest(ROOT / 'data_processed/amis_mandarin/validation.csv'),
         'diagnostic_alpha': cfg['rewire_strength'], 'precision': 'train_BF16_TF32off_checkpointing_disabled',
         'note': 'Offline diagnostic, not identical to checkpointed training; no effect on optimizer/RNG.'})
    # Full validation replay proves the production-selected weights were actually reloaded.
    base = strict_model(folder / 'best_model')
    routed = CrossLayerResidualRewire(base, distance=2, strength=cfg['rewire_strength'], stack='encoder')
    actual = predictions(routed, tokenizer, rows)
    expected_rows = read(folder / 'validation_predictions.csv')
    assert len(expected_rows) == len(rows) and all(r['source'] == e['source'] and r['target'] == e['target']
          for r, e in zip(rows, expected_rows))
    assert actual == [r['prediction'].strip() for r in expected_rows], 'Best-model reload prediction mismatch'
    save(folder / 'BEST_RELOAD_PASS.json', {'status': 'PASS', 'rows': len(rows), 'beams': 1,
         'precision': 'fp32', 'prediction_sha256': digest(folder / 'validation_predictions.csv')})
    for h in routed._handles:
        h.remove()
    del routed, base
    gc.collect()
    torch.cuda.empty_cache()
    # Reuse the tested measurement implementation; only replace its HF downloader with local input.
    extracted = {}
    def local_checkpoint(entry, destination, receipts):
        return extracted[entry['id']]
    measurements.load_checkpoint = local_checkpoint
    archives = sorted((folder / 'trajectory').glob('*.zip'))
    assert archives, 'No trajectory archives preserved'
    all_receipts = []
    for archive in archives:
        receipt = json.loads(archive.with_suffix('.json').read_text())
        assert digest(archive) == receipt['sha256'] and archive.stat().st_size == receipt['archive_bytes']
        name = f'{folder.name}-{archive.stem}'
        unpack = folder / 'probe_extract' / archive.stem
        unpack.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(archive) as z:
            assert all((unpack / n).resolve().is_relative_to(unpack.resolve()) for n in z.namelist())
            z.extractall(unpack)
        modes = [('eval_alpha_fixed', cfg['rewire_strength'])]
        train_alpha = receipt['alpha_train_last_update']
        if abs(train_alpha - cfg['rewire_strength']) > 1e-12:
            modes.append(('last_training_alpha', train_alpha))
        for mode, alpha in modes:
            condition_name = f'{name}-{mode}'
            extracted[condition_name] = unpack
            entry = {'id': condition_name, 'family': 'warmup_full' if cfg['jepa_weight'] else 'warmup_clrr_only',
                     'seed': cfg['seed'], 'alpha': alpha, 'lambda': cfg['jepa_weight']}
            measurements.run_comprehensive_f3(tokenizer, rows, [entry])
            probe = measurements.OUT_BASE / 'probes/runs' / condition_name / 'probe_results.json'
            record = json.loads(probe.read_text())
            record['diagnostic_alpha_mode'] = mode
            record['weight_receipt'] = receipt
            record['fixed_validation_indices'] = measurements.FIXED_BATCHES
            record['runtime_receipt_sha256'] = digest(measurements.OUT_BASE / 'runtime.json')
            # Native CE is token weighted; gradient records remain per-batch diagnostics.
            numerator = count = 0
            for b in record['gradient_batches']:
                ids = tokenizer(text_target=[rows[i]['target'] for i in b['indices']], padding=True,
                                truncation=True, max_length=256, return_tensors='pt')['input_ids']
                n = int(ids.ne(tokenizer.pad_token_id).sum())
                b['label_tokens'] = n
                numerator += b['ce'] * n
                count += n
            record['teacher_ce_token_weighted_fixed32'] = numerator / count
            save(probe, record)
            all_receipts.append({'name': condition_name, 'probe_sha256': digest(probe), 'weight_sha256': receipt['sha256']})
        shutil.rmtree(unpack)
    save(folder / 'TRAJECTORY_PROBES_PASS.json', {'status': 'PASS', 'receipts': all_receipts,
          'best_reloaded': True, 'counterfactual_lsr_lambda_clrr_only': 0.1})


if __name__ == '__main__':
    main()
