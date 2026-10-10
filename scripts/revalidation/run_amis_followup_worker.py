"""Unified worker script running F1, F4, and F3 on Colab A100 GPU.

Implements Sections 14.4 (F1), 14.5 (F3), and 14.6 (F4) of docs/operations/CONFIGURATION_SEARCH_HANDOFF.md.
"""
from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from amis_rewire.modeling import (
    CrossLayerResidualRewire,
    JEPAGuidedSeq2SeqLM,
    _find_stack_layers,
    _masked_mean_pool,
)
from amis_rewire.ablation import AttachedSourceResidualRewire
from amis_rewire.evaluation import generation_precision

HF_REPO = 'FiveC/amis-rewire-checkpoints'
OUT_BASE = ROOT / 'outputs_rebuttal/amis_followup_20261004_v2'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def fresh(model):
    return model._fresh_cache() if hasattr(model, '_fresh_cache') else contextlib.nullcontext()


class LayerMeasurements:
    def __init__(self, model, rewired):
        self.rewired = rewired
        self.enabled = False
        self.phase = ''
        self.row_start = 0
        self.mask = None
        self.pre = {}
        self.rows = []
        self.pooled = {}
        self.handles = []
        for index, layer in enumerate(_find_stack_layers(model, 'encoder')):
            self.handles.append(layer.register_forward_hook(self.before(index), prepend=True))
            self.handles.append(layer.register_forward_hook(self.after(index)))

    def before(self, index):
        def hook(_module, _inputs, output):
            if self.enabled:
                self.pre[index] = (output[0] if isinstance(output, (tuple, list)) else output).detach()
        return hook

    def after(self, index):
        def hook(_module, _inputs, output):
            if not self.enabled:
                return
            post = (output[0] if isinstance(output, (tuple, list)) else output).detach().float()
            pre = self.pre.pop(index).float()
            mask = self.mask.to(pre.device).float().unsqueeze(-1)
            denom = (mask.sum(dim=1).squeeze(-1) * pre.shape[-1]).clamp_min(1)
            rms = lambda t: ((t.square() * mask).sum((1, 2)) / denom).sqrt()
            pre_rms, post_rms = rms(pre), rms(post)
            ratio = torch.zeros_like(pre_rms)
            skip_rms = torch.zeros_like(pre_rms)
            cosine = None
            if self.rewired is not None and index >= self.rewired.distance:
                skip = self.rewired._cache['encoder'][index - self.rewired.distance].float()
                residual = self.rewired.strength * skip
                skip_rms = rms(skip)
                ratio = rms(residual) / pre_rms.clamp_min(1e-12)
                dot = (pre * skip * mask).sum((1, 2))
                cosine = dot / ((pre.square() * mask).sum((1, 2)).sqrt() * (skip.square() * mask).sum((1, 2)).sqrt()).clamp_min(1e-12)
            for i in range(len(pre_rms)):
                self.rows.append({
                    'phase': self.phase,
                    'row_index': self.row_start + i,
                    'layer': index + 1,
                    'pre_rms': float(pre_rms[i]),
                    'post_rms': float(post_rms[i]),
                    'skip_rms': float(skip_rms[i]),
                    'residual_to_current_rms': float(ratio[i]),
                    'skip_current_cosine': float(cosine[i]) if cosine is not None else None,
                })
            pooled = _masked_mean_pool(post, self.mask).cpu().numpy()
            self.pooled.setdefault((self.phase, index + 1), []).append(pooled)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def spectrum_diagnostics(representations):
    raw = np.asarray(representations, dtype='float64')
    centered = raw - raw.mean(axis=0, keepdims=True)
    singular = torch.linalg.svdvals(torch.from_numpy(centered).cuda()).cpu().numpy()
    numerical_tolerance = singular[0] * max(centered.shape) * np.finfo('float64').eps

    def erank(values):
        if values.sum() == 0:
            return 0.0
        p = values[values > 0] / values.sum()
        return float(np.exp(-(p * np.log(p)).sum()))

    energy = singular ** 2
    norm = np.linalg.norm(raw, axis=1)
    unit = raw / np.maximum(norm[:, None], 1e-12)
    pairs = (unit @ unit.T)[np.triu_indices(len(raw), 1)]
    return dict(
        rows=len(raw),
        hidden_dim=raw.shape[1],
        maximum_centered_rank=min(len(raw) - 1, raw.shape[1]),
        numerical_rank=int((singular > numerical_tolerance).sum()),
        numerical_tolerance=float(numerical_tolerance),
        effective_rank_singular_values=erank(singular),
        effective_rank_covariance_energy=erank(energy),
        participation_ratio=float(energy.sum() ** 2 / (energy ** 2).sum()) if energy.sum() > 0 else 0.0,
        covariance_trace=float(energy.sum() / (len(raw) - 1)),
        top1_variance_fraction=float(energy[0] / energy.sum()) if energy.sum() > 0 else 0.0,
        top10_variance_fraction=float(energy[:10].sum() / energy.sum()) if energy.sum() > 0 else 0.0,
        mean_pairwise_cosine=float(pairs.mean()),
        std_pairwise_cosine=float(pairs.std()),
        pooled_norm_mean=float(norm.mean()),
        pooled_norm_std=float(norm.std()),
        singular_values=singular.tolist()
    )


def gradient_group(name, parameter, embedding):
    if parameter is embedding:
        return 'shared_input_embeddings'
    if 'encoder' in name:
        return 'encoder_without_shared_embeddings'
    if 'decoder' in name:
        return 'decoder_without_shared_embeddings'
    return 'other_parameters'


def download_and_extract(archive_path, expected_sha, extract_root):
    token = os.environ.get('HF_TOKEN')
    print(f'Downloading {archive_path} from {HF_REPO}...', flush=True)
    local_zip = Path(hf_hub_download(repo_id=HF_REPO, filename=archive_path, token=token))
    actual_sha = digest(local_zip)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f'SHA mismatch for {archive_path}: expected {expected_sha}, got {actual_sha}')

    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(local_zip) as z:
        z.extractall(extract_root)

    candidates = [p.parent for p in extract_root.rglob('*') if p.name in ['pytorch_model.bin', 'model.safetensors']]
    if candidates:
        return candidates[0]
    if (extract_root / 'best_model').exists():
        return extract_root / 'best_model'
    ckpt_dirs = sorted(extract_root.glob('checkpoint-*'))
    if ckpt_dirs:
        return ckpt_dirs[-1]
    return extract_root


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    val_csv = ROOT / 'data_processed/amis_mandarin/validation.csv'
    test_csv = ROOT / 'data_processed/amis_mandarin/test.csv'
    val_rows = read_csv(val_csv)
    test_rows = read_csv(test_csv)
    assert len(val_rows) == 576
    assert len(test_rows) == 575
    val_refs = [r['target'].strip() for r in val_rows]
    test_refs = [r['target'].strip() for r in test_rows]

    inv_path = OUT_BASE / 'validation_inventory_21.json'
    assert inv_path.exists(), f"Missing {inv_path}"
    inventory = json.loads(inv_path.read_text(encoding='utf-8'))
    assert len(inventory) == 21

    trajectory_summary_path = ROOT / 'outputs_rebuttal/amis_confirmation_20261003/trajectory_summary.json'
    assert trajectory_summary_path.exists()
    traj_data = json.loads(trajectory_summary_path.read_text(encoding='utf-8'))
    fixed_val_batches = traj_data['fixed_validation_indices']  # 8 batches of 4 indices

    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt', src_lang='tl_XX', tgt_lang='zh_CN')
    zh_token_id = tokenizer.convert_tokens_to_ids('zh_CN')

    start_time = time.time()
    status_file = OUT_BASE / 'status.json'

    # =========================================================================
    # STEP F1: Evaluate 3 Full lambda=0.03 test checkpoints
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP F1: Evaluating 3 Full lambda=0.03 test checkpoints (FP32, beam 4)")
    print("=" * 60, flush=True)

    f1_models = [m for m in inventory if m['family'] == 'full_detached_l003']
    assert len(f1_models) == 3

    f1_results = {}
    for m in f1_models:
        model_id = m['id']
        run_out = OUT_BASE / 'supplemental_test/runs' / model_id
        run_out.mkdir(parents=True, exist_ok=True)

        if (run_out / 'test_metrics.json').exists():
            print(f'F1 ALREADY DONE: {model_id}', flush=True)
            f1_results[model_id] = json.loads((run_out / 'test_metrics.json').read_text())
            continue

        print(f"\n--- [F1] Evaluating {model_id} (seed {m['seed']}) ---", flush=True)
        extract_dir = ROOT / 'scratch/f1_extract' / model_id
        ckpt_dir = download_and_extract(m['archive_path'], m.get('archive_sha256'), extract_dir)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).cuda()
        model = CrossLayerResidualRewire(base_model, distance=2, strength=0.10, stack='encoder')
        model.eval()

        predictions = []
        batch_size = 8
        for s in range(0, len(test_rows), batch_size):
            chunk = test_rows[s:s + batch_size]
            inputs = tokenizer([r['source'] for r in chunk], padding=True, truncation=True, max_length=256, pad_to_multiple_of=8, return_tensors='pt')
            inputs = {k: v.cuda() for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
            with torch.no_grad(), generation_precision(model, 'fp32'), fresh(model):
                ids = model.generate(**inputs, num_beams=4, max_length=256, forced_bos_token_id=zh_token_id)
            decoded = tokenizer.batch_decode(ids, skip_special_tokens=True)
            predictions.extend(p.strip() for p in decoded)

        # Save test predictions
        pred_records = [
            {'index': i, 'source': r['source'], 'target': r['target'], 'prediction': p}
            for i, (r, p) in enumerate(zip(test_rows, predictions))
        ]
        with (run_out / 'test_predictions.csv').open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['index', 'source', 'target', 'prediction'])
            w.writeheader()
            w.writerows(pred_records)

        bleu = sacrebleu.corpus_bleu(predictions, [test_refs], tokenize='zh').score
        chrfpp = sacrebleu.corpus_chrf(predictions, [test_refs], word_order=2).score
        chrf0 = sacrebleu.corpus_chrf(predictions, [test_refs], word_order=0).score

        metrics = {
            'id': model_id, 'family': m['family'], 'seed': m['seed'],
            'alpha': m['alpha'], 'lambda': m['lambda'], 'detach': m['detach'],
            'test_rows': len(test_rows), 'test_bleu': bleu, 'test_chrf++': chrfpp, 'test_chrf0': chrf0,
            'prediction_sha256': digest(run_out / 'test_predictions.csv')
        }
        save_json(run_out / 'test_metrics.json', metrics)
        f1_results[model_id] = metrics
        print(f"  F1 Result {model_id}: BLEU={bleu:.4f}, chrF++={chrfpp:.4f}, chrF0={chrf0:.4f}", flush=True)

        # Cleanup
        del model, base_model
        torch.cuda.empty_cache()
        gc.collect()
        shutil.rmtree(extract_dir, ignore_errors=True)

    save_json(OUT_BASE / 'supplemental_test/f1_test_summary.json', f1_results)

    # =========================================================================
    # STEP F4: Route-on vs Route-off on validation for 3 Full lambda=0.10 seeds
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP F4: Route-on vs Route-off on validation (beam 1, 576 rows)")
    print("=" * 60, flush=True)

    f4_models = [m for m in inventory if m['family'] == 'original_full_detached']
    assert len(f4_models) == 3

    f4_results = {}
    for m in f4_models:
        model_id = m['id']
        run_out = OUT_BASE / 'route_off_validation/runs' / model_id
        run_out.mkdir(parents=True, exist_ok=True)

        if (run_out / 'route_comparison.json').exists():
            print(f'F4 ALREADY DONE: {model_id}', flush=True)
            f4_results[model_id] = json.loads((run_out / 'route_comparison.json').read_text())
            continue

        print(f"\n--- [F4] Route-on/off for {model_id} (seed {m['seed']}) ---", flush=True)
        extract_dir = ROOT / 'scratch/f4_extract' / model_id
        ckpt_dir = download_and_extract(m['archive_path'], m.get('archive_sha256'), extract_dir)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).cuda()

        # Run with alpha = 0.10 (Route-on)
        model_on = CrossLayerResidualRewire(base_model, distance=2, strength=0.10, stack='encoder')
        model_on.eval()
        preds_on = []
        for s in range(0, len(val_rows), 8):
            chunk = val_rows[s:s + 8]
            inputs = tokenizer([r['source'] for r in chunk], padding=True, truncation=True, max_length=256, pad_to_multiple_of=8, return_tensors='pt')
            inputs = {k: v.cuda() for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
            with torch.no_grad(), generation_precision(model_on, 'fp32'), fresh(model_on):
                ids = model_on.generate(**inputs, num_beams=1, max_length=256, forced_bos_token_id=zh_token_id)
            preds_on.extend(p.strip() for p in tokenizer.batch_decode(ids, skip_special_tokens=True))

        # Run with alpha = 0.00 (Route-off)
        model_off = CrossLayerResidualRewire(base_model, distance=2, strength=0.00, stack='encoder')
        model_off.eval()
        preds_off = []
        for s in range(0, len(val_rows), 8):
            chunk = val_rows[s:s + 8]
            inputs = tokenizer([r['source'] for r in chunk], padding=True, truncation=True, max_length=256, pad_to_multiple_of=8, return_tensors='pt')
            inputs = {k: v.cuda() for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
            with torch.no_grad(), generation_precision(model_off, 'fp32'), fresh(model_off):
                ids = model_off.generate(**inputs, num_beams=1, max_length=256, forced_bos_token_id=zh_token_id)
            preds_off.extend(p.strip() for p in tokenizer.batch_decode(ids, skip_special_tokens=True))

        changed = sum(p1 != p2 for p1, p2 in zip(preds_on, preds_off))

        bleu_on = sacrebleu.corpus_bleu(preds_on, [val_refs], tokenize='zh').score
        chrfpp_on = sacrebleu.corpus_chrf(preds_on, [val_refs], word_order=2).score
        chrf0_on = sacrebleu.corpus_chrf(preds_on, [val_refs], word_order=0).score

        bleu_off = sacrebleu.corpus_bleu(preds_off, [val_refs], tokenize='zh').score
        chrfpp_off = sacrebleu.corpus_chrf(preds_off, [val_refs], word_order=2).score
        chrf0_off = sacrebleu.corpus_chrf(preds_off, [val_refs], word_order=0).score

        # Save predictions CSVs
        for fname, preds in [('val_predictions_route_on.csv', preds_on), ('val_predictions_route_off.csv', preds_off)]:
            records = [{'index': i, 'source': r['source'], 'target': r['target'], 'prediction': p} for i, (r, p) in enumerate(zip(val_rows, preds))]
            with (run_out / fname).open('w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['index', 'source', 'target', 'prediction'])
                w.writeheader()
                w.writerows(records)

        comp = {
            'id': model_id, 'seed': m['seed'], 'validation_rows': len(val_rows),
            'sentences_changed': changed, 'pct_changed': changed / len(val_rows) * 100.0,
            'route_on': {'bleu': bleu_on, 'chrf++': chrfpp_on, 'chrf0': chrf0_on},
            'route_off': {'bleu': bleu_off, 'chrf++': chrfpp_off, 'chrf0': chrf0_off},
            'delta_on_minus_off': {'bleu': bleu_on - bleu_off, 'chrf++': chrfpp_on - chrfpp_off, 'chrf0': chrf0_on - chrf0_off},
        }
        save_json(run_out / 'route_comparison.json', comp)
        f4_results[model_id] = comp
        print(f"  F4 Result {model_id}: changed={changed}/{len(val_rows)}, delta_BLEU={comp['delta_on_minus_off']['bleu']:+.4f}, delta_chrF++={comp['delta_on_minus_off']['chrf++']:+.4f}", flush=True)

        del model_on, model_off, base_model
        torch.cuda.empty_cache()
        gc.collect()
        shutil.rmtree(extract_dir, ignore_errors=True)

    save_json(OUT_BASE / 'route_off_validation/f4_summary.json', f4_results)

    # =========================================================================
    # STEP F3: Gradient, residual, and rank probes across all 21 best checkpoints
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP F3: Gradient, residual, and rank probes across 21 best checkpoints")
    print("=" * 60, flush=True)

    f3_summary = {}

    for idx, m in enumerate(inventory):
        model_id = m['id']
        run_out = OUT_BASE / 'probes/runs' / model_id
        run_out.mkdir(parents=True, exist_ok=True)

        if (run_out / 'probe_results.json').exists():
            print(f'[{idx+1}/21] F3 ALREADY DONE: {model_id}', flush=True)
            f3_summary[model_id] = json.loads((run_out / 'probe_results.json').read_text())
            continue

        print(f"\n--- [{idx+1}/21 F3] Probing {model_id} (family={m['family']}, seed={m['seed']}, alpha={m['alpha']}, lambda={m['lambda']}) ---", flush=True)

        extract_dir = ROOT / 'scratch/f3_extract' / model_id
        ckpt_dir = download_and_extract(m['archive_path'], m.get('archive_sha256'), extract_dir)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).cuda()

        # Wrap with routing
        rewired_layer = None
        if m['family'] == 'full_source_attached':
            model = AttachedSourceResidualRewire(base_model, distance=2, strength=m['alpha'], stack='encoder')
            rewired_layer = model
        elif m['alpha'] > 0:
            model = CrossLayerResidualRewire(base_model, distance=2, strength=m['alpha'], stack='encoder')
            rewired_layer = model
        else:
            model = base_model

        # Wrap with JEPAGuidedSeq2SeqLM if lambda > 0
        if m['lambda'] > 0:
            wrapper = JEPAGuidedSeq2SeqLM(model, jepa_weight=m['lambda'])
        else:
            wrapper = model

        # ---------------------------------------------------------------------
        # 1. Residual & Representation Diagnostics (576 validation rows)
        # ---------------------------------------------------------------------
        print("  Extracting representations and residual dynamics on validation set...", flush=True)
        wrapper.eval()
        collector = LayerMeasurements(base_model, rewired_layer)

        val_reps = {}
        for phase, col in [('source', 'source'), ('target', 'target')]:
            reps_list = []
            for s in range(0, len(val_rows), 16):
                chunk = val_rows[s:s + 16]
                texts = [r[col] for r in chunk]
                if phase == 'target':
                    tok_out = tokenizer(text_target=texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
                else:
                    tok_out = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
                batch = {k: v.cuda() for k, v in tok_out.items() if k in ['input_ids', 'attention_mask']}
                collector.enabled = True
                collector.phase = phase
                collector.row_start = s
                collector.mask = batch['attention_mask']
                with torch.no_grad(), fresh(wrapper):
                    enc_out = base_model.get_encoder()(**batch, return_dict=True)
                    pooled = _masked_mean_pool(enc_out.last_hidden_state.float(), batch['attention_mask']).cpu().numpy()
                    reps_list.append(pooled)
                collector.enabled = False
            val_reps[phase] = np.concatenate(reps_list)

        collector.close()

        # Spectrum and effective rank
        src_spectrum = spectrum_diagnostics(val_reps['source'])
        tgt_spectrum = spectrum_diagnostics(val_reps['target'])

        # Paired source-target cosine
        src_norm = val_reps['source'] / np.maximum(np.linalg.norm(val_reps['source'], axis=1, keepdims=True), 1e-12)
        tgt_norm = val_reps['target'] / np.maximum(np.linalg.norm(val_reps['target'], axis=1, keepdims=True), 1e-12)
        paired_cosines = (src_norm * tgt_norm).sum(axis=1)

        # Residual metrics summary
        residual_df = pd.DataFrame(collector.rows)
        residual_summary = {}
        if not residual_df.empty:
            residual_df.to_csv(run_out / 'residual_per_example.csv', index=False)
            ratio_vals = residual_df[residual_df['residual_to_current_rms'] > 0]['residual_to_current_rms']
            residual_summary = {
                'mean_residual_ratio': float(ratio_vals.mean()) if not ratio_vals.empty else 0.0,
                'max_residual_ratio': float(ratio_vals.max()) if not ratio_vals.empty else 0.0,
                'p95_residual_ratio': float(ratio_vals.quantile(0.95)) if not ratio_vals.empty else 0.0,
            }

        # ---------------------------------------------------------------------
        # 2. Gradient Diagnostics on 32 fixed validation rows (8 batches x 4)
        # ---------------------------------------------------------------------
        print("  Measuring native CE vs weighted LSR gradients on fixed 32 validation rows...", flush=True)
        wrapper.train()
        base_model.gradient_checkpointing_disable()
        base_model.config.use_cache = False

        names, parameters = zip(*wrapper.named_parameters())
        embedding = base_model.get_input_embeddings().weight
        groups = [gradient_group(n, p, embedding) for n, p in zip(names, parameters)]

        gradient_results = []
        for batch_id, indices in enumerate(fixed_val_batches):
            batch_rows = [val_rows[i] for i in indices]
            x = tokenizer([r['source'] for r in batch_rows], padding=True, truncation=True, max_length=256, return_tensors='pt')
            y = tokenizer(text_target=[r['target'] for r in batch_rows], padding=True, truncation=True, max_length=256, return_tensors='pt')['input_ids']
            y[y == tokenizer.pad_token_id] = -100
            inputs = {k: v.cuda() for k, v in x.items() if k in ['input_ids', 'attention_mask']}
            labels = y.cuda()

            target_captures = []
            native_ce = []

            def capture_enc(_module, _inputs, output):
                target_captures.append(output.last_hidden_state)
            enc_handle = base_model.get_encoder().register_forward_hook(capture_enc)

            underlying = wrapper
            while isinstance(underlying, (CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM, AttachedSourceResidualRewire)):
                underlying = underlying.base_model

            def capture_loss(_module, _inputs, output):
                native_ce.append(output.loss)
            loss_handle = underlying.register_forward_hook(capture_loss)

            torch.manual_seed(1000 + batch_id)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = wrapper(**inputs, labels=labels, output_hidden_states=True, return_dict=True)
                assert len(native_ce) == 1
                ce = native_ce[0]

                if m['lambda'] > 0:
                    weighted_lsr = outputs.loss - ce
                    lsr = weighted_lsr / m['lambda']
                else:
                    # Counterfactual LSR with lambda=0.1
                    target_ids = labels.clone()
                    target_ids[target_ids == -100] = base_model.config.pad_token_id
                    target_mask = target_ids.ne(base_model.config.pad_token_id).long()
                    with torch.no_grad(), fresh(wrapper):
                        target = base_model.get_encoder()(input_ids=target_ids, attention_mask=target_mask, return_dict=True).last_hidden_state
                    src = F.normalize(_masked_mean_pool(outputs.encoder_last_hidden_state, inputs['attention_mask']), dim=-1)
                    tgt = F.normalize(_masked_mean_pool(target, target_mask), dim=-1)
                    lsr = (1 - (src * tgt).sum(-1)).mean()
                    weighted_lsr = 0.1 * lsr.to(ce.dtype)

            enc_handle.remove()
            loss_handle.remove()

            ce_grads = torch.autograd.grad(ce, parameters, retain_graph=True, allow_unused=True)
            lsr_grads = torch.autograd.grad(weighted_lsr, parameters, allow_unused=True)

            sums = {key: [0.0, 0.0, 0.0] for key in ['all_parameters', *sorted(set(groups))]}
            for grp, gce, glsr in zip(groups, ce_grads, lsr_grads):
                nce = float(gce.float().norm() ** 2) if gce is not None else 0.0
                nlsr = float(glsr.float().norm() ** 2) if glsr is not None else 0.0
                dot = float((gce.float() * glsr.float()).sum()) if gce is not None and glsr is not None else 0.0
                for k in ['all_parameters', grp]:
                    sums[k][0] += nce
                    sums[k][1] += nlsr
                    sums[k][2] += dot

            b_res = {
                'batch_id': batch_id,
                'indices': indices,
                'ce_loss': float(ce.detach()),
                'lsr_loss': float(lsr.detach()),
                'weighted_lsr_loss': float(weighted_lsr.detach()),
                'groups': {}
            }
            for k, (a, b, c) in sums.items():
                b_res['groups'][k] = {
                    'ce_gradient_norm': a ** 0.5,
                    'weighted_lsr_gradient_norm': b ** 0.5,
                    'lsr_to_ce_ratio': (b / a) ** 0.5 if a > 0 else None,
                    'gradient_cosine': c / (a * b) ** 0.5 if a > 0 and b > 0 else None,
                }
            gradient_results.append(b_res)
            wrapper.zero_grad(set_to_none=True)

        # Consolidate probe output for this model
        probe_record = {
            'id': model_id,
            'family': m['family'],
            'seed': m['seed'],
            'alpha': m['alpha'],
            'lambda': m['lambda'],
            'detach': m['detach'],
            'best_step': m['best_step'],
            'residual_summary': residual_summary,
            'source_spectrum': {k: v for k, v in src_spectrum.items() if k != 'singular_values'},
            'target_spectrum': {k: v for k, v in tgt_spectrum.items() if k != 'singular_values'},
            'paired_source_target_cosine_mean': float(paired_cosines.mean()),
            'paired_source_target_cosine_std': float(paired_cosines.std()),
            'gradient_batches': gradient_results,
            'mean_all_param_gradient_cosine': float(np.mean([b['groups']['all_parameters']['gradient_cosine'] for b in gradient_results if b['groups']['all_parameters']['gradient_cosine'] is not None])),
            'mean_all_param_lsr_to_ce_ratio': float(np.mean([b['groups']['all_parameters']['lsr_to_ce_ratio'] for b in gradient_results if b['groups']['all_parameters']['lsr_to_ce_ratio'] is not None])),
        }
        save_json(run_out / 'probe_results.json', probe_record)
        f3_summary[model_id] = probe_record

        print(f"  F3 Done {model_id}: mean_grad_cos={probe_record['mean_all_param_gradient_cosine']:.4f}, mean_lsr_ratio={probe_record['mean_all_param_lsr_to_ce_ratio']:.4f}, src_erank={src_spectrum['effective_rank_singular_values']:.2f}", flush=True)

        # Cleanup
        del wrapper, model, base_model
        torch.cuda.empty_cache()
        gc.collect()
        shutil.rmtree(extract_dir, ignore_errors=True)

    save_json(OUT_BASE / 'probes/probe_summary.json', f3_summary)

    # Write overall completion receipt
    elapsed = time.time() - start_time
    complete_info = {
        'status': 'ALL_GPU_SUITES_COMPLETE',
        'f1_test_runs': len(f1_results),
        'f4_route_on_off_runs': len(f4_results),
        'f3_probed_models': len(f3_summary),
        'elapsed_seconds': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    save_json(OUT_BASE / 'COMPLETE_GPU.json', complete_info)
    print("\n" + "=" * 60)
    print(f"ALL GPU SUITES COMPLETE (F1, F4, F3) in {elapsed:.1f}s!")
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
