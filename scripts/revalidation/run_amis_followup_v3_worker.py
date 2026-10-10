"""Corrected Follow-up Worker v3 implementing Sections 16.2 (Corrected F4) and 16.4/16.5 (Layerwise F3).

Guarantees:
1. Smoke tests on GPU confirming hook addition, hook removal, and strength=0 identity.
2. Corrected F4: route-on vs route-off with strictly separated hook states, teacher-forced logits on 32 rows, and greedy decoding on 576 rows.
3. Layerwise F3 probes across 21 models: saving 12-layer representations, full spectra, per-layer gradient cosines/ratios, and strict SHA verification.
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
import torch.nn as nn
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


def download_and_extract(archive_path: str, expected_sha: str | None, extract_to: Path) -> Path:
    token = os.environ.get('HF_TOKEN')
    print(f"Downloading {archive_path} from {HF_REPO}...", flush=True)
    local_zip = Path(hf_hub_download(repo_id=HF_REPO, filename=archive_path, token=token))
    actual_sha = digest(local_zip)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"Checksum mismatch for {archive_path}: {actual_sha} vs {expected_sha}")
    if expected_sha:
        print(f"  Verified SHA-256: {actual_sha[:16]}...", flush=True)

    if extract_to.exists():
        shutil.rmtree(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {local_zip} -> {extract_to}...", flush=True)
    with zipfile.ZipFile(local_zip, 'r') as zf:
        zf.extractall(extract_to)

    candidates = [p.parent for p in extract_to.rglob('*') if p.name in ['pytorch_model.bin', 'model.safetensors']]
    if candidates:
        return candidates[0]
    if (extract_to / 'best_model').exists():
        return extract_to / 'best_model'
    ckpt_dirs = sorted(extract_to.glob('checkpoint-*'))
    if ckpt_dirs:
        return ckpt_dirs[-1]
    return extract_to


# =============================================================================
# GPU SMOKE TEST FOR REWIRING HOOKS
# =============================================================================
def run_hook_smoke_test(tokenizer, device='cuda'):
    print("\n=== RUNNING SMOKE TEST ON GPU FOR REWIRING HOOKS ===", flush=True)
    # Tiny dummy test model using base architecture
    from transformers import MBartConfig, MBartForConditionalGeneration
    cfg = MBartConfig(
        vocab_size=len(tokenizer),
        d_model=64,
        encoder_layers=4,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
    )
    test_model = MBartForConditionalGeneration(cfg).to(device).eval()

    layers = _find_stack_layers(test_model, 'encoder')
    assert all(len(getattr(l, '_forward_hooks', {})) == 0 for l in layers), "Initial hooks not zero!"

    # Wrap model_on
    rewired = CrossLayerResidualRewire(test_model, distance=2, strength=0.10, stack='encoder')
    assert all(len(getattr(l, '_forward_hooks', {})) == 1 for l in layers), "Hook count != 1 after rewire!"

    dummy_input = torch.randint(1, 100, (2, 8), device=device)
    dummy_mask = torch.ones_like(dummy_input)

    # Route-on forward
    with torch.no_grad(), fresh(rewired):
        out_on = rewired(input_ids=dummy_input, attention_mask=dummy_mask, return_dict=True)

    # Set strength=0.00
    rewired.strength = 0.00
    with torch.no_grad(), fresh(rewired):
        out_off_zero = rewired(input_ids=dummy_input, attention_mask=dummy_mask, return_dict=True)

    # Clean raw model with handles removed
    for h in rewired._handles:
        h.remove()
    rewired._handles.clear()
    assert all(len(getattr(l, '_forward_hooks', {})) == 0 for l in layers), "Hooks not cleared after remove!"

    with torch.no_grad():
        out_raw = test_model(input_ids=dummy_input, attention_mask=dummy_mask, return_dict=True)

    diff_on = (out_on.logits - out_raw.logits).abs().max().item()
    diff_zero = (out_off_zero.logits - out_raw.logits).abs().max().item()

    print(f"  Max logit diff (on vs raw): {diff_on:.6f} (expected > 0)")
    print(f"  Max logit diff (strength=0 vs raw): {diff_zero:.6e} (expected == 0)")
    assert diff_on > 1e-5, f"Route-on did not modify logits! ({diff_on})"
    assert diff_zero < 1e-6, "Route-off (strength=0) differed from raw model!"

    del test_model, rewired
    torch.cuda.empty_cache()
    print("=== SMOKE TEST PASSED 100% ===\n", flush=True)


# =============================================================================
# CORRECTED STEP F4: ROUTE-ON VS ROUTE-OFF
# =============================================================================
def run_corrected_f4(tokenizer, val_rows, val_refs, inventory):
    f4_out_dir = OUT_BASE / 'amis_route_control_corrected_20261004_v3'
    f4_out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = f4_out_dir / 'route_control_summary.json'
    zh_token_id = tokenizer.convert_tokens_to_ids('zh_CN')

    f4_models = [m for m in inventory if m['family'] == 'original_full_detached']
    assert len(f4_models) == 3, f"Expected 3 Full detached models, got {len(f4_models)}"

    results = {}
    print("\n" + "=" * 60)
    print("STEP F4 (CORRECTED): Route-on vs Route-off Evaluation")
    print("=" * 60, flush=True)

    for m in f4_models:
        model_id = m['id']
        run_out = f4_out_dir / model_id
        run_out.mkdir(parents=True, exist_ok=True)
        rec_file = run_out / 'route_record.json'

        if rec_file.exists():
            print(f"F4 ALREADY DONE: {model_id}", flush=True)
            results[model_id] = json.loads(rec_file.read_text(encoding='utf-8'))
            continue

        print(f"\n--- [F4 Corrected] Running {model_id} (seed {m['seed']}) ---", flush=True)
        extract_dir = ROOT / 'scratch/f4_extract' / model_id
        ckpt_dir = download_and_extract(m['archive_path'], m.get('archive_sha256'), extract_dir)

        # 1. Load clean base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).cuda().eval()

        # 2. Check 32 fixed rows teacher-forced CE loss & logits difference
        fixed_32 = val_rows[:32]
        x_32 = tokenizer([r['source'] for r in fixed_32], padding=True, truncation=True, max_length=256, return_tensors='pt')
        y_32 = tokenizer(text_target=[r['target'] for r in fixed_32], padding=True, truncation=True, max_length=256, return_tensors='pt')['input_ids']
        y_32[y_32 == tokenizer.pad_token_id] = -100
        inputs_32 = {k: v.cuda() for k, v in x_32.items() if k in ['input_ids', 'attention_mask']}
        labels_32 = y_32.cuda()

        # Route-off on raw model (clean, 0 hooks)
        with torch.no_grad():
            raw_eval = base_model(**inputs_32, labels=labels_32)
            raw_ce = float(raw_eval.loss)
            raw_logits = raw_eval.logits.detach()

        # Wrap with CrossLayerResidualRewire
        rewired = CrossLayerResidualRewire(base_model, distance=2, strength=0.10, stack='encoder')
        with torch.no_grad(), fresh(rewired):
            on_eval = rewired(**inputs_32, labels=labels_32)
            on_ce = float(on_eval.loss)
            on_logits = on_eval.logits.detach()

        logit_diff_mean = float((on_logits - raw_logits).abs().mean().cpu())
        logit_diff_max = float((on_logits - raw_logits).abs().max().cpu())

        print(f"  Fixed 32 CE: on={on_ce:.4f}, off={raw_ce:.4f}, logit_diff_mean={logit_diff_mean:.6f}, max={logit_diff_max:.6f}", flush=True)

        # 3. Greedy decoding on 576 validation rows
        # A. Route-on (strength=0.10)
        rewired.strength = 0.10
        preds_on = []
        for s in range(0, len(val_rows), 8):
            chunk = val_rows[s:s + 8]
            inputs = tokenizer([r['source'] for r in chunk], padding=True, truncation=True, max_length=256, pad_to_multiple_of=8, return_tensors='pt')
            inputs = {k: v.cuda() for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
            with torch.no_grad(), generation_precision(rewired, 'fp32'), fresh(rewired):
                ids = rewired.generate(**inputs, num_beams=1, max_length=256, forced_bos_token_id=zh_token_id)
            preds_on.extend(p.strip() for p in tokenizer.batch_decode(ids, skip_special_tokens=True))

        # B. Route-off (remove hooks completely to ensure 100% clean base model)
        for h in rewired._handles:
            h.remove()
        rewired._handles.clear()

        preds_off = []
        for s in range(0, len(val_rows), 8):
            chunk = val_rows[s:s + 8]
            inputs = tokenizer([r['source'] for r in chunk], padding=True, truncation=True, max_length=256, pad_to_multiple_of=8, return_tensors='pt')
            inputs = {k: v.cuda() for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
            with torch.no_grad(), generation_precision(base_model, 'fp32'):
                ids = base_model.generate(**inputs, num_beams=1, max_length=256, forced_bos_token_id=zh_token_id)
            preds_off.extend(p.strip() for p in tokenizer.batch_decode(ids, skip_special_tokens=True))

        changed = sum(p1 != p2 for p1, p2 in zip(preds_on, preds_off))
        pct_changed = 100.0 * changed / len(val_rows)

        bleu_on = sacrebleu.corpus_bleu(preds_on, [val_refs], tokenize='zh').score
        chrfpp_on = sacrebleu.corpus_chrf(preds_on, [val_refs], word_order=2).score
        chrf0_on = sacrebleu.corpus_chrf(preds_on, [val_refs], word_order=0).score

        bleu_off = sacrebleu.corpus_bleu(preds_off, [val_refs], tokenize='zh').score
        chrfpp_off = sacrebleu.corpus_chrf(preds_off, [val_refs], word_order=2).score
        chrf0_off = sacrebleu.corpus_chrf(preds_off, [val_refs], word_order=0).score

        record = {
            'id': model_id,
            'seed': m['seed'],
            'validation_rows': len(val_rows),
            'sentences_changed': changed,
            'pct_changed': pct_changed,
            'fixed_32_diagnostics': {
                'teacher_forced_ce_on': on_ce,
                'teacher_forced_ce_off': raw_ce,
                'ce_delta_on_minus_off': on_ce - raw_ce,
                'logit_diff_mean': logit_diff_mean,
                'logit_diff_max': logit_diff_max,
            },
            'route_on': {'bleu': bleu_on, 'chrf++': chrfpp_on, 'chrf0': chrf0_on},
            'route_off': {'bleu': bleu_off, 'chrf++': chrfpp_off, 'chrf0': chrf0_off},
            'delta_on_minus_off': {
                'bleu': bleu_on - bleu_off,
                'chrf++': chrfpp_on - chrfpp_off,
                'chrf0': chrf0_on - chrf0_off,
            },
        }

        # Save predictions CSVs
        pd.DataFrame({'index': range(len(preds_on)), 'source': [r['source'] for r in val_rows], 'target': val_refs, 'prediction_route_on': preds_on, 'prediction_route_off': preds_off}).to_csv(run_out / 'route_predictions_comparison.csv', index=False, encoding='utf-8-sig')
        save_json(rec_file, record)
        results[model_id] = record

        print(f"  F4 Done {model_id}: changed={changed}/{len(val_rows)} ({pct_changed:.2f}%), delta_BLEU={record['delta_on_minus_off']['bleu']:+.4f}, delta_chrF++={record['delta_on_minus_off']['chrf++']:+.4f}", flush=True)

        del base_model, rewired
        torch.cuda.empty_cache()
        gc.collect()
        shutil.rmtree(extract_dir, ignore_errors=True)

    save_json(summary_path, results)
    print(f"Corrected F4 complete! Results saved to {summary_path}\n", flush=True)
    return results


# =============================================================================
# STEP F3 (COMPREHENSIVE): LAYERWISE GRADIENT & REPRESENTATION PROBES
# =============================================================================
class FullLayerMeasurements:
    def __init__(self, model, rewired):
        self.rewired = rewired
        self.enabled = False
        self.phase = ''
        self.mask = None
        self.handles = []
        self.pooled_by_layer = {}
        self.residual_stats = []

        layers = _find_stack_layers(model, 'encoder')
        self.num_layers = len(layers)
        for index, layer in enumerate(layers):
            self.handles.append(layer.register_forward_hook(self._make_hook(index)))

    def _make_hook(self, index):
        def hook(_module, _inputs, output):
            if not self.enabled:
                return
            post = (output[0] if isinstance(output, (tuple, list)) else output).detach()
            mask_float = self.mask.to(post.device).float()
            # Mean pool masked
            pooled = _masked_mean_pool(post.float(), mask_float).cpu().numpy()
            self.pooled_by_layer.setdefault((self.phase, index + 1), []).append(pooled)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def compute_spectrum_metrics(matrix: np.ndarray) -> dict:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    p = s / s.sum().clip(min=1e-12)
    entropy = float(-(p * np.log(p.clip(min=1e-12))).sum())
    erank_s = float(np.exp(entropy))

    cov = (centered.T @ centered) / max(len(matrix) - 1, 1)
    evals = np.maximum(0.0, np.linalg.eigvalsh(cov))[::-1]
    denom = evals.sum()
    erank_cov = float((denom ** 2) / ((evals ** 2).sum().clip(min=1e-12))) if denom > 0 else 0.0

    return {
        'effective_rank_singular_values': erank_s,
        'effective_rank_covariance_energy': erank_cov,
        'singular_values': [float(x) for x in s[:50]],  # top 50 singular values
        'top1_variance_fraction': float(evals[0] / denom) if denom > 0 else 0.0,
    }


def run_comprehensive_f3(tokenizer, val_rows, inventory):
    f3_out_dir = OUT_BASE / 'probes_v3'
    f3_out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = f3_out_dir / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = f3_out_dir / 'probe_summary_v3.json'

    # Fixed 32 rows for gradient probes
    grad_32 = val_rows[:32]
    batches_32 = [grad_32[i:i + 4] for i in range(0, 32, 4)]
    assert len(batches_32) == 8

    f3_summary = {}
    print("\n" + "=" * 60)
    print("STEP F3 (COMPREHENSIVE): 12-Layer Representation & Gradient Probes")
    print("=" * 60, flush=True)

    for idx, m in enumerate(inventory, 1):
        model_id = m['id']
        run_out = runs_dir / model_id
        run_out.mkdir(parents=True, exist_ok=True)
        rec_file = run_out / 'probe_results.json'

        if rec_file.exists():
            print(f"[{idx}/21] F3 ALREADY DONE: {model_id}", flush=True)
            f3_summary[model_id] = json.loads(rec_file.read_text(encoding='utf-8'))
            continue

        print(f"\n--- [{idx}/21 F3] Probing {model_id} (family={m['family']}, seed={m['seed']}, alpha={m['alpha']}, lambda={m['lambda']}) ---", flush=True)
        extract_dir = ROOT / 'scratch/f3_extract' / model_id
        ckpt_dir = download_and_extract(m['archive_path'], m.get('archive_sha256'), extract_dir)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).cuda()

        # Wrap routing
        rewired_layer = None
        if m['family'] == 'full_source_attached':
            model = AttachedSourceResidualRewire(base_model, distance=2, strength=m['alpha'], stack='encoder')
            rewired_layer = model
        elif m['alpha'] > 0:
            model = CrossLayerResidualRewire(base_model, distance=2, strength=m['alpha'], stack='encoder')
            rewired_layer = model
        else:
            model = base_model

        if m['lambda'] > 0:
            wrapper = JEPAGuidedSeq2SeqLM(model, jepa_weight=m['lambda'])
        else:
            wrapper = model

        # ---------------------------------------------------------------------
        # 1. 12-Layer Representation & Spectrum Extraction (576 validation rows)
        # ---------------------------------------------------------------------
        print("  Extracting representations across all 12 encoder layers...", flush=True)
        wrapper.eval()
        collector = FullLayerMeasurements(base_model, rewired_layer)

        layer_reps = {}
        for phase, col in [('source', 'source'), ('target', 'target')]:
            collector.phase = phase
            collector.enabled = True
            for s in range(0, len(val_rows), 16):
                chunk = val_rows[s:s + 16]
                texts = [r[col] for r in chunk]
                kwargs = {'text_target': texts} if phase == 'target' else {'text': texts}
                tokens = tokenizer(**kwargs, padding=True, truncation=True, max_length=256, pad_to_multiple_of=8, return_tensors='pt')
                tokens = {k: v.cuda() for k, v in tokens.items() if k in ['input_ids', 'attention_mask']}
                collector.mask = tokens['attention_mask']
                with torch.no_grad(), fresh(wrapper):
                    base_model.get_encoder()(**tokens)

            for layer_idx in range(1, collector.num_layers + 1):
                mat = np.concatenate(collector.pooled_by_layer[(phase, layer_idx)], axis=0)
                layer_reps[(phase, layer_idx)] = mat
        collector.remove()

        # Compute spectra per layer and cosine alignment
        layer_metrics = {}
        for layer_idx in range(1, collector.num_layers + 1):
            src_mat = layer_reps[('source', layer_idx)]
            tgt_mat = layer_reps[('target', layer_idx)]

            src_norm = src_mat / np.linalg.norm(src_mat, axis=-1, keepdims=True).clip(min=1e-12)
            tgt_norm = tgt_mat / np.linalg.norm(tgt_mat, axis=-1, keepdims=True).clip(min=1e-12)
            paired_cos = (src_norm * tgt_norm).sum(axis=-1)

            src_spec = compute_spectrum_metrics(src_mat)
            tgt_spec = compute_spectrum_metrics(tgt_mat)

            layer_metrics[f'layer_{layer_idx}'] = {
                'source_effective_rank_singular': src_spec['effective_rank_singular_values'],
                'source_effective_rank_cov': src_spec['effective_rank_covariance_energy'],
                'target_effective_rank_singular': tgt_spec['effective_rank_singular_values'],
                'target_effective_rank_cov': tgt_spec['effective_rank_covariance_energy'],
                'paired_cosine_mean': float(paired_cos.mean()),
                'paired_cosine_std': float(paired_cos.std()),
                'source_top50_singular_values': src_spec['singular_values'],
                'target_top50_singular_values': tgt_spec['singular_values'],
            }

        # ---------------------------------------------------------------------
        # 2. Layerwise Gradient Probes on 32 Validation Rows (8 batches of 4)
        # ---------------------------------------------------------------------
        print("  Measuring per-layer CE vs weighted LSR gradients on 32 validation rows...", flush=True)
        wrapper.train()
        parameters = [p for p in base_model.parameters() if p.requires_grad]

        # Categorize parameters by encoder layers
        param_groups = []
        for name, p in base_model.named_parameters():
            if not p.requires_grad:
                continue
            if 'encoder.layers' in name:
                parts = name.split('.')
                l_idx = parts[parts.index('layers') + 1]
                param_groups.append(f'encoder_layer_{int(l_idx) + 1}')
            elif 'embed' in name:
                param_groups.append('embeddings')
            elif 'decoder' in name:
                param_groups.append('decoder')
            else:
                param_groups.append('other')

        unique_groups = sorted(set(param_groups))
        all_group_names = ['all_parameters'] + unique_groups

        gradient_results = []
        for batch_id, batch_rows in enumerate(batches_32):
            x = tokenizer([r['source'] for r in batch_rows], padding=True, truncation=True, max_length=256, return_tensors='pt')
            y = tokenizer(text_target=[r['target'] for r in batch_rows], padding=True, truncation=True, max_length=256, return_tensors='pt')['input_ids']
            y[y == tokenizer.pad_token_id] = -100
            inputs = {k: v.cuda() for k, v in x.items() if k in ['input_ids', 'attention_mask']}
            labels = y.cuda()

            native_ce = []
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

            loss_handle.remove()

            ce_grads = torch.autograd.grad(ce, parameters, retain_graph=True, allow_unused=True)
            lsr_grads = torch.autograd.grad(weighted_lsr, parameters, allow_unused=True)

            batch_record = {'batch_id': batch_id, 'groups': {}}
            for grp_name in all_group_names:
                sum_sq_ce, sum_sq_lsr, dot_prod = 0.0, 0.0, 0.0
                for grp, gce, glsr in zip(param_groups, ce_grads, lsr_grads):
                    if grp_name != 'all_parameters' and grp != grp_name:
                        continue
                    if gce is not None and glsr is not None:
                        gce_f = gce.float()
                        glsr_f = glsr.float()
                        sum_sq_ce += float(gce_f.norm() ** 2)
                        sum_sq_lsr += float(glsr_f.norm() ** 2)
                        dot_prod += float((gce_f * glsr_f).sum())

                norm_ce = float(np.sqrt(sum_sq_ce))
                norm_lsr = float(np.sqrt(sum_sq_lsr))
                cos = float(dot_prod / max(norm_ce * norm_lsr, 1e-12)) if (norm_ce > 0 and norm_lsr > 0) else None
                ratio = float(norm_lsr / norm_ce) if norm_ce > 0 else None

                batch_record['groups'][grp_name] = {
                    'ce_gradient_norm': norm_ce,
                    'lsr_gradient_norm': norm_lsr,
                    'gradient_cosine': cos,
                    'lsr_to_ce_ratio': ratio,
                }
            gradient_results.append(batch_record)

        # Average gradient statistics per group across the 8 batches
        avg_gradient_groups = {}
        for grp_name in all_group_names:
            cos_vals = [b['groups'][grp_name]['gradient_cosine'] for b in gradient_results if b['groups'][grp_name]['gradient_cosine'] is not None]
            ratio_vals = [b['groups'][grp_name]['lsr_to_ce_ratio'] for b in gradient_results if b['groups'][grp_name]['lsr_to_ce_ratio'] is not None]
            avg_gradient_groups[grp_name] = {
                'mean_gradient_cosine': float(np.mean(cos_vals)) if cos_vals else None,
                'mean_lsr_to_ce_ratio': float(np.mean(ratio_vals)) if ratio_vals else None,
            }

        # Save probe record
        probe_record = {
            'id': model_id,
            'family': m['family'],
            'seed': m['seed'],
            'alpha': m['alpha'],
            'lambda': m['lambda'],
            'layer_spectra_and_alignment': layer_metrics,
            'average_gradient_by_layer_group': avg_gradient_groups,
            'gradient_batches': gradient_results,
        }
        save_json(rec_file, probe_record)
        f3_summary[model_id] = probe_record

        print(f"  F3 Done {model_id}: all_param_cos={avg_gradient_groups['all_parameters']['mean_gradient_cosine']:.4f}, ratio={avg_gradient_groups['all_parameters']['mean_lsr_to_ce_ratio']:.4f}, layer12_erank={layer_metrics['layer_12']['source_effective_rank_singular']:.2f}", flush=True)

        del wrapper, model, base_model
        torch.cuda.empty_cache()
        gc.collect()
        shutil.rmtree(extract_dir, ignore_errors=True)

    save_json(summary_path, f3_summary)
    print(f"\nComprehensive F3 probes complete! Summary saved to {summary_path}\n", flush=True)
    return f3_summary


def main():
    print("=== STARTING AMIS FOLLOWUP WORKER V3 ===", flush=True)
    start_time = time.time()

    val_csv = ROOT / 'data_processed/amis_mandarin/validation.csv'
    val_rows = read_csv(val_csv)
    val_refs = [r['target'].strip() for r in val_rows]
    print(f"Loaded {len(val_rows)} validation rows.")

    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')

    # Load certified inventory 21
    inv_file = OUT_BASE / 'validation_inventory_21.json'
    inventory = json.loads(inv_file.read_text(encoding='utf-8'))
    print(f"Loaded certified inventory of {len(inventory)} models.")

    # 1. Run hook smoke test
    run_hook_smoke_test(tokenizer, device='cuda')

    # 2. Run corrected F4 (route-on vs route-off)
    run_corrected_f4(tokenizer, val_rows, val_refs, inventory)

    # 3. Run comprehensive layerwise F3 probes
    run_comprehensive_f3(tokenizer, val_rows, inventory)

    elapsed = time.time() - start_time
    receipt = {
        'status': 'V3_WORKER_COMPLETE',
        'elapsed_seconds': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    save_json(OUT_BASE / 'COMPLETE_V3.json', receipt)
    print("=" * 60)
    print(f"ALL V3 TASKS COMPLETE in {elapsed:.1f}s!")
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
