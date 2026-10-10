"""Verified-protocol Follow-up Worker v4 implementing Sections 16.2 (Corrected F4) and 16.4/16.5 (Layerwise F3).

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
OUT_BASE = ROOT / 'outputs_rebuttal/amis_followup_20261004_v4'
from amis_v4_measurements import (spectrum, gradient_group, Collector, load_checkpoint, strict_model, runtime, local_checks, teacher_diagnostics)


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
    f4_out_dir = OUT_BASE / 'route_control'
    f4_out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = f4_out_dir / 'route_control_summary.json'
    zh_token_id = tokenizer.convert_tokens_to_ids('zh_CN')

    f4_models = [m for m in inventory if m['family'] == 'original_full_detached']
    assert len(f4_models) in (1, 3), f"Expected sentinel or 3 Full models, got {len(f4_models)}"

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
        ckpt_dir = load_checkpoint(m, extract_dir, OUT_BASE / 'loading_receipts')

        # 1. Load clean base model
        base_model = strict_model(ckpt_dir).eval()

        rewired = CrossLayerResidualRewire(base_model, distance=2, strength=0.10, stack='encoder')
        diag = teacher_diagnostics(base_model, rewired, tokenizer, val_rows, FIXED_BATCHES)
        print('  Teacher diagnostics:', diag['ce_delta_on_minus_off'], 'alpha0/raw:', diag['alpha0_vs_clean_raw_max'], flush=True)

        # 3. Greedy decoding on 576 validation rows
        # A. Route-on (strength=0.10)
        rewired.strength = 0.10
        preds_on = []
        expected_rows = read_csv(ROOT / 'replay_reference' / model_id / 'val_predictions_route_on.csv')
        assert len(expected_rows) == len(val_rows)
        expected = [r['prediction'].strip() for r in expected_rows]
        for s in range(0, len(val_rows), 8):
            chunk = val_rows[s:s + 8]
            inputs = tokenizer([r['source'] for r in chunk], padding=True, truncation=True, max_length=256, pad_to_multiple_of=8, return_tensors='pt')
            inputs = {k: v.cuda() for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
            with torch.no_grad(), generation_precision(rewired, 'fp32'), fresh(rewired):
                ids = rewired.generate(**inputs, num_beams=1, max_length=256, forced_bos_token_id=zh_token_id)
            decoded = [p.strip() for p in tokenizer.batch_decode(ids, skip_special_tokens=True)]
            assert decoded == expected[s:s + len(decoded)], f'Route-on replay mismatch: {model_id}, rows {s}..{s+len(decoded)}'
            preds_on.extend(decoded)

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
            'route_on_replay': 'EXACT_PREDICTION_MATCH_V2',
            'fixed_validation_indices': FIXED_BATCHES,
            'sentences_changed': changed,
            'pct_changed': pct_changed,
            'fixed_32_diagnostics': diag,
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




def run_comprehensive_f3(tokenizer, val_rows, inventory):
    f3_out_dir = OUT_BASE / 'probes'
    f3_out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = f3_out_dir / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = f3_out_dir / 'probe_summary.json'

    # Fixed 32 rows for gradient probes
    batches_32 = [[val_rows[i] for i in indices] for indices in FIXED_BATCHES]
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
        ckpt_dir = load_checkpoint(m, extract_dir, OUT_BASE / 'loading_receipts')

        base_model = strict_model(ckpt_dir)

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
        collector = Collector(base_model, rewired_layer)

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
                collector.row_start = s
                with torch.no_grad(), fresh(wrapper):
                    base_model.get_encoder()(**tokens)

            for layer_idx in [*range(1, collector.num_layers + 1), 'final']:
                mat = np.concatenate(collector.pooled_by_layer[(phase, layer_idx)], axis=0)
                layer_reps[(phase, layer_idx)] = mat
        collector.remove()
        np.savez_compressed(run_out / 'representations.npz', **{f'{phase}_{site}': values for (phase, site), values in layer_reps.items()})
        pd.DataFrame(collector.residual_stats).to_csv(run_out / 'residuals.csv', index=False)

        # Compute spectra per layer and cosine alignment
        layer_metrics = {}
        for layer_idx in [*range(1, collector.num_layers + 1), 'final']:
            src_mat = layer_reps[('source', layer_idx)]
            tgt_mat = layer_reps[('target', layer_idx)]

            src_norm = src_mat / np.linalg.norm(src_mat, axis=-1, keepdims=True).clip(min=1e-12)
            tgt_norm = tgt_mat / np.linalg.norm(tgt_mat, axis=-1, keepdims=True).clip(min=1e-12)
            paired_cos = (src_norm * tgt_norm).sum(axis=-1)

            src_spec = spectrum(src_mat)
            tgt_spec = spectrum(tgt_mat)

            layer_metrics[f'layer_{layer_idx}'] = {
                'source_effective_rank_singular': src_spec['effective_rank_singular_values'],
                'source_effective_rank_cov': src_spec['effective_rank_covariance_energy'],
                'target_effective_rank_singular': tgt_spec['effective_rank_singular_values'],
                'target_effective_rank_cov': tgt_spec['effective_rank_covariance_energy'],
                'paired_cosine_mean': float(paired_cos.mean()),
                'paired_cosine_std': float(paired_cos.std()),
                'source_full_spectrum': src_spec,
                'site': 'final_encoder_output_after_LayerNorm' if layer_idx == 'final' else 'encoder_block_output_after_routing_before_final_LayerNorm',
                'target_full_spectrum': tgt_spec,
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
            encoder_grad_flags = []
            def capture_encoder_grad(_module, _inputs, output):
                encoder_grad_flags.append(output.last_hidden_state.requires_grad)
            encoder_handle = underlying.get_encoder().register_forward_hook(capture_encoder_grad)

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
                    assert not target.requires_grad
                    lsr = (1 - (src * tgt).sum(-1)).mean()
                    weighted_lsr = 0.1 * lsr.to(ce.dtype)

            loss_handle.remove()
            encoder_handle.remove()
            assert encoder_grad_flags == [True, False], encoder_grad_flags

            ce_grads = torch.autograd.grad(ce, parameters, retain_graph=True, allow_unused=True)
            lsr_grads = torch.autograd.grad(weighted_lsr, parameters, allow_unused=True)

            batch_record = {'batch_id': batch_id, 'indices': FIXED_BATCHES[batch_id], 'ce': float(ce.detach()), 'weighted_lsr': float(weighted_lsr.detach()), 'encoder_requires_grad': encoder_grad_flags, 'groups': {}}
            for grp_name in all_group_names:
                batch_record['groups'][grp_name] = gradient_group(param_groups, ce_grads, lsr_grads, grp_name)
            assert batch_record['groups']['decoder']['ce_gradient_norm'] > 0, 'decoder CE was omitted'
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
            'gradient_mode': 'train_BF16_TF32off_checkpointing_disabled',
            'counterfactual_lsr_lambda': 0.1 if m['lambda'] == 0 else None,
            'parameter_groups': [{'name': name, 'group': group, 'numel': p.numel()} for (name, p), group in zip([(n,p) for n,p in base_model.named_parameters() if p.requires_grad], param_groups)],
            'representations_sha256': digest(run_out / 'representations.npz'),
        }
        save_json(rec_file, probe_record)
        f3_summary[model_id] = probe_record

        print(f"  F3 Done {model_id}: all_param_cos={avg_gradient_groups['all_parameters']['mean_gradient_cosine']:.4f}, ratio={avg_gradient_groups['all_parameters']['mean_lsr_to_ce_ratio']:.4f}, layer12_erank={layer_metrics['layer_12']['source_effective_rank_singular']:.2f}", flush=True)

        if rewired_layer is not None:
            for handle in rewired_layer._handles:
                handle.remove()
            rewired_layer._handles.clear()
        del wrapper, model, base_model, rewired_layer, collector, underlying, parameters
        del ce_grads, lsr_grads, outputs, ce, weighted_lsr, native_ce, capture_loss, loss_handle
        del encoder_handle, capture_encoder_grad
        del layer_reps, src_mat, tgt_mat, src_norm, tgt_norm
        if m['lambda'] == 0:
            del target, src, tgt, lsr
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(extract_dir, ignore_errors=True)

    save_json(summary_path, f3_summary)
    print(f"\nComprehensive F3 probes complete! Summary saved to {summary_path}\n", flush=True)
    return f3_summary


def main():
    global FIXED_BATCHES
    print('=== STARTING AMIS FOLLOWUP WORKER V4 ===', flush=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    FIXED_BATCHES = json.loads((ROOT / 'trajectory_summary.json').read_text())['fixed_validation_indices']
    assert len(FIXED_BATCHES) == 8 and all(len(b) == 4 for b in FIXED_BATCHES)
    save_json(OUT_BASE / 'local_checks.json', local_checks())
    start_time = time.time()

    val_csv = ROOT / 'data_processed/amis_mandarin/validation.csv'
    val_rows = read_csv(val_csv)
    val_refs = [r['target'].strip() for r in val_rows]
    print(f"Loaded {len(val_rows)} validation rows.")

    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt', revision='e30b6cb8eb0d43a0b73cab73c7676b9863223a30', src_lang='tl_XX', tgt_lang='zh_CN')
    save_json(OUT_BASE / 'runtime.json', runtime(tokenizer, ROOT))
    assert tokenizer('source')['input_ids'][0] == 250047
    assert tokenizer(text_target='target')['input_ids'][0] == 250025

    # Load certified inventory 21
    inv_file = ROOT / 'validation_inventory_21.json'
    inventory = json.loads(inv_file.read_text(encoding='utf-8'))
    print(f"Loaded certified inventory of {len(inventory)} models.")

    # 1. Run hook smoke test
    run_hook_smoke_test(tokenizer, device='cuda')
    save_json(OUT_BASE / 'hook_smoke.json', {'status': 'PASS', 'alpha0_matches_raw': True, 'one_hook_per_layer': True, 'removed_hook_count': 0})

    # Two sentinels must pass before the rest of the queue.
    full42 = [m for m in inventory if m['family'] == 'original_full_detached' and m['seed'] == 42]
    baseline42 = [m for m in inventory if m['family'] == 'baseline' and m['seed'] == 42]
    assert len(full42) == len(baseline42) == 1
    run_corrected_f4(tokenizer, val_rows, val_refs, full42)
    run_comprehensive_f3(tokenizer, val_rows, baseline42)
    save_json(OUT_BASE / 'SENTINELS_PASS.json', {'route_replay_full42': 'PASS', 'baseline42_gradient_union': 'PASS'})

    # 2. Run corrected F4 (route-on vs route-off)
    run_corrected_f4(tokenizer, val_rows, val_refs, inventory)

    # 3. Run comprehensive layerwise F3 probes
    run_comprehensive_f3(tokenizer, val_rows, inventory)

    elapsed = time.time() - start_time
    receipt = {
        'status': 'V4_WORKER_COMPLETE',
        'elapsed_seconds': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    save_json(OUT_BASE / 'COMPLETE_V4.json', receipt)
    print("=" * 60)
    print(f"ALL V4 TASKS COMPLETE in {elapsed:.1f}s!")
    print("=" * 60, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save_json(OUT_BASE / 'FAILED.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
