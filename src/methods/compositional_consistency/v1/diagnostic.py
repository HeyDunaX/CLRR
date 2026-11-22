"""Zero-optimizer-step Step-0 diagnostic for CoCC on real mBART-50 / A100 checkpoints.

Without performing any optimizer update (0 optimizer steps, no test split access):
  1. Measures full-backbone gradient norm ratios rho_grad = ||grad L_KL||_2 / ||grad L_CE||_2
     at pretrained initialization across fixed train microbatches to verify CoCC beta calibration
     against single-sentence ISO_SYM_SG (beta=2.50).
  2. Restores public A100 comparator checkpoint GOC_SYM_B250_S42 (FiveC/amis-rewire-checkpoints)
     and measures S0 vs S1 compositional context-sensitivity KL across validation pairs
     (Seg 1 body/EOS distraction and Seg 2 y_{j,1}/interior).
"""
import gc
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from ...gold_ordered_consistency.v1.model import goc_loss
from .model import (
    build_composite_batch_s1,
    chunked_aligned_kl_loss,
    chunked_ce_loss,
)


GOC_SYM_B250_S42_REVISION = 'dfc11d02c7fe17fccd8d699b46c406d9bb3af1fe'
GOC_SYM_B250_S42_ARCHIVE = (
    'checkpoints/amis_mandarin/mbart-large-50/'
    'symmetric-stopgrad-consistency-v1/seed-42/'
    'a100-bf16-sdpa-b32-ga4-es10-select-bleu-beta2.50/best.zip'
)


def _kl_vec(za, zb):
    la = F.log_softmax(za.float(), dim=-1)
    lb = F.log_softmax(zb.float(), dim=-1)
    return (la.exp() * (la - lb)).sum(dim=-1)


def _ent_vec(za):
    la = F.log_softmax(za.float(), dim=-1)
    return -(la.exp() * la).sum(dim=-1)


def _measure_init_grad_ratio(model, train_batches, device, max_source_length=256, max_target_length=256):
    """Measure full-backbone init gradient norm ratio of CoCC KL vs ISO_SYM_SG KL and CE."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    def grad_vec():
        vecs = [p.grad.detach().float() if p.grad is not None else None for p in params]
        sq = sum(float(g.pow(2).sum().item()) for g in vecs if g is not None)
        return math.sqrt(sq), vecs

    ce_vals, kl_iso_vals, kl_cocc_vals = [], [], []
    g_ce_norms, g_iso_norms, g_cocc_norms, cos_list = [], [], [], []
    pad_id = int(model.config.pad_token_id)
    dec_start_cfg = getattr(model.config, 'decoder_start_token_id', None)
    dec_start_id = int(model.config.eos_token_id if dec_start_cfg is None else dec_start_cfg)

    for b_idx, batch in enumerate(train_batches):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        dec_iso = model.prepare_decoder_input_ids_from_labels(labels)

        torch.manual_seed(1234 + b_idx)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(1234 + b_idx)
        comp = build_composite_batch_s1(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pad_token_id=pad_id,
            decoder_start_token_id=dec_start_id,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            mask_y_j1=True,
            layout='s1',
        )
        if comp is None:
            continue

        # 1) CE gradient
        model.zero_grad(set_to_none=True)
        torch.manual_seed(2026 + b_idx)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(2026 + b_idx)
        z_iso = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec_iso, use_cache=False).logits
        ce = chunked_ce_loss(z_iso, labels, chunk_size=32)
        del z_iso
        ce.backward()
        g_ce, vec_ce = grad_vec()
        ce_vals.append(float(ce.detach()))
        g_ce_norms.append(g_ce)

        # 2) ISO_SYM_SG pure KL gradient (two dropout views of isolated batch)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(2026 + b_idx)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(2026 + b_idx)
        z1 = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec_iso, use_cache=False).logits
        z2 = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec_iso, use_cache=False).logits
        _, _, kl_iso, _ = goc_loss(z1, z2, labels, mode='sym_sg', beta=1.0, chunk_size=32)
        del z1, z2
        kl_iso.backward()
        g_iso, _ = grad_vec()
        kl_iso_vals.append(float(kl_iso.detach()))
        g_iso_norms.append(g_iso)

        # 3) COCC_SYM pure KL gradient (Layout S1 + mask_y_j1=True)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(2026 + b_idx)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(2026 + b_idx)
        zi = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec_iso, use_cache=False).logits
        zc = model(
            input_ids=comp['input_ids'],
            attention_mask=comp['attention_mask'],
            decoder_input_ids=comp['decoder_input_ids'],
            use_cache=False,
        ).logits
        kl_cocc = chunked_aligned_kl_loss(
            zi, zc, comp['iso_flat_idx'], comp['comp_flat_idx'], direction='sym_sg', chunk_size=32
        )
        del zi, zc
        kl_cocc.backward()
        g_cocc, vec_cocc = grad_vec()
        kl_cocc_vals.append(float(kl_cocc.detach()))
        g_cocc_norms.append(g_cocc)

        dot = sum(
            float((a * b).sum().item())
            for a, b in zip(vec_ce, vec_cocc)
            if a is not None and b is not None
        )
        cos_list.append(dot / max(1e-12, g_ce * g_cocc))
        model.zero_grad(set_to_none=True)
        del vec_ce, vec_cocc

    mean_g_ce = float(np.mean(g_ce_norms))
    mean_g_iso = float(np.mean(g_iso_norms))
    mean_g_cocc = float(np.mean(g_cocc_norms))
    rho_iso = mean_g_iso / max(1e-12, mean_g_ce)
    rho_cocc = mean_g_cocc / max(1e-12, mean_g_ce)
    rec_beta = round(min(2.5, max(0.5, 2.5 * rho_iso / max(1e-12, rho_cocc))) * 4.0) / 4.0
    return dict(
        num_batches=len(ce_vals),
        ce_iso=round(float(np.mean(ce_vals)), 4),
        kl_iso_sym_sg=round(float(np.mean(kl_iso_vals)), 4),
        kl_cocc_sym=round(float(np.mean(kl_cocc_vals)), 4),
        grad_norm_ce=round(mean_g_ce, 4),
        grad_norm_kl_iso=round(mean_g_iso, 4),
        grad_norm_kl_cocc=round(mean_g_cocc, 4),
        rho_iso=round(rho_iso, 4),
        rho_cocc=round(rho_cocc, 4),
        ratio_cocc_over_iso=round(rho_cocc / max(1e-12, rho_iso), 4),
        cos_cocc_ce=round(float(np.mean(cos_list)), 4),
        recommended_beta_matched_to_250=float(rec_beta),
    )


def _probe_s0_vs_s1_eval(model, val_rows, tokenizer, device, max_pairs=128):
    """Measure S0 vs S1 context-sensitivity KL on frozen checkpoint in eval mode."""
    model.eval()
    eos_id = int(tokenizer.eos_token_id)
    results = {}
    n_pairs = min(int(max_pairs), len(val_rows) // 2)
    pairs = [(2 * k, 2 * k + 1) for k in range(n_pairs)]

    for mode in ('S0', 'S1'):
        s1_body, s1_eos, s2_y1, s2_int = [], [], [], []
        eos_iso_probs, eos_comp_probs, ent_iso_list, ent_comp_list = [], [], [], []
        with torch.no_grad():
            for i, j in pairs:
                xi = tokenizer(val_rows['source'].iloc[i], truncation=True, max_length=256)['input_ids']
                xj = tokenizer(val_rows['source'].iloc[j], truncation=True, max_length=256)['input_ids']
                yi = tokenizer(text_target=val_rows['target'].iloc[i], truncation=True, max_length=256)['input_ids']
                yj = tokenizer(text_target=val_rows['target'].iloc[j], truncation=True, max_length=256)['input_ids']
                if len(yi) < 3 or len(yj) < 3 or len(xi) + len(xj) > 256 or len(yi) + len(yj) > 256:
                    continue

                zi = model(input_ids=torch.tensor([xi], device=device), labels=torch.tensor([yi], device=device)).logits[0]
                zj = model(input_ids=torch.tensor([xj], device=device), labels=torch.tensor([yj], device=device)).logits[0]
                if mode == 'S0':
                    xc = xi[:-1] + xj[1:]
                    yc = yi[:-1] + yj[1:]
                    idx_i = list(range(1, len(yi) - 1))
                    c_i = list(range(1, len(yi) - 1))
                    idx_j = list(range(1, len(yj)))
                    c_j = list(range(len(yi) - 1, len(yi) - 1 + len(yj) - 1))
                    yc_tensor = torch.tensor([yc], device=device)
                    dec_c = model.prepare_decoder_input_ids_from_labels(yc_tensor)
                    zc = model(
                        input_ids=torch.tensor([xc], device=device),
                        decoder_input_ids=dec_c,
                    ).logits[0]
                else:
                    xc = xi + xj
                    yc = yi + yj
                    idx_i = list(range(1, len(yi)))
                    c_i = list(range(1, len(yi)))
                    idx_j = list(range(1, len(yj)))
                    c_j = list(range(len(yi) + 1, len(yi) + len(yj)))
                    yc_tensor = torch.tensor([yc], device=device)
                    dec_c = model.prepare_decoder_input_ids_from_labels(yc_tensor)
                    zc = model(
                        input_ids=torch.tensor([xc], device=device),
                        decoder_input_ids=dec_c,
                    ).logits[0]

                k1 = _kl_vec(zi[idx_i], zc[c_i])
                k2 = _kl_vec(zj[idx_j], zc[c_j])
                if mode == 'S1':
                    s1_body.extend(k1[:-1].tolist())
                    s1_eos.append(float(k1[-1].item()))
                    eos_iso_probs.append(float(F.softmax(zi[idx_i[-1]].float(), -1)[eos_id].item()))
                    eos_comp_probs.append(float(F.softmax(zc[c_i[-1]].float(), -1)[eos_id].item()))
                else:
                    s1_body.extend(k1.tolist())
                    eos_iso_probs.append(float(F.softmax(zj[idx_j[0]].float(), -1)[eos_id].item()))
                    eos_comp_probs.append(float(F.softmax(zc[c_j[0]].float(), -1)[eos_id].item()))

                s2_y1.append(float(k2[0].item()))
                if len(k2) > 1:
                    s2_int.extend(k2[1:].tolist())
                ent_iso_list.append(float(_ent_vec(zi[idx_i]).mean().item()))
                ent_comp_list.append(float(_ent_vec(zc[c_i]).mean().item()))
                ent_iso_list.append(float(_ent_vec(zj[idx_j]).mean().item()))
                ent_comp_list.append(float(_ent_vec(zc[c_j]).mean().item()))

        results[mode] = dict(
            pairs_measured=len(s2_y1),
            s1_body_kl=round(float(np.mean(s1_body)), 4) if s1_body else 0.0,
            s1_eos_kl=round(float(np.mean(s1_eos)), 4) if s1_eos else None,
            s2_y1_kl=round(float(np.mean(s2_y1)), 4) if s2_y1 else 0.0,
            s2_int_kl=round(float(np.mean(s2_int)), 4) if s2_int else 0.0,
            ent_iso=round(float(np.mean(ent_iso_list)), 4) if ent_iso_list else 0.0,
            ent_comp=round(float(np.mean(ent_comp_list)), 4) if ent_comp_list else 0.0,
            eos_prob_iso=round(float(np.mean(eos_iso_probs)), 4) if eos_iso_probs else 0.0,
            eos_prob_comp=round(float(np.mean(eos_comp_probs)), 4) if eos_comp_probs else 0.0,
        )
    return results


def run_step0_diagnostic(config_path, num_val_pairs=128, num_grad_batches=4):
    """Run 0-optimizer-step Step-0 calibration & trained context-sensitivity probe."""
    from experiment_runner import engine
    from experiment_runner.config import verify_frozen
    from experiment_runner.integrity import restore_archive

    out, plan = verify_frozen(config_path)
    engine.configure(out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    started = time.time()

    cocc_spec = next((r for r in plan['runs'] if r['method'] == 'cocc-sym-v1'), plan['runs'][0])
    use_bf16 = bool(plan['protocol'].get('bf16') and device.type == 'cuda')

    # 1) Pretrained init gradient-norm calibration on real train microbatches (0 optimizer steps)
    base_spec = dict(cocc_spec, method='baseline')
    model, tokenizer, _ = engine.load_model(base_spec)
    model.to(device)
    if use_bf16:
        model.to(dtype=torch.bfloat16)

    _, train_ds = engine.dataset('train', tokenizer, base_spec)
    raw_val, _ = engine.dataset('validation', tokenizer, base_spec)
    coll = engine.collator(model, tokenizer, base_spec)
    bsz = int(base_spec['microbatch'])
    train_batches = [
        coll([train_ds[i] for i in range(k * bsz, min((k + 1) * bsz, len(train_ds)))])
        for k in range(int(num_grad_batches))
        if min((k + 1) * bsz, len(train_ds)) - k * bsz >= 2
    ]

    init_grad = _measure_init_grad_ratio(
        model,
        train_batches,
        device,
        max_source_length=int(cocc_spec.get('max_source_length', 256)),
        max_target_length=int(cocc_spec.get('max_target_length', 256)),
    )
    init_probe = _probe_s0_vs_s1_eval(model, raw_val, tokenizer, device, max_pairs=min(32, num_val_pairs))
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # 2) Public converged GOC_SYM_B250_S42 checkpoint probe (S0 vs S1 context sensitivity)
    trained_probe = {}
    try:
        with tempfile.TemporaryDirectory(prefix='cocc_step0_') as tmpdir:
            tmp = Path(tmpdir)
            archive_path = hf_hub_download(
                repo_id=plan['repo'],
                filename=GOC_SYM_B250_S42_ARCHIVE,
                revision=GOC_SYM_B250_S42_REVISION,
                local_dir=str(tmp / 'dl'),
            )
            ckpt_dir = tmp / 'restored'
            restore_archive(Path(archive_path), ckpt_dir)
            sym_spec = dict(
                cocc_spec,
                method='symmetric-stopgrad-consistency-v1',
                distance=2,
                strength=0.0,
                jepa_weight=0.0,
                consistency_beta=2.5,
            )
            sym_model, sym_tok, _ = engine.load_model(sym_spec, folder=ckpt_dir)
            sym_model.to(device)
            if use_bf16:
                sym_model.to(dtype=torch.bfloat16)
            trained_probe = {
                'status': 'MEASURED',
                'checkpoint_id': 'GOC_SYM_B250_S42',
                'revision': GOC_SYM_B250_S42_REVISION,
                'probe': _probe_s0_vs_s1_eval(sym_model, raw_val, sym_tok, device, max_pairs=num_val_pairs),
            }
            del sym_model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    except Exception as exc:
        trained_probe = {
            'status': 'SKIPPED_WARNING',
            'checkpoint_id': 'GOC_SYM_B250_S42',
            'error': str(exc),
        }

    configured_beta = float(cocc_spec.get('consistency_beta', 1.25))
    report = dict(
        status='COMPLETE',
        optimizer_steps=0,
        test_split_opened=False,
        configured_cocc_beta=configured_beta,
        init_gradient_calibration=init_grad,
        init_s0_vs_s1_probe=init_probe,
        trained_sym_sg_probe=trained_probe,
        seconds=round(time.time() - started, 2),
    )
    out_path = out / 'step0_diagnostic.json'
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('COCC_STEP0_DIAGNOSTIC_COMPLETE', json.dumps(report, indent=2), flush=True)
    return report
