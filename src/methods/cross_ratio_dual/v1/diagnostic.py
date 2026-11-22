"""Zero-optimizer-step Step-0 diagnostic for CR-Dual on real mBART-50/NLLB-200 checkpoints.

Measures on length-matched validation & train pairs without modifying model weights:
  1. Distribution and sign of normalized cross-ratios M_fwd/L_ij vs M_rev/L_ij.
  2. Paired-RNG shared dropout variance reduction across K=8 dropout repeats
     (target: >= 70% reduction vs independent dropout masks).
  3. Full-backbone gradient norm ratio rho_grad = ||grad L_CR||_2 / ||grad L_CE||_2
     and cosine similarity cos(grad L_CR, grad L_CE_fwd) to verify lambda_max calibration.
"""
import gc
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

from .model import (
    CrossRatioDualMT,
    build_length_matched_involution,
    chunked_seq_logprob_and_ce,
    cross_ratio_huber_loss,
)


def _eval_cr_batches(model, batches, device, paired_rng=True, eval_dropout=True, seed=42):
    """Collect per-pair normalized M_fwd and M_rev across batches."""
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model.train(eval_dropout)
    prev_paired = model.paired_rng
    model.paired_rng = paired_rng
    m_fwds, m_revs, diffs = [], [], []
    with torch.no_grad():
        for batch in batches:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            dec_fwd = model.prepare_decoder_input_ids_from_labels(labels)
            src_lengths = attention_mask.sum(dim=1)
            tgt_lengths = labels.ne(-100).sum(dim=1)
            idx_i, idx_j, pi = build_length_matched_involution(src_lengths, tgt_lengths)
            if int(idx_i.numel()) == 0:
                continue

            enc_x = model.base_model.get_encoder()(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state
            rng_f = (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
            lg_fd = model._decode_from_encoder(enc_x, attention_mask, dec_fwd)
            lp_fd, len_y, _ = chunked_seq_logprob_and_ce(lg_fd, labels, chunk_size=model.logit_chunk_tokens)
            del lg_fd
            if paired_rng:
                torch.set_rng_state(rng_f[0])
                if device.type == 'cuda' and rng_f[1] is not None:
                    torch.cuda.set_rng_state(rng_f[1], device)
            lg_fo = model._decode_from_encoder(
                enc_x.index_select(0, pi), attention_mask.index_select(0, pi), dec_fwd
            )
            lp_fo, _, _ = chunked_seq_logprob_and_ce(lg_fo, labels, chunk_size=model.logit_chunk_tokens)
            del lg_fo

            rev_ids, rev_mask, rev_labels = model._build_reverse_batch(input_ids, attention_mask, labels)
            dec_rev = model.prepare_decoder_input_ids_from_labels(rev_labels)
            enc_y = model.base_model.get_encoder()(
                input_ids=rev_ids, attention_mask=rev_mask, return_dict=True
            ).last_hidden_state
            rng_r = (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
            lg_rd = model._decode_from_encoder(enc_y, rev_mask, dec_rev)
            lp_rd, len_x, _ = chunked_seq_logprob_and_ce(lg_rd, rev_labels, chunk_size=model.logit_chunk_tokens)
            del lg_rd
            if paired_rng:
                torch.set_rng_state(rng_r[0])
                if device.type == 'cuda' and rng_r[1] is not None:
                    torch.cuda.set_rng_state(rng_r[1], device)
            lg_ro = model._decode_from_encoder(
                enc_y.index_select(0, pi), rev_mask.index_select(0, pi), dec_rev
            )
            lp_ro, _, _ = chunked_seq_logprob_and_ce(lg_ro, rev_labels, chunk_size=model.logit_chunk_tokens)
            del lg_ro

            m_f = (lp_fd[idx_i] + lp_fd[idx_j]) - (lp_fo[idx_i] + lp_fo[idx_j])
            m_r = (lp_rd[idx_i] + lp_rd[idx_j]) - (lp_ro[idx_i] + lp_ro[idx_j])
            l_ij = 0.5 * (len_x[idx_i] + len_x[idx_j] + len_y[idx_i] + len_y[idx_j]).clamp(min=1.0)
            nf = (m_f / l_ij).cpu().numpy()
            nr = (m_r / l_ij).cpu().numpy()
            m_fwds.append(nf)
            m_revs.append(nr)
            diffs.append(nf - nr)
    model.paired_rng = prev_paired
    mf = np.concatenate(m_fwds) if m_fwds else np.zeros(0)
    mr = np.concatenate(m_revs) if m_revs else np.zeros(0)
    df = np.concatenate(diffs) if diffs else np.zeros(0)
    return mf, mr, df


def _measure_grad_ratio(model, batch, device, alpha_rev=0.5, delta=1.0):
    """Measure ||grad L_CR||_2 / ||grad L_CE||_2 and cos(grad L_CR, grad L_CE_fwd) with 0 optimizer steps."""
    model.train()
    model.zero_grad(set_to_none=True)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    dec_fwd = model.prepare_decoder_input_ids_from_labels(labels)
    idx_i, idx_j, pi = build_length_matched_involution(attention_mask.sum(dim=1), labels.ne(-100).sum(dim=1))

    # 1) Gradient of CE_fwd + alpha_rev * CE_rev (deterministic seed)
    torch.manual_seed(1234)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(1234)
    enc_x = model.base_model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
    lg_fd = model._decode_from_encoder(enc_x, attention_mask, dec_fwd)
    _, _, ce_fwd = chunked_seq_logprob_and_ce(lg_fd, labels, chunk_size=model.logit_chunk_tokens)
    del lg_fd
    rev_ids, rev_mask, rev_labels = model._build_reverse_batch(input_ids, attention_mask, labels)
    dec_rev = model.prepare_decoder_input_ids_from_labels(rev_labels)
    enc_y = model.base_model.get_encoder()(input_ids=rev_ids, attention_mask=rev_mask, return_dict=True).last_hidden_state
    lg_rd = model._decode_from_encoder(enc_y, rev_mask, dec_rev)
    _, _, ce_rev = chunked_seq_logprob_and_ce(lg_rd, rev_labels, chunk_size=model.logit_chunk_tokens)
    del lg_rd
    (ce_fwd + alpha_rev * ce_rev).backward()

    params = [p for p in model.parameters() if p.requires_grad]
    g_ce = [p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32) for p in params]
    ce_norm = math.sqrt(sum(float(g.pow(2).sum().item()) for g in g_ce))
    model.zero_grad(set_to_none=True)

    # 2) Gradient of pure L_CR (same seed, Paired-RNG)
    torch.manual_seed(1234)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(1234)
    enc_x = model.base_model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
    rng_f = (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
    lg_fd = model._decode_from_encoder(enc_x, attention_mask, dec_fwd)
    lp_fd, len_y, _ = chunked_seq_logprob_and_ce(lg_fd, labels, chunk_size=model.logit_chunk_tokens)
    del lg_fd
    torch.set_rng_state(rng_f[0])
    if device.type == 'cuda' and rng_f[1] is not None:
        torch.cuda.set_rng_state(rng_f[1], device)
    lg_fo = model._decode_from_encoder(enc_x.index_select(0, pi), attention_mask.index_select(0, pi), dec_fwd)
    lp_fo, _, _ = chunked_seq_logprob_and_ce(lg_fo, labels, chunk_size=model.logit_chunk_tokens)
    del lg_fo

    enc_y = model.base_model.get_encoder()(input_ids=rev_ids, attention_mask=rev_mask, return_dict=True).last_hidden_state
    rng_r = (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
    lg_rd = model._decode_from_encoder(enc_y, rev_mask, dec_rev)
    lp_rd, len_x, _ = chunked_seq_logprob_and_ce(lg_rd, rev_labels, chunk_size=model.logit_chunk_tokens)
    del lg_rd
    torch.set_rng_state(rng_r[0])
    if device.type == 'cuda' and rng_r[1] is not None:
        torch.cuda.set_rng_state(rng_r[1], device)
    lg_ro = model._decode_from_encoder(enc_y.index_select(0, pi), rev_mask.index_select(0, pi), dec_rev)
    lp_ro, _, _ = chunked_seq_logprob_and_ce(lg_ro, rev_labels, chunk_size=model.logit_chunk_tokens)
    del lg_ro

    cr_loss, cr_stats = cross_ratio_huber_loss(
        lp_fd, lp_fo, lp_rd, lp_ro, len_x, len_y, idx_i, idx_j, delta=delta, shuffle_reverse=False
    )
    cr_loss.backward()
    g_cr = [p.grad.detach().float() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32) for p in params]
    cr_norm = math.sqrt(sum(float(g.pow(2).sum().item()) for g in g_cr))
    dot = sum(float((a * b).sum().item()) for a, b in zip(g_ce, g_cr))
    cos = dot / max(1e-12, ce_norm * cr_norm)
    model.zero_grad(set_to_none=True)
    del g_ce, g_cr

    rho_grad = cr_norm / max(1e-12, ce_norm)
    return dict(
        ce_fwd=float(ce_fwd.detach()),
        ce_rev=float(ce_rev.detach()),
        cr_loss=float(cr_loss.detach()),
        grad_norm_ce=ce_norm,
        grad_norm_cr=cr_norm,
        rho_grad=rho_grad,
        cos_cr_ce=cos,
        recommended_lambda_for_10pct_ratio=round(0.10 / max(1e-6, rho_grad), 4),
        cr_stats=cr_stats,
    )


def run_step0_diagnostic(config_path, bit_run_id='BIT_CE_S42', num_val_pairs=100, dropout_repeats=8):
    """Run zero-optimizer-step Step-0 diagnostic on trained BIT_CE checkpoint (restored from HF if cleaned)."""
    from experiment_runner import engine
    from experiment_runner.config import verify_frozen
    from experiment_runner.integrity import restore_archive

    out, plan = verify_frozen(config_path)
    engine.configure(out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bit_spec = next((r for r in plan['runs'] if r['id'] == bit_run_id), None)
    if bit_spec is None:
        raise ValueError(f'Run {bit_run_id} not found in plan')
    val_receipt_path = out / 'runs' / bit_run_id / 'VAL_COMPLETE.json'
    if not val_receipt_path.exists():
        raise FileNotFoundError(f'Complete {bit_run_id} before running Step-0 diagnostic')
    val_receipt = json.loads(val_receipt_path.read_text(encoding='utf-8'))

    started = time.time()
    with tempfile.TemporaryDirectory(prefix='cr_dual_step0_') as tmpdir:
        tmp = Path(tmpdir)
        local_export = out / 'runs' / bit_run_id / 'best_model'
        if local_export.exists():
            ckpt_dir = local_export
        else:
            archive_path = hf_hub_download(
                repo_id=plan['repo'],
                filename=bit_spec['remote_prefix'] + '/best.zip',
                revision=val_receipt['backup']['revision'],
                local_dir=str(tmp / 'dl'),
            )
            ckpt_dir = tmp / 'restored'
            restore_archive(Path(archive_path), ckpt_dir)

        cr_spec = next((r for r in plan['runs'] if r['method'] == 'cross-ratio-dual-v1'), bit_spec)
        model, tokenizer, _ = engine.load_model(bit_spec, folder=ckpt_dir)
        model.to(device)
        if plan['protocol'].get('bf16') and device.type == 'cuda':
            model.to(dtype=torch.bfloat16)

        _, val_ds = engine.dataset('validation', tokenizer, bit_spec)
        collator = engine.collator(model, tokenizer, bit_spec)
        bsz = min(int(bit_spec['microbatch']), 32)
        num_rows = min(len(val_ds), (num_val_pairs * 2 // bsz + 1) * bsz)
        num_rows = (num_rows // 2) * 2
        val_batches = [
            collator([val_ds[i] for i in range(s, min(s + bsz, num_rows))])
            for s in range(0, num_rows, bsz)
            if min(s + bsz, num_rows) - s >= 2
        ]

        # 1) Eval-mode (no dropout) true cross-ratio gap on BIT_CE checkpoint
        mf_det, mr_det, df_det = _eval_cr_batches(model, val_batches, device, paired_rng=True, eval_dropout=False)

        # 2) Paired-RNG vs Independent-RNG variance over K dropout repeats on first 2 batches
        probe_batches = val_batches[:max(1, min(2, len(val_batches)))]
        rep_paired = np.stack([
            _eval_cr_batches(model, probe_batches, device, paired_rng=True, eval_dropout=True, seed=100 + k)[0]
            for k in range(dropout_repeats)
        ], axis=0)
        rep_indep = np.stack([
            _eval_cr_batches(model, probe_batches, device, paired_rng=False, eval_dropout=True, seed=100 + k)[0]
            for k in range(dropout_repeats)
        ], axis=0)
        var_paired = float(np.mean(np.var(rep_paired, axis=0)))
        var_indep = float(np.mean(np.var(rep_indep, axis=0)))
        var_reduction = 1.0 - (var_paired / max(1e-12, var_indep))

        # 3) Gradient norm calibration on 1 validation batch (0 optimizer steps)
        grad_info = _measure_grad_ratio(
            model,
            val_batches[0],
            device,
            alpha_rev=float(cr_spec.get('alpha_rev', 0.5)),
            delta=float(cr_spec.get('cr_delta', 1.0)),
        )
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    configured_lambda = float(cr_spec.get('cr_lambda', 0.25))
    effective_ratio = configured_lambda * grad_info['rho_grad']
    report = dict(
        status='COMPLETE',
        optimizer_steps=0,
        test_split_opened=False,
        checkpoint_run_id=bit_run_id,
        checkpoint_val_bleu=val_receipt['scores'].get('bleu'),
        checkpoint_val_chrf0=val_receipt['scores'].get('chrf0'),
        num_val_pairs=int(len(mf_det)),
        deterministic_cross_ratio=dict(
            mean_m_fwd=float(np.mean(mf_det)),
            mean_m_rev=float(np.mean(mr_det)),
            mean_m_fwd_minus_m_rev=float(np.mean(df_det)),
            median_m_fwd_minus_m_rev=float(np.median(df_det)),
            mean_abs_violation=float(np.mean(np.abs(df_det))),
            frac_m_fwd_lt_m_rev=float(np.mean(df_det < 0.0)),
        ),
        paired_rng_variance_check=dict(
            dropout_repeats=dropout_repeats,
            mean_var_independent=var_indep,
            mean_var_paired_rng=var_paired,
            variance_reduction_fraction=var_reduction,
            passed_70pct_target=bool(var_reduction >= 0.70),
        ),
        gradient_calibration=dict(
            **grad_info,
            configured_cr_lambda=configured_lambda,
            configured_effective_grad_ratio=effective_ratio,
            within_safe_band_002_to_030=bool(0.02 <= effective_ratio <= 0.30),
        ),
        seconds=round(time.time() - started, 2),
    )
    out_path = out / 'step0_diagnostic.json'
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('STEP0_DIAGNOSTIC_COMPLETE', json.dumps(report, indent=2), flush=True)
    return report
