"""Zero-optimizer-step Step-0 diagnostic for Off-EOS Stopping (`off_eos_stopping` v1).

Without performing any optimizer update (0 optimizer steps, no test split access):
  1. Restores verified public A100 checkpoint BASELINE_S42 (FiveC/amis-rewire-checkpoints)
     and measures two-sided stopping oracle headroom on amis_mandarin validation (576 rows):
       - native_greedy baseline BLEU / chrF0 / chrF++ / BP / LR
       - truncate_only_oracle (k <= k_eos)
       - extend_only_oracle (k >= k_eos, plus anti-repetition guard triggers)
       - two_sided_oracle (1 <= k <= T_b via sentence chrF0)
       - length_only_oracle (gold lexical body length clamp(L_b*, 1, T_b))
  2. Benchmarks full-backbone gradient norm ratios (rho = ||grad L_stop||_2 / ||grad L_CE||_2)
     and microbatch wall-clock timing breakdown (rollout, chrF scoring, Pass 1+2 backward)
     on real B=32 training microbatches at both pretrained init and BASELINE_S42.
"""
import gc
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import sacrebleu
import torch
from huggingface_hub import hf_hub_download

from .model import (
    OffEosStoppingMT,
    detect_repetition_horizon,
    extract_eos_margins,
    off_eos_stop_loss,
    select_chrf_stop_set,
)
from ...compositional_consistency.v1.model import chunked_ce_loss


BASELINE_S42_REVISION = '01a0b7a709ea5fcd24a95982f22c8150adaca4d5'
BASELINE_S42_ARCHIVE = (
    'checkpoints/amis_mandarin/mbart-large-50/'
    'baseline/seed-42/'
    'a100-bf16-sdpa-b32-ga4-es10-select-bleu/best.zip'
)
BASELINE_S42_SHA_PREFIX = '985f9fc4'


def _corpus_metrics(preds, refs, bleu_tokenizer='zh'):
    clean_p = [s.strip() for s in preds]
    clean_r = [s.strip() for s in refs]
    bleu = sacrebleu.metrics.BLEU(tokenize=bleu_tokenizer).corpus_score(clean_p, [clean_r])
    chrf0 = sacrebleu.metrics.CHRF(word_order=0).corpus_score(clean_p, [clean_r]).score
    chrfpp = sacrebleu.metrics.CHRF(word_order=2).corpus_score(clean_p, [clean_r]).score
    return dict(
        bleu=round(float(bleu.score), 4),
        chrf0=round(float(chrf0), 4),
        chrfpp=round(float(chrfpp), 4),
        bp=round(float(bleu.bp), 4),
        lr=round(float(bleu.sys_len) / float(max(1, bleu.ref_len)), 4),
        sys_len=int(bleu.sys_len),
        ref_len=int(bleu.ref_len),
    )


def _measure_two_sided_val_oracle(
    base_model,
    val_ds,
    raw_val,
    tokenizer,
    collator,
    device,
    spec,
    bleu_tokenizer='zh',
    eval_batch_size=16,
):
    """Run FP32 eval greedy off-EOS rollout across validation split and measure 2-sided headroom."""
    wrapper = OffEosStoppingMT(
        base_model,
        method='off-eos-chrf-v1',
        stop_weight=float(spec.get('stop_weight', 0.10)),
        stop_margin=float(spec.get('stop_margin', 0.1)),
        stop_tau=float(spec.get('stop_tau', 0.25)),
        chrf_band=float(spec.get('chrf_band', 0.5)),
        extra_past_eos=int(spec.get('extra_past_eos', 12)),
        max_rollout_cap=int(spec.get('max_rollout_cap', 96)),
        max_len_ratio=float(spec.get('max_len_ratio', 1.5)),
        stop_warmup_fraction=0.0,
        max_target_length=int(spec.get('max_target_length', 256)),
        logit_chunk_tokens=int(spec.get('logit_chunk_tokens', 32)),
        tokenizer=tokenizer,
    )
    wrapper.eval()
    wrapper.base_model.to(dtype=torch.float32)

    _, _, _, non_lex_ids = wrapper._resolve_special_ids()
    non_lex_set = set(non_lex_ids)
    chrf_metric = sacrebleu.metrics.CHRF(word_order=0)
    refs = [s.strip() for s in raw_val['target'].tolist()]

    preds_native = []
    preds_trunc = []
    preds_extend = []
    preds_two_sided = []
    preds_length = []

    trunc_gt1, trunc_gains = 0, []
    extend_gt1, extend_gains = 0, []
    two_gt1, two_gains = 0, []
    pos_counts = {'earlier': 0, 'exact': 0, 'later': 0}
    rep_triggered = 0
    native_eos_hits = 0

    started = time.perf_counter()
    n_rows = len(val_ds)
    for start in range(0, n_rows, int(eval_batch_size)):
        end = min(n_rows, start + int(eval_batch_size))
        batch = collator([val_ds[i] for i in range(start, end)])
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        _, gold_lens = wrapper._extract_gold_body(labels, non_lex_set)
        traj_tokens, native_eos_steps, rep_flags = wrapper._greedy_rollout_past_eos(
            input_ids, attention_mask, gold_lens
        )

        for b_idx in range(end - start):
            row_idx = start + b_idx
            toks = traj_tokens[b_idx]
            k_eos = native_eos_steps[b_idx]
            if k_eos is not None:
                native_eos_hits += 1
            if rep_flags[b_idx]:
                rep_triggered += 1

            # Evaluate every candidate lexical prefix length k in 1..len(toks)
            ref = refs[row_idx]
            prefix_texts = [tokenizer.decode(toks[:k], skip_special_tokens=True).strip() for k in range(1, len(toks) + 1)]
            prefix_scores = [
                float(chrf_metric.sentence_score(txt, [ref]).score) if (txt and ref) else 0.0
                for txt in prefix_texts
            ]

            if k_eos is None:
                native_k = len(toks)
                native_text = prefix_texts[-1]
                native_sc = prefix_scores[-1]
            elif k_eos == 0:
                native_k = 0
                native_text = ''
                native_sc = 0.0
            else:
                native_k = min(k_eos, len(toks))
                native_text = prefix_texts[native_k - 1]
                native_sc = prefix_scores[native_k - 1]

            # Truncation-only best: k in 1..max(1, native_k)
            hi_trunc = max(1, native_k)
            trunc_best_sc = max(prefix_scores[:hi_trunc])
            trunc_k = prefix_scores[:hi_trunc].index(trunc_best_sc) + 1
            if native_k == 0 or native_sc >= trunc_best_sc:
                preds_trunc.append(native_text if native_k == 0 or native_sc > trunc_best_sc else prefix_texts[trunc_k - 1])
                trunc_gain = 0.0 if native_sc >= trunc_best_sc else (trunc_best_sc - native_sc)
            else:
                preds_trunc.append(prefix_texts[trunc_k - 1])
                trunc_gain = trunc_best_sc - native_sc
            trunc_gains.append(trunc_gain)
            if trunc_gain > 1.0:
                trunc_gt1 += 1

            # Extension-only best: k in max(1, native_k)..len(toks)
            lo_ext = max(1, native_k)
            ext_slice = prefix_scores[lo_ext - 1:]
            ext_best_sc = max(ext_slice)
            ext_k = lo_ext + ext_slice.index(ext_best_sc)
            if ext_best_sc > native_sc:
                preds_extend.append(prefix_texts[ext_k - 1])
                ext_gain = ext_best_sc - native_sc
            else:
                preds_extend.append(native_text)
                ext_gain = 0.0
            extend_gains.append(ext_gain)
            if ext_gain > 1.0:
                extend_gt1 += 1

            # Two-sided best: any k in 1..len(toks)
            best_sc = max(prefix_scores)
            best_k = prefix_scores.index(best_sc) + 1
            if best_sc > native_sc:
                preds_two_sided.append(prefix_texts[best_k - 1])
                two_gain = best_sc - native_sc
            else:
                best_k = native_k
                preds_two_sided.append(native_text)
                two_gain = 0.0
            two_gains.append(two_gain)
            if two_gain > 1.0:
                two_gt1 += 1
            if best_k < native_k:
                pos_counts['earlier'] += 1
            elif best_k > native_k:
                pos_counts['later'] += 1
            else:
                pos_counts['exact'] += 1

            len_k = max(1, min(gold_lens[b_idx], len(toks)))
            preds_length.append(prefix_texts[len_k - 1])
            preds_native.append(native_text)

    m_native = _corpus_metrics(preds_native, refs, bleu_tokenizer)
    m_trunc = _corpus_metrics(preds_trunc, refs, bleu_tokenizer)
    m_ext = _corpus_metrics(preds_extend, refs, bleu_tokenizer)
    m_two = _corpus_metrics(preds_two_sided, refs, bleu_tokenizer)
    m_len = _corpus_metrics(preds_length, refs, bleu_tokenizer)

    return dict(
        rows=n_rows,
        seconds=round(time.perf_counter() - started, 2),
        native_eos_hit_rate=round(native_eos_hits / max(1, n_rows), 4),
        repetition_guard_triggered=rep_triggered,
        repetition_guard_rate=round(rep_triggered / max(1, n_rows), 4),
        native_greedy=m_native,
        truncate_only_oracle={
            **m_trunc,
            'rows_gain_gt_1_chrf': trunc_gt1,
            'frac_gain_gt_1_chrf': round(trunc_gt1 / max(1, n_rows), 4),
            'mean_chrf_gain': round(float(np.mean(trunc_gains)), 4),
            'delta_bleu_vs_native': round(m_trunc['bleu'] - m_native['bleu'], 4),
        },
        extend_only_oracle={
            **m_ext,
            'rows_gain_gt_1_chrf': extend_gt1,
            'frac_gain_gt_1_chrf': round(extend_gt1 / max(1, n_rows), 4),
            'mean_chrf_gain': round(float(np.mean(extend_gains)), 4),
            'delta_bleu_vs_native': round(m_ext['bleu'] - m_native['bleu'], 4),
        },
        two_sided_oracle={
            **m_two,
            'rows_gain_gt_1_chrf': two_gt1,
            'frac_gain_gt_1_chrf': round(two_gt1 / max(1, n_rows), 4),
            'mean_chrf_gain': round(float(np.mean(two_gains)), 4),
            'delta_bleu_vs_native': round(m_two['bleu'] - m_native['bleu'], 4),
            'delta_chrf0_vs_native': round(m_two['chrf0'] - m_native['chrf0'], 4),
            'stop_position_counts': pos_counts,
        },
        length_only_oracle={
            **m_len,
            'delta_bleu_vs_native': round(m_len['bleu'] - m_native['bleu'], 4),
        },
    )


def _measure_grad_and_timing(base_model, train_batches, tokenizer, device, spec, use_ckpt=True):
    """Benchmark full-backbone gradient norm ratio and per-stage microbatch latency."""
    if use_ckpt:
        base_model.enable_input_require_grads()
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    wrapper = OffEosStoppingMT(
        base_model,
        method='off-eos-chrf-v1',
        stop_weight=1.0,
        stop_margin=float(spec.get('stop_margin', 0.1)),
        stop_tau=float(spec.get('stop_tau', 0.25)),
        chrf_band=float(spec.get('chrf_band', 0.5)),
        extra_past_eos=int(spec.get('extra_past_eos', 12)),
        max_rollout_cap=int(spec.get('max_rollout_cap', 96)),
        max_len_ratio=float(spec.get('max_len_ratio', 1.5)),
        stop_warmup_fraction=0.0,
        max_target_length=int(spec.get('max_target_length', 256)),
        logit_chunk_tokens=int(spec.get('logit_chunk_tokens', 32)),
        tokenizer=tokenizer,
    )
    wrapper.train()
    params = [p for p in wrapper.parameters() if p.requires_grad]

    def grad_vec():
        vecs = [p.grad.detach().float() if p.grad is not None else None for p in params]
        sq = sum(float(g.pow(2).sum().item()) for g in vecs if g is not None)
        return math.sqrt(sq), vecs

    eos_id = int(wrapper.base_model.config.eos_token_id)
    pad_id, dec_start_id, forced_bos_id, non_lex_ids = wrapper._resolve_special_ids()
    non_lex_set = set(non_lex_ids)
    prefix = [dec_start_id, forced_bos_id] if forced_bos_id is not None else [dec_start_id]

    ce_vals, stop_chrf_vals, stop_len_vals, active_fracs = [], [], [], []
    g_ce_norms, g_chrf_norms, g_len_norms, cos_list = [], [], [], []
    t_pass1, t_rollout, t_chrf, t_pass2 = [], [], [], []

    for b_idx, batch in enumerate(train_batches):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        dec_gold = wrapper.prepare_decoder_input_ids_from_labels(labels)

        # 1) Pass 1 CE
        wrapper.zero_grad(set_to_none=True)
        torch.manual_seed(2026 + b_idx)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(2026 + b_idx)
        t0 = time.perf_counter()
        z_gold = wrapper._forward_logits(input_ids, attention_mask, dec_gold)
        ce = chunked_ce_loss(z_gold, labels, chunk_size=wrapper.logit_chunk_tokens)
        del z_gold
        ce.backward()
        t_pass1.append(time.perf_counter() - t0)
        g_ce, vec_ce = grad_vec()
        ce_vals.append(float(ce.detach()))
        g_ce_norms.append(g_ce)

        # 2) Greedy off-EOS rollout
        t0 = time.perf_counter()
        gold_tokens, gold_lens = wrapper._extract_gold_body(labels, non_lex_set)
        traj_tokens, _, _ = wrapper._greedy_rollout_past_eos(input_ids, attention_mask, gold_lens)
        t_rollout.append(time.perf_counter() - t0)

        # 3) chrF stop set selection
        t0 = time.perf_counter()
        bsz = int(input_ids.shape[0])
        chrf_sets, active_rows = [], []
        for b in range(bsz):
            ref_txt = wrapper._decode_tokens(gold_tokens[b])
            fset, _, _ = select_chrf_stop_set(
                traj_tokens[b], ref_txt, wrapper._decode_tokens, chrf_band=wrapper.chrf_band, chrf_metric=wrapper._chrf_metric
            )
            chrf_sets.append(fset)
            if fset:
                active_rows.append(b)
        t_chrf.append(time.perf_counter() - t0)
        active_fracs.append(float(len(active_rows)) / float(max(1, bsz)))

        # 4a) Pass 2 backward for chrF stop loss
        if active_rows:
            wrapper.zero_grad(set_to_none=True)
            torch.manual_seed(2026 + b_idx)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(2026 + b_idx)
            t0 = time.perf_counter()
            act_traj = [traj_tokens[b] for b in active_rows]
            max_t = max(len(t) for t in act_traj)
            dec_traj = torch.full((len(active_rows), len(prefix) + max_t), pad_id, dtype=torch.long, device=device)
            for i, toks in enumerate(act_traj):
                seq = prefix + toks
                dec_traj[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            act_idx = torch.tensor(active_rows, dtype=torch.long, device=device)
            z_traj = wrapper._forward_logits(input_ids.index_select(0, act_idx), attention_mask.index_select(0, act_idx), dec_traj)
            margins = extract_eos_margins(z_traj[:, len(prefix) - 1:, :], eos_token_id=eos_id, non_lexical_ids=non_lex_ids)
            del z_traj
            stop_c = torch.stack([
                off_eos_stop_loss(margins[i, :len(traj_tokens[b]) + 1], chrf_sets[b], delta=wrapper.stop_margin, tau=wrapper.stop_tau)
                for i, b in enumerate(active_rows)
            ]).mean()
            stop_c.backward()
            t_pass2.append(time.perf_counter() - t0)
            g_c, vec_c = grad_vec()
            stop_chrf_vals.append(float(stop_c.detach()))
            g_chrf_norms.append(g_c)
            dot = sum(float((a * b).sum().item()) for a, b in zip(vec_ce, vec_c) if a is not None and b is not None)
            cos_list.append(dot / max(1e-12, g_ce * g_c))
            del vec_c

        # 4b) Pass 2 backward for length control
        wrapper.zero_grad(set_to_none=True)
        torch.manual_seed(2026 + b_idx)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(2026 + b_idx)
        max_t_all = max(len(t) for t in traj_tokens)
        dec_all = torch.full((bsz, len(prefix) + max_t_all), pad_id, dtype=torch.long, device=device)
        for i, toks in enumerate(traj_tokens):
            seq = prefix + toks
            dec_all[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        z_all = wrapper._forward_logits(input_ids, attention_mask, dec_all)
        margins_all = extract_eos_margins(z_all[:, len(prefix) - 1:, :], eos_token_id=eos_id, non_lexical_ids=non_lex_ids)
        del z_all
        stop_l = torch.stack([
            off_eos_stop_loss(margins_all[b, :len(traj_tokens[b]) + 1], [max(1, min(gold_lens[b], len(traj_tokens[b])))], delta=wrapper.stop_margin, tau=wrapper.stop_tau)
            for b in range(bsz)
        ]).mean()
        stop_l.backward()
        g_l, _ = grad_vec()
        stop_len_vals.append(float(stop_l.detach()))
        g_len_norms.append(g_l)

        wrapper.zero_grad(set_to_none=True)
        del vec_ce

    mean_g_ce = float(np.mean(g_ce_norms))
    mean_g_chrf = float(np.mean(g_chrf_norms)) if g_chrf_norms else 0.0
    mean_g_len = float(np.mean(g_len_norms))
    rho_chrf = mean_g_chrf / max(1e-12, mean_g_ce)
    rho_len = mean_g_len / max(1e-12, mean_g_ce)
    rec_weight = round(min(0.50, max(0.02, 0.15 / max(1e-12, rho_chrf))), 3) if rho_chrf > 0 else float(spec.get('stop_weight', 0.10))
    sec_per_mb = float(np.mean(t_pass1) + np.mean(t_rollout) + np.mean(t_chrf) + (np.mean(t_pass2) if t_pass2 else np.mean(t_pass1)))

    return dict(
        num_batches=len(ce_vals),
        ce=round(float(np.mean(ce_vals)), 4),
        stop_loss_chrf=round(float(np.mean(stop_chrf_vals)), 4) if stop_chrf_vals else 0.0,
        stop_loss_length=round(float(np.mean(stop_len_vals)), 4),
        active_fraction_chrf=round(float(np.mean(active_fracs)), 4),
        grad_norm_ce=round(mean_g_ce, 4),
        grad_norm_stop_chrf=round(mean_g_chrf, 4),
        grad_norm_stop_length=round(mean_g_len, 4),
        rho_chrf=round(rho_chrf, 4),
        rho_length=round(rho_len, 4),
        cos_chrf_ce=round(float(np.mean(cos_list)), 4) if cos_list else 0.0,
        recommended_stop_weight_for_rho_015=rec_weight,
        timing=dict(
            sec_per_microbatch_pass1=round(float(np.mean(t_pass1)), 3),
            sec_per_microbatch_rollout=round(float(np.mean(t_rollout)), 3),
            sec_per_microbatch_chrf=round(float(np.mean(t_chrf)), 3),
            sec_per_microbatch_pass2=round(float(np.mean(t_pass2)), 3) if t_pass2 else 0.0,
            sec_per_microbatch_total=round(sec_per_mb, 3),
            estimated_minutes_per_epoch_144mb=round((sec_per_mb * 144.0) / 60.0, 2),
        ),
    )


def run_step0_diagnostic(config_path, num_grad_batches=4):
    """Run 0-optimizer-step Step-0 two-sided oracle probe, grad calibration & latency check."""
    from experiment_runner import engine
    from experiment_runner.config import verify_frozen
    from experiment_runner.integrity import restore_archive, sha256

    out, plan = verify_frozen(config_path)
    engine.configure(out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    started = time.time()

    run_spec = next((r for r in plan['runs'] if r['method'] == 'off-eos-chrf-v1'), plan['runs'][0])
    use_bf16 = bool(plan['protocol'].get('bf16') and device.type == 'cuda')
    bleu_tok = plan.get('protocol', {}).get('bleu_tokenizer', 'zh')

    # 1) Pretrained initialization gradient & wall-time check (0 optimizer steps)
    base_spec = dict(run_spec, method='baseline')
    init_model, tokenizer, _ = engine.load_model(base_spec)
    init_model.to(device)
    if use_bf16:
        init_model.to(dtype=torch.bfloat16)

    _, train_ds = engine.dataset('train', tokenizer, base_spec)
    raw_val, val_ds = engine.dataset('validation', tokenizer, base_spec)
    coll = engine.collator(init_model, tokenizer, base_spec)
    bsz = int(base_spec['microbatch'])
    train_batches = [
        coll([train_ds[i] for i in range(k * bsz, min((k + 1) * bsz, len(train_ds)))])
        for k in range(int(num_grad_batches))
        if min((k + 1) * bsz, len(train_ds)) - k * bsz >= 1
    ]

    init_grad = _measure_grad_and_timing(init_model, train_batches, tokenizer, device, run_spec)
    del init_model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # 2) Restored BASELINE_S42 two-sided validation oracle + baseline gradient/timing
    baseline_probe = {}
    try:
        with tempfile.TemporaryDirectory(prefix='off_eos_step0_') as tmpdir:
            tmp = Path(tmpdir)
            archive_path = hf_hub_download(
                repo_id=plan['repo'],
                filename=BASELINE_S42_ARCHIVE,
                revision=BASELINE_S42_REVISION,
                local_dir=str(tmp / 'dl'),
            )
            arch_sha = sha256(Path(archive_path))
            if not arch_sha.startswith(BASELINE_S42_SHA_PREFIX):
                raise ValueError(f'BASELINE_S42 archive SHA mismatch: {arch_sha}')
            ckpt_dir = tmp / 'restored'
            restore_archive(Path(archive_path), ckpt_dir)
            base_ckpt, base_tok, _ = engine.load_model(base_spec, folder=ckpt_dir)
            base_ckpt.to(device)

            val_oracle = _measure_two_sided_val_oracle(
                base_ckpt, val_ds, raw_val, base_tok, coll, device, run_spec, bleu_tokenizer=bleu_tok
            )
            if use_bf16:
                base_ckpt.to(dtype=torch.bfloat16)
            ckpt_grad = _measure_grad_and_timing(base_ckpt, train_batches, base_tok, device, run_spec)
            baseline_probe = {
                'status': 'MEASURED',
                'checkpoint_id': 'BASELINE_S42',
                'revision': BASELINE_S42_REVISION,
                'archive_sha256': arch_sha[:12],
                'validation_two_sided_oracle': val_oracle,
                'train_grad_and_timing': ckpt_grad,
            }
            del base_ckpt
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    except Exception as exc:
        baseline_probe = {
            'status': 'SKIPPED_WARNING',
            'checkpoint_id': 'BASELINE_S42',
            'error': str(exc),
        }

    report = dict(
        status='COMPLETE',
        optimizer_steps=0,
        test_split_opened=False,
        configured_stop_weight=float(run_spec.get('stop_weight', 0.10)),
        init_grad_and_timing=init_grad,
        baseline_s42_probe=baseline_probe,
        seconds=round(time.time() - started, 2),
    )
    out_path = out / 'step0_diagnostic.json'
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('OFF_EOS_STEP0_DIAGNOSTIC_COMPLETE', json.dumps(report, indent=2), flush=True)
    return report
