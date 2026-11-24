"""Zero-optimizer-step Step-0 diagnostic for Round 5 2x2 Lexical Fork Preference.

Measures on verified public A100 checkpoint BASELINE_S42 (without any optimizer update
or test.csv access):
  1. Reproduces native greedy FP32 validation metrics on amis_mandarin (576 rows).
  2. Runs reference-free prefix selection (random_prefix + min_margin_prefix) on both
     validation (576 rows) and a seeded 576-row train subset using compatible top-2
     lexical rivals (matching sentencepiece word-boundary `▁` vs continuation).
  3. Executes unpadded bucketed greedy branch rollouts u_a, u_b and evaluates 6-way
     float64 corpus BLEU (zh) + chrF0 sufficient-statistics concordance (`2x2`),
     stratified 500-draw matched natural-only controls, and disjoint half-corpus sign
     stability.
  4. Audits hybrid sequence off-support NLL (`h b u_a` vs `h a u_a` and `h a u_b` vs
     `h b u_b`), reversed-batch replica disagreement, and wall-clock rollout latency.
"""
import argparse
import csv
import hashlib
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from experiment_runner.integrity import restore_archive, sha256  # noqa: E402


REPO_ID = 'FiveC/amis-rewire-checkpoints'
MODEL_ID = 'facebook/mbart-large-50-many-to-many-mmt'
MODEL_REVISION = 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30'
BASELINE_S42_REVISION = '01a0b7a709ea5fcd24a95982f22c8150adaca4d5'
BASELINE_S42_ARCHIVE = (
    'checkpoints/amis_mandarin/mbart-large-50/'
    'baseline/seed-42/'
    'a100-bf16-sdpa-b32-ga4-es10-select-bleu/best.zip'
)
BASELINE_S42_SHA_PREFIX = '985f9fc4'
EXPECTED_SPLIT_SHA = {
    'validation': ('576', 'eac078ab237b55c10614f4db5ff026dc3a96a09a69c648ced29b5cdd8d4d11b4'),
    'train': ('4600', '4e13d83f035a18c67e1b2e3990baff3aa92154a34b754a4d0cdef9a71d85b8e6'),
}


def load_csv_split(split: str):
    if split not in EXPECTED_SPLIT_SHA:
        raise ValueError(f'Step-0 diagnostic refuses split={split!r}; test split is strictly off-limits')
    path = ROOT / 'data_processed/amis_mandarin' / f'{split}.csv'
    expected_rows, expected_sha = EXPECTED_SPLIT_SHA[split]
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    # Allow LF/CRLF normalization only if exact SHA matches either way
    if actual_sha != expected_sha:
        lf_sha = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        crlf_sha = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()
        if expected_sha not in (lf_sha, crlf_sha):
            raise ValueError(f'Dataset SHA mismatch for {path}: {actual_sha}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if str(len(rows)) != expected_rows:
        raise ValueError(f'Expected {expected_rows} rows in {path}, got {len(rows)}')
    return rows


def build_lexical_boundary_masks(tokenizer, vocab_size: int, extra_special_ids=()):
    """Partition content lexical IDs into word-initial (`▁`) and continuation tokens."""
    special = {int(x) for x in (getattr(tokenizer, 'all_special_ids', None) or ())}
    for attr in ('pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id'):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            special.add(int(val))
    for val in extra_special_ids:
        if val is not None:
            special.add(int(val))

    bow_mask = torch.zeros(vocab_size, dtype=torch.bool)
    cont_mask = torch.zeros(vocab_size, dtype=torch.bool)
    for tid in range(vocab_size):
        if tid in special:
            continue
        piece = tokenizer.convert_ids_to_tokens(tid)
        if not piece or (piece.startswith('<') and piece.endswith('>')):
            continue
        if not any(ch.isalpha() for ch in piece):
            continue
        if piece.startswith('\u2581'):
            bow_mask[tid] = True
        else:
            cont_mask[tid] = True
    return bow_mask, cont_mask


def _resolve_decoder_prefix(model):
    cfg = model.config
    eos_id = int(cfg.eos_token_id)
    pad_id = int(cfg.pad_token_id)
    dec_start = getattr(cfg, 'decoder_start_token_id', None)
    dec_start_id = int(eos_id if dec_start is None else dec_start)
    forced_bos = getattr(cfg, 'forced_bos_token_id', None)
    if forced_bos is None and getattr(model, 'generation_config', None) is not None:
        forced_bos = getattr(model.generation_config, 'forced_bos_token_id', None)
    prefix = [dec_start_id, int(forced_bos)] if forced_bos is not None else [dec_start_id]
    return pad_id, eos_id, prefix


def _decode_body(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def greedy_native_with_rivals(
    model,
    tokenizer,
    sources,
    bow_mask: torch.Tensor,
    cont_mask: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
    max_source_length: int = 256,
    max_new_tokens: int = 254,
):
    """Run deterministic FP32 batched greedy rollout tracking eligible forks and rivals."""
    model.eval()
    pad_id, eos_id, prefix = _resolve_decoder_prefix(model)
    max_pos = int(getattr(model.config, 'max_position_embeddings', max_new_tokens + len(prefix) + 2))
    step_cap = min(int(max_new_tokens), max(1, max_pos - len(prefix) - 2))
    bow_dev = bow_mask.to(device)
    cont_dev = cont_mask.to(device)
    results = []

    with torch.no_grad():
        for start in range(0, len(sources), int(batch_size)):
            end = min(len(sources), start + int(batch_size))
            enc_in = tokenizer(
                sources[start:end],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_source_length,
            )
            input_ids = enc_in['input_ids'].to(device)
            attention_mask = enc_in['attention_mask'].to(device)
            enc_out = model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            bsz = end - start
            step_in = torch.tensor([prefix], dtype=torch.long, device=device).expand(bsz, len(prefix))
            pkv = None

            seq_tokens = [[] for _ in range(bsz)]
            rivals = [[] for _ in range(bsz)]
            margins = [[] for _ in range(bsz)]
            finished = [False] * bsz

            for _ in range(step_cap):
                out = model(
                    encoder_outputs=enc_out,
                    attention_mask=attention_mask,
                    decoder_input_ids=step_in,
                    past_key_values=pkv,
                    use_cache=True,
                    return_dict=True,
                )
                pkv = out.past_key_values
                step_logits = out.logits[:, -1, :].float()
                top1 = step_logits.argmax(dim=-1)

                bow_vals, bow_ids = step_logits.masked_fill(~bow_dev, float('-inf')).topk(2, dim=-1)
                cont_vals, cont_ids = step_logits.masked_fill(~cont_dev, float('-inf')).topk(2, dim=-1)

                top1_list = top1.tolist()
                bow_ids_l, bow_vals_l = bow_ids.tolist(), bow_vals.tolist()
                cont_ids_l, cont_vals_l = cont_ids.tolist(), cont_vals.tolist()

                all_done = True
                for b in range(bsz):
                    if finished[b]:
                        continue
                    a = int(top1_list[b])
                    if a == eos_id:
                        finished[b] = True
                        continue
                    all_done = False
                    seq_tokens[b].append(a)
                    if bool(bow_mask[a].item()):
                        if int(bow_ids_l[b][0]) == a:
                            r_id, r_val = int(bow_ids_l[b][1]), float(bow_vals_l[b][1])
                        else:
                            r_id, r_val = int(bow_ids_l[b][0]), float(bow_vals_l[b][0])
                        if r_val > -1e30:
                            rivals[b].append(r_id)
                            margins[b].append(float(step_logits[b, a].item()) - r_val)
                        else:
                            rivals[b].append(-1)
                            margins[b].append(float('inf'))
                    elif bool(cont_mask[a].item()):
                        if int(cont_ids_l[b][0]) == a:
                            r_id, r_val = int(cont_ids_l[b][1]), float(cont_vals_l[b][1])
                        else:
                            r_id, r_val = int(cont_ids_l[b][0]), float(cont_vals_l[b][0])
                        if r_val > -1e30:
                            rivals[b].append(r_id)
                            margins[b].append(float(step_logits[b, a].item()) - r_val)
                        else:
                            rivals[b].append(-1)
                            margins[b].append(float('inf'))
                    else:
                        rivals[b].append(-1)
                        margins[b].append(float('inf'))

                if all_done:
                    break
                step_in = top1.unsqueeze(1)

            for b in range(bsz):
                elig = [j for j, r_id in enumerate(rivals[b]) if r_id >= 0]
                results.append(
                    dict(
                        row_idx=start + b,
                        native_tokens=seq_tokens[b],
                        rivals=rivals[b],
                        margins=margins[b],
                        eligible_positions=elig,
                        native_text=_decode_body(tokenizer, seq_tokens[b]),
                    )
                )
    return results


def select_fork_positions(native_rows, policy: str, seed: int = 42):
    """Select at most one reference-free fork position per eligible row."""
    rng = np.random.default_rng(seed)
    forks = []
    for row in native_rows:
        elig = row['eligible_positions']
        if not elig:
            continue
        if policy == 'random_prefix':
            j = int(rng.choice(elig))
        elif policy == 'min_margin_prefix':
            j = int(min(elig, key=lambda idx: (row['margins'][idx], idx)))
        else:
            raise ValueError(f'Unknown prefix selection policy: {policy}')
        forks.append(
            dict(
                row_idx=row['row_idx'],
                j=j,
                a=int(row['native_tokens'][j]),
                b=int(row['rivals'][j]),
                margin=float(row['margins'][j]),
            )
        )
    return forks


def greedy_branch_rollouts(
    model,
    tokenizer,
    sources,
    native_rows,
    forks,
    device: torch.device,
    batch_size: int = 32,
    max_source_length: int = 256,
    max_target_length: int = 256,
):
    """Run unpadded prefix-length-bucketed greedy continuations u_b after [h, b]."""
    model.eval()
    _, eos_id, prefix = _resolve_decoder_prefix(model)
    max_pos = int(getattr(model.config, 'max_position_embeddings', max_target_length + 2))
    eff_max_tgt = min(int(max_target_length), max(len(prefix) + 2, max_pos - 2))
    buckets = defaultdict(list)
    for f in forks:
        r_idx = f['row_idx']
        j = f['j']
        b_tok = f['b']
        dec_prefix = prefix + native_rows[r_idx]['native_tokens'][:j] + [b_tok]
        if len(dec_prefix) < eff_max_tgt:
            buckets[len(dec_prefix)].append((f, dec_prefix))

    tails_by_key = {}
    with torch.no_grad():
        for lp in sorted(buckets):
            items = buckets[lp]
            max_tail = max(0, eff_max_tgt - lp)
            for start in range(0, len(items), int(batch_size)):
                chunk = items[start:start + int(batch_size)]
                chunk_srcs = [sources[it[0]['row_idx']] for it in chunk]
                enc_in = tokenizer(
                    chunk_srcs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_source_length,
                )
                input_ids = enc_in['input_ids'].to(device)
                attention_mask = enc_in['attention_mask'].to(device)
                enc_out = model.get_encoder()(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                step_in = torch.tensor([it[1] for it in chunk], dtype=torch.long, device=device)
                pkv = None
                m = len(chunk)
                tails = [[] for _ in range(m)]
                finished = [False] * m

                for _ in range(max_tail):
                    out = model(
                        encoder_outputs=enc_out,
                        attention_mask=attention_mask,
                        decoder_input_ids=step_in,
                        past_key_values=pkv,
                        use_cache=True,
                        return_dict=True,
                    )
                    pkv = out.past_key_values
                    next_tok = out.logits[:, -1, :].float().argmax(dim=-1)
                    next_l = next_tok.tolist()
                    all_done = True
                    for idx in range(m):
                        if finished[idx]:
                            continue
                        tok = int(next_l[idx])
                        if tok == eos_id:
                            finished[idx] = True
                        else:
                            tails[idx].append(tok)
                            all_done = False
                    if all_done:
                        break
                    step_in = next_tok.unsqueeze(1)

                for idx, (f, _) in enumerate(chunk):
                    tails_by_key[(f['row_idx'], f['j'], f['b'])] = tails[idx]
    return tails_by_key


def _corpus_metrics_from_texts(hyps, refs, bleu_tok='zh'):
    clean_h = [s.strip() for s in hyps]
    clean_r = [s.strip() for s in refs]
    bleu = BLEU(tokenize=bleu_tok).corpus_score(clean_h, [clean_r])
    chrf0 = CHRF(word_order=0).corpus_score(clean_h, [clean_r]).score
    return dict(
        bleu=round(float(bleu.score), 4),
        chrf0=round(float(chrf0), 4),
        bp=round(float(bleu.bp), 4),
        lr=round(float(bleu.sys_len) / float(max(1, bleu.ref_len)), 4),
        sys_len=int(bleu.sys_len),
        ref_len=int(bleu.ref_len),
    )


def _score_stat_vec(bleu_m, chrf_m, st_b_vec, st_c_vec):
    b = bleu_m._compute_score_from_stats(st_b_vec.tolist())
    c = chrf_m._compute_score_from_stats(st_c_vec.tolist())
    return b, c


def _compute_6way_sign(bleu_m, chrf_m, tot_b, tot_c, diff_b, diff_c, tol=1e-10):
    # diff_b, diff_c are sequences of 4 diffs matching [v00, v10, v01, v11]
    b_scores = [bleu_m._compute_score_from_stats((tot_b + diff_b[k]).tolist()).score for k in range(4)]
    c_scores = [chrf_m._compute_score_from_stats((tot_c + diff_c[k]).tolist()).score for k in range(4)]

    b_deltas = (b_scores[1] - b_scores[0], b_scores[3] - b_scores[2], b_scores[3] - b_scores[0])
    c_deltas = (c_scores[1] - c_scores[0], c_scores[3] - c_scores[2], c_scores[3] - c_scores[0])
    all_six = (*b_deltas, *c_deltas)
    if all(d > tol for d in all_six):
        s_6way = 1
    elif all(d < -tol for d in all_six):
        s_6way = -1
    else:
        s_6way = 0

    if b_deltas[2] > tol and c_deltas[2] > tol:
        s_nat = 1
    elif b_deltas[2] < -tol and c_deltas[2] < -tol:
        s_nat = -1
    else:
        s_nat = 0
    return s_6way, s_nat, b_deltas[2], c_deltas[2]


def evaluate_lexical_forks(
    tokenizer,
    refs,
    native_rows,
    forks,
    tails_by_key,
    bleu_tok: str = 'zh',
    tie_tol: float = 1e-10,
    n_draws: int = 500,
):
    """Score exact float64 2x2 corpus metrics, split-background stability, and controls."""
    bleu_m = BLEU(tokenize=bleu_tok)
    chrf_m = CHRF(word_order=0)
    clean_refs = [r.strip() for r in refs]
    native_texts = [r['native_text'].strip() for r in native_rows]

    st_b = np.stack([np.asarray(bleu_m._extract_corpus_statistics([h], [[r]])[0], dtype=np.int64)
                     for h, r in zip(native_texts, clean_refs)], axis=0)
    st_c = np.stack([np.asarray(chrf_m._extract_corpus_statistics([h], [[r]])[0], dtype=np.float64)
                     for h, r in zip(native_texts, clean_refs)], axis=0)
    tot_b = st_b.sum(axis=0)
    tot_c = st_c.sum(axis=0)

    # Disjoint half-corpus backgrounds (even vs odd indices)
    even_mask = np.arange(len(native_texts)) % 2 == 0
    odd_mask = ~even_mask
    tot_b_even = st_b[even_mask].sum(axis=0)
    tot_c_even = st_c[even_mask].sum(axis=0)
    tot_b_odd = st_b[odd_mask].sum(axis=0)
    tot_c_odd = st_c[odd_mask].sum(axis=0)

    fork_records = []
    texts_1tok_2x2 = native_texts.copy()
    texts_branch_2x2 = native_texts.copy()
    texts_1tok_nat = native_texts.copy()
    texts_branch_nat = native_texts.copy()
    nat_pos_pool = []

    n_pos = n_neg = n_zero = 0
    n_nat_pos = n_nat_neg = 0
    split_total_nz = split_stable_nz = 0
    split_total_pos = split_stable_pos = 0

    for f in forks:
        i = f['row_idx']
        j = f['j']
        a = f['a']
        b_tok = f['b']
        nat_toks = native_rows[i]['native_tokens']
        h = nat_toks[:j]
        u_a = nat_toks[j + 1:]
        u_b = tails_by_key.get((i, j, b_tok), [])

        variants = (
            _decode_body(tokenizer, h + [a] + u_a),  # v00 (native)
            _decode_body(tokenizer, h + [b_tok] + u_a),  # v10 (1-token hybrid)
            _decode_body(tokenizer, h + [a] + u_b),  # v01 (a with u_b hybrid)
            _decode_body(tokenizer, h + [b_tok] + u_b),  # v11 (natural alternative)
        )
        v_b = [np.asarray(bleu_m._extract_corpus_statistics([z], [[clean_refs[i]]])[0], dtype=np.int64) for z in variants]
        v_c = [np.asarray(chrf_m._extract_corpus_statistics([z], [[clean_refs[i]]])[0], dtype=np.float64) for z in variants]
        diff_b = [vb - st_b[i] for vb in v_b]
        diff_c = [vc - st_c[i] for vc in v_c]

        s_6, s_nat, d_b_nat, d_c_nat = _compute_6way_sign(bleu_m, chrf_m, tot_b, tot_c, diff_b, diff_c, tol=tie_tol)

        # Evaluate on disjoint half-backgrounds (other even rows + candidate, vs other odd rows + candidate)
        bg_b_e = tot_b_even + (st_b[i] if i % 2 == 1 else 0)
        bg_c_e = tot_c_even + (st_c[i] if i % 2 == 1 else 0)
        bg_b_o = tot_b_odd + (st_b[i] if i % 2 == 0 else 0)
        bg_c_o = tot_c_odd + (st_c[i] if i % 2 == 0 else 0)
        s_even, _, _, _ = _compute_6way_sign(bleu_m, chrf_m, bg_b_e, bg_c_e, diff_b, diff_c, tol=tie_tol)
        s_odd, _, _, _ = _compute_6way_sign(bleu_m, chrf_m, bg_b_o, bg_c_o, diff_b, diff_c, tol=tie_tol)
        if s_6 != 0:
            split_total_nz += 1
            if s_even == s_6 and s_odd == s_6:
                split_stable_nz += 1
        if s_6 == 1:
            n_pos += 1
            split_total_pos += 1
            if s_even == 1 and s_odd == 1:
                split_stable_pos += 1
            texts_1tok_2x2[i] = variants[1]
            texts_branch_2x2[i] = variants[3]
        elif s_6 == -1:
            n_neg += 1
        else:
            n_zero += 1

        if s_nat == 1:
            n_nat_pos += 1
            texts_1tok_nat[i] = variants[1]
            texts_branch_nat[i] = variants[3]
            nat_pos_pool.append((d_b_nat, diff_b[1], diff_c[1], diff_b[3], diff_c[3], s_6 == 1))
        elif s_nat == -1:
            n_nat_neg += 1

        fork_records.append(
            dict(
                row_idx=i,
                j=j,
                a=a,
                b=b_tok,
                u_a=u_a,
                u_b=u_b,
                s_6way=s_6,
                s_nat=s_nat,
                margin=f['margin'],
            )
        )

    m_native = _corpus_metrics_from_texts(native_texts, clean_refs, bleu_tok)
    m_1tok_2x2 = _corpus_metrics_from_texts(texts_1tok_2x2, clean_refs, bleu_tok)
    m_br_2x2 = _corpus_metrics_from_texts(texts_branch_2x2, clean_refs, bleu_tok)
    m_1tok_nat = _corpus_metrics_from_texts(texts_1tok_nat, clean_refs, bleu_tok)
    m_br_nat = _corpus_metrics_from_texts(texts_branch_nat, clean_refs, bleu_tok)

    # Matched natural-only 5-quintile stratified control (500 draws, RNG 42)
    matched_ctrl_1tok = dict(m_native)
    matched_ctrl_branch = dict(m_native)
    if n_pos > 0 and len(nat_pos_pool) >= n_pos:
        order = np.argsort([item[0] for item in nat_pos_pool], kind='stable')
        bins = np.array_split(order, 5)
        rng = np.random.default_rng(42)
        draws_1tok, draws_branch = [], []
        for _ in range(int(n_draws)):
            picked = [
                int(idx)
                for group in bins
                for idx in rng.choice(group, size=sum(1 for idx in group if nat_pos_pool[idx][5]), replace=False)
            ]
            tb1 = tot_b + sum((nat_pos_pool[k][1] for k in picked), np.zeros_like(tot_b))
            tc1 = tot_c + sum((nat_pos_pool[k][2] for k in picked), np.zeros_like(tot_c))
            b1, c1 = _score_stat_vec(bleu_m, chrf_m, tb1, tc1)
            draws_1tok.append([b1.score, c1.score, b1.bp, b1.sys_len / max(1, b1.ref_len)])

            tbb = tot_b + sum((nat_pos_pool[k][3] for k in picked), np.zeros_like(tot_b))
            tcb = tot_c + sum((nat_pos_pool[k][4] for k in picked), np.zeros_like(tot_c))
            bb, cb = _score_stat_vec(bleu_m, chrf_m, tbb, tcb)
            draws_branch.append([bb.score, cb.score, bb.bp, bb.sys_len / max(1, bb.ref_len)])

        med1 = np.median(np.asarray(draws_1tok), axis=0)
        medb = np.median(np.asarray(draws_branch), axis=0)
        matched_ctrl_1tok = dict(bleu=round(float(med1[0]), 4), chrf0=round(float(med1[1]), 4), bp=round(float(med1[2]), 4), lr=round(float(med1[3]), 4))
        matched_ctrl_branch = dict(bleu=round(float(medb[0]), 4), chrf0=round(float(medb[1]), 4), bp=round(float(medb[2]), 4), lr=round(float(medb[3]), 4))

    total_rows = len(native_rows)
    n_forks = len(forks)
    summary = dict(
        rows=total_rows,
        eligible_rows=n_forks,
        counts=dict(
            pos_2x2=n_pos,
            neg_2x2=n_neg,
            abstain_2x2=n_zero,
            nonzero_2x2=n_pos + n_neg,
            pos_natural=n_nat_pos,
            neg_natural=n_nat_neg,
            rho_active_over_forks=round((n_pos + n_neg) / max(1, n_forks), 4),
            rho_pos_over_forks=round(n_pos / max(1, n_forks), 4),
            rho_active_over_rows=round((n_pos + n_neg) / max(1, total_rows), 4),
        ),
        split_background_stability=dict(
            nonzero_checked=split_total_nz,
            nonzero_stable=split_stable_nz,
            nonzero_stability_rate=round(split_stable_nz / max(1, split_total_nz), 4),
            pos_checked=split_total_pos,
            pos_stable=split_stable_pos,
            pos_stability_rate=round(split_stable_pos / max(1, split_total_pos), 4),
        ),
        metrics=dict(
            native=m_native,
            oracle_one_token_2x2=m_1tok_2x2,
            oracle_natural_branch_2x2=m_br_2x2,
            matched_natural_only_one_token_median_500=matched_ctrl_1tok,
            matched_natural_only_branch_median_500=matched_ctrl_branch,
            all_natural_pos_one_token=m_1tok_nat,
            all_natural_pos_branch=m_br_nat,
        ),
        deltas=dict(
            one_token_2x2_minus_native_bleu=round(m_1tok_2x2['bleu'] - m_native['bleu'], 4),
            one_token_2x2_minus_native_chrf0=round(m_1tok_2x2['chrf0'] - m_native['chrf0'], 4),
            one_token_2x2_minus_matched_nat_bleu=round(m_1tok_2x2['bleu'] - matched_ctrl_1tok['bleu'], 4),
            one_token_2x2_minus_matched_nat_chrf0=round(m_1tok_2x2['chrf0'] - matched_ctrl_1tok['chrf0'], 4),
            branch_2x2_minus_native_bleu=round(m_br_2x2['bleu'] - m_native['bleu'], 4),
        ),
    )
    return summary, fork_records


def _tail_nll_per_row(
    model,
    tokenizer,
    sources,
    records_with_seq,
    device: torch.device,
    prefix,
    eos_id: int,
    batch_size: int = 32,
    max_source_length: int = 256,
    max_target_length: int = 256,
):
    """Compute exact unpadded FP32 per-token NLL of tail + EOS given decoder prefix [h, tok]."""
    model.eval()
    buckets = defaultdict(list)
    for rec in records_with_seq:
        dec_pref = prefix + rec['h'] + [rec['pivot']]
        tail_eval = rec['tail'] + [eos_id]
        if len(dec_pref) + len(tail_eval) > max_target_length:
            keep_tail = max(1, max_target_length - len(dec_pref))
            tail_eval = tail_eval[:keep_tail]
        dec_in = dec_pref + tail_eval[:-1]
        buckets[(len(dec_pref), len(tail_eval))].append((rec['key'], dec_in, tail_eval, rec['row_idx']))

    out_stats = {}
    with torch.no_grad():
        for (lp, lu), items in sorted(buckets.items()):
            for start in range(0, len(items), int(batch_size)):
                chunk = items[start:start + int(batch_size)]
                enc_in = tokenizer(
                    [sources[it[3]] for it in chunk],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_source_length,
                )
                input_ids = enc_in['input_ids'].to(device)
                attention_mask = enc_in['attention_mask'].to(device)
                dec_t = torch.tensor([it[1] for it in chunk], dtype=torch.long, device=device)
                tgt_t = torch.tensor([it[2] for it in chunk], dtype=torch.long, device=device)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_t,
                    use_cache=False,
                    return_dict=True,
                ).logits[:, lp - 1:lp - 1 + lu, :].float()
                tok_nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tgt_t.reshape(-1),
                    reduction='none',
                ).reshape(len(chunk), lu)
                mean_nll = tok_nll.mean(dim=1).tolist()
                junc_nll = tok_nll[:, 0].tolist()
                for idx, it in enumerate(chunk):
                    out_stats[it[0]] = (float(mean_nll[idx]), float(junc_nll[idx]))
    return out_stats


def measure_hybrid_nll(
    model,
    tokenizer,
    sources,
    native_rows,
    fork_records,
    device: torch.device,
    batch_size: int = 32,
    max_source_length: int = 256,
    max_target_length: int = 256,
):
    """Compare NLL/token and junction-token NLL of tails under native vs crossed prefixes."""
    _, eos_id, prefix = _resolve_decoder_prefix(model)
    requests = []
    for f in fork_records:
        i = f['row_idx']
        h = native_rows[i]['native_tokens'][:f['j']]
        for tag, piv, tl in (
            ('ua_nat', f['a'], f['u_a']),
            ('ua_cross', f['b'], f['u_a']),
            ('ub_nat', f['b'], f['u_b']),
            ('ub_cross', f['a'], f['u_b']),
        ):
            requests.append(dict(key=(i, tag), row_idx=i, h=h, pivot=piv, tail=tl))

    stats = _tail_nll_per_row(
        model, tokenizer, sources, requests, device, prefix, eos_id,
        batch_size=batch_size, max_source_length=max_source_length, max_target_length=max_target_length,
    )

    def summarize(subset):
        d_ua, d_ub, j_ua, j_ub = [], [], [], []
        for f in subset:
            i = f['row_idx']
            if all((i, k) in stats for k in ('ua_nat', 'ua_cross', 'ub_nat', 'ub_cross')):
                m0, j0 = stats[(i, 'ua_nat')]
                m1, j1 = stats[(i, 'ua_cross')]
                m2, j2 = stats[(i, 'ub_nat')]
                m3, j3 = stats[(i, 'ub_cross')]
                d_ua.append(m1 - m0)
                d_ub.append(m3 - m2)
                j_ua.append(j1 - j0)
                j_ub.append(j3 - j2)
        if not d_ua:
            return {'count': 0}
        both = d_ua + d_ub
        return dict(
            count=len(d_ua),
            delta_nll_ua_mean=round(float(np.mean(d_ua)), 4),
            delta_nll_ua_median=round(float(np.median(d_ua)), 4),
            delta_nll_ub_mean=round(float(np.mean(d_ub)), 4),
            delta_nll_ub_median=round(float(np.median(d_ub)), 4),
            delta_nll_pooled_mean=round(float(np.mean(both)), 4),
            delta_nll_pooled_median=round(float(np.median(both)), 4),
            junction_delta_ua_mean=round(float(np.mean(j_ua)), 4),
            junction_delta_ub_mean=round(float(np.mean(j_ub)), 4),
        )

    return dict(
        all_forks=summarize(fork_records),
        nonzero_2x2=summarize([f for f in fork_records if f['s_6way'] != 0]),
        pos_2x2=summarize([f for f in fork_records if f['s_6way'] == 1]),
    )


def measure_replica_disagreement(
    model,
    tokenizer,
    sources,
    refs,
    bow_mask,
    cont_mask,
    native_rows,
    fork_records,
    device: torch.device,
    bleu_tok: str = 'zh',
    n_rows: int = 128,
    batch_size_rep2: int = 8,
    max_source_length: int = 256,
    max_target_length: int = 256,
    max_new_tokens: int = 254,
):
    """Compare original vs reversed/re-batched rollout tokens and 2x2 signs."""
    m = min(int(n_rows), len(sources))
    if m == 0:
        return {'rows': 0, 'token_disagreement_rate': 0.0, 'sign_disagreement_rate': 0.0}

    rev_indices = list(range(m - 1, -1, -1))
    rev_sources = [sources[idx] for idx in rev_indices]
    rep2_raw = greedy_native_with_rivals(
        model, tokenizer, rev_sources, bow_mask, cont_mask, device,
        batch_size=batch_size_rep2, max_source_length=max_source_length, max_new_tokens=max_new_tokens,
    )
    rep2_by_orig = {}
    for item in rep2_raw:
        orig_idx = rev_indices[item['row_idx']]
        item_copy = dict(item, row_idx=orig_idx)
        rep2_by_orig[orig_idx] = item_copy

    sub_forks = []
    token_mismatches = 0
    for f in fork_records:
        i = f['row_idx']
        if i >= m:
            continue
        r2 = rep2_by_orig[i]
        if r2['native_tokens'] != native_rows[i]['native_tokens'] or f['j'] >= len(r2['rivals']) or r2['rivals'][f['j']] != f['b']:
            token_mismatches += 1
        if f['j'] < len(r2['rivals']) and r2['rivals'][f['j']] >= 0:
            sub_forks.append(
                dict(
                    row_idx=i,
                    j=f['j'],
                    a=int(r2['native_tokens'][f['j']]),
                    b=int(r2['rivals'][f['j']]),
                    margin=float(r2['margins'][f['j']]),
                )
            )
    rep2_list = [rep2_by_orig[i] for i in range(m)] + native_rows[m:]
    tails_rep2 = greedy_branch_rollouts(
        model, tokenizer, sources, rep2_list, list(reversed(sub_forks)), device,
        batch_size=batch_size_rep2, max_source_length=max_source_length, max_target_length=max_target_length,
    )
    _, rep2_records = evaluate_lexical_forks(tokenizer, refs, rep2_list, sub_forks, tails_rep2, bleu_tok=bleu_tok, n_draws=10)
    rep2_map = {rec['row_idx']: rec['s_6way'] for rec in rep2_records}

    checked = 0
    sign_mismatches = 0
    for f in fork_records:
        i = f['row_idx']
        if i >= m:
            continue
        checked += 1
        if rep2_map.get(i, 0) != f['s_6way']:
            sign_mismatches += 1

    return dict(
        rows_checked=m,
        forks_checked=checked,
        native_token_or_rival_disagreement=token_mismatches,
        native_token_disagreement_rate=round(token_mismatches / max(1, m), 4),
        sign_disagreement=sign_mismatches,
        sign_disagreement_rate=round(sign_mismatches / max(1, checked), 4),
    )


def run_policy_pipeline(
    model,
    tokenizer,
    rows,
    bow_mask,
    cont_mask,
    device: torch.device,
    native_rows,
    t_native: float,
    policy: str,
    seed: int,
    bleu_tok: str = 'zh',
    max_source_length: int = 256,
    max_target_length: int = 256,
):
    sources = [r['source'] for r in rows]
    refs = [r['target'] for r in rows]

    t0 = time.perf_counter()
    forks = select_fork_positions(native_rows, policy=policy, seed=seed)
    tails = greedy_branch_rollouts(
        model, tokenizer, sources, native_rows, forks, device,
        batch_size=32, max_source_length=max_source_length, max_target_length=max_target_length,
    )
    t_branch = time.perf_counter() - t0

    t0 = time.perf_counter()
    summary, fork_records = evaluate_lexical_forks(tokenizer, refs, native_rows, forks, tails, bleu_tok=bleu_tok)
    t_score = time.perf_counter() - t0

    t0 = time.perf_counter()
    hybrid_nll = measure_hybrid_nll(
        model, tokenizer, sources, native_rows, fork_records, device,
        batch_size=32, max_source_length=max_source_length, max_target_length=max_target_length,
    )
    t_nll = time.perf_counter() - t0

    n_rows = len(rows)
    sec_per_32 = (t_native + t_branch + t_score) * (32.0 / max(1, n_rows))
    summary['timing'] = dict(
        seconds_native_rollout=round(t_native, 2),
        seconds_branch_rollout=round(t_branch, 2),
        seconds_corpus_scoring=round(t_score, 2),
        seconds_hybrid_nll=round(t_nll, 2),
        extrapolated_rollout_score_sec_per_32mb=round(sec_per_32, 3),
        estimated_extra_minutes_per_epoch_144mb=round((sec_per_32 * 144.0) / 60.0, 2),
    )
    summary['hybrid_off_support_nll'] = hybrid_nll
    return summary, fork_records


def load_baseline_s42(checkpoint_dir=None, tmp_dir=None):
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
        arch_sha = 'local'
    else:
        archive_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=BASELINE_S42_ARCHIVE,
            revision=BASELINE_S42_REVISION,
            local_dir=str(Path(tmp_dir) / 'dl'),
        )
        arch_sha = sha256(Path(archive_path))
        if not arch_sha.startswith(BASELINE_S42_SHA_PREFIX):
            raise ValueError(f'BASELINE_S42 archive SHA mismatch: {arch_sha}')
        ckpt_path = Path(tmp_dir) / 'restored'
        restore_archive(Path(archive_path), ckpt_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_path), attn_implementation='sdpa')
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), use_fast=False)

    tokenizer.src_lang = 'tl_XX'
    tokenizer.tgt_lang = 'zh_CN'
    tgt_id = tokenizer.convert_tokens_to_ids('zh_CN')
    assert tgt_id != tokenizer.unk_token_id
    model.config.forced_bos_token_id = tgt_id
    model.generation_config.forced_bos_token_id = tgt_id
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    model.generation_config.decoder_start_token_id = tokenizer.eos_token_id
    return model, tokenizer, arch_sha


def run_step0(checkpoint_dir=None, out_path=None, seed=42, train_sample_size=576):
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_rows = load_csv_split('validation')
    train_all = load_csv_split('train')
    rng = np.random.default_rng(int(seed))
    train_idx = sorted(int(i) for i in rng.choice(len(train_all), size=min(int(train_sample_size), len(train_all)), replace=False))
    train_rows = [train_all[i] for i in train_idx]

    with tempfile.TemporaryDirectory(prefix='lexfork_step0_') as tmpdir:
        model, tokenizer, arch_sha = load_baseline_s42(checkpoint_dir=checkpoint_dir, tmp_dir=tmpdir)
        model.to(device=device, dtype=torch.float32)
        model.eval()
        _, eos_id, prefix = _resolve_decoder_prefix(model)
        bow_mask, cont_mask = build_lexical_boundary_masks(
            tokenizer, int(model.config.vocab_size), extra_special_ids=[eos_id, *prefix]
        )

        # 1) Validation 576 rows
        val_sources = [r['source'] for r in val_rows]
        val_refs = [r['target'] for r in val_rows]
        t0 = time.perf_counter()
        val_native = greedy_native_with_rivals(model, tokenizer, val_sources, bow_mask, cont_mask, device, batch_size=16)
        t_val_nat = time.perf_counter() - t0

        val_rand, val_rand_forks = run_policy_pipeline(
            model, tokenizer, val_rows, bow_mask, cont_mask, device, val_native, t_val_nat, 'random_prefix', seed, 'zh'
        )
        val_minm, val_minm_forks = run_policy_pipeline(
            model, tokenizer, val_rows, bow_mask, cont_mask, device, val_native, t_val_nat, 'min_margin_prefix', seed, 'zh'
        )
        replica_audit = measure_replica_disagreement(
            model, tokenizer, val_sources, val_refs, bow_mask, cont_mask, val_native, val_rand_forks, device, bleu_tok='zh'
        )

        # 2) Train subset (576 rows, strictly disjoint from validation/test)
        tr_sources = [r['source'] for r in train_rows]
        t0 = time.perf_counter()
        tr_native = greedy_native_with_rivals(model, tokenizer, tr_sources, bow_mask, cont_mask, device, batch_size=16)
        t_tr_nat = time.perf_counter() - t0
        tr_rand, _ = run_policy_pipeline(
            model, tokenizer, train_rows, bow_mask, cont_mask, device, tr_native, t_tr_nat, 'random_prefix', seed, 'zh'
        )
        tr_minm, _ = run_policy_pipeline(
            model, tokenizer, train_rows, bow_mask, cont_mask, device, tr_native, t_tr_nat, 'min_margin_prefix', seed, 'zh'
        )

    # Evaluate pre-registered Step-0 gates on validation/random_prefix
    m_vr = val_rand['metrics']
    c_vr = val_rand['counts']
    nll_pos = val_rand['hybrid_off_support_nll'].get('pos_2x2', {}).get('delta_nll_pooled_mean')
    nll_ok = (nll_pos is not None) and (nll_pos < 0.50)

    gates = {
        'gate_native_reproduced': abs(m_vr['native']['bleu'] - 17.5373) < 0.15,
        'gate_firing_rate_ge_2pct': c_vr['rho_active_over_forks'] >= 0.02 and c_vr['pos_2x2'] >= 10,
        'gate_one_token_bleu_ge_030': (
            val_rand['deltas']['one_token_2x2_minus_native_bleu'] >= 0.30
            and val_rand['deltas']['one_token_2x2_minus_native_chrf0'] > 0.0
        ),
        'gate_beats_matched_natural_control': (
            val_rand['deltas']['one_token_2x2_minus_matched_nat_bleu'] > 0.0
            and val_rand['deltas']['one_token_2x2_minus_matched_nat_chrf0'] > 0.0
        ),
        'gate_lr_preserved': 0.93 <= m_vr['oracle_one_token_2x2']['lr'] <= 1.05,
        'gate_off_support_nll_lt_050': bool(nll_ok),
        'gate_replica_disagreement_lt_5pct': replica_audit['sign_disagreement_rate'] < 0.05,
    }
    verdict = 'GO_PILOT' if all(gates.values()) else (
        'REVIEW_MIN_MARGIN_OR_REFINEMENT'
        if (
            val_minm['deltas']['one_token_2x2_minus_native_bleu'] >= 0.30
            and val_minm['counts']['pos_2x2'] >= 10
            and val_minm['deltas']['one_token_2x2_minus_matched_nat_bleu'] > 0.0
        )
        else 'NO_GO'
    )

    report = dict(
        status='COMPLETE',
        optimizer_steps=0,
        test_split_opened=False,
        checkpoint_id='BASELINE_S42',
        revision=BASELINE_S42_REVISION,
        archive_sha256=arch_sha[:12],
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        verdict=verdict,
        gates_on_validation_random_prefix=gates,
        replica_disagreement_audit=replica_audit,
        validation={
            'random_prefix': val_rand,
            'min_margin_prefix': val_minm,
        },
        train_subset_576={
            'sample_seed': seed,
            'random_prefix': tr_rand,
            'min_margin_prefix': tr_minm,
        },
        seconds=round(time.time() - started, 2),
    )
    target = Path(out_path) if out_path else (ROOT / 'outputs/experiments/round5_lexical_fork_step0/STEP0_REPORT.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('LEXICAL_FORK_STEP0_COMPLETE', json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', default=None, help='Optional local BASELINE_S42 directory instead of HF download')
    parser.add_argument('--out', default=None, help='Optional output JSON path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-sample-size', type=int, default=576)
    args = parser.parse_args()
    run_step0(checkpoint_dir=args.checkpoint_dir, out_path=args.out, seed=args.seed, train_sample_size=args.train_sample_size)


if __name__ == '__main__':
    main()
