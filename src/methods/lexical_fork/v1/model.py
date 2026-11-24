"""2x2 Cross-Tail Lexical Fork Preference (`lexical_fork` v1) and matched natural control.

Methods:
  - `lexical-fork-2x2-v1`     ('2x2')     : 6-way Pareto concordance over (h a u_a, h b u_a, h a u_b, h b u_b)
                                            for both corpus BLEU and corpus chrF0 against fixed epoch-train background.
  - `lexical-fork-natural-v1` ('natural') : Matched control using only natural diagonal (h b u_b vs h a u_a),
                                            paired in exact positive/negative counts per margin stratum.

Inference (eval / generate) delegates directly to the unwrapped native backbone with zero
extra parameters and zero inference overhead.
"""
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sacrebleu.metrics import BLEU, CHRF
from transformers.modeling_outputs import Seq2SeqLMOutput

from ...compositional_consistency.v1.model import chunked_ce_loss


METHODS = {
    'lexical-fork-2x2-v1': '2x2',
    'lexical-fork-natural-v1': 'natural',
}


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

    bow_mask = torch.zeros(int(vocab_size), dtype=torch.bool)
    cont_mask = torch.zeros(int(vocab_size), dtype=torch.bool)
    for tid in range(int(vocab_size)):
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


def hash_prefix_index(seed: int, epoch_idx: int, row_id: int, eligible_positions):
    """Deterministically select one eligible lexical fork position via blake2b."""
    if not eligible_positions:
        raise ValueError('eligible_positions must be non-empty')
    digest = hashlib.blake2b(
        f'{int(seed)}:{int(epoch_idx)}:{int(row_id)}'.encode('utf-8'),
        digest_size=8,
    ).digest()
    idx = int.from_bytes(digest, 'big') % len(eligible_positions)
    return int(eligible_positions[idx])


def hash_selection_key(seed: int, epoch_idx: int, mode: str, sign: int, row_id: int):
    """Deterministic tie-broken hash sort key for sign/stratum-balanced fork selection."""
    digest = hashlib.blake2b(
        f'{int(seed)}:{int(epoch_idx)}:{mode}:{int(sign)}:{int(row_id)}'.encode('utf-8'),
        digest_size=8,
    ).digest()
    return (int.from_bytes(digest, 'big'), int(row_id))


def _resolve_prefix_ids(base_model):
    cfg = base_model.config
    pad_id = int(cfg.pad_token_id)
    eos_id = int(cfg.eos_token_id)
    dec_start = getattr(cfg, 'decoder_start_token_id', None)
    dec_start_id = int(eos_id if dec_start is None else dec_start)
    forced_bos = getattr(cfg, 'forced_bos_token_id', None)
    if forced_bos is None and getattr(base_model, 'generation_config', None) is not None:
        forced_bos = getattr(base_model.generation_config, 'forced_bos_token_id', None)
    prefix = [dec_start_id, int(forced_bos)] if forced_bos is not None else [dec_start_id]
    return pad_id, eos_id, prefix


def _decode_body(tokenizer, token_ids):
    if tokenizer is None:
        return ' '.join(str(x) for x in token_ids)
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def _greedy_native_with_rivals(
    base_model,
    tokenizer,
    sources,
    bow_mask,
    cont_mask,
    device,
    row_ids=None,
    batch_size=64,
    max_source_length=256,
    max_new_tokens=254,
):
    _, eos_id, prefix = _resolve_prefix_ids(base_model)
    max_pos = int(getattr(base_model.config, 'max_position_embeddings', max_new_tokens + len(prefix) + 2))
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
            enc_out = base_model.get_encoder()(
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
                out = base_model(
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

                top1_l = top1.tolist()
                bow_il, bow_vl = bow_ids.tolist(), bow_vals.tolist()
                cont_il, cont_vl = cont_ids.tolist(), cont_vals.tolist()

                all_done = True
                for b in range(bsz):
                    if finished[b]:
                        continue
                    a = int(top1_l[b])
                    if a == eos_id:
                        finished[b] = True
                        continue
                    all_done = False
                    seq_tokens[b].append(a)
                    if bool(bow_mask[a].item()):
                        if int(bow_il[b][0]) == a:
                            r_id, r_val = int(bow_il[b][1]), float(bow_vl[b][1])
                        else:
                            r_id, r_val = int(bow_il[b][0]), float(bow_vl[b][0])
                    elif bool(cont_mask[a].item()):
                        if int(cont_il[b][0]) == a:
                            r_id, r_val = int(cont_il[b][1]), float(cont_vl[b][1])
                        else:
                            r_id, r_val = int(cont_il[b][0]), float(cont_vl[b][0])
                    else:
                        r_id, r_val = -1, float('-inf')

                    if r_id >= 0 and r_val > -1e30:
                        rivals[b].append(r_id)
                        margins[b].append(float(step_logits[b, a].item()) - r_val)
                    else:
                        rivals[b].append(-1)
                        margins[b].append(float('inf'))

                if all_done:
                    break
                step_in = top1.unsqueeze(1)

            for b in range(bsz):
                pos = start + b
                elig = [j for j, r_id in enumerate(rivals[b]) if r_id >= 0]
                results.append(
                    dict(
                        pos=pos,
                        row_id=int(row_ids[pos]) if row_ids is not None else pos,
                        native_tokens=seq_tokens[b],
                        rivals=rivals[b],
                        margins=margins[b],
                        eligible_positions=elig,
                        native_text=_decode_body(tokenizer, seq_tokens[b]),
                    )
                )
    return results


def _greedy_branch_rollouts(
    base_model,
    tokenizer,
    sources,
    native_rows,
    forks,
    device,
    batch_size=64,
    max_source_length=256,
    max_target_length=256,
):
    _, eos_id, prefix = _resolve_prefix_ids(base_model)
    max_pos = int(getattr(base_model.config, 'max_position_embeddings', max_target_length + 2))
    eff_max_tgt = min(int(max_target_length), max(len(prefix) + 2, max_pos - 2))
    buckets = defaultdict(list)
    for f in forks:
        pos = f['pos']
        j = f['j']
        b_tok = f['b']
        dec_prefix = prefix + native_rows[pos]['native_tokens'][:j] + [b_tok]
        if len(dec_prefix) < eff_max_tgt:
            buckets[len(dec_prefix)].append((f, dec_prefix))

    tails_by_key = {}
    with torch.no_grad():
        for lp in sorted(buckets):
            items = buckets[lp]
            max_tail = max(0, eff_max_tgt - lp)
            for start in range(0, len(items), int(batch_size)):
                chunk = items[start:start + int(batch_size)]
                enc_in = tokenizer(
                    [sources[it[0]['pos']] for it in chunk],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_source_length,
                )
                input_ids = enc_in['input_ids'].to(device)
                attention_mask = enc_in['attention_mask'].to(device)
                enc_out = base_model.get_encoder()(
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
                    out = base_model(
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
                    tails_by_key[(f['row_id'], f['j'], f['b'])] = tails[idx]
    return tails_by_key


def select_balanced_epoch_forks(
    scored_forks,
    mode: str,
    seed: int,
    epoch_idx: int,
    num_strata: int = 5,
    min_pos_2x2: int = 32,
):
    """Balance +1/-1 counts per margin stratum matching 2x2's realized min(|p2|, |n2|) quota."""
    if mode not in ('2x2', 'natural'):
        raise ValueError(f'Unknown selection mode: {mode}')

    raw_pos_2x2 = sum(1 for x in scored_forks if x['s_2x2'] == 1)
    raw_neg_2x2 = sum(1 for x in scored_forks if x['s_2x2'] == -1)
    raw_abstain_2x2 = sum(1 for x in scored_forks if x['s_2x2'] == 0)
    raw_pos_nat = sum(1 for x in scored_forks if x['s_nat'] == 1)
    raw_neg_nat = sum(1 for x in scored_forks if x['s_nat'] == -1)
    raw_abstain_nat = sum(1 for x in scored_forks if x['s_nat'] == 0)

    stats = dict(
        epoch=int(epoch_idx),
        mode=mode,
        eligible_forks=len(scored_forks),
        raw_pos_2x2=raw_pos_2x2,
        raw_neg_2x2=raw_neg_2x2,
        raw_abstain_2x2=raw_abstain_2x2,
        raw_pos_nat=raw_pos_nat,
        raw_neg_nat=raw_neg_nat,
        raw_abstain_nat=raw_abstain_nat,
        min_pos_2x2_threshold=int(min_pos_2x2),
        selected_pos=0,
        selected_neg=0,
        selected_total=0,
        active=False,
        deactivated_reason=None,
        strata=[],
    )
    if not scored_forks or raw_pos_2x2 < int(min_pos_2x2):
        stats['deactivated_reason'] = 'raw_pos_2x2_below_threshold'
        return {}, stats

    ordered = sorted(scored_forks, key=lambda x: (float(x['margin']), int(x['row_id'])))
    k_strata = min(int(num_strata), len(ordered))
    groups = np.array_split(np.arange(len(ordered)), k_strata)

    selected_map = {}
    sel_pos_total = 0
    sel_neg_total = 0
    for g_idx, idxs in enumerate(groups):
        items = [ordered[int(i)] for i in idxs]
        p2 = [x for x in items if x['s_2x2'] == 1]
        n2 = [x for x in items if x['s_2x2'] == -1]
        pn = [x for x in items if x['s_nat'] == 1]
        nn = [x for x in items if x['s_nat'] == -1]

        quota_2x2 = min(len(p2), len(n2))
        pos_pool, neg_pool = (p2, n2) if mode == '2x2' else (pn, nn)
        quota_eff = min(quota_2x2, len(pos_pool), len(neg_pool))

        chosen_pos = sorted(
            pos_pool,
            key=lambda x: hash_selection_key(seed, epoch_idx, mode, 1, x['row_id']),
        )[:quota_eff]
        chosen_neg = sorted(
            neg_pool,
            key=lambda x: hash_selection_key(seed, epoch_idx, mode, -1, x['row_id']),
        )[:quota_eff]

        for x in chosen_pos:
            selected_map[int(x['row_id'])] = dict(
                j=int(x['j']),
                h=list(x['h']),
                a=int(x['a']),
                b=int(x['b']),
                sign=1,
                margin=float(x['margin']),
                stratum=int(g_idx),
            )
        for x in chosen_neg:
            selected_map[int(x['row_id'])] = dict(
                j=int(x['j']),
                h=list(x['h']),
                a=int(x['a']),
                b=int(x['b']),
                sign=-1,
                margin=float(x['margin']),
                stratum=int(g_idx),
            )

        sel_pos_total += len(chosen_pos)
        sel_neg_total += len(chosen_neg)
        stats['strata'].append(
            dict(
                stratum=int(g_idx),
                size=len(items),
                raw_pos_2x2=len(p2),
                raw_neg_2x2=len(n2),
                raw_pos_nat=len(pn),
                raw_neg_nat=len(nn),
                quota_2x2=quota_2x2,
                selected_each_sign=quota_eff,
            )
        )

    stats['selected_pos'] = sel_pos_total
    stats['selected_neg'] = sel_neg_total
    stats['selected_total'] = sel_pos_total + sel_neg_total
    stats['active'] = stats['selected_total'] > 0
    if not stats['active']:
        stats['deactivated_reason'] = 'zero_balanced_strata_quota'
    return selected_map, stats


def build_epoch_fork_table(
    model,
    tokenizer,
    sources,
    refs,
    mode: str,
    seed: int,
    epoch_idx: int,
    row_ids=None,
    bleu_tok: str = 'zh',
    num_strata: int = 5,
    min_pos_2x2: int = 32,
    batch_size: int = 64,
    max_source_length: int = 256,
    max_target_length: int = 256,
    tie_tol: float = 1e-10,
):
    """Build per-epoch frozen-snapshot balanced fork labels over the full training split."""
    if len(sources) != len(refs) or not sources:
        raise ValueError('sources and refs must have identical nonzero length')
    if row_ids is not None and len(row_ids) != len(sources):
        raise ValueError('row_ids must match sources length')
    base_model = model.base_model if hasattr(model, 'base_model') else model
    device = next(base_model.parameters()).device
    orig_training = base_model.training
    orig_dtype = next(base_model.parameters()).dtype
    orig_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    orig_tf32_cn = torch.backends.cudnn.allow_tf32

    started = time.perf_counter()
    try:
        base_model.eval()
        if orig_dtype != torch.float32:
            base_model.to(dtype=torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        with torch.amp.autocast(device_type=device.type, enabled=False):
            _, eos_id, prefix = _resolve_prefix_ids(base_model)
            if hasattr(model, '_bow_mask') and model._bow_mask is not None and model._cont_mask is not None:
                bow_mask, cont_mask = model._bow_mask, model._cont_mask
            else:
                bow_mask, cont_mask = build_lexical_boundary_masks(
                    tokenizer, int(base_model.config.vocab_size), extra_special_ids=[eos_id, *prefix]
                )

            t0 = time.perf_counter()
            native_rows = _greedy_native_with_rivals(
                base_model, tokenizer, sources, bow_mask, cont_mask, device, row_ids=row_ids,
                batch_size=batch_size, max_source_length=max_source_length, max_new_tokens=max_target_length - len(prefix),
            )
            t_native = time.perf_counter() - t0

            t0 = time.perf_counter()
            forks = []
            for row in native_rows:
                elig = row['eligible_positions']
                if not elig:
                    continue
                j = hash_prefix_index(seed, epoch_idx, row['row_id'], elig)
                forks.append(
                    dict(
                        pos=int(row['pos']),
                        row_id=int(row['row_id']),
                        j=j,
                        a=int(row['native_tokens'][j]),
                        b=int(row['rivals'][j]),
                        margin=float(row['margins'][j]),
                    )
                )
            tails = _greedy_branch_rollouts(
                base_model, tokenizer, sources, native_rows, forks, device,
                batch_size=batch_size, max_source_length=max_source_length, max_target_length=max_target_length,
            )
            t_branch = time.perf_counter() - t0

            t0 = time.perf_counter()
            bleu_m = BLEU(tokenize=bleu_tok)
            chrf_m = CHRF(word_order=0)
            clean_refs = [r.strip() for r in refs]
            native_texts = [r['native_text'].strip() for r in native_rows]

            st_b = np.stack(
                [np.asarray(bleu_m._extract_corpus_statistics([h], [[r]])[0], dtype=np.int64)
                 for h, r in zip(native_texts, clean_refs)],
                axis=0,
            )
            st_c = np.stack(
                [np.asarray(chrf_m._extract_corpus_statistics([h], [[r]])[0], dtype=np.float64)
                 for h, r in zip(native_texts, clean_refs)],
                axis=0,
            )
            tot_b = st_b.sum(axis=0)
            tot_c = st_c.sum(axis=0)
            native_bleu = float(bleu_m._compute_score_from_stats(tot_b.tolist()).score)
            native_chrf0 = float(chrf_m._compute_score_from_stats(tot_c.tolist()).score)

            scored_forks = []
            for f in forks:
                pos = f['pos']
                i = f['row_id']
                j = f['j']
                a = f['a']
                b_tok = f['b']
                nat_toks = native_rows[pos]['native_tokens']
                h = nat_toks[:j]
                u_a = nat_toks[j + 1:]
                u_b = tails.get((i, j, b_tok), [])

                variants = (
                    _decode_body(tokenizer, h + [a] + u_a),
                    _decode_body(tokenizer, h + [b_tok] + u_a),
                    _decode_body(tokenizer, h + [a] + u_b),
                    _decode_body(tokenizer, h + [b_tok] + u_b),
                )
                vb = [np.asarray(bleu_m._extract_corpus_statistics([z], [[clean_refs[pos]]])[0], dtype=np.int64) - st_b[pos] for z in variants]
                vc = [np.asarray(chrf_m._extract_corpus_statistics([z], [[clean_refs[pos]]])[0], dtype=np.float64) - st_c[pos] for z in variants]
                bs = [bleu_m._compute_score_from_stats((tot_b + db).tolist()).score for db in vb]
                cs = [chrf_m._compute_score_from_stats((tot_c + dc).tolist()).score for dc in vc]

                all_six = (
                    bs[1] - bs[0], bs[3] - bs[2], bs[3] - bs[0],
                    cs[1] - cs[0], cs[3] - cs[2], cs[3] - cs[0],
                )
                if all(d > tie_tol for d in all_six):
                    s_2x2 = 1
                elif all(d < -tie_tol for d in all_six):
                    s_2x2 = -1
                else:
                    s_2x2 = 0

                dn_b, dn_c = bs[3] - bs[0], cs[3] - cs[0]
                if dn_b > tie_tol and dn_c > tie_tol:
                    s_nat = 1
                elif dn_b < -tie_tol and dn_c < -tie_tol:
                    s_nat = -1
                else:
                    s_nat = 0

                scored_forks.append(
                    dict(
                        row_id=i,
                        j=j,
                        h=h,
                        a=a,
                        b=b_tok,
                        margin=f['margin'],
                        s_2x2=s_2x2,
                        s_nat=s_nat,
                    )
                )
            selected_map, stats = select_balanced_epoch_forks(
                scored_forks, mode=mode, seed=seed, epoch_idx=epoch_idx, num_strata=num_strata, min_pos_2x2=min_pos_2x2
            )
            t_scoring = time.perf_counter() - t0
            stats.update(
                native_train_bleu=round(native_bleu, 4),
                native_train_chrf0=round(native_chrf0, 4),
                seconds_native=round(t_native, 2),
                seconds_branch=round(t_branch, 2),
                seconds_scoring=round(t_scoring, 2),
                seconds_total=round(time.perf_counter() - started, 2),
            )
        return selected_map, stats
    finally:
        if orig_dtype != torch.float32:
            base_model.to(dtype=orig_dtype)
        base_model.train(orig_training)
        torch.backends.cuda.matmul.allow_tf32 = orig_tf32_mm
        torch.backends.cudnn.allow_tf32 = orig_tf32_cn


class LexicalForkMT(nn.Module):
    """Unidirectional 2x2 Lexical Fork Preference wrapper and matched Natural Control."""

    accepts_loss_kwargs = False

    def __init__(
        self,
        base_model,
        method='lexical-fork-2x2-v1',
        seed=42,
        lambda_epoch2=0.05,
        lambda_epoch3_plus=0.10,
        min_pos_2x2=32,
        num_margin_strata=5,
        rollout_batch_size=64,
        max_source_length=256,
        max_target_length=256,
        logit_chunk_tokens=32,
        bleu_tokenizer='zh',
        tokenizer=None,
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f'Unknown lexical_fork method: {method}')
        mode = METHODS[method]
        if not math.isfinite(lambda_epoch2) or lambda_epoch2 < 0.0:
            raise ValueError('lambda_epoch2 must be finite and nonnegative')
        if not math.isfinite(lambda_epoch3_plus) or lambda_epoch3_plus < 0.0:
            raise ValueError('lambda_epoch3_plus must be finite and nonnegative')
        if int(min_pos_2x2) < 1 or int(num_margin_strata) < 1:
            raise ValueError('min_pos_2x2 and num_margin_strata must be positive')
        if int(rollout_batch_size) < 1 or int(max_source_length) < 4 or int(max_target_length) < 4:
            raise ValueError('Invalid rollout_batch_size or sequence length bounds')

        self.base_model = base_model
        self.method = method
        self.mode = mode
        self.seed = int(seed)
        self.lambda_epoch2 = float(lambda_epoch2)
        self.lambda_epoch3_plus = float(lambda_epoch3_plus)
        self.min_pos_2x2 = int(min_pos_2x2)
        self.num_margin_strata = int(num_margin_strata)
        self.rollout_batch_size = int(rollout_batch_size)
        self.max_source_length = int(max_source_length)
        self.max_target_length = int(max_target_length)
        self.logit_chunk_tokens = int(logit_chunk_tokens)
        self.bleu_tokenizer = str(bleu_tokenizer)

        self._tokenizer = None
        self._bow_mask = None
        self._cont_mask = None
        if tokenizer is not None:
            self.set_tokenizer(tokenizer)

        self.active_epoch_idx = 0
        self.active_lambda = 0.0
        self.selected_map = {}
        self.last_diagnostics = {}

    @property
    def config(self):
        return self.base_model.config

    @property
    def generation_config(self):
        return self.base_model.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.base_model.generation_config = value

    def get_encoder(self):
        return self.base_model.get_encoder()

    def get_decoder(self):
        return self.base_model.get_decoder()

    def prepare_decoder_input_ids_from_labels(self, labels):
        if hasattr(self.base_model, 'prepare_decoder_input_ids_from_labels'):
            return self.base_model.prepare_decoder_input_ids_from_labels(labels)
        pad_id, eos_id, _ = _resolve_prefix_ids(self.base_model)
        dec_start = getattr(self.base_model.config, 'decoder_start_token_id', None)
        dec_start_id = int(eos_id if dec_start is None else dec_start)
        shifted = labels.new_full(labels.shape, pad_id)
        if labels.shape[1] > 1:
            shifted[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1] == -100, pad_id)
        shifted[:, 0] = dec_start_id
        return shifted

    def gradient_checkpointing_enable(self, *args, **kwargs):
        return self.base_model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        return self.base_model.gradient_checkpointing_disable(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def extra_parameter_count(self):
        return 0

    def num_parameters(self, only_trainable=False, exclude_embeddings=False, **kwargs):
        if hasattr(self.base_model, 'num_parameters'):
            return self.base_model.num_parameters(
                only_trainable=only_trainable, exclude_embeddings=exclude_embeddings, **kwargs
            )
        params = [p for p in self.parameters() if (not only_trainable or p.requires_grad)]
        return sum(p.numel() for p in params)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def enable_input_require_grads(self):
        return self.base_model.enable_input_require_grads()

    def disable_input_require_grads(self):
        return self.base_model.disable_input_require_grads()

    def state_dict(self, *args, **kwargs):
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.base_model.load_state_dict(state_dict, strict=strict)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer
        _, eos_id, prefix = _resolve_prefix_ids(self.base_model)
        self._bow_mask, self._cont_mask = build_lexical_boundary_masks(
            tokenizer, int(self.base_model.config.vocab_size), extra_special_ids=[eos_id, *prefix]
        )

    def set_epoch_fork_state(self, epoch_idx: int, active_lambda: float, selected_map: dict):
        self.active_epoch_idx = int(epoch_idx)
        self.active_lambda = float(active_lambda)
        self.selected_map = {int(k): dict(v) for k, v in selected_map.items()}

    def restore_fork_state(self, folder):
        state_path = Path(folder) / 'lexical_fork_state.json'
        if not state_path.is_file():
            self.set_epoch_fork_state(0, 0.0, {})
            return
        payload = json.loads(state_path.read_text(encoding='utf-8'))
        self.set_epoch_fork_state(
            payload.get('active_epoch_idx', 0),
            payload.get('active_lambda', 0.0),
            payload.get('selected_map', {}),
        )

    def scheduled_lambda(self, epoch_idx: int) -> float:
        if int(epoch_idx) <= 1:
            return 0.0
        if int(epoch_idx) == 2:
            return self.lambda_epoch2
        return self.lambda_epoch3_plus

    def _forward_logits(self, input_ids, attention_mask, decoder_input_ids):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        ).logits

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        row_ids=None,
        fork_window_count=None,
        fork_window_gas=None,
        **kwargs,
    ):
        kwargs.pop('num_items_in_batch', None)
        if not self.training or labels is None:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                **kwargs,
            )

        pad_id, _, prefix = _resolve_prefix_ids(self.base_model)
        if attention_mask is None:
            attention_mask = input_ids.ne(pad_id).long()
        dec_gold = (
            self.prepare_decoder_input_ids_from_labels(labels)
            if decoder_input_ids is None else decoder_input_ids
        )

        # Pass 1: standard teacher-forced CE over valid gold tokens.
        z_gold = self._forward_logits(input_ids, attention_mask, dec_gold)
        ce = chunked_ce_loss(z_gold, labels, chunk_size=self.logit_chunk_tokens)
        del z_gold

        device = input_ids.device
        active_indices = []
        active_entries = []
        if self.active_lambda > 0.0 and row_ids is not None and self.selected_map:
            rid_list = row_ids.tolist() if isinstance(row_ids, torch.Tensor) else list(row_ids)
            for idx, rid in enumerate(rid_list):
                entry = self.selected_map.get(int(rid))
                if entry is not None:
                    active_indices.append(idx)
                    active_entries.append(entry)

        mb_sel = len(active_entries)
        k_w = int(fork_window_count.item()) if isinstance(fork_window_count, torch.Tensor) else (
            int(fork_window_count) if fork_window_count is not None else mb_sel
        )
        g_w = int(fork_window_gas.item()) if isinstance(fork_window_gas, torch.Tensor) else (
            int(fork_window_gas) if fork_window_gas is not None else 1
        )
        pos_in_mb = sum(1 for e in active_entries if int(e['sign']) == 1)
        neg_in_mb = sum(1 for e in active_entries if int(e['sign']) == -1)

        if self.active_lambda > 0.0 and mb_sel > 0 and k_w > 0:
            seqs = [prefix + list(e['h']) for e in active_entries]
            max_l = max(len(s) for s in seqs)
            dec_fork = torch.full((mb_sel, max_l), pad_id, dtype=torch.long, device=device)
            pred_pos = torch.empty(mb_sel, dtype=torch.long, device=device)
            tok_a = torch.empty(mb_sel, dtype=torch.long, device=device)
            tok_b = torch.empty(mb_sel, dtype=torch.long, device=device)
            signs = torch.empty(mb_sel, dtype=torch.float32, device=device)

            for m_idx, (seq, e) in enumerate(zip(seqs, active_entries)):
                dec_fork[m_idx, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                pred_pos[m_idx] = len(seq) - 1
                tok_a[m_idx] = int(e['a'])
                tok_b[m_idx] = int(e['b'])
                signs[m_idx] = float(e['sign'])

            act_t = torch.tensor(active_indices, dtype=torch.long, device=device)
            z_fork = self._forward_logits(
                input_ids.index_select(0, act_t),
                attention_mask.index_select(0, act_t),
                dec_fork,
            )
            rows_ar = torch.arange(mb_sel, dtype=torch.long, device=device)
            piv_logits = z_fork[rows_ar, pred_pos].to(
                torch.float64 if z_fork.dtype == torch.float64 else torch.float32
            )
            del z_fork

            z_a = piv_logits.gather(-1, tok_a.unsqueeze(-1)).squeeze(-1)
            z_b = piv_logits.gather(-1, tok_b.unsqueeze(-1)).squeeze(-1)
            signs_c = signs.to(piv_logits.dtype)
            per_fork = F.softplus(-signs_c * (z_b - z_a))
            fork_mean_mb = per_fork.mean()
            # Multiply by g_w / k_w so Trainer's division by g_w yields exact mean over the full window k_w.
            fork_window_scaled = (float(g_w) / float(k_w)) * per_fork.sum()
            loss = ce + float(self.active_lambda) * fork_window_scaled.to(ce.dtype)
        else:
            fork_mean_mb = ce.new_zeros(())
            fork_window_scaled = ce.new_zeros(())
            loss = ce

        self.last_diagnostics = dict(
            mode=self.mode,
            epoch_idx=int(self.active_epoch_idx),
            active_lambda=float(self.active_lambda),
            ce=float(ce.detach()),
            fork_mean_in_mb=float(fork_mean_mb.detach()),
            fork_window_scaled=float(fork_window_scaled.detach()),
            mb_selected=int(mb_sel),
            window_selected=int(k_w),
            window_gas=int(g_w),
            pos_in_mb=int(pos_in_mb),
            neg_in_mb=int(neg_in_mb),
        )
        return Seq2SeqLMOutput(loss=loss)

    def method_metadata(self):
        return dict(
            version=1,
            method=self.method,
            mode=self.mode,
            seed=self.seed,
            lambda_epoch2=self.lambda_epoch2,
            lambda_epoch3_plus=self.lambda_epoch3_plus,
            min_pos_2x2=self.min_pos_2x2,
            num_margin_strata=self.num_margin_strata,
            rollout_batch_size=self.rollout_batch_size,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            logit_chunk_tokens=self.logit_chunk_tokens,
            bleu_tokenizer=self.bleu_tokenizer,
            prefix_policy='blake2b-epoch-row-random',
            extra_parameters=0,
            external_lm=False,
            inference_route='native',
            attribution=(
                '2x2 Cross-Tail Lexical Fork Preference (corpus BLEU + chrF0 6-way concordance)'
                if self.mode == '2x2'
                else 'Matched Natural-Diagonal Control (5-stratum sign-balanced parity with 2x2 quota)'
            ),
        )

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder, 'lexical_fork_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8'
        )
        Path(folder, 'lexical_fork_state.json').write_text(
            json.dumps(
                dict(
                    active_epoch_idx=self.active_epoch_idx,
                    active_lambda=self.active_lambda,
                    selected_map={str(k): v for k, v in self.selected_map.items()},
                )
            ),
            encoding='utf-8',
        )

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(
            Path(folder, 'lexical_fork_config.json').read_text(encoding='utf-8')
        )
        if saved != self.method_metadata():
            raise ValueError('Lexical fork checkpoint recipe mismatch')
