"""Marginal-Free Cross-Ratio Dual Consistency (CR-Dual) and matched controls.

Based on the Arnold & Press (1989) conditional compatibility identity applied to
Dual Supervised Learning (Xia et al., ICML 2017) inside a single shared
many-to-many seq2seq backbone (mBART-50 / NLLB-200):

    M_fwd(i, j) = log P(y_i|x_i) - log P(y_i|x_j) - log P(y_j|x_i) + log P(y_j|x_j)
    M_rev(i, j) = log P(x_i|y_i) - log P(x_i|y_j) - log P(x_j|y_i) + log P(x_j|y_j)

Both unknown marginal language models P(x) and P(y) cancel identically. Dividing
both sides by the shared symmetric pair length L_ij = 0.5*(|x_i|+|x_j|+|y_i|+|y_j|)
preserves exact Bayesian invariance regardless of tokenization ratio |x| != |y|.

Matched pilot modes:
  - bidirectional-ce-v1       ('bit_ce')      : CE_fwd + alpha_rev * CE_rev (Ding 2021 BiT)
  - cross-ratio-dual-v1       ('cr_dual')     : BIT_CE + lambda(s) * Huber((M_fwd - M_rev)/L_ij)
  - cross-ratio-shuffled-v1   ('cr_shuffled') : Identical compute/Huber, pairing M_fwd(i,j) with M_rev(k,l)
  - unidirectional-contrast-v1('uni_contrast'): CE_fwd + lambda(s) * ReLU(gamma - (lp_diag - lp_off)/|y|)

Inference (eval / generate) executes the unwrapped native backbone with zero overhead.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


METHODS = {
    'bidirectional-ce-v1': 'bit_ce',
    'cross-ratio-dual-v1': 'cr_dual',
    'cross-ratio-shuffled-v1': 'cr_shuffled',
    'unidirectional-contrast-v1': 'uni_contrast',
}


def build_length_matched_involution(src_lengths, tgt_lengths):
    """Build a disjoint length-matched involution pi over batch rows.

    Sorts rows by combined sequence length (|x| + |y|) and pairs adjacent rows
    (order[0], order[1]), (order[2], order[3]), ... so pi(pi(k)) == k and
    pi(k) != k for all paired rows. If batch size B is odd, the final longest
    row maps to itself in pi and is excluded from (idx_i, idx_j).
    """
    if src_lengths.ndim != 1 or tgt_lengths.ndim != 1 or src_lengths.shape != tgt_lengths.shape:
        raise ValueError('src_lengths and tgt_lengths must be 1-D tensors of identical shape')
    bsz = int(src_lengths.shape[0])
    device = src_lengths.device
    if bsz < 2:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, torch.arange(bsz, dtype=torch.long, device=device)
    combined = (src_lengths.long() + tgt_lengths.long())
    order = torch.argsort(combined, stable=True)
    usable = (bsz // 2) * 2
    idx_i = order[0:usable:2]
    idx_j = order[1:usable:2]
    pi = torch.arange(bsz, dtype=torch.long, device=device)
    pi[idx_i] = idx_j
    pi[idx_j] = idx_i
    return idx_i, idx_j, pi


def _chunk_tok_logprob(logits_chunk, labels_chunk):
    """FP32 token log-probability gathered at supervised labels (0 on -100)."""
    logp = F.log_softmax(logits_chunk.float(), dim=-1)
    mask = labels_chunk.ne(-100)
    safe = labels_chunk.masked_fill(~mask, 0).unsqueeze(-1)
    tok_lp = logp.gather(-1, safe).squeeze(-1)
    return torch.where(mask, tok_lp, torch.zeros_like(tok_lp))


def chunked_seq_logprob_and_ce(logits, labels, chunk_size=64):
    """Compute exact FP32 sequence log-probabilities, valid lengths, and mean CE.

    Uses chunked FP32 log_softmax with reentrant checkpointing to keep peak
    VRAM small over 250k-token multilingual vocabularies while guaranteeing
    exact zero gradient on masked (-100) positions.
    """
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError('Expected [batch, time, vocab] logits and [batch, time] labels')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    bsz, seq_len, vocab = logits.shape
    mask = labels.ne(-100)
    valid_count = mask.sum()
    if not valid_count:
        raise ValueError('At least one supervised token required')
    if ((labels[mask] < 0) | (labels[mask] >= vocab)).any():
        raise ValueError('Supervised label outside vocabulary')

    flat_logits = logits.reshape(bsz * seq_len, vocab)
    flat_labels = labels.reshape(bsz * seq_len)
    parts = []
    use_ckpt = torch.is_grad_enabled() and flat_logits.requires_grad
    for offset in range(0, flat_labels.numel(), chunk_size):
        lg = flat_logits[offset:offset + chunk_size]
        lb = flat_labels[offset:offset + chunk_size]
        if use_ckpt:
            tok_lp = checkpoint(_chunk_tok_logprob, lg, lb, use_reentrant=True, preserve_rng_state=True)
        else:
            tok_lp = _chunk_tok_logprob(lg, lb)
        parts.append(tok_lp)
    tok_logp = torch.cat(parts, dim=0).reshape(bsz, seq_len)
    seq_logp = tok_logp.sum(dim=1)
    seq_lens = mask.sum(dim=1).to(torch.float32)
    mean_ce = -seq_logp.sum() / seq_lens.sum()
    return seq_logp, seq_lens, mean_ce


def cross_ratio_huber_loss(
    lp_f_diag, lp_f_off, lp_r_diag, lp_r_off,
    len_x, len_y, idx_i, idx_j,
    delta=1.0, shuffle_reverse=False,
):
    """Symmetric length-normalized 4-point cross-ratio Huber compatibility loss.

    For each disjoint pair (i, j):
      M_fwd(i, j) = (lp_f_diag[i] + lp_f_diag[j]) - (lp_f_off[i] + lp_f_off[j])
      M_rev(i, j) = (lp_r_diag[i] + lp_r_diag[j]) - (lp_r_off[i] + lp_r_off[j])
      L_ij        = 0.5 * (len_x[i] + len_x[j] + len_y[i] + len_y[j])

    Because lp_f_off[i] = log P(y_i | x_j) and lp_f_off[j] = log P(y_j | x_i)
    under the involution pi(i)=j, pi(j)=i, both directions share the exact same
    symmetric normalizer L_ij.

    If shuffle_reverse=True (Control 1 falsifier), compares M_fwd(i,j)/L_ij
    against M_rev(k,l)/L_kl rolled by 1 across pairs (k,l) != (i,j).
    """
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError('cr_delta must be finite and positive')
    num_pairs = int(idx_i.numel())
    if num_pairs == 0:
        zero = lp_f_diag.new_zeros((), dtype=torch.float32)
        return zero, dict(num_pairs=0, mean_m_fwd=0.0, mean_m_rev=0.0,
                          mean_m_diff=0.0, abs_cr_violation=0.0)
    m_fwd = (lp_f_diag[idx_i] + lp_f_diag[idx_j]) - (lp_f_off[idx_i] + lp_f_off[idx_j])
    m_rev = (lp_r_diag[idx_i] + lp_r_diag[idx_j]) - (lp_r_off[idx_i] + lp_r_off[idx_j])
    l_ij = 0.5 * (len_x[idx_i] + len_x[idx_j] + len_y[idx_i] + len_y[idx_j]).clamp(min=1.0)
    norm_fwd = m_fwd / l_ij
    norm_rev = m_rev / l_ij
    target_rev = torch.roll(norm_rev, shifts=1, dims=0) if (shuffle_reverse and num_pairs >= 2) else norm_rev
    loss = F.huber_loss(norm_fwd, target_rev, reduction='mean', delta=float(delta))
    true_diff = (norm_fwd - norm_rev).detach()
    stats = dict(
        num_pairs=num_pairs,
        mean_m_fwd=float(norm_fwd.detach().mean()),
        mean_m_rev=float(norm_rev.detach().mean()),
        mean_m_diff=float(true_diff.mean()),
        abs_cr_violation=float(true_diff.abs().mean()),
    )
    return loss, stats


def unidirectional_contrast_loss(lp_f_diag, lp_f_off, len_y, idx_i, idx_j, margin=2.0):
    """Control 3 falsifier: unidirectional SCD-style off-diagonal margin loss."""
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError('contrast_margin must be finite and positive')
    if int(idx_i.numel()) == 0:
        zero = lp_f_diag.new_zeros((), dtype=torch.float32)
        return zero, dict(num_pairs=0, mean_uni_margin=0.0)
    active = torch.cat([idx_i, idx_j], dim=0)
    norm_margin = (lp_f_diag[active] - lp_f_off[active]) / len_y[active].clamp(min=1.0)
    loss = F.relu(float(margin) - norm_margin).mean()
    return loss, dict(num_pairs=int(idx_i.numel()), mean_uni_margin=float(norm_margin.detach().mean()))


def _snapshot_rng(device):
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    return cpu_state, cuda_state


def _restore_rng(state, device):
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if device.type == 'cuda' and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


class CrossRatioDualMT(nn.Module):
    """Many-to-many wrapper implementing CR-Dual and matched pilot controls.

    Zero extra parameters, zero external LMs, zero inference overhead.
    """

    accepts_loss_kwargs = False

    def __init__(
        self,
        base_model,
        method='cross-ratio-dual-v1',
        alpha_rev=0.5,
        cr_lambda=0.25,
        cr_delta=1.0,
        cr_warmup_fraction=0.2,
        contrast_margin=2.0,
        paired_rng=True,
        logit_chunk_tokens=64,
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f'Unknown CR-Dual method: {method}')
        if not math.isfinite(alpha_rev) or alpha_rev < 0:
            raise ValueError('alpha_rev must be finite and nonnegative')
        if not math.isfinite(cr_lambda) or cr_lambda < 0:
            raise ValueError('cr_lambda must be finite and nonnegative')
        if not math.isfinite(cr_delta) or cr_delta <= 0:
            raise ValueError('cr_delta must be finite and positive')
        if not math.isfinite(cr_warmup_fraction) or not (0.0 <= cr_warmup_fraction <= 1.0):
            raise ValueError('cr_warmup_fraction must be in [0, 1]')
        if not math.isfinite(contrast_margin) or contrast_margin <= 0:
            raise ValueError('contrast_margin must be finite and positive')
        if not isinstance(logit_chunk_tokens, int) or logit_chunk_tokens < 1:
            raise ValueError('logit_chunk_tokens must be a positive integer')

        self.base_model = base_model
        self.method = method
        self.mode = METHODS[method]
        self.alpha_rev = float(alpha_rev)
        self.cr_lambda = float(cr_lambda)
        self.cr_delta = float(cr_delta)
        self.cr_warmup_fraction = float(cr_warmup_fraction)
        self.contrast_margin = float(contrast_margin)
        self.paired_rng = bool(paired_rng)
        self.logit_chunk_tokens = int(logit_chunk_tokens)

        # Standalone calls (e.g. unit tests) default to full lambda unless
        # ControlTrainer sets explicit optimizer step progress.
        self._current_step = None
        self._max_steps = None
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
        return self.base_model.prepare_decoder_input_ids_from_labels(labels)

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

    def set_training_progress(self, current_step, max_steps):
        self._current_step = int(current_step)
        self._max_steps = int(max_steps) if max_steps is not None else None

    def current_lambda(self):
        if self.mode == 'bit_ce' or self.cr_lambda == 0.0:
            return 0.0
        if self._current_step is None or self._max_steps is None or self._max_steps <= 0:
            return self.cr_lambda
        if self.cr_warmup_fraction <= 0.0:
            return self.cr_lambda
        warmup_steps = max(1, int(round(self.cr_warmup_fraction * self._max_steps)))
        scale = min(1.0, max(0.0, float(self._current_step) / float(warmup_steps)))
        return self.cr_lambda * scale

    def _build_reverse_batch(self, input_ids, attention_mask, labels):
        """Construct byte-exact reverse-direction (Y -> X) tensors for mBART-50 / NLLB-200.

        In mBART-50 and NLLB-200, input_ids = [src_lang, x_1..x_m, eos, pad...]
        and labels = [tgt_lang, y_1..y_n, eos, -100...]. Swapping them with
        pad_token_id <-> -100 preserves exact source/target language tags and EOS.
        """
        pad_id = self.base_model.config.pad_token_id
        if pad_id is None:
            raise ValueError('Backbone config must define pad_token_id for bidirectional swap')
        rev_attention_mask = labels.ne(-100).long()
        rev_input_ids = labels.masked_fill(labels.eq(-100), int(pad_id))
        if attention_mask is None:
            src_valid = input_ids.ne(int(pad_id))
        else:
            src_valid = attention_mask.ne(0)
        rev_labels = input_ids.masked_fill(~src_valid, -100)
        return rev_input_ids, rev_attention_mask, rev_labels

    def _decode_from_encoder(self, hidden_states, encoder_attention_mask, decoder_input_ids):
        return self.base_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden_states),
            attention_mask=encoder_attention_mask,
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

        device = input_ids.device
        pad_id = self.base_model.config.pad_token_id
        if attention_mask is None:
            attention_mask = input_ids.ne(int(pad_id)).long() if pad_id is not None else torch.ones_like(input_ids)
        dec_fwd = (
            self.prepare_decoder_input_ids_from_labels(labels)
            if decoder_input_ids is None else decoder_input_ids
        )
        src_lengths = attention_mask.sum(dim=1)
        tgt_lengths = labels.ne(-100).sum(dim=1)
        idx_i, idx_j, pi = build_length_matched_involution(src_lengths, tgt_lengths)

        # 1) Encode X once; reuse H_X for both diagonal and off-diagonal forward decoder passes.
        enc_x = self.base_model.get_encoder()(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state

        need_off_fwd = self.mode in ('cr_dual', 'cr_shuffled', 'uni_contrast') and int(idx_i.numel()) > 0
        eff_lambda = self.current_lambda()

        rng_fwd = _snapshot_rng(device) if (need_off_fwd and self.paired_rng) else None
        logits_f_diag = self._decode_from_encoder(enc_x, attention_mask, dec_fwd)
        lp_f_diag, len_y, ce_fwd = chunked_seq_logprob_and_ce(
            logits_f_diag, labels, chunk_size=self.logit_chunk_tokens
        )
        del logits_f_diag

        if need_off_fwd:
            if rng_fwd is not None:
                _restore_rng(rng_fwd, device)
            logits_f_off = self._decode_from_encoder(
                enc_x.index_select(0, pi),
                attention_mask.index_select(0, pi),
                dec_fwd,
            )
            lp_f_off, _, _ = chunked_seq_logprob_and_ce(
                logits_f_off, labels, chunk_size=self.logit_chunk_tokens
            )
            del logits_f_off
        else:
            lp_f_off = None

        if self.mode == 'uni_contrast':
            if lp_f_off is not None:
                aux_loss, aux_stats = unidirectional_contrast_loss(
                    lp_f_diag, lp_f_off, len_y, idx_i, idx_j, margin=self.contrast_margin
                )
            else:
                aux_loss = ce_fwd.new_zeros(())
                aux_stats = dict(num_pairs=0, mean_uni_margin=0.0)
            total = ce_fwd + eff_lambda * aux_loss
            self.last_diagnostics = dict(
                mode=self.mode,
                ce_fwd=float(ce_fwd.detach()),
                ce_rev=0.0,
                aux_loss=float(aux_loss.detach()),
                weighted_aux=float((eff_lambda * aux_loss).detach()),
                effective_lambda=float(eff_lambda),
                alpha_rev=0.0,
                paired_rng=self.paired_rng,
                **aux_stats,
            )
            return Seq2SeqLMOutput(loss=total)

        # 2) Bidirectional reverse direction (Y -> X): encode Y once.
        rev_input_ids, rev_attention_mask, rev_labels = self._build_reverse_batch(
            input_ids, attention_mask, labels
        )
        dec_rev = self.prepare_decoder_input_ids_from_labels(rev_labels)
        enc_y = self.base_model.get_encoder()(
            input_ids=rev_input_ids, attention_mask=rev_attention_mask, return_dict=True
        ).last_hidden_state

        need_off_rev = self.mode in ('cr_dual', 'cr_shuffled') and int(idx_i.numel()) > 0
        rng_rev = _snapshot_rng(device) if (need_off_rev and self.paired_rng) else None
        logits_r_diag = self._decode_from_encoder(enc_y, rev_attention_mask, dec_rev)
        lp_r_diag, len_x, ce_rev = chunked_seq_logprob_and_ce(
            logits_r_diag, rev_labels, chunk_size=self.logit_chunk_tokens
        )
        del logits_r_diag

        if need_off_rev:
            if rng_rev is not None:
                _restore_rng(rng_rev, device)
            logits_r_off = self._decode_from_encoder(
                enc_y.index_select(0, pi),
                rev_attention_mask.index_select(0, pi),
                dec_rev,
            )
            lp_r_off, _, _ = chunked_seq_logprob_and_ce(
                logits_r_off, rev_labels, chunk_size=self.logit_chunk_tokens
            )
            del logits_r_off
            cr_loss, cr_stats = cross_ratio_huber_loss(
                lp_f_diag, lp_f_off, lp_r_diag, lp_r_off,
                len_x, len_y, idx_i, idx_j,
                delta=self.cr_delta,
                shuffle_reverse=(self.mode == 'cr_shuffled'),
            )
        else:
            cr_loss = ce_fwd.new_zeros(())
            cr_stats = dict(num_pairs=int(idx_i.numel()), mean_m_fwd=0.0, mean_m_rev=0.0,
                            mean_m_diff=0.0, abs_cr_violation=0.0)

        total = ce_fwd + self.alpha_rev * ce_rev + eff_lambda * cr_loss
        self.last_diagnostics = dict(
            mode=self.mode,
            ce_fwd=float(ce_fwd.detach()),
            ce_rev=float(ce_rev.detach()),
            aux_loss=float(cr_loss.detach()),
            weighted_aux=float((eff_lambda * cr_loss).detach()),
            effective_lambda=float(eff_lambda),
            alpha_rev=self.alpha_rev,
            paired_rng=self.paired_rng,
            **cr_stats,
        )
        return Seq2SeqLMOutput(loss=total)

    def method_metadata(self):
        return dict(
            version=1,
            method=self.method,
            mode=self.mode,
            alpha_rev=self.alpha_rev if self.mode != 'uni_contrast' else 0.0,
            cr_lambda=self.cr_lambda if self.mode != 'bit_ce' else 0.0,
            cr_delta=self.cr_delta,
            cr_warmup_fraction=self.cr_warmup_fraction,
            contrast_margin=self.contrast_margin,
            paired_rng=self.paired_rng,
            logit_chunk_tokens=self.logit_chunk_tokens,
            symmetric_length_norm='0.5*(|x_i|+|x_j|+|y_i|+|y_j|)',
            pairing='length-sorted disjoint involution pi',
            extra_parameters=0,
            external_lm=False,
            inference_route='native',
            attribution=(
                'Marginal-Free Cross-Ratio Dual Consistency (CR-Dual) via Arnold & Press (1989) '
                '4-point conditional compatibility eliminating external LMs in Xia et al. (ICML 2017 DSL)'
            ),
        )

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder, 'cross_ratio_dual_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8'
        )

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(
            Path(folder, 'cross_ratio_dual_config.json').read_text(encoding='utf-8')
        )
        if saved != self.method_metadata():
            raise ValueError('CR-Dual checkpoint recipe mismatch')
