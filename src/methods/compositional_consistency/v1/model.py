"""Compositional Context-Invariance Consistency (CoCC v1) and matched controls.

Unidirectional (X -> Y) output-distribution invariance across isolated sentences
and dynamically formed disjoint length-guarded concatenated sentence pairs using
Layout S1 (full native sequences [lang, body..., eos] concatenated without dropping
internal EOS tokens, masking only internal <tgt_lang> targets in CE_comp):

  - concat-ce-v1         ('concat_ce')       : Nguyen et al. (IWSLT 2021) RAND S1 CE (beta=0)
  - concat-symsg-v1      ('concat_symsg')    : Unaligned within-view SYM-SG on iso + comp (4 passes)
  - cocc-sym-v1          ('cocc_sym')        : Primary candidate: 2-pass S1 + mask_y_j1 + 0.5/0.5 stop-grad KL
  - cocc-iso-teacher-v1  ('cocc_iso_teacher'): Ablation V2: 2-pass S1 + mask_y_j1 + 1-way KL(sg(p_iso) || p_comp)

Inference (eval / generate) delegates directly to the unwrapped backbone with zero overhead.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import Seq2SeqLMOutput

from ...gold_ordered_consistency.v1.model import goc_loss


METHODS = {
    'concat-ce-v1': 'concat_ce',
    'concat-symsg-v1': 'concat_symsg',
    'cocc-sym-v1': 'cocc_sym',
    'cocc-iso-teacher-v1': 'cocc_iso_teacher',
}


def build_composite_batch_s1(
    input_ids,
    attention_mask,
    labels,
    pad_token_id,
    decoder_start_token_id,
    max_source_length=256,
    max_target_length=256,
    mask_y_j1=True,
    layout='s1',
):
    """Build disjoint length-guarded composite pairs and exact 1-to-1 KL indices.

    Uses PyTorch RNG (``torch.randperm`` on CPU) so HuggingFace Trainer's saved
    ``rng_state.pth`` guarantees bit-exact deterministic resume replay.
    Returns ``None`` if fewer than one valid length-guarded pair exists in the batch.
    """
    if layout not in ('s1', 's0'):
        raise ValueError(f'Unknown composite layout: {layout}')
    bsz, tgt_width = labels.shape
    device = input_ids.device
    if bsz < 2:
        return None
    if pad_token_id is None or decoder_start_token_id is None:
        raise ValueError('pad_token_id and decoder_start_token_id must be defined')

    if attention_mask is None:
        attention_mask = input_ids.ne(int(pad_token_id)).long()

    src_seqs = []
    tgt_seqs = []
    tgt_pos = []
    for k in range(bsz):
        src_mask = attention_mask[k].ne(0)
        x_k = input_ids[k][src_mask]
        lab_mask = labels[k].ne(-100)
        y_k = labels[k][lab_mask]
        pos_k = torch.nonzero(lab_mask, as_tuple=False).squeeze(-1)
        src_seqs.append(x_k)
        tgt_seqs.append(y_k)
        tgt_pos.append(pos_k)

    # Sample pairing order strictly via torch RNG so resume RNG replay matches.
    perm = torch.randperm(bsz, device='cpu').tolist()
    used = [False] * bsz
    pairs = []
    for a_idx in range(bsz):
        i = perm[a_idx]
        if used[i]:
            continue
        sx_i = int(src_seqs[i].numel())
        sy_i = int(tgt_seqs[i].numel())
        if sx_i < 2 or sy_i < 2:
            continue
        for b_idx in range(a_idx + 1, bsz):
            j = perm[b_idx]
            if used[j]:
                continue
            sx_j = int(src_seqs[j].numel())
            sy_j = int(tgt_seqs[j].numel())
            if sx_j < 2 or sy_j < 2:
                continue
            if layout == 's1':
                comp_sx = sx_i + sx_j
                comp_sy = sy_i + sy_j
            else:
                comp_sx = sx_i + sx_j - 2
                comp_sy = sy_i + sy_j - 2
            if comp_sx <= max_source_length and comp_sy <= max_target_length:
                used[i] = True
                used[j] = True
                pairs.append((i, j))
                break

    if not pairs:
        return None

    x_comps = []
    raw_y_comps = []
    if layout == 's1':
        max_xc = max(int(src_seqs[i].numel() + src_seqs[j].numel()) for i, j in pairs)
        max_yc = max(int(tgt_seqs[i].numel() + tgt_seqs[j].numel()) for i, j in pairs)
    else:
        max_xc = max(int(src_seqs[i].numel() + src_seqs[j].numel() - 2) for i, j in pairs)
        max_yc = max(int(tgt_seqs[i].numel() + tgt_seqs[j].numel() - 2) for i, j in pairs)

    num_pairs = len(pairs)
    comp_input_ids = torch.full((num_pairs, max_xc), int(pad_token_id), dtype=input_ids.dtype, device=device)
    comp_attention_mask = torch.zeros((num_pairs, max_xc), dtype=attention_mask.dtype, device=device)
    raw_y_padded = torch.full((num_pairs, max_yc), int(pad_token_id), dtype=labels.dtype, device=device)
    comp_labels = torch.full((num_pairs, max_yc), -100, dtype=labels.dtype, device=device)

    iso_indices = []
    comp_indices = []
    start_j = 2 if mask_y_j1 else 1

    for r, (i, j) in enumerate(pairs):
        xi, xj = src_seqs[i], src_seqs[j]
        yi, yj = tgt_seqs[i], tgt_seqs[j]
        pi_pos, pj_pos = tgt_pos[i], tgt_pos[j]
        sy_i = int(yi.numel())
        sy_j = int(yj.numel())

        if layout == 's1':
            xc = torch.cat([xi, xj], dim=0)
            yc = torch.cat([yi, yj], dim=0)
            lxc = int(xc.numel())
            lyc = int(yc.numel())
            comp_input_ids[r, :lxc] = xc
            comp_attention_mask[r, :lxc] = 1
            raw_y_padded[r, :lyc] = yc
            comp_labels[r, :lyc] = yc
            # Mask only the internal <tgt_lang> target token at position sy_i;
            # both </s>_i (sy_i - 1) and </s>_j (sy_i + sy_j - 1) remain supervised.
            comp_labels[r, sy_i] = -100

            # Segment 1: y_{i,1..m_i} and </s>_i (indices 1..sy_i-1)
            iso_s1 = i * tgt_width + pi_pos[1:sy_i]
            comp_s1 = r * max_yc + torch.arange(1, sy_i, dtype=torch.long, device=device)
            iso_indices.append(iso_s1)
            comp_indices.append(comp_s1)

            # Segment 2: y_{j,start_j..m_j} and </s>_j (indices start_j..sy_j-1)
            if start_j < sy_j:
                iso_s2 = j * tgt_width + pj_pos[start_j:sy_j]
                comp_s2 = r * max_yc + (sy_i + torch.arange(start_j, sy_j, dtype=torch.long, device=device))
                iso_indices.append(iso_s2)
                comp_indices.append(comp_s2)
        else:
            # S0 boundary ablation: strip internal </s>_i and <lang>_j
            xc = torch.cat([xi[:-1], xj[1:]], dim=0)
            yc = torch.cat([yi[:-1], yj[1:]], dim=0)
            lxc = int(xc.numel())
            lyc = int(yc.numel())
            comp_input_ids[r, :lxc] = xc
            comp_attention_mask[r, :lxc] = 1
            raw_y_padded[r, :lyc] = yc
            comp_labels[r, :lyc] = yc

            s1_end = sy_i - 1
            if s1_end > 1:
                iso_s1 = i * tgt_width + pi_pos[1:s1_end]
                comp_s1 = r * max_yc + torch.arange(1, s1_end, dtype=torch.long, device=device)
                iso_indices.append(iso_s1)
                comp_indices.append(comp_s1)

            if start_j < sy_j:
                offset = s1_end - 1
                iso_s2 = j * tgt_width + pj_pos[start_j:sy_j]
                comp_s2 = r * max_yc + (offset + torch.arange(start_j, sy_j, dtype=torch.long, device=device))
                iso_indices.append(iso_s2)
                comp_indices.append(comp_s2)

        x_comps.append(lxc)
        raw_y_comps.append(lyc)

    # Shift full unmasked raw_y_padded BEFORE applying internal -100 target mask so
    # right before predicting y_{j,1} the decoder receives [..., </s>_i, <tgt_lang>].
    comp_decoder_input_ids = torch.full((num_pairs, max_yc), int(pad_token_id), dtype=labels.dtype, device=device)
    comp_decoder_input_ids[:, 0] = int(decoder_start_token_id)
    if max_yc > 1:
        comp_decoder_input_ids[:, 1:] = raw_y_padded[:, :-1]

    iso_flat_idx = torch.cat(iso_indices, dim=0) if iso_indices else torch.empty(0, dtype=torch.long, device=device)
    comp_flat_idx = torch.cat(comp_indices, dim=0) if comp_indices else torch.empty(0, dtype=torch.long, device=device)

    return {
        'input_ids': comp_input_ids,
        'attention_mask': comp_attention_mask,
        'decoder_input_ids': comp_decoder_input_ids,
        'labels': comp_labels,
        'pairs': pairs,
        'iso_flat_idx': iso_flat_idx,
        'comp_flat_idx': comp_flat_idx,
    }


def _chunk_nll_sum(logits_chunk, labels_chunk):
    """Sum FP32 (or FP64 if given FP64) token cross-entropy over non-(-100) targets."""
    compute_dtype = torch.float64 if logits_chunk.dtype == torch.float64 else torch.float32
    logp = F.log_softmax(logits_chunk.to(compute_dtype), dim=-1)
    mask = labels_chunk.ne(-100)
    safe = labels_chunk.masked_fill(~mask, 0).unsqueeze(-1)
    nll = -logp.gather(-1, safe).squeeze(-1)
    return torch.where(mask, nll, torch.zeros_like(nll)).sum()


def chunked_ce_loss(logits, labels, chunk_size=32):
    """Memory-bounded FP32 valid-token mean CE matching HuggingFace Trainer."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError('Expected [batch, time, vocab] logits and [batch, time] labels')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    mask = labels.ne(-100)
    valid_count = mask.sum()
    if not valid_count:
        raise ValueError('At least one supervised token required')
    vocab = logits.shape[-1]
    if ((labels[mask] < 0) | (labels[mask] >= vocab)).any():
        raise ValueError('Supervised label outside vocabulary')

    compute_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    flat_logits = logits.reshape(-1, vocab)
    flat_labels = labels.reshape(-1)
    total_nll = logits.new_zeros((), dtype=compute_dtype)
    use_ckpt = torch.is_grad_enabled() and flat_logits.requires_grad
    for offset in range(0, flat_labels.numel(), chunk_size):
        lg = flat_logits[offset:offset + chunk_size]
        lb = flat_labels[offset:offset + chunk_size]
        nll = checkpoint(_chunk_nll_sum, lg, lb, use_reentrant=True, preserve_rng_state=True) if use_ckpt else _chunk_nll_sum(lg, lb)
        total_nll = total_nll + nll
    return total_nll / valid_count.to(compute_dtype)


def _chunk_aligned_kl_sum(za_chunk, zb_chunk, direction):
    """Exact stop-grad KL sum over one chunk of aligned [chunk, vocab] logit pairs."""
    compute_dtype = torch.float64 if (za_chunk.dtype == torch.float64 and zb_chunk.dtype == torch.float64) else torch.float32
    log_p = F.log_softmax(za_chunk.to(compute_dtype), dim=-1)
    log_q = F.log_softmax(zb_chunk.to(compute_dtype), dim=-1)
    p = log_p.exp()
    kl_i2c = (p.detach() * (log_p.detach() - log_q)).sum(dim=-1)
    if direction == 'iso_teacher':
        return kl_i2c.sum()
    if direction == 'sym_sg':
        q = log_q.exp()
        kl_c2i = (q.detach() * (log_q.detach() - log_p)).sum(dim=-1)
        return (0.5 * (kl_i2c + kl_c2i)).sum()
    raise ValueError(f'Unknown CoCC KL direction: {direction}')


def chunked_aligned_kl_loss(
    logits_iso,
    logits_comp,
    iso_flat_idx,
    comp_flat_idx,
    direction='sym_sg',
    chunk_size=32,
):
    """Mean FP32 stop-grad KL over 1-to-1 aligned isolated and composite tokens."""
    if direction not in ('sym_sg', 'iso_teacher'):
        raise ValueError(f'Unknown CoCC KL direction: {direction}')
    if iso_flat_idx.shape != comp_flat_idx.shape:
        raise ValueError('iso_flat_idx and comp_flat_idx must have identical shape')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    compute_dtype = torch.float64 if (logits_iso.dtype == torch.float64 and logits_comp.dtype == torch.float64) else torch.float32
    num_aligned = int(iso_flat_idx.numel())
    if num_aligned == 0:
        return logits_iso.new_zeros((), dtype=compute_dtype)

    flat_iso = logits_iso.reshape(-1, logits_iso.shape[-1]).index_select(0, iso_flat_idx)
    flat_comp = logits_comp.reshape(-1, logits_comp.shape[-1]).index_select(0, comp_flat_idx)
    kl_sum = logits_iso.new_zeros((), dtype=compute_dtype)
    use_ckpt = torch.is_grad_enabled() and (flat_iso.requires_grad or flat_comp.requires_grad)
    for offset in range(0, num_aligned, chunk_size):
        za = flat_iso[offset:offset + chunk_size]
        zb = flat_comp[offset:offset + chunk_size]
        if use_ckpt:
            part = checkpoint(
                lambda a, b: _chunk_aligned_kl_sum(a, b, direction),
                za, zb, use_reentrant=True, preserve_rng_state=True,
            )
        else:
            part = _chunk_aligned_kl_sum(za, zb, direction)
        kl_sum = kl_sum + part
    return kl_sum / float(num_aligned)


class CompositionalConsistencyMT(nn.Module):
    """Unidirectional CoCC wrapper and matched pilot controls (zero extra params)."""

    accepts_loss_kwargs = False

    def __init__(
        self,
        base_model,
        method='cocc-sym-v1',
        beta=1.25,
        layout='s1',
        mask_y_j1=True,
        max_source_length=256,
        max_target_length=256,
        logit_chunk_tokens=32,
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f'Unknown CoCC method: {method}')
        mode = METHODS[method]
        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError('consistency_beta must be finite and nonnegative')
        if mode == 'concat_ce' and beta != 0.0:
            raise ValueError('concat-ce-v1 requires consistency_beta == 0.0')
        if mode != 'concat_ce' and beta <= 0.0:
            raise ValueError(f'{method} requires consistency_beta > 0.0')
        if layout not in ('s1', 's0'):
            raise ValueError(f'Unknown CoCC layout: {layout}')
        if not isinstance(max_source_length, int) or max_source_length < 4:
            raise ValueError('max_source_length must be an integer >= 4')
        if not isinstance(max_target_length, int) or max_target_length < 4:
            raise ValueError('max_target_length must be an integer >= 4')
        if not isinstance(logit_chunk_tokens, int) or logit_chunk_tokens < 1:
            raise ValueError('logit_chunk_tokens must be a positive integer')

        self.base_model = base_model
        self.method = method
        self.mode = mode
        self.beta = float(beta)
        self.layout = str(layout)
        self.mask_y_j1 = bool(mask_y_j1)
        self.max_source_length = int(max_source_length)
        self.max_target_length = int(max_target_length)
        self.logit_chunk_tokens = int(logit_chunk_tokens)
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
        pad_id, dec_start_id = self._resolve_special_ids()
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

    def _resolve_special_ids(self):
        pad_id = self.base_model.config.pad_token_id
        dec_start = getattr(self.base_model.config, 'decoder_start_token_id', None)
        if dec_start is None:
            dec_start = self.base_model.config.eos_token_id
        if pad_id is None or dec_start is None:
            raise ValueError('Backbone config must define pad_token_id and decoder_start/eos token IDs')
        return int(pad_id), int(dec_start)

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

        pad_id, dec_start_id = self._resolve_special_ids()
        if attention_mask is None:
            attention_mask = input_ids.ne(pad_id).long()
        dec_iso = (
            self.prepare_decoder_input_ids_from_labels(labels)
            if decoder_input_ids is None else decoder_input_ids
        )
        comp = build_composite_batch_s1(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pad_token_id=pad_id,
            decoder_start_token_id=dec_start_id,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            mask_y_j1=self.mask_y_j1,
            layout=self.layout,
        )

        bsz = int(input_ids.shape[0])
        if comp is None:
            if self.mode == 'concat_symsg':
                z1 = self._forward_logits(input_ids, attention_mask, dec_iso)
                z2 = self._forward_logits(input_ids, attention_mask, dec_iso)
                loss, ce_iso, kl, _ = goc_loss(
                    z1, z2, labels, mode='sym_sg', beta=self.beta, chunk_size=self.logit_chunk_tokens
                )
            else:
                z_iso = self._forward_logits(input_ids, attention_mask, dec_iso)
                ce_iso = chunked_ce_loss(z_iso, labels, chunk_size=self.logit_chunk_tokens)
                kl = ce_iso.new_zeros(())
                loss = ce_iso
            self.last_diagnostics = dict(
                mode=self.mode,
                beta=self.beta,
                layout=self.layout,
                mask_y_j1=self.mask_y_j1,
                num_pairs=0,
                paired_fraction=0.0,
                aligned_tokens=0,
                ce_iso=float(ce_iso.detach()),
                ce_comp=0.0,
                kl=float(kl.detach()),
                weighted_kl=float((self.beta * kl).detach()) if self.mode == 'concat_symsg' else 0.0,
            )
            return Seq2SeqLMOutput(loss=loss)

        num_pairs = len(comp['pairs'])
        paired_fraction = float(2 * num_pairs) / float(max(1, bsz))
        num_aligned = int(comp['iso_flat_idx'].numel())

        if self.mode == 'concat_symsg':
            # Unaligned stacked baseline (4 forward passes): within-view SYM-SG separately on iso and comp.
            z_i1 = self._forward_logits(input_ids, attention_mask, dec_iso)
            z_i2 = self._forward_logits(input_ids, attention_mask, dec_iso)
            loss_iso, ce_iso, kl_iso, _ = goc_loss(
                z_i1, z_i2, labels, mode='sym_sg', beta=self.beta, chunk_size=self.logit_chunk_tokens
            )
            del z_i1, z_i2
            z_c1 = self._forward_logits(comp['input_ids'], comp['attention_mask'], comp['decoder_input_ids'])
            z_c2 = self._forward_logits(comp['input_ids'], comp['attention_mask'], comp['decoder_input_ids'])
            loss_comp, ce_comp, kl_comp, _ = goc_loss(
                z_c1, z_c2, comp['labels'], mode='sym_sg', beta=self.beta, chunk_size=self.logit_chunk_tokens
            )
            del z_c1, z_c2
            loss = 0.5 * (loss_iso + loss_comp)
            kl_val = 0.5 * (kl_iso + kl_comp)
            self.last_diagnostics = dict(
                mode=self.mode,
                beta=self.beta,
                layout=self.layout,
                mask_y_j1=self.mask_y_j1,
                num_pairs=num_pairs,
                paired_fraction=paired_fraction,
                aligned_tokens=0,
                ce_iso=float(ce_iso.detach()),
                ce_comp=float(ce_comp.detach()),
                kl_iso=float(kl_iso.detach()),
                kl_comp=float(kl_comp.detach()),
                kl=float(kl_val.detach()),
                weighted_kl=float((self.beta * kl_val).detach()),
            )
            return Seq2SeqLMOutput(loss=loss)

        # 2 forward passes total: Pass 1 isolated (B, L), Pass 2 composite (B/2, ~2L) -> ~2*B*L total tokens.
        z_iso = self._forward_logits(input_ids, attention_mask, dec_iso)
        ce_iso = chunked_ce_loss(z_iso, labels, chunk_size=self.logit_chunk_tokens)
        z_comp = self._forward_logits(comp['input_ids'], comp['attention_mask'], comp['decoder_input_ids'])
        ce_comp = chunked_ce_loss(z_comp, comp['labels'], chunk_size=self.logit_chunk_tokens)
        ce = 0.5 * (ce_iso + ce_comp)

        if self.mode == 'concat_ce':
            del z_iso, z_comp
            kl = ce.new_zeros(())
            loss = ce
        else:
            direction = 'sym_sg' if self.mode == 'cocc_sym' else 'iso_teacher'
            kl = chunked_aligned_kl_loss(
                z_iso,
                z_comp,
                comp['iso_flat_idx'],
                comp['comp_flat_idx'],
                direction=direction,
                chunk_size=self.logit_chunk_tokens,
            )
            del z_iso, z_comp
            loss = ce + self.beta * kl

        self.last_diagnostics = dict(
            mode=self.mode,
            beta=self.beta,
            layout=self.layout,
            mask_y_j1=self.mask_y_j1,
            num_pairs=num_pairs,
            paired_fraction=paired_fraction,
            aligned_tokens=num_aligned,
            ce_iso=float(ce_iso.detach()),
            ce_comp=float(ce_comp.detach()),
            kl=float(kl.detach()),
            weighted_kl=float((self.beta * kl).detach()),
        )
        return Seq2SeqLMOutput(loss=loss)

    def method_metadata(self):
        kl_policy = {
            'concat_ce': 'none (Nguyen et al. IWSLT 2021 RAND composite CE only)',
            'concat_symsg': 'within-view 0.5/0.5 stop-grad SYM-SG on iso and comp separately (no iso<->comp alignment)',
            'cocc_sym': 'token-aligned iso<->comp 0.5/0.5 symmetric stop-grad KL under Layout S1',
            'cocc_iso_teacher': 'token-aligned iso->comp one-way detached KL(sg(p_iso) || p_comp) under Layout S1',
        }[self.mode]
        return dict(
            version=1,
            method=self.method,
            mode=self.mode,
            beta=self.beta,
            layout=self.layout,
            mask_y_j1=self.mask_y_j1,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            logit_chunk_tokens=self.logit_chunk_tokens,
            kl_policy=kl_policy,
            pairing='torch-rng length-guarded disjoint pairing',
            extra_parameters=0,
            external_lm=False,
            inference_route='native',
            attribution=(
                'Compositional Context-Invariance Consistency (CoCC): token-aligned '
                'isolated-vs-concatenated distribution invariance under native Layout S1'
            ),
        )

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder, 'compositional_consistency_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8'
        )

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(
            Path(folder, 'compositional_consistency_config.json').read_text(encoding='utf-8')
        )
        if saved != self.method_metadata():
            raise ValueError('CoCC checkpoint recipe mismatch')
