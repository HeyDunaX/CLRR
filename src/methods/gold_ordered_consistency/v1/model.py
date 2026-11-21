"""Gold-ordered asymmetric dropout consistency and matched teacher-selection controls.

At each supervised token position t, let view v in {1, 2} have log-distribution
log p_{v,t}. The gold-winner view w_t has the strictly larger gold log-probability
log p_{w_t,t}(y_t); exact ties use weight 0.5 for each direction. GOC transfers
one-way detached KL(sg(p_{w_t,t}) || p_{l_t,t}) from the gold-winner teacher to
the gold-loser student. Matched controls use identical two-pass native dropout,
FP32 chunked reduction, valid-token mean, and native single-pass inference.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import Seq2SeqLMOutput

from ...paired_route.v1.model import PairedRouteMT


METHODS = {
    'gold-ordered-consistency-v1': 'goc',
    'symmetric-stopgrad-consistency-v1': 'sym_sg',
    'inverted-gold-consistency-v1': 'inv',
    'rdrop-consistency-v1': 'rdrop',
}


def token_loss_and_stats(left, right, labels, mode):
    """Per-token CE, auxiliary KL, and detached winner/tie counts in FP32."""
    z1, z2 = left.float(), right.float()
    mask = labels.ne(-100)
    safe = labels.masked_fill(~mask, 0).unsqueeze(-1)
    a, b = F.log_softmax(z1, -1), F.log_softmax(z2, -1)
    p, q = a.exp(), b.exp()
    ag, bg = a.gather(-1, safe).squeeze(-1), b.gather(-1, safe).squeeze(-1)
    ce = -.5 * (ag + bg)
    winner = torch.where(ag > bg, 1., torch.where(ag < bg, 0., .5)).detach()
    win1 = ((ag > bg) & mask).sum().detach()
    ties = ((ag == bg) & mask).sum().detach()
    if mode == 'rdrop':
        kl = .5 * ((p - q) * (a - b)).sum(-1)
    else:
        if mode == 'goc':
            direction = winner
        elif mode == 'sym_sg':
            direction = torch.full_like(winner, .5)
        elif mode == 'inv':
            direction = 1. - winner
        else:
            raise ValueError(f'Unknown gold-ordered consistency mode: {mode}')
        ab = (p.detach() * (a.detach() - b)).sum(-1)
        ba = (q.detach() * (b.detach() - a)).sum(-1)
        kl = direction * ab + (1. - direction) * ba
    return (ce * mask).sum(), (kl * mask).sum(), win1, ties


def goc_loss(left, right, labels, mode='goc', beta=1.25, chunk_size=32):
    if mode not in ('goc', 'sym_sg', 'inv', 'rdrop'):
        raise ValueError(f'Unknown consistency mode: {mode}')
    if not math.isfinite(beta) or beta < 0:
        raise ValueError('consistency_beta must be finite and nonnegative')
    if left.shape != right.shape or left.shape[:-1] != labels.shape or left.shape[-1] < 2:
        raise ValueError('Expected paired [batch,time,vocab>=2] logits and aligned labels')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    mask = labels.ne(-100)
    valid_count = mask.sum()
    if not valid_count:
        raise ValueError('At least one supervised token required')
    if ((labels[mask] < 0) | (labels[mask] >= left.shape[-1])).any():
        raise ValueError('Supervised label outside vocabulary')
    a, b, y = left.flatten(0, 1), right.flatten(0, 1), labels.flatten()
    ce = left.new_zeros((), dtype=torch.float32)
    kl = left.new_zeros((), dtype=torch.float32)
    win1 = left.new_zeros((), dtype=torch.float32)
    ties = left.new_zeros((), dtype=torch.float32)
    for offset in range(0, y.numel(), chunk_size):
        args = (a[offset:offset+chunk_size], b[offset:offset+chunk_size],
                y[offset:offset+chunk_size])
        def reduced(x, z, target):
            return token_loss_and_stats(x, z, target, mode)
        if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
            c, k, w1, ti = checkpoint(reduced, *args, use_reentrant=True, preserve_rng_state=True)
        else:
            c, k, w1, ti = reduced(*args)
        ce, kl, win1, ties = ce + c, kl + k, win1 + w1, ties + ti
    denom = valid_count.to(torch.float32)
    ce, kl = ce / denom, kl / denom
    stats = dict(
        valid_tokens=int(valid_count.detach()),
        winner_view1_fraction=float((win1 / denom).detach()),
        tie_fraction=float((ties / denom).detach()),
    )
    return ce + beta * kl, ce, kl, stats


class GoldOrderedConsistencyMT(PairedRouteMT):
    def __init__(self, base_model, method='gold-ordered-consistency-v1', beta=1.25):
        if method not in METHODS:
            raise ValueError(f'Unknown GOC method: {method}')
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError('GOC consistency_beta must be finite and positive')
        super().__init__(base_model, 'paired-native-rdrop-v1')
        self.method = method
        self.mode = METHODS[method]
        self.beta = float(beta)

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                decoder_input_ids=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        if not self.training or labels is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                labels=labels, decoder_input_ids=decoder_input_ids, **kwargs)
        decoder = (self.prepare_decoder_input_ids_from_labels(labels)
                   if decoder_input_ids is None else decoder_input_ids)
        logits = []
        for _ in range(2):
            with self._route(False):
                logits.append(self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                    decoder_input_ids=decoder, use_cache=False, return_dict=True).logits)
        loss, ce, kl, stats = goc_loss(*logits, labels, mode=self.mode, beta=self.beta)
        self.last_diagnostics = dict(
            ce=float(ce.detach()),
            kl=float(kl.detach()),
            weighted_kl=float((self.beta * kl).detach()),
            beta=self.beta,
            mode=self.mode,
            views=2,
            second_view_rewired=False,
            checkpoint_context_empty=not self._checkpoint_context,
            **stats,
        )
        return Seq2SeqLMOutput(loss=loss)

    def method_metadata(self):
        teacher_policy = {
            'goc': 'per-token gold log-prob winner detached as teacher (0.5/0.5 on exact tie)',
            'sym_sg': 'deterministic 0.5/0.5 two-way detached KL (EDWARD-NMT Eq. 13)',
            'inv': 'per-token gold log-prob loser detached as teacher (0.5/0.5 on exact tie; causal falsifier)',
            'rdrop': 'undirected symmetric KL with both arguments differentiable (R-Drop comparator)',
        }[self.mode]
        return dict(
            version=1,
            method=self.method,
            mode=self.mode,
            beta=self.beta,
            temperature=1.0,
            views='two native passes; independent dropout',
            ce='mean two views; valid target-token mean per microbatch',
            auxiliary='full-vocabulary per-token KL; FP32 log_softmax chunked reduction',
            teacher_policy=teacher_policy,
            stop_gradient=self.mode != 'rdrop',
            masks='labels != -100, including supervised special tokens',
            logit_chunk_tokens=32,
            extra_parameters=0,
            original_lsr_weight=0.0,
            inference_route='native',
            strength=0.0,
            attribution='Gold-Ordered Asymmetric Dropout Consistency (GOC) vs EDWARD-NMT Eq. 13 and R-Drop controls',
        )

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder, 'gold_ordered_consistency_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8')

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(
            Path(folder, 'gold_ordered_consistency_config.json').read_text(encoding='utf-8'))
        if saved != self.method_metadata():
            raise ValueError('Gold-ordered consistency checkpoint recipe mismatch')
