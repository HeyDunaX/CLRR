"""R-Drop engineering ablations using the KL chain-rule split from DKD."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import Seq2SeqLMOutput

from ...paired_route.v1.model import PairedRouteMT


METHODS = {'gold-consistency-v1': ('gold', 2.5),
           'rest-consistency-v1': ('rest', 2.5),
           'weak-full-consistency-v1': ('full', 1.25)}


def token_components(left, right, labels):
    """Per-token CE, binary gold/rest KL and probability-weighted rest KL.

    Keep both views and non-gold masses differentiable. Work entirely in log
    space, including when FP32 p_gold rounds to one. Ignore padding only after
    replacing its index with a safe vocabulary index.
    """
    a, b = F.log_softmax(left.float(), -1), F.log_softmax(right.float(), -1)
    safe = labels.masked_fill(labels.eq(-100), 0).unsqueeze(-1)
    nongold = torch.ones_like(a, dtype=torch.bool).scatter_(-1, safe, False)
    ag, bg = a.gather(-1, safe), b.gather(-1, safe)
    ar = a.masked_fill(~nongold, -torch.inf).logsumexp(-1, keepdim=True)
    br = b.masked_fill(~nongold, -torch.inf).logsumexp(-1, keepdim=True)
    u, v = torch.cat((ag, ar), -1), torch.cat((bg, br), -1)
    gold = .5 * ((u.exp() - v.exp()) * (u - v)).sum(-1)
    conditional_difference = ((a - ar) - (b - br)).masked_fill(~nongold, 0.)
    rest = .5 * ((a.exp() - b.exp()) * conditional_difference).sum(-1)
    ce = -.5 * (ag + bg).squeeze(-1)
    return ce, gold, rest


def consistency_loss(left, right, labels, component, beta, chunk_size=32):
    if component not in ('gold', 'rest', 'full'):
        raise ValueError('Unknown consistency component')
    if left.shape != right.shape or left.shape[:-1] != labels.shape or left.shape[-1] < 2:
        raise ValueError('Expected paired [batch,time,vocab>=2] logits and labels')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    mask = labels.ne(-100)
    if not mask.any():
        raise ValueError('At least one supervised token required')
    if ((labels[mask] < 0) | (labels[mask] >= left.shape[-1])).any():
        raise ValueError('Supervised label outside vocabulary')
    left, right, labels = left.flatten(0, 1), right.flatten(0, 1), labels.flatten()
    sums = [left.new_zeros((), dtype=torch.float32) for _ in range(3)]
    for offset in range(0, labels.numel(), chunk_size):
        args = (left[offset:offset+chunk_size], right[offset:offset+chunk_size],
                labels[offset:offset+chunk_size])
        def reduced(a, b, y):
            valid = y.ne(-100)
            return tuple((value * valid).sum() for value in token_components(a, b, y))
        if torch.is_grad_enabled() and (left.requires_grad or right.requires_grad):
            values = checkpoint(reduced, *args, use_reentrant=True, preserve_rng_state=True)
        else:
            values = reduced(*args)
        sums = [total + value for total, value in zip(sums, values)]
    ce, gold, rest = [value / mask.sum() for value in sums]
    auxiliary = {'gold': gold, 'rest': rest, 'full': gold + rest}[component]
    return ce + beta * auxiliary, ce, gold, rest


class DecoupledConsistencyMT(PairedRouteMT):
    def __init__(self, base_model, method):
        # Reuse verified native two-pass checkpoint/RNG/cache handling unchanged.
        super().__init__(base_model, 'paired-native-rdrop-v1')
        self.method = method
        self.component, self.beta = METHODS[method]

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                decoder_input_ids=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        if not self.training or labels is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids, labels=labels, **kwargs)
        decoder_input_ids = (self.prepare_decoder_input_ids_from_labels(labels)
                             if decoder_input_ids is None else decoder_input_ids)
        logits = []
        for _ in range(2):
            with self._route(False):
                logits.append(self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids, use_cache=False, return_dict=True).logits)
        loss, ce, gold, rest = consistency_loss(*logits, labels, self.component, self.beta)
        selected = {'gold': gold, 'rest': rest, 'full': gold + rest}[self.component]
        self.last_diagnostics = dict(ce=float(ce.detach()), kl=float(selected.detach()),
            gold=float(gold.detach()), rest=float(rest.detach()), beta=self.beta,
            component=self.component, views=2, second_view_rewired=False,
            checkpoint_context_empty=not self._checkpoint_context)
        return Seq2SeqLMOutput(loss=loss)

    def method_metadata(self):
        return dict(version=1, method=self.method, component=self.component, beta=self.beta,
            temperature=1., views='two native passes; independent dropout',
            ce='mean two views; valid target-token mean per microbatch',
            rest='weighted conditional symmetric KL; gradients through both views and masses',
            masks='labels != -100, including supervised special tokens',
            logit_chunk_tokens=32, extra_parameters=0, original_lsr_weight=0.,
            inference_route='native', strength=0.,
            attribution='R-Drop (NeurIPS 2021); KL split follows DKD (CVPR 2022); engineering ablation')

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder, 'decoupled_consistency_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8')

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(Path(folder, 'decoupled_consistency_config.json').read_text(encoding='utf-8'))
        if saved != self.method_metadata():
            raise ValueError('Decoupled consistency checkpoint recipe mismatch')
