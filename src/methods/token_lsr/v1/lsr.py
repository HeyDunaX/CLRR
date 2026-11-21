"""Detached balanced token transport, v1: no learned alignment parameters."""
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from ...original.lsr import JEPAGuidedSeq2SeqLM


def transport_plan(cost, source_mask, target_mask, epsilon=.10, iterations=200):
    if epsilon <= 0 or iterations < 1:
        raise ValueError('Positive epsilon and iteration count required')
    if not source_mask.any(1).all() or not target_mask.any(1).all():
        raise ValueError('Every sentence must contain at least one content token')
    with torch.no_grad(), torch.autocast(device_type=cost.device.type, enabled=False):
        a = source_mask.float() / source_mask.sum(1, keepdim=True)
        b = target_mask.float() / target_mask.sum(1, keepdim=True)
        kernel = torch.exp(-cost.detach().float() / epsilon)
        kernel = kernel * source_mask.unsqueeze(2) * target_mask.unsqueeze(1)
        v = target_mask.float()
        for _ in range(iterations):
            u = a / torch.bmm(kernel, v.unsqueeze(2)).squeeze(2).clamp_min(1e-30)
            v = b / torch.bmm(kernel.transpose(1, 2), u.unsqueeze(2)).squeeze(2).clamp_min(1e-30)
        plan = u.unsqueeze(2) * kernel * v.unsqueeze(1)
        error = torch.maximum((plan.sum(2) - a).abs().amax(1),
                              (plan.sum(1) - b).abs().amax(1))
    return plan.detach(), error


def token_alignment_loss(source, target, source_mask, target_mask):
    """Mean per-sentence transport-weighted cosine cost, not Sinkhorn divergence."""
    with torch.autocast(device_type=source.device.type, enabled=False):
        cost = 1 - torch.bmm(F.normalize(source.float(), dim=-1),
                            F.normalize(target.detach().float(), dim=-1).transpose(1, 2))
        plan, error = transport_plan(cost, source_mask, target_mask)
        if not torch.isfinite(plan).all() or error.max() >= 1e-3:
            raise RuntimeError('Token transport failed the finite/marginal correctness gate')
        return (plan * cost).sum((1, 2)).mean()


class TokenLSRSeq2SeqLM(JEPAGuidedSeq2SeqLM):
    """CE + .10 Token-LSR; target and matching plan have no gradient."""
    def __init__(self, base_model, special_token_ids):
        super().__init__(base_model, jepa_weight=.10)
        self.special_token_ids = tuple(sorted(set(special_token_ids)))

    def content_mask(self, ids, attention_mask):
        mask = attention_mask.bool().clone()
        for token_id in self.special_token_ids:
            mask &= ids != token_id
        return mask

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None, output_hidden_states=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
            labels=labels, output_hidden_states=output_hidden_states, **kwargs)
        if labels is None or outputs.loss is None:
            return outputs
        pad = self.config.pad_token_id
        source_attention = attention_mask if attention_mask is not None else (input_ids != pad)
        target_ids = labels.masked_fill(labels == -100, pad)
        target_attention = (target_ids != pad).long()
        context = self.base_model._fresh_cache() if hasattr(self.base_model, '_fresh_cache') else nullcontext()
        with context, torch.no_grad():
            target_hidden = self.get_encoder()(input_ids=target_ids,
                attention_mask=target_attention).last_hidden_state.detach()
        auxiliary = token_alignment_loss(outputs.encoder_last_hidden_state, target_hidden,
            self.content_mask(input_ids, source_attention), self.content_mask(target_ids, target_attention))
        outputs.loss = outputs.loss + self.jepa_weight * auxiliary.to(outputs.loss.dtype)
        return outputs
