"""Experimental bilingual CLRR response alignment; not a certified novelty claim."""
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F

from ...original.clrr import CrossLayerResidualRewire
from ...original.lsr import JEPAGuidedSeq2SeqLM
from ...common.helpers import _masked_mean_pool


def paired_contrastive_loss(source, target, temperature):
    """One-way in-microbatch InfoNCE; shared encoder target is stop-gradient."""
    if temperature <= 0 or source.shape != target.shape or source.shape[0] < 2:
        raise ValueError('Need positive temperature and >=2 paired vectors of equal shape')
    logits = F.normalize(source.float(), dim=-1) @ F.normalize(target.detach().float(), dim=-1).T
    return F.cross_entropy(logits / temperature, torch.arange(source.shape[0], device=source.device))


class ResidualEffectSeq2SeqLM(JEPAGuidedSeq2SeqLM):
    """CE + optional original LSR + paired ordinary/effect alignment.

    Auxiliary views run before CE with encoder dropout/checkpointing disabled.
    Source ON and OFF are differentiable; both target views are detached.
    No queue: accumulation does not enlarge the contrastive negative pool.
    """
    def __init__(self, base_model, *, mode, jepa_weight, alignment_weight,
                 temperature, special_token_ids):
        super().__init__(base_model, jepa_weight=jepa_weight)
        if mode not in ('ordinary', 'effect') or alignment_weight <= 0 or temperature <= 0:
            raise ValueError('Invalid alignment mode, weight or temperature')
        if mode == 'effect' and not isinstance(base_model, CrossLayerResidualRewire):
            raise ValueError('Effect alignment requires encoder CLRR')
        self.mode = mode
        self.alignment_weight = alignment_weight
        self.temperature = temperature
        self.special_token_ids = tuple(special_token_ids)

    def content_mask(self, ids, attention):
        mask = attention.bool().clone()
        for token in self.special_token_ids:
            mask &= ids != token
        if not mask.any(1).all():
            raise ValueError('Every auxiliary sentence must contain content tokens')
        return mask

    @contextmanager
    def deterministic_view(self, rewired):
        encoder = self.get_encoder()
        flags = [(module, module.training) for module in encoder.modules()]
        rewire = self.base_model if isinstance(self.base_model, CrossLayerResidualRewire) else None
        strength = rewire.strength if rewire is not None else None
        cache = rewire._fresh_cache() if rewire is not None else nullcontext()
        try:
            encoder.eval()
            if rewire is not None and not rewired:
                rewire.strength = 0.0
            with cache:
                yield
        finally:
            if rewire is not None:
                rewire.strength = strength
            for module, flag in flags:
                module.training = flag

    def representation(self, ids, attention, rewired):
        mask = self.content_mask(ids, attention)
        with self.deterministic_view(rewired):
            hidden = self.get_encoder()(input_ids=ids, attention_mask=attention).last_hidden_state
        return _masked_mean_pool(hidden, mask)

    def auxiliary_loss(self, input_ids, attention_mask, labels):
        # A final evaluation batch can contain one sentence and no negatives.
        # Its contrastive CE is zero; do not reject otherwise valid inference.
        if not self.training and input_ids.shape[0] == 1:
            return self.get_input_embeddings().weight.new_zeros(())
        pad = self.config.pad_token_id
        attention = attention_mask if attention_mask is not None else (input_ids != pad).long()
        target_ids = labels.masked_fill(labels == -100, pad)
        target_attention = (target_ids != pad).long()
        source = self.representation(input_ids, attention, True)
        if self.mode == 'effect':
            source = source - self.representation(input_ids, attention, False)
        with torch.no_grad():
            target = self.representation(target_ids, target_attention, True)
            if self.mode == 'effect':
                target = target - self.representation(target_ids, target_attention, False)
        return paired_contrastive_loss(source, target, self.temperature)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        auxiliary = None
        if labels is not None:
            auxiliary = self.auxiliary_loss(input_ids, attention_mask, labels)
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                  labels=labels, **kwargs)
        if auxiliary is not None:
            outputs.loss = outputs.loss + self.alignment_weight * auxiliary.to(outputs.loss.dtype)
        return outputs
