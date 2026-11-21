"""Frozen v1 definitions: NCE only, CLRR+NCE, and attached-donor P3."""
from .clrr import AttachedResidualRewire
from ...original.clrr import CrossLayerResidualRewire
from ...residual_effect.v1.lsr import ResidualEffectSeq2SeqLM


METHODS = frozenset({'nce-only-v1', 'clrr-nce-v1', 'attached-ordinary-full-v1'})


def wrap_control(base, spec):
    method = spec['method']
    if method not in METHODS:
        raise ValueError(f'Unknown contrastive control: {method}')
    if method != 'nce-only-v1':
        route = AttachedResidualRewire if method == 'attached-ordinary-full-v1' else CrossLayerResidualRewire
        base = route(base, distance=2, strength=.10, stack='encoder')
    special = [getattr(base.config, key, None) for key in ('pad_token_id', 'bos_token_id', 'eos_token_id')]
    return ResidualEffectSeq2SeqLM(base, mode='ordinary',
        jepa_weight=.10 if method == 'attached-ordinary-full-v1' else 0.,
        alignment_weight=spec['alignment_weight'], temperature=spec['temperature'],
        special_token_ids=[s for s in special if s is not None])
