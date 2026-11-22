"""Stable method IDs; legacy candidate IDs retain their frozen definitions."""
from .original import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM
from .precache.v1.clrr import PrecacheResidualRewire
from .centered_lsr.v1.lsr import CenteredLSRSeq2SeqLM
from .token_lsr.v1.lsr import TokenLSRSeq2SeqLM
from .residual_effect.v1.lsr import ResidualEffectSeq2SeqLM
from .contrastive_controls.v1 import METHODS as CONTRASTIVE_CONTROL_METHODS, wrap_control
from .grounded_interaction.v1 import METHODS as INTERACTION_METHODS, GroundedInteractionSeq2SeqLM
from .sequence_risk.v1 import METHODS as SEQUENCE_METHODS, SequenceRiskSeq2SeqLM
from .depth_difference.v2.model import METHODS as DEPTH_METHODS, ValueScaledDepthDifference
from .alignment_residual import METHODS as ALIGNMENT_RESIDUAL_METHODS, model_class as residual_model_class
from .grounded_interaction.v1 import METHODS as INTERACTION_METHODS, GroundedInteractionSeq2SeqLM

from .decoder_residual_bridge.v1 import DecoderResidualBridge, METHODS as DRB_METHODS
from .difference_preserving.v1 import DifferencePreservingSeq2SeqLM, METHODS as DIFFERENCE_METHODS
from .difference_preserving.v2.model import RelationalSeq2SeqLM, METHODS as RELATIONAL_METHODS
from .transition_lsr import TransitionLSRSeq2SeqLM, METHODS as TRANSITION_METHODS
from .learned_strength import LearnedStrengthCLRR, METHODS as STRENGTH_METHODS
from .paired_route import PairedRouteMT, METHODS as PAIRED_ROUTE_METHODS
from .decoupled_consistency import DecoupledConsistencyMT, METHODS as CONSISTENCY_METHODS
from .prefix_consistency import PrefixConsistencyMT, METHODS as PREFIX_METHODS
from .continuation_consistency import model_class as cmc_model_class, METHODS as CMC_METHODS
from .corrective_order import CorrectiveOrderMT, METHODS as ORDER_METHODS
from .virtual_corrective_order.v1.model import VirtualCorrectiveOrderMT, METHODS as VIRTUAL_ORDER_METHODS

from .antisymmetric_consistency import AntisymmetricConsistencyMT, METHODS as AC_METHODS
from .decision_residual_memory import DecisionResidualMemoryMT, METHODS as DRM_METHODS
from .gold_ordered_consistency import GoldOrderedConsistencyMT, METHODS as GOC_METHODS
from .cross_ratio_dual import CrossRatioDualMT, METHODS as CR_DUAL_METHODS
TWO_METHODS = AC_METHODS.keys() | DRM_METHODS.keys()

ALIASES = {
    'clrr-precache': 'precache-clrr-v1',
    'full-precache': 'precache-full-v1',
    'lsr-centered': 'centered-lsr-v1',
    'full-centered': 'centered-full-v1',
}
CANDIDATES = frozenset(ALIASES.values())
ORIGINAL_ALIASES = {'clrr': 'original-clrr', 'clrr-encoder': 'original-clrr',
                    'lsr': 'original-lsr', 'full': 'original-full'}
TOKEN_METHODS = frozenset({'token-lsr-v1', 'token-full-v1'})
EFFECT_METHODS = frozenset({'ordinary-lsr-v1', 'ordinary-full-v1',
                          'effect-clrr-v1', 'effect-full-v1'})
CONTRASTIVE_METHODS = EFFECT_METHODS | CONTRASTIVE_CONTROL_METHODS


def wrap_method(base, spec):
    """Wrap loaded backbone; candidates retain distance=2 and weight=.10."""
    method = ORIGINAL_ALIASES.get(spec['method'], spec['method'])
    method = ALIASES.get(method, method)
    if method == 'baseline':
        return base
    if method in VIRTUAL_ORDER_METHODS:
        return VirtualCorrectiveOrderMT(base, rank=spec['virtual_rank'], basis_seed=spec['basis_seed'],
            eta=spec['virtual_eta'], coefficient=spec['order_coefficient'], chunk=spec['order_chunk'])
    if method in ORDER_METHODS:
        return CorrectiveOrderMT(base, method, eta=spec['virtual_eta'],
            coefficient=spec.get('order_coefficient', 1.),
            multiplier=spec['order_multiplier'], beta=spec.get('consistency_beta', 1.25),
            offload=spec.get('order_offload', 'cpu'))
    if method in AC_METHODS:
        return AntisymmetricConsistencyMT(base, beta=spec['consistency_beta'])
    if method in DRM_METHODS:
        return DecisionResidualMemoryMT(base, eta=spec['memory_eta'], beta=spec['consistency_beta'])
    if method in GOC_METHODS:
        return GoldOrderedConsistencyMT(base, method=method, beta=spec['consistency_beta'])
    if method in CR_DUAL_METHODS:
        return CrossRatioDualMT(
            base,
            method=method,
            alpha_rev=spec.get('alpha_rev', 0.5),
            cr_lambda=spec.get('cr_lambda', 0.25),
            cr_delta=spec.get('cr_delta', 1.0),
            cr_warmup_fraction=spec.get('cr_warmup_fraction', 0.2),
            contrast_margin=spec.get('contrast_margin', 2.0),
            paired_rng=spec.get('paired_rng', True),
            logit_chunk_tokens=spec.get('logit_chunk_tokens', 64),
        )
    if method in CMC_METHODS:
        return cmc_model_class(method)(base, method, top_k=spec['continuation_top_k'],
            beta=spec['consistency_beta'], probe_chunk=spec['probe_chunk'])
    if method in PREFIX_METHODS:
        return PrefixConsistencyMT(base, method, weight=spec['path_weight'], beta=spec['consistency_beta'])
    if method in CONSISTENCY_METHODS:
        return DecoupledConsistencyMT(base, method)
    if method in PAIRED_ROUTE_METHODS:
        return PairedRouteMT(base, method)
    if method in STRENGTH_METHODS:
        base = LearnedStrengthCLRR(base, distance=spec['distance'])
        if STRENGTH_METHODS[method]:
            base = JEPAGuidedSeq2SeqLM(base, jepa_weight=spec['jepa_weight'])
        return base
    if method in TRANSITION_METHODS:
        if TRANSITION_METHODS[method]:
            base = CrossLayerResidualRewire(base, distance=2, strength=.10, stack='encoder')
        return TransitionLSRSeq2SeqLM(base, weight=spec['alignment_weight'], distance=2)
    if method in RELATIONAL_METHODS:
        objective, rewired = RELATIONAL_METHODS[method]
        if rewired:
            base = CrossLayerResidualRewire(base, distance=2, strength=.10, stack='encoder')
        return RelationalSeq2SeqLM(base, objective=objective, weight=spec['alignment_weight'])
    if method in DIFFERENCE_METHODS:
        if DIFFERENCE_METHODS[method][1]:
            base = CrossLayerResidualRewire(base, distance=2, strength=.10, stack='encoder')
        return DifferencePreservingSeq2SeqLM(base, weight=spec['alignment_weight'])
    if method in DRB_METHODS:
        return DecoderResidualBridge(base, method, donor_layer=spec['donor_layer'],
            decoder_layer=spec['decoder_layer'], rank=spec['adapter_rank'], module_seed=spec['module_seed'])
    if method in ALIGNMENT_RESIDUAL_METHODS:
        return residual_model_class(method)(base, method, rank=spec['adapter_rank'],
            module_seed=spec['module_seed'], distance=spec['distance'],
            strength=spec['strength'], jepa_weight=spec['jepa_weight'])
    if method in DEPTH_METHODS:
        return ValueScaledDepthDifference(base, donor_layer=spec['donor_layer'],
            decoder_layer=spec['decoder_layer'], mode=DEPTH_METHODS[method])
    if method in SEQUENCE_METHODS:
        return SequenceRiskSeq2SeqLM(base, rewired='clrr' in method,
            weight=spec['risk_weight'], alpha=spec['risk_alpha'], score_chunk=spec['score_chunk'],
            distance=spec.get('distance', 2), strength=spec.get('strength', .1))
    if method in INTERACTION_METHODS:
        return GroundedInteractionSeq2SeqLM(base,
            mode={'route-paired-ce-v1': 'paired-ce', 'route-source-contrast-v1': 'source',
                  'route-source-interaction-v1': 'interaction'}[method],
            weight=spec.get('interaction_weight', .1), temperature=spec.get('temperature', 1.),
            distance=spec.get('distance', 2), strength=spec.get('strength', .1))
    if method in INTERACTION_METHODS:
        return GroundedInteractionSeq2SeqLM(base,
            mode={'route-paired-ce-v1': 'paired-ce', 'route-source-contrast-v1': 'source',
                  'route-source-interaction-v1': 'interaction'}[method],
            weight=spec.get('interaction_weight', .1), temperature=spec.get('temperature', 1.),
            distance=spec.get('distance', 2), strength=spec.get('strength', .1))
    if method in CONTRASTIVE_CONTROL_METHODS:
        return wrap_control(base, spec)
    if method in EFFECT_METHODS:
        if method != 'ordinary-lsr-v1':
            base = CrossLayerResidualRewire(base, distance=2, strength=.10, stack='encoder')
        special = [getattr(base.config, key, None) for key in ('pad_token_id', 'bos_token_id', 'eos_token_id')]
        return ResidualEffectSeq2SeqLM(base,
            mode='effect' if method.startswith('effect-') else 'ordinary',
            jepa_weight=0.0 if method == 'effect-clrr-v1' else .10,
            alignment_weight=spec['alignment_weight'], temperature=spec['temperature'],
            special_token_ids=[s for s in special if s is not None])
    if method in TOKEN_METHODS:
        if method == 'token-full-v1':
            base = CrossLayerResidualRewire(base, distance=2, strength=.10, stack='encoder')
        special = [getattr(base.config, key, None) for key in ('pad_token_id', 'bos_token_id', 'eos_token_id')]
        return TokenLSRSeq2SeqLM(base, special_token_ids=[s for s in special if s is not None])
    if method in ('original-clrr', 'original-lsr', 'original-full'):
        if method in ('original-clrr', 'original-full'):
            base = CrossLayerResidualRewire(base, distance=spec.get('distance', 2),
                strength=spec.get('strength', .10), stack=spec.get('stack', 'encoder'))
        if method in ('original-lsr', 'original-full'):
            base = JEPAGuidedSeq2SeqLM(base, jepa_weight=spec.get('jepa_weight', .10))
        return base
    if method not in CANDIDATES:
        raise ValueError(f"Unknown method: {spec['method']}")
    if method in ('precache-clrr-v1', 'precache-full-v1'):
        base = PrecacheResidualRewire(base, distance=2, strength=.10, stack='encoder')
    elif method == 'centered-full-v1':
        base = CrossLayerResidualRewire(base, distance=2, strength=.10, stack='encoder')
    if method == 'precache-full-v1':
        base = JEPAGuidedSeq2SeqLM(base, jepa_weight=.10)
    elif method in ('centered-lsr-v1', 'centered-full-v1'):
        base = CenteredLSRSeq2SeqLM(base, jepa_weight=.10)
    return base


def wrap_candidate(base, spec):
    """Compatibility entry point for the four implemented candidates."""
    if ALIASES.get(spec['method'], spec['method']) not in CANDIDATES:
        raise ValueError(f"Unknown candidate: {spec['method']}")
    return wrap_method(base, spec)
