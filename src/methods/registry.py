"""Stable method IDs; legacy candidate IDs retain their frozen definitions."""
from .original import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM
from .precache.v1.clrr import PrecacheResidualRewire
from .centered_lsr.v1.lsr import CenteredLSRSeq2SeqLM

ALIASES = {
    'clrr-precache': 'precache-clrr-v1',
    'full-precache': 'precache-full-v1',
    'lsr-centered': 'centered-lsr-v1',
    'full-centered': 'centered-full-v1',
}
CANDIDATES = frozenset(ALIASES.values())
ORIGINAL_ALIASES = {'clrr': 'original-clrr', 'clrr-encoder': 'original-clrr',
                    'lsr': 'original-lsr', 'full': 'original-full'}


def wrap_method(base, spec):
    """Wrap loaded backbone; candidates retain distance=2 and weight=.10."""
    method = ORIGINAL_ALIASES.get(spec['method'], spec['method'])
    method = ALIASES.get(method, method)
    if method == 'baseline':
        return base
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
