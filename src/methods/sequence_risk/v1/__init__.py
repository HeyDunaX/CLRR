"""Known sequence-risk control; not registered as a trainable new method."""

from .loss import candidate_set_risk
from .model import SequenceRiskSeq2SeqLM

METHODS = frozenset({'sequence-ce-v1', 'sequence-risk-v1',
    'sequence-clrr-ce-v1', 'sequence-clrr-risk-v1'})

__all__ = ["candidate_set_risk", "SequenceRiskSeq2SeqLM", "METHODS"]
