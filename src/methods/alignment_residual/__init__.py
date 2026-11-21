"""Versioned experimental CLRR/LSR residual modules."""
from .v1 import AlignmentResidualMT, METHODS as PILOT_METHODS
from .controls import ResidualCEControl, METHODS as CONTROL_METHODS
from .learned_donor import LearnedDonorCE, METHODS as LEARNED_DONOR_METHODS
from .matched_baseline import MatchedBaselineCE, METHODS as MATCHED_BASELINE_METHODS

METHODS = {**PILOT_METHODS, **CONTROL_METHODS, **LEARNED_DONOR_METHODS, **MATCHED_BASELINE_METHODS}


def model_class(method):
    if method in MATCHED_BASELINE_METHODS:
        return MatchedBaselineCE
    if method in LEARNED_DONOR_METHODS:
        return LearnedDonorCE
    return ResidualCEControl if method in CONTROL_METHODS else AlignmentResidualMT
