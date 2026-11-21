from .v1.model import ContinuationConsistencyMT, METHODS as V1_METHODS
from .v2.model import EstimatorConsistencyMT, METHODS as V2_METHODS

METHODS = {**V1_METHODS, **V2_METHODS}


def model_class(method):
    return EstimatorConsistencyMT if method in V2_METHODS else ContinuationConsistencyMT
