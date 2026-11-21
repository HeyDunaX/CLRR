"""Native CE function retaining the experimental family's target dummy pass."""
from ..controls import ResidualCEControl

METHODS = {'alignment-matched-ce-control-v1': 'matched-baseline-capacity'}


class MatchedBaselineCE(ResidualCEControl):
    def __init__(self, base_model, method, rank=64, module_seed=10042,
                 distance=2, strength=.1, jepa_weight=0.):
        if method not in METHODS or jepa_weight != 0:
            raise ValueError('Matched baseline requires its identifier and zero auxiliary weight')
        super().__init__(base_model, 'alignment-donor-ce-control-v1', rank,
                         module_seed, distance, strength, jepa_weight)
        self.get_encoder().residual_q = 0
        self.method_id = method

    def method_metadata(self):
        return dict(method=self.method_id, rank=self.rank, module_seed=self.module_seed,
            distance=self.distance, strength=self.strength, jepa_weight=0.,
            attention_implementation=self.config._attn_implementation,
            gradient_policy=METHODS[self.method_id], target_no_grad=True,
            source_rng_replay=True, capacity_target_dummy=True,
            extra_parameters=self.extra_parameter_count())
