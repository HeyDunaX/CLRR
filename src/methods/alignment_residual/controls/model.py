"""Matched CE controls; original v1 implementation and identifiers stay intact."""
import torch
from torch import nn
from ..v1 import AlignmentResidualMT

METHODS = {'alignment-local-ce-control-v1': 'local-capacity',
           'alignment-donor-ce-control-v1': 'donor-capacity'}


class ZeroCorrection(nn.Module):
    def forward(self, value):
        return torch.zeros_like(value)


class ResidualCEControl(AlignmentResidualMT):
    def __init__(self, base_model, method, rank=64, module_seed=10042,
                 distance=2, strength=.1, jepa_weight=0.):
        if method not in METHODS or jepa_weight != 0:
            raise ValueError('CE control requires a control identifier and zero auxiliary weight')
        local = METHODS[method] == 'local-capacity'
        super().__init__(base_model,
            'alignment-local-confined-v1' if local else 'alignment-residual-capacity-v1',
            rank, module_seed, distance, strength, jepa_weight)
        if not local:
            encoder = self.get_encoder()
            encoder.residual_modules = nn.ModuleDict({k: ZeroCorrection() for k in encoder.residual_modules})
        self.method_id = method

    def method_metadata(self):
        return dict(method=self.method_id, rank=self.rank, module_seed=self.module_seed,
            distance=self.distance, strength=self.strength, jepa_weight=0.,
            attention_implementation=self.config._attn_implementation,
            gradient_policy=METHODS[self.method_id], target_no_grad=True,
            source_rng_replay=True, capacity_target_dummy=True,
            extra_parameters=self.extra_parameter_count())

    def auxiliary_loss(self, *args, **kwargs):
        raise ValueError('CE controls do not define an auxiliary objective')
