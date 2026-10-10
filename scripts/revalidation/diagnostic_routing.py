"""Inference-only interventions; the training implementation stays unchanged."""
from amis_rewire.modeling import CrossLayerResidualRewire, _hidden_and_repack


class DiagnosticRouting(CrossLayerResidualRewire):
    def __init__(self, base_model, active_layers=range(3, 13), strength=.1):
        self.active_layers = frozenset(active_layers)
        super().__init__(base_model, distance=2, strength=strength, stack='encoder')

    def _make_hook(self, stack_name, index):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            source = self._cache[stack_name].get(index - self.distance)
            if index + 1 in self.active_layers and source is not None and source.shape == hidden.shape:
                hidden = hidden + self.strength * source.to(dtype=hidden.dtype)
            # Inactive blocks still contribute their outputs to downstream cache.
            self._cache[stack_name][index] = hidden.detach()
            return _hidden_and_repack(output, hidden)
        return hook
