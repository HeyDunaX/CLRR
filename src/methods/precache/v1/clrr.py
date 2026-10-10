"""Pre-addition detached donor cache; historical U1 definition."""
import torch
from ...original.clrr import CrossLayerResidualRewire
from ...common.helpers import _hidden_and_repack

class PrecacheResidualRewire(CrossLayerResidualRewire):
    def _make_hook(self, stack_name, index):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden, torch.Tensor):
                return output
            donor = hidden.detach()
            source = self._cache[stack_name].get(index - self.distance)
            if source is not None and source.shape == hidden.shape:
                hidden = hidden + self.strength * source.to(dtype=hidden.dtype)
            self._cache[stack_name][index] = donor
            return _hidden_and_repack(output, hidden)
        return hook
