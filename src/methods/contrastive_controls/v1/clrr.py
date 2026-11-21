"""Source-attached identity route, with donor addition outside checkpoint recompute."""
import torch
from ...original.clrr import CrossLayerResidualRewire
from ...common.helpers import _hidden_and_repack, _find_stack_layers


class AttachedResidualRewire(CrossLayerResidualRewire):
    def __init__(self, *args, **kwargs):
        self._inside_checkpoint = False
        super().__init__(*args, **kwargs)

    def _route(self, output, stack_name, index):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(hidden, torch.Tensor):
            return output
        source = self._cache[stack_name].get(index - self.distance)
        if source is not None and source.shape == hidden.shape:
            hidden = hidden + self.strength * source.to(dtype=hidden.dtype)
        self._cache[stack_name][index] = hidden
        return _hidden_and_repack(output, hidden)

    def _make_hook(self, stack_name, index):
        def hook(_module, _inputs, output):
            return output if self._inside_checkpoint else self._route(output, stack_name, index)
        return hook

    def _checkpoint_function(self, checkpoint, stack_name, index):
        def run(block, *args, **kwargs):
            def raw_block(*inputs, **options):
                previous = self._inside_checkpoint
                self._inside_checkpoint = True
                try:
                    return block(*inputs, **options)
                finally:
                    self._inside_checkpoint = previous
            output = checkpoint(raw_block, *args, **kwargs)
            # This addition is in the autograd graph, including the donor input.
            # No mutable donor cache is read or written during backward recompute.
            return self._route(output, stack_name, index)
        return run

    def gradient_checkpointing_enable(self, *args, **kwargs):
        result = self.base_model.gradient_checkpointing_enable(*args, **kwargs)
        stacks = ('encoder', 'decoder') if self.stack == 'both' else (self.stack,)
        for stack_name in stacks:
            for index, layer in enumerate(_find_stack_layers(self.base_model, stack_name)):
                if not hasattr(layer, '_gradient_checkpointing_func'):
                    raise RuntimeError('Attached route requires checkpoint-aware Transformer layers')
                layer._gradient_checkpointing_func = self._checkpoint_function(
                    layer._gradient_checkpointing_func, stack_name, index)
        return result
