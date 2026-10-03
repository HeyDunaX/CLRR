"""Attached-source CLRR control with routing outside checkpointed block calls."""
from contextlib import contextmanager
import torch
from .modeling import CrossLayerResidualRewire, _find_stack_layers, _hidden_and_repack


class AttachedSourceResidualRewire(CrossLayerResidualRewire):
    """Keep CLRR forward unchanged while allowing gradients through cached sources.

    Reentrant checkpoint forwards run under no_grad. Routing must therefore be
    applied outside the checkpoint call to retain the added gradient path.
    """

    def __init__(self, base_model, distance=2, strength=.1, stack='encoder'):
        if stack != 'encoder':
            raise ValueError('This matched ablation currently supports encoder routing only.')
        self._suspended = False
        super().__init__(base_model, distance=distance, strength=strength, stack=stack)

    def _route(self, index, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(hidden, torch.Tensor):
            return output
        source = self._cache['encoder'].get(index-self.distance)
        if source is not None and source.shape == hidden.shape:
            hidden = hidden+self.strength*source.to(dtype=hidden.dtype)
        self._cache['encoder'][index] = hidden
        return _hidden_and_repack(output, hidden)

    def _make_hook(self, stack_name, index):
        def hook(_module, _inputs, output):
            return output if self._suspended else self._route(index, output)
        return hook

    @contextmanager
    def _without_routing_hooks(self):
        previous = self._suspended
        self._suspended = True
        try:
            yield
        finally:
            self._suspended = previous

    def gradient_checkpointing_enable(self, *args, **kwargs):
        result = self.base_model.gradient_checkpointing_enable(*args, **kwargs)
        def wrap_checkpoint(index, original_checkpoint):
            def checkpoint_with_attached_route(function, *inputs, **options):
                def original_block(*values, **block_options):
                    with self._without_routing_hooks():
                        return function(*values, **block_options)
                output = original_checkpoint(original_block, *inputs, **options)
                return self._route(index, output)
            return checkpoint_with_attached_route
        for index, layer in enumerate(_find_stack_layers(self.base_model, 'encoder')):
            if not hasattr(layer, '_gradient_checkpointing_func'):
                raise RuntimeError('Attached control requires the pinned Transformers layer checkpoint API.')
            layer._gradient_checkpointing_func = wrap_checkpoint(index, layer._gradient_checkpointing_func)
        return result
