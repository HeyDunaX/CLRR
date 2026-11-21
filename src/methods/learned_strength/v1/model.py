"""Bounded engineering pilot: zero-initialized learned CLRR edge scalars."""
import json
from pathlib import Path

import torch
from torch import nn

from ...common.helpers import _find_stack_layers, _hidden_and_repack
from ...original.clrr import CrossLayerResidualRewire


METHODS = {'learned-strength-clrr-v1': False, 'learned-strength-full-v1': True}
STATE_KEY = '_clrr_route_strengths'


class LearnedStrengthCLRR(CrossLayerResidualRewire):
    def __init__(self, base_model, distance=2):
        # Parent installs hooks without consuming random numbers.
        super().__init__(base_model, distance=distance, strength=0., stack='encoder')
        layers = _find_stack_layers(base_model, 'encoder')
        self.route_strengths = nn.Parameter(torch.zeros(len(layers) - distance))
        self._checkpoint_context = {}
        self._handles.append(self.get_encoder().register_forward_pre_hook(self._start_encoder))

    def _start_encoder(self, module, args):
        self._cache['encoder'] = {}

    def _make_hook(self, stack_name, index):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            context = self._checkpoint_context.get(index)
            source = context[0] if context is not None else self._cache[stack_name].get(index - self.distance)
            if index >= self.distance:
                if source is None or source.shape != hidden.shape:
                    raise RuntimeError('Missing or mismatched CLRR donor')
                hidden = hidden + self.route_strengths[index - self.distance] * source.detach().to(hidden.dtype)
            if context is None or not context[1]:
                self._cache[stack_name][index] = hidden.detach()
            return _hidden_and_repack(output, hidden)
        return hook

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.base_model.gradient_checkpointing_enable(*args, **kwargs)
        for index, layer in enumerate(_find_stack_layers(self.base_model, 'encoder')):
            layer._gradient_checkpointing_func = self._checkpoint_function(
                index, layer._gradient_checkpointing_func)

    def _checkpoint_function(self, index, checkpoint):
        def checkpoint_with_donor(function, *args, **kwargs):
            # Capture this invocation's detached donor, including when LSR runs
            # a second target encoder pass before CE backward. Never replay
            # against a mutable cache or a donor from another microbatch.
            donor = self._cache['encoder'].get(index - self.distance)
            calls = 0
            def replay(*values):
                nonlocal calls
                self._checkpoint_context[index] = (donor, calls > 0)
                calls += 1
                try:
                    return function(*values)
                finally:
                    self._checkpoint_context.pop(index, None)
            return checkpoint(replay, *args, **kwargs)
        return checkpoint_with_donor

    def extra_parameter_count(self):
        return self.route_strengths.numel()

    def num_parameters(self, only_trainable=False, exclude_embeddings=False, **kwargs):
        return self.base_model.num_parameters(only_trainable=only_trainable,
            exclude_embeddings=exclude_embeddings, **kwargs) + self.extra_parameter_count()

    def method_metadata(self):
        return dict(version=1, distance=self.distance, stack='encoder', initial_strength=0.,
            learned_strength=True, scalar_constraint='unconstrained signed',
            extra_parameters=self.extra_parameter_count(), donor='detached post-rewire',
            checkpoint_donor='captured per invocation; identical on recomputation')

    def state_dict(self, *args, **kwargs):
        state = self.base_model.state_dict(*args, **kwargs)
        state[STATE_KEY] = self.route_strengths if kwargs.get('keep_vars', False) else self.route_strengths.detach()
        return state

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        state = dict(state_dict)
        scalars = state.pop(STATE_KEY)
        if scalars.shape != self.route_strengths.shape:
            raise ValueError('CLRR scalar checkpoint shape mismatch')
        with torch.no_grad():
            self.route_strengths.copy_(scalars)
        return self.base_model.load_state_dict(state, strict=strict, **kwargs)

    def save_pretrained(self, folder, **kwargs):
        kwargs['state_dict'] = kwargs.get('state_dict') or self.state_dict()
        kwargs['safe_serialization'] = False
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder, 'learned_strength_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8')

    def restore_scalars(self, folder):
        saved = json.loads(Path(folder, 'learned_strength_config.json').read_text())
        if saved != self.method_metadata():
            raise ValueError('Learned strength checkpoint recipe mismatch')
        state = torch.load(Path(folder, 'pytorch_model.bin'), map_location='cpu', weights_only=True)
        self.load_state_dict(state)
