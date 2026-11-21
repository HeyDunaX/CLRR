"""Attributed CLRR engineering experiment; symmetric KL follows R-Drop."""
from contextlib import contextmanager
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import Seq2SeqLMOutput

from ...common.helpers import _find_stack_layers, _hidden_and_repack
from ...original.clrr import CrossLayerResidualRewire


# (second view rewired, symmetric KL coefficient). First view is always OFF.
METHODS = {'paired-native-ce-v1': (False, 0.),
           'paired-native-rdrop-v1': (False, 2.5),
           'paired-clrr-ce-v1': (True, 0.),
           'paired-clrr-rdrop-v1': (True, 2.5)}


def token_sums(left, right, labels, consistency):
    """Exact full-vocabulary objective; both KL arguments remain differentiable."""
    a, b = F.log_softmax(left.float(), -1), F.log_softmax(right.float(), -1)
    mask = labels.ne(-100)
    safe = labels.masked_fill(~mask, 0).unsqueeze(-1)
    ce = -.5 * (a.gather(-1, safe).squeeze(-1) + b.gather(-1, safe).squeeze(-1))
    kl = .5 * ((a.exp() - b.exp()) * (a - b)).sum(-1) if consistency else ce * 0.
    return (ce * mask).sum(), (kl * mask).sum()


def paired_loss(left, right, labels, beta, chunk_size=32):
    if left.shape != right.shape or left.shape[:-1] != labels.shape:
        raise ValueError('Expected paired [batch,time,vocab] logits and aligned labels')
    count = labels.ne(-100).sum()
    if count == 0:
        raise ValueError('At least one supervised token required')
    left, right, labels = left.flatten(0, 1), right.flatten(0, 1), labels.flatten()
    ce, kl = left.new_zeros(()), left.new_zeros(())
    for offset in range(0, labels.numel(), chunk_size):
        args = (left[offset:offset+chunk_size], right[offset:offset+chunk_size],
                labels[offset:offset+chunk_size])
        def reduced(a, b, y):
            return token_sums(a, b, y, beta != 0.)
        # Recompute log-softmax chunks rather than retaining full-vocab auxiliary
        # intermediates. Backbone per-layer reentrant checkpointing is unchanged.
        if torch.is_grad_enabled() and (left.requires_grad or right.requires_grad):
            c, k = checkpoint(reduced, *args, use_reentrant=True, preserve_rng_state=True)
        else:
            c, k = reduced(*args)
        ce, kl = ce + c, kl + k
    ce, kl = ce / count, kl / count
    return ce + beta * kl, ce, kl


class PairedRouteMT(CrossLayerResidualRewire):
    def __init__(self, base_model, method):
        self.method = method
        self.rewired, self.beta = METHODS[method]
        super().__init__(base_model, distance=2, strength=.1 if self.rewired else 0., stack='encoder')
        self._checkpoint_context = {}
        self.last_diagnostics = {}

    def _make_hook(self, stack_name, index):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            context = self._checkpoint_context.get(index)
            donor, strength, replay = context if context is not None else (
                self._cache[stack_name].get(index - self.distance), self.strength, False)
            if strength and index >= self.distance:
                if donor is None or donor.shape != hidden.shape:
                    raise RuntimeError('Missing paired-route donor')
                hidden = hidden + strength * donor.detach().to(hidden.dtype)
            if not replay:
                self._cache[stack_name][index] = hidden.detach()
            return _hidden_and_repack(output, hidden)
        return hook

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.base_model.gradient_checkpointing_enable(*args, **kwargs)
        for index, layer in enumerate(_find_stack_layers(self.base_model, 'encoder')):
            layer._gradient_checkpointing_func = self._checkpoint_function(
                index, layer._gradient_checkpointing_func)

    def _checkpoint_function(self, index, checkpoint_function):
        def captured(function, *args, **kwargs):
            donor = self._cache['encoder'].get(index - self.distance)
            strength, calls = self.strength, 0
            def replay(*values):
                nonlocal calls
                self._checkpoint_context[index] = (donor, strength, calls > 0)
                calls += 1
                try:
                    return function(*values)
                finally:
                    self._checkpoint_context.pop(index, None)
            return checkpoint_function(replay, *args, **kwargs)
        return captured

    @contextmanager
    def _route(self, enabled):
        previous = self.strength
        try:
            self.strength = .1 if enabled else 0.
            with self._fresh_cache():
                yield
        finally:
            self.strength = previous

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                decoder_input_ids=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        if not self.training or labels is None:
            with self._route(self.rewired):
                return self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids, labels=labels, **kwargs)
        decoder_input_ids = (self.prepare_decoder_input_ids_from_labels(labels)
                             if decoder_input_ids is None else decoder_input_ids)
        logits = []
        for enabled in (False, self.rewired):
            with self._route(enabled):
                logits.append(self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids, use_cache=False, return_dict=True).logits)
        loss, ce, kl = paired_loss(*logits, labels, self.beta)
        self.last_diagnostics = dict(ce=float(ce.detach()), kl=float(kl.detach()),
            beta=self.beta, views=2, second_view_rewired=self.rewired,
            checkpoint_context_empty=not self._checkpoint_context)
        return Seq2SeqLMOutput(loss=loss)

    def method_metadata(self):
        return dict(version=1, method=self.method, beta=self.beta, temperature=1.,
            first_view='OFF', second_view='ON' if self.rewired else 'OFF',
            inference_route='ON' if self.rewired else 'OFF',
            ce='mean two views; baseline valid target-token mean per microbatch',
            kl='full-vocabulary symmetric; both arguments differentiable',
            independent_dropout=True, logit_chunk_tokens=32,
            backbone_checkpointing='per-layer reentrant; per-invocation donor/route capture',
            distance=2, strength=.1 if self.rewired else 0.,
            donor='detached post-rewire', extra_parameters=0, original_lsr_weight=0.,
            attribution='R-Drop (NeurIPS 2021); engineering CLRR refinement, not novelty certification')

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder, 'paired_route_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8')

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(Path(folder, 'paired_route_config.json').read_text(encoding='utf-8'))
        if saved != self.method_metadata():
            raise ValueError('Paired route checkpoint recipe mismatch')
