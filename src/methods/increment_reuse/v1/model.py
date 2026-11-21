"""One-edge increment read/write ablations of pretrained mBART.

RO is locally expressible with Hyper-Connections read/write coefficients.
This module tests an empirical hypothesis; it does not establish novelty.
Donors are differentiable, including with reentrant checkpointing.
"""
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn
from transformers.modeling_outputs import Seq2SeqLMOutput

from ...decoupled_consistency.v1.model import consistency_loss


class IncrementReuseMT(nn.Module):
    def __init__(self, base_model, variant, donor_layer, receiver_layer, special_token_ids):
        super().__init__()
        if base_model.config.model_type != 'mbart':
            raise ValueError('Only the audited pre-LN mBART implementation is supported')
        layers = base_model.get_encoder().layers
        if getattr(base_model.get_encoder(), '_increment_reuse_installed', False):
            raise ValueError('Construct each variant from an unwrapped backbone, not a copied wrapper')
        if any(layer.gradient_checkpointing for layer in layers):
            raise ValueError('Enable checkpointing through this wrapper after construction')
        if variant not in ('ro', 'dw', 'rw'):
            raise ValueError('Expected ro, dw or rw')
        if not 0 <= donor_layer < receiver_layer < len(layers):
            raise ValueError('Require zero-based donor < receiver < encoder depth')
        if base_model.config.encoder_layerdrop:
            raise ValueError('LayerDrop would make the fixed donor unavailable')
        if not special_token_ids:
            raise ValueError('Supply tokenizer.all_special_ids, including the source proxy')
        self.base_model = base_model
        self.variant, self.donor_layer, self.receiver_layer = variant, donor_layer, receiver_layer
        self.special_token_ids = tuple(sorted(set(map(int, special_token_ids))))
        self.alpha = nn.Parameter(next(base_model.parameters()).new_zeros(()))
        self._donor = self._mask = None
        self._inside_checkpoint = False
        self._checkpoint_installed = False
        self.last_diagnostics = {}
        encoder = base_model.get_encoder()
        encoder._increment_reuse_installed = True
        original_encoder = encoder.forward

        def encoder_forward(input_ids=None, attention_mask=None, **kwargs):
            if input_ids is None:
                raise ValueError('Input IDs required to exclude special tokens from donor reuse')
            if self._mask is not None:
                raise RuntimeError('Nested encoder invocation is unsupported')
            mask = input_ids.ne(self.special_token_ids[0])
            for token_id in self.special_token_ids[1:]:
                mask = mask & input_ids.ne(token_id)
            if attention_mask is not None:
                mask = mask & attention_mask.bool()
            self._mask = mask.unsqueeze(-1)
            try:
                return original_encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            finally:
                self._donor = self._mask = None

        encoder.forward = encoder_forward
        for index in (donor_layer, receiver_layer):
            layer = layers[index]
            layer.forward = self._layer_forward(index, layer, layer.forward)

    @property
    def config(self):
        return self.base_model.config

    @property
    def generation_config(self):
        return self.base_model.generation_config

    def get_encoder(self):
        return self.base_model.get_encoder()

    def prepare_decoder_input_ids_from_labels(self, labels):
        return self.base_model.prepare_decoder_input_ids_from_labels(labels)

    @contextmanager
    def _receiver(self, layer, donor, mask):
        if donor is None or mask is None:
            raise RuntimeError('Receiver invoked without its donor and source mask')
        increment = self.alpha * donor * mask.to(donor.dtype)
        handles = []
        if self.variant in ('ro', 'rw'):
            handles.append(layer.final_layer_norm.register_forward_pre_hook(
                lambda module, args: (args[0] + increment,)))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _write(self, output, donor, mask):
        if self.variant in ('dw', 'rw'):
            return (output[0] + self.alpha * donor * mask.to(donor.dtype), *output[1:])
        return output

    def _layer_forward(self, index, layer, original):
        def forward(hidden_states, *args, **kwargs):
            if self._inside_checkpoint:
                return original(hidden_states, *args, **kwargs)
            if index == self.receiver_layer:
                with self._receiver(layer, self._donor, self._mask):
                    output = original(hidden_states, *args, **kwargs)
                    return self._write(output, self._donor, self._mask)
            output = original(hidden_states, *args, **kwargs)
            self._donor = output[0] - hidden_states
            return output
        return forward

    def gradient_checkpointing_enable(self, **kwargs):
        if self._checkpoint_installed:
            raise RuntimeError('Checkpointing already installed')
        self.base_model.gradient_checkpointing_enable(**kwargs)
        for index, layer in enumerate(self.get_encoder().layers):
            layer._gradient_checkpointing_func = self._checkpoint_function(
                index, layer, layer._gradient_checkpointing_func)
        self._checkpoint_installed = True

    def _checkpoint_function(self, index, layer, checkpoint_function):
        def captured(function, *args, **kwargs):
            mask = self._mask
            donor = self._donor if index == self.receiver_layer else None

            def replay(*values):
                previous = self._inside_checkpoint
                self._inside_checkpoint = True
                try:
                    if index == self.receiver_layer:
                        # Explicit checkpoint input preserves the donor's backward
                        # edge. Capturing a tensor in a closure is insufficient.
                        with self._receiver(layer, values[-1], mask):
                            output = function(*values[:-1])
                            return self._write(output, values[-1], mask)
                    return function(*values)
                finally:
                    self._inside_checkpoint = previous

            inputs = (*args, donor) if index == self.receiver_layer else args
            output = checkpoint_function(replay, *inputs, **kwargs)
            if index == self.donor_layer:
                self._donor = output[0] - args[0]
            return output
        return captured

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                decoder_input_ids=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        if not self.training or labels is None:
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                labels=labels, decoder_input_ids=decoder_input_ids, **kwargs)
        decoder_input_ids = (self.prepare_decoder_input_ids_from_labels(labels)
                             if decoder_input_ids is None else decoder_input_ids)
        logits = [self.base_model(input_ids=input_ids, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids, use_cache=False, return_dict=True).logits
            for _ in range(2)]
        loss, ce, gold, rest = consistency_loss(*logits, labels, 'full', 1.25)
        self.last_diagnostics = dict(ce=float(ce.detach()), kl=float((gold + rest).detach()),
            alpha=float(self.alpha.detach()), beta=1.25, variant=self.variant,
            cache_empty=self._donor is None and self._mask is None)
        return Seq2SeqLMOutput(loss=loss)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def method_metadata(self):
        return dict(version=1, variant=self.variant, donor_layer=self.donor_layer,
            receiver_layer=self.receiver_layer, special_token_ids=list(self.special_token_ids),
            donor='differentiable layer output minus input', alpha_initial=0.,
            beta=1.25, views=2, extra_parameters=1,
            scope='engineering ablation; not standalone architectural novelty')

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        metadata = {**self.method_metadata(), 'alpha': float(self.alpha.detach().cpu())}
        Path(folder, 'increment_reuse_config.json').write_text(
            json.dumps(metadata, indent=2), encoding='utf-8')

    @classmethod
    def from_pretrained(cls, folder, **kwargs):
        from transformers import MBartForConditionalGeneration
        metadata = json.loads(Path(folder, 'increment_reuse_config.json').read_text(encoding='utf-8'))
        base = MBartForConditionalGeneration.from_pretrained(folder, **kwargs)
        model = cls(base, metadata['variant'], metadata['donor_layer'],
                    metadata['receiver_layer'], metadata['special_token_ids'])
        expected = {key: value for key, value in metadata.items() if key != 'alpha'}
        if expected != model.method_metadata():
            raise ValueError('Increment reuse metadata mismatch')
        with torch.no_grad():
            model.alpha.fill_(metadata['alpha'])
        return model
