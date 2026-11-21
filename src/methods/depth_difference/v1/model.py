"""U2 correctness prototype for eager mBART in Transformers 4.57.6.

Recipient/detached donor travel as one packed decoder-memory tensor. They are
never stored on a module. Only the selected cross-attention layer caches both
K/V pairs; ordinary layers cache recipient K/V exactly as before.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_model, save_model
from transformers import GenerationConfig, MBartConfig, MBartForConditionalGeneration
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.mbart.modeling_mbart import eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


@dataclass
class DepthEncoderOutput(BaseModelOutput):
    donor_hidden_state: torch.Tensor | None = None


class PackedCrossAttention(nn.Module):
    """Read explicit packed memory; add U2 at exactly one decoder layer."""

    def __init__(self, attention, selected=False):
        super().__init__()
        self.attention = attention
        self.selected = selected
        if selected:
            self.gates = nn.Parameter(attention.q_proj.weight.new_zeros(attention.num_heads))
        self.collect_diagnostics = False
        self.last_diagnostics = None

    def donor_value_input(self, recipient, donor, attention_mask):
        return donor

    def correction(self, top_probability, donor_probability, value_top, value_donor):
        return torch.matmul(donor_probability - top_probability, value_donor - value_top)

    def forward(self, hidden_states, key_value_states=None, past_key_values=None,
                attention_mask=None, layer_head_mask=None, output_attentions=False,
                cache_position=None, **kwargs):
        attn = self.attention
        if key_value_states is None or key_value_states.shape[-1] != 2 * attn.embed_dim:
            raise ValueError('U2 requires explicitly packed recipient/donor encoder memory')
        recipient, donor = key_value_states.split(attn.embed_dim, dim=-1)
        donor = donor.detach()
        if not self.selected:
            return attn(hidden_states=hidden_states, key_value_states=recipient,
                        past_key_values=past_key_values, attention_mask=attention_mask,
                        layer_head_mask=layer_head_mask, output_attentions=output_attentions,
                        cache_position=cache_position, **kwargs)

        bsz, tgt_len, _ = hidden_states.shape
        src_len = recipient.shape[1]
        def heads(tensor, length):
            return tensor.view(bsz, length, attn.num_heads, attn.head_dim).transpose(1, 2)
        query = heads(attn.q_proj(hidden_states), tgt_len)
        cache = None
        if past_key_values is not None:
            if not isinstance(past_key_values, EncoderDecoderCache):
                raise TypeError('U2 prototype supports EncoderDecoderCache only')
            cache = past_key_values.cross_attention_cache
            if not isinstance(cache, DynamicCache):
                raise TypeError('U2 prototype supports dynamic cross-attention cache only')
        if cache is not None and past_key_values.is_updated.get(attn.layer_idx, False):
            keys = cache.layers[attn.layer_idx].keys
            values = cache.layers[attn.layer_idx].values
        else:
            keys = torch.cat((heads(attn.k_proj(recipient), src_len),
                              heads(attn.k_proj(donor), src_len)), dim=-1)
            values = torch.cat((heads(attn.v_proj(recipient), src_len),
                                heads(attn.v_proj(self.donor_value_input(
                                    recipient, donor, attention_mask)), src_len)), dim=-1)
            if cache is not None:
                keys, values = cache.update(keys, values, attn.layer_idx)
                past_key_values.is_updated[attn.layer_idx] = True
        key_top, key_donor = keys.split(attn.head_dim, dim=-1)
        value_top, value_donor = values.split(attn.head_dim, dim=-1)
        interface = eager_attention_forward if attn.config._attn_implementation == 'eager' else ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]
        main, weights = interface(
            attn, query, key_top, value_top, attention_mask, scaling=attn.scaling,
            dropout=attn.dropout if self.training else 0., head_mask=layer_head_mask)
        def probabilities(key):
            logits = torch.matmul(query, key.transpose(2, 3)) * attn.scaling
            if attention_mask is not None:
                logits = logits + attention_mask
            return torch.softmax(logits, dim=-1)
        top_probability = probabilities(key_top)
        donor_probability = probabilities(key_donor)
        delta = self.correction(top_probability, donor_probability, value_top, value_donor)
        if layer_head_mask is not None:
            delta = delta * layer_head_mask.view(1, -1, 1, 1)
        correction = delta.transpose(1, 2)
        if self.collect_diagnostics:
            head_main = torch.linalg.vector_norm(main.detach().float(), dim=(0, 1, 3))
            head_delta = torch.linalg.vector_norm(correction.detach().float(), dim=(0, 1, 3))
            self.last_diagnostics = {
                'base_context_norm': float(main.detach().float().norm()),
                'correction_norm': float(correction.detach().float().norm()),
                'gate_abs_max': float(self.gates.detach().abs().max()),
                'recipient_hidden_norm': float(recipient.detach().float().norm()),
                'donor_hidden_norm': float(donor.detach().float().norm()),
                'top_value_norm': float(value_top.detach().float().norm()),
                'donor_value_norm': float(value_donor.detach().float().norm()),
                'attention_difference_l1_mean': float(
                    (donor_probability - top_probability).detach().float().abs().sum(-1).mean()),
                'head_correction_to_base_norm': (head_delta / head_main.clamp_min(1e-30)).tolist(),
            }
        combined = main + correction * self.gates.view(1, 1, -1, 1)
        return attn.out_proj(combined.reshape(bsz, tgt_len, attn.embed_dim)), weights


class DepthDifferencePrototype(nn.Module):
    """CE-only U2; explicit custom save/load, no training-registry entry."""

    accepts_loss_kwargs = False
    attention_class = PackedCrossAttention
    method_id = 'depth-difference-prototype-v1'
    supported_backends = ('eager',)

    def __init__(self, base_model, donor_layer=6, decoder_layer=None):
        super().__init__()
        if not isinstance(base_model, MBartForConditionalGeneration):
            raise TypeError('U2 v1 prototype supports mBART only')
        if base_model.config._attn_implementation not in self.supported_backends:
            raise ValueError(f'Unsupported attention backend; expected {self.supported_backends}')
        if base_model.config.encoder_layerdrop or base_model.config.decoder_layerdrop:
            raise ValueError('U2 prototype requires encoder/decoder LayerDrop = 0')
        decoder_layer = decoder_layer or base_model.config.decoder_layers
        if not 1 <= donor_layer < base_model.config.encoder_layers:
            raise ValueError('Donor must be an intermediate encoder layer, counted from 1')
        if not 1 <= decoder_layer <= base_model.config.decoder_layers:
            raise ValueError('Invalid decoder layer, counted from 1')
        if any(isinstance(layer.encoder_attn, PackedCrossAttention)
               for layer in base_model.model.decoder.layers):
            raise ValueError('Backbone already wrapped by U2')
        self.base_model = base_model
        self.donor_layer = donor_layer
        self.decoder_layer = decoder_layer
        for index, layer in enumerate(base_model.model.decoder.layers, 1):
            layer.encoder_attn = self.attention_class(layer.encoder_attn, index == decoder_layer)
        self.train(base_model.training)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    @property
    def correction_attention(self):
        return self.base_model.model.decoder.layers[self.decoder_layer - 1].encoder_attn

    def encode(self, input_ids=None, attention_mask=None, inputs_embeds=None,
               head_mask=None, output_hidden_states=False, output_attentions=False):
        if attention_mask is not None and (attention_mask.ndim != 2 or
                                          not attention_mask.bool().any(-1).all()):
            raise ValueError('Every source requires at least one unmasked position')
        outputs = self.base_model.get_encoder()(
            input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds,
            head_mask=head_mask, output_hidden_states=True,
            output_attentions=output_attentions, return_dict=True)
        return DepthEncoderOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
            donor_hidden_state=outputs.hidden_states[self.donor_layer].detach())

    def _packed(self, memory):
        if not isinstance(memory, DepthEncoderOutput) or memory.donor_hidden_state is None:
            raise ValueError('Precomputed encoder_outputs must come from prototype.encode()')
        if memory.last_hidden_state.shape != memory.donor_hidden_state.shape:
            raise ValueError('Recipient and donor must have matching shapes')
        return BaseModelOutput(
            last_hidden_state=torch.cat((memory.last_hidden_state,
                                         memory.donor_hidden_state.detach()), dim=-1),
            hidden_states=memory.hidden_states, attentions=memory.attentions)

    def forward(self, input_ids=None, attention_mask=None, encoder_outputs=None,
                inputs_embeds=None, output_hidden_states=None, output_attentions=None,
                head_mask=None, return_dict=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        hidden = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        attentions = self.config.output_attentions if output_attentions is None else output_attentions
        memory = encoder_outputs
        if memory is None:
            memory = self.encode(input_ids, attention_mask, inputs_embeds, head_mask, hidden, attentions)
        result = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            encoder_outputs=self._packed(memory), inputs_embeds=inputs_embeds,
            output_hidden_states=hidden, output_attentions=attentions,
            head_mask=head_mask, return_dict=True, **kwargs)
        result.encoder_last_hidden_state = memory.last_hidden_state
        as_dict = self.config.use_return_dict if return_dict is None else return_dict
        return result if as_dict else result.to_tuple()

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, encoder_outputs=None, **kwargs):
        memory = encoder_outputs
        if memory is None:
            memory = self.encode(input_ids, attention_mask,
                inputs_embeds=kwargs.pop('inputs_embeds', None),
                head_mask=kwargs.get('head_mask'),
                output_hidden_states=kwargs.get('output_hidden_states', False),
                output_attentions=kwargs.get('output_attentions', False))
        return self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                        encoder_outputs=self._packed(memory), **kwargs)

    def extra_parameter_count(self):
        return self.correction_attention.gates.numel()

    def save_pretrained(self, folder):
        """Prototype-only format: generic HF loading must not silently omit U2."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if any((folder / name).exists() for name in ('model.safetensors', 'pytorch_model.bin')):
            raise ValueError('Use a separate prototype folder, not a native checkpoint directory')
        self.config.save_pretrained(folder)
        self.generation_config.save_pretrained(folder)
        (folder / 'prototype_config.json').write_text(json.dumps({
            'method': self.method_id, 'donor_layer': self.donor_layer,
            'decoder_layer': self.decoder_layer, 'attention_implementation': self.config._attn_implementation,
        }, indent=2), encoding='utf-8')
        save_model(self, str(folder / 'prototype.safetensors'))

    @classmethod
    def from_pretrained(cls, folder):
        folder = Path(folder)
        recipe = json.loads((folder / 'prototype_config.json').read_text(encoding='utf-8'))
        if recipe['method'] != cls.method_id:
            raise ValueError('Not a U2 prototype checkpoint')
        config = MBartConfig.from_pretrained(folder, local_files_only=True)
        config._attn_implementation = 'eager'
        model = cls(MBartForConditionalGeneration(config), recipe['donor_layer'], recipe['decoder_layer'])
        model.base_model.generation_config = GenerationConfig.from_pretrained(folder, local_files_only=True)
        load_model(model, str(folder / 'prototype.safetensors'), strict=True)
        return model
