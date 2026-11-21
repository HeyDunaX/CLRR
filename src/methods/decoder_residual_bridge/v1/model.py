"""DRB: graph-local packed source memory, native SDPA and detached side reads."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import MBartConfig, MBartForConditionalGeneration, GenerationConfig
from transformers.modeling_outputs import BaseModelOutput
from methods.depth_difference.v1.model import DepthDifferencePrototype, DepthEncoderOutput

METHODS = {'drb-local-v1': 'local', 'drb-donor-v1': 'donor',
           'drb-source-v1': 'source', 'drb-query-v1': 'query'}


def norm(value):
    return value * (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt()


@dataclass
class BridgeEncoderOutput(DepthEncoderOutput):
    content_mask: torch.Tensor | None = None


class BridgeAttention(nn.Module):
    def __init__(self, attention, selected, mode, rank, generator):
        super().__init__()
        self.attention, self.selected, self.mode = attention, selected, mode
        if selected:
            width = attention.embed_dim
            self.down = nn.Parameter(torch.empty(rank, width))
            self.query = nn.Parameter(torch.empty(rank, width))
            self.up = nn.Parameter(torch.zeros(width, rank))
            for parameter in (self.down, self.query):
                nn.init.uniform_(parameter, -1/math.sqrt(width), 1/math.sqrt(width), generator=generator)
            self.to(device=attention.q_proj.weight.device, dtype=attention.q_proj.weight.dtype)

    def side_features(self, hidden, top, donor, content, mask, head_mask):
        attn = self.attention
        b, t, width = hidden.shape
        def heads(value):
            return value.reshape(b, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
        with torch.no_grad():
            logits = heads(attn.q_proj(hidden)) @ heads(attn.k_proj(top)).transpose(-1, -2)
            logits = logits * attn.scaling
            if mask is not None:
                logits = logits + mask
            probability = logits.softmax(-1)
            if head_mask is not None:
                probability = probability * head_mask.reshape(1, -1, 1, 1)
            # All arms execute both reads; no side-path dropout/RNG consumption.
            low = (probability @ heads(attn.v_proj(norm(donor)))).transpose(1, 2).reshape(b, t, width)
            high = (probability @ heads(attn.v_proj(norm(top)))).transpose(1, 2).reshape(b, t, width)
            value = high if self.mode == 'local' else low if self.mode == 'donor' else low-high
            weight = content.to(top.dtype).unsqueeze(-1)
            source = (norm(top) * weight).sum(1, keepdim=True) / weight.sum(1, keepdim=True).clamp_min(1)
            query = source.expand(-1, t, -1) if self.mode == 'source' else hidden
            return norm(value).detach(), norm(query).detach()

    def forward(self, hidden_states, key_value_states=None, past_key_values=None,
                attention_mask=None, layer_head_mask=None, output_attentions=False,
                cache_position=None, **kwargs):
        width = self.attention.embed_dim
        if key_value_states is None or key_value_states.shape[-1] != 2*width+1:
            raise ValueError('DRB requires its explicit packed encoder memory')
        top, donor, content = key_value_states.split([width, width, 1], dim=-1)
        result, weights = self.attention(hidden_states=hidden_states,
            key_value_states=top, past_key_values=past_key_values,
            attention_mask=attention_mask, layer_head_mask=layer_head_mask,
            output_attentions=output_attentions, cache_position=cache_position, **kwargs)
        if self.selected:
            value, query = self.side_features(hidden_states, top, donor, content[..., 0],
                                               attention_mask, layer_head_mask)
            delta = F.linear(F.gelu(F.linear(value, self.down)) *
                             torch.sigmoid(F.linear(query, self.query)), self.up)
            result = result + delta
        return result, weights


class DecoderResidualBridge(DepthDifferencePrototype):
    """One CE-only bridge; inherited encode/forward/generate packing contracts."""
    def __init__(self, base_model, method='drb-query-v1', donor_layer=6,
                 decoder_layer=12, rank=64, module_seed=10042):
        nn.Module.__init__(self)
        if not isinstance(base_model, MBartForConditionalGeneration) or method not in METHODS:
            raise ValueError('DRB requires mBART and a registered method')
        if base_model.config._attn_implementation != 'sdpa':
            raise ValueError('DRB pilot locks native SDPA')
        if base_model.config.encoder_layerdrop or base_model.config.decoder_layerdrop:
            raise ValueError('DRB requires LayerDrop=0')
        if not 1 <= donor_layer < base_model.config.encoder_layers or not 1 <= decoder_layer <= base_model.config.decoder_layers:
            raise ValueError('Invalid donor/decoder layer')
        if rank < 1:
            raise ValueError('Rank must be positive')
        self.base_model, self.method_id = base_model, method
        self.donor_layer, self.decoder_layer = donor_layer, decoder_layer
        self.rank, self.module_seed = rank, module_seed
        self.special_ids = sorted({x for x in (base_model.config.pad_token_id,
            base_model.config.bos_token_id, base_model.config.eos_token_id) if x is not None})
        generator = torch.Generator().manual_seed(module_seed)
        for index, layer in enumerate(base_model.model.decoder.layers, 1):
            layer.encoder_attn = BridgeAttention(layer.encoder_attn, index == decoder_layer,
                                                METHODS[method], rank, generator)
        self.train(base_model.training)

    def encode(self, input_ids=None, attention_mask=None, inputs_embeds=None,
               head_mask=None, output_hidden_states=False, output_attentions=False):
        output = super().encode(input_ids, attention_mask, inputs_embeds, head_mask,
                                output_hidden_states, output_attentions)
        valid = torch.ones(output.last_hidden_state.shape[:2], dtype=torch.bool,
                           device=output.last_hidden_state.device) if attention_mask is None else attention_mask.bool()
        content = valid.clone()
        if input_ids is not None:
            for token in self.special_ids:
                content &= input_ids.ne(token)
        content = torch.where(content.any(-1, keepdim=True), content, valid)
        return BridgeEncoderOutput(last_hidden_state=output.last_hidden_state,
            hidden_states=output.hidden_states, attentions=output.attentions,
            donor_hidden_state=output.donor_hidden_state, content_mask=content)

    def _packed(self, memory):
        if not isinstance(memory, BridgeEncoderOutput):
            raise ValueError('Encoder memory must come from DRB.encode')
        return BaseModelOutput(last_hidden_state=torch.cat((memory.last_hidden_state,
            memory.donor_hidden_state.detach(), memory.content_mask[..., None].to(memory.last_hidden_state.dtype)), -1),
            hidden_states=memory.hidden_states, attentions=memory.attentions)

    def extra_parameter_count(self):
        a = self.correction_attention
        return a.down.numel() + a.query.numel() + a.up.numel()

    def method_metadata(self):
        return dict(method=self.method_id, donor_layer=self.donor_layer,
            decoder_layer=self.decoder_layer, rank=self.rank, module_seed=self.module_seed,
            special_ids=self.special_ids, attention_implementation=self.config._attn_implementation,
            side_gradient='detached', side_attention='pre-dropout',
            extra_parameters=self.extra_parameter_count())

    def save_pretrained(self, folder, state_dict=None, **kwargs):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(folder)
        self.generation_config.save_pretrained(folder)
        (folder/'drb_config.json').write_text(json.dumps(self.method_metadata(), indent=2), encoding='utf-8')
        torch.save(self.state_dict() if state_dict is None else state_dict, folder/'pytorch_model.bin')

    @classmethod
    def from_pretrained(cls, folder):
        folder = Path(folder)
        metadata = json.loads((folder/'drb_config.json').read_text(encoding='utf-8'))
        config = MBartConfig.from_pretrained(folder, local_files_only=True)
        config._attn_implementation = metadata['attention_implementation']
        model = cls(MBartForConditionalGeneration(config), **{k:metadata[k] for k in
            ('method','donor_layer','decoder_layer','rank','module_seed')})
        model.special_ids = metadata['special_ids']
        model.base_model.generation_config = GenerationConfig.from_pretrained(folder, local_files_only=True)
        model.load_state_dict(torch.load(folder/'pytorch_model.bin', map_location='cpu', weights_only=True), strict=True)
        if model.method_metadata() != metadata:
            raise ValueError('DRB checkpoint metadata mismatch')
        return model
