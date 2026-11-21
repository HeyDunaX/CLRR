"""mBART-only residual alignment experiment; graph-local donors, no hooks.

The encoder loop follows Transformers 4.57.6 MBartEncoder (LayerDrop=0).
Each checkpoint closure binds its own block and parameter view. Reentrant
recomputation cannot read donor state from a later source or target pass.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import MBartConfig, MBartForConditionalGeneration, GenerationConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.mbart.modeling_mbart import MBartEncoder

from .module import ResidualBottleneck

METHODS = {
    'alignment-residual-capacity-v1': 'capacity',
    'alignment-residual-fullgrad-v1': 'fullgrad',
    'alignment-residual-confined-v1': 'confined',
    'alignment-local-confined-v1': 'local',
}


class _BlockForward(nn.Module):
    """Bypass HF's implicit checkpoint wrapper; our closure owns checkpointing."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        return self.block.forward(*args, **kwargs)


def _constant_call(module, *args, **kwargs):
    return functional_call(module, {n: p.detach() for n, p in module.named_parameters()},
                           args, kwargs, strict=False)


class ResidualEncoder(MBartEncoder):
    def forward(self, input_ids=None, attention_mask=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, frozen_backbone=False):
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('Provide exactly one of input_ids and inputs_embeds')
        call = _constant_call if frozen_backbone else lambda module, *a, **k: module(*a, **k)
        if inputs_embeds is None:
            position_input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            inputs_embeds = call(self.embed_tokens, input_ids)
        else:
            position_input = inputs_embeds[:, :, -1]
        positions = call(self.embed_positions, position_input)
        hidden = call(self.layernorm_embedding, inputs_embeds + positions.to(inputs_embeds.device))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        mask = self._update_full_mask(attention_mask, inputs_embeds)
        if head_mask is not None and head_mask.shape[0] != len(self.layers):
            raise ValueError('Wrong encoder head-mask layer count')
        states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        donors = []
        for index, layer in enumerate(self.layers):
            if output_hidden_states:
                states += (hidden,)
            # Native mBART consumes this draw even when LayerDrop is zero.
            if self.training:
                torch.rand([])
            proxy = _BlockForward(layer)
            layer_head = head_mask[index] if head_mask is not None else None
            # Bind every argument, including frozen policy, before backward replay.
            def block(value, proxy=proxy, layer_head=layer_head, mask=mask,
                      output_attentions=output_attentions, frozen=frozen_backbone):
                invoke = _constant_call if frozen else lambda m, *a, **k: m(*a, **k)
                return invoke(proxy, value, mask, layer_head_mask=layer_head,
                              output_attentions=output_attentions)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                if not hidden.requires_grad:
                    hidden = hidden.detach().requires_grad_(True)
                result = checkpoint(block, hidden, use_reentrant=True, preserve_rng_state=True)
            else:
                result = block(hidden)
            hidden = result[0]
            if index >= self.residual_distance:
                donor = donors[index - self.residual_distance] if self.residual_q else torch.zeros_like(hidden)
                correction = self.residual_modules[str(index)](donor - hidden)
                hidden = hidden + self.residual_strength * (donor + correction)
            donors.append(hidden.detach())
            if output_attentions:
                attentions += (result[1],)
        hidden = call(self.layer_norm, hidden)
        if output_hidden_states:
            states += (hidden,)
        output = BaseModelOutput(last_hidden_state=hidden, hidden_states=states, attentions=attentions)
        return output if return_dict else output.to_tuple()


class AlignmentResidualMT(nn.Module):
    accepts_loss_kwargs = False

    def __init__(self, base_model, method, rank=64, module_seed=10042,
                 distance=2, strength=.1, jepa_weight=.1):
        super().__init__()
        if not isinstance(base_model, MBartForConditionalGeneration) or method not in METHODS:
            raise ValueError('Alignment residual v1 requires mBART and a registered method')
        if base_model.config.encoder_layerdrop or base_model.config.decoder_layerdrop:
            raise ValueError('Alignment residual v1 requires LayerDrop=0')
        if distance < 1 or distance >= base_model.config.encoder_layers or rank < 1:
            raise ValueError('Invalid residual distance/rank')
        if strength < 0 or jepa_weight < 0:
            raise ValueError('Negative residual strength or LSR weight')
        if METHODS[method] == 'capacity' and jepa_weight != 0:
            raise ValueError('Capacity control must have zero LSR weight')
        encoder = base_model.get_encoder()
        if isinstance(encoder, ResidualEncoder):
            raise ValueError('Encoder already wrapped')
        self.base_model = base_model
        self.method_id, self.rank, self.module_seed = method, rank, module_seed
        self.distance, self.strength, self.jepa_weight = distance, strength, jepa_weight
        encoder.__class__ = ResidualEncoder
        encoder.residual_distance, encoder.residual_strength = distance, strength
        encoder.residual_q = 0 if METHODS[method] == 'local' else 1
        generator = torch.Generator(device='cpu').manual_seed(module_seed)
        encoder.residual_modules = nn.ModuleDict({str(i): ResidualBottleneck(
            base_model.config.d_model, rank, generator) for i in range(distance, len(encoder.layers))})
        encoder.residual_modules.to(device=encoder.embed_tokens.weight.device,
                                    dtype=encoder.embed_tokens.weight.dtype)
        self.last_diagnostics = None
        self.train(base_model.training)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def extra_parameter_count(self):
        return sum(p.numel() for p in self.get_encoder().residual_modules.parameters())

    def method_metadata(self):
        return dict(method=self.method_id, rank=self.rank, module_seed=self.module_seed,
                    distance=self.distance, strength=self.strength, jepa_weight=self.jepa_weight,
                    attention_implementation=self.config._attn_implementation,
                    gradient_policy=METHODS[self.method_id], target_no_grad=True,
                    source_rng_replay=True, capacity_target_dummy=True,
                    extra_parameters=self.extra_parameter_count())

    @staticmethod
    def pooled(hidden, mask):
        weight = mask.to(hidden.dtype).unsqueeze(-1)
        mean = (hidden * weight).sum(1) / weight.sum(1).clamp_min(1e-6)
        return F.normalize(mean, dim=-1, eps=1e-12)

    def auxiliary_loss(self, input_ids, attention_mask, target_vector):
        hidden = self.get_encoder()(input_ids=input_ids, attention_mask=attention_mask,
            frozen_backbone=METHODS[self.method_id] in ('confined', 'local'),
            return_dict=True).last_hidden_state
        return (1 - (self.pooled(hidden, attention_mask) * target_vector.detach()).sum(-1)).mean()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        if labels is None:
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if kwargs.get('encoder_outputs') is not None or kwargs.get('inputs_embeds') is not None:
            raise ValueError('Alignment training requires original source IDs')
        as_dict = kwargs.pop('return_dict', self.config.use_return_dict)
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).long()
        device = input_ids.device
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
        result = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                 labels=labels, return_dict=True, **kwargs)
        # Replay the source draw without advancing the CE stream twice.
        source = None
        if self.jepa_weight:
            with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng, device)
                source = self.get_encoder()(input_ids=input_ids, attention_mask=attention_mask,
                    frozen_backbone=METHODS[self.method_id] in ('confined', 'local'),
                    return_dict=True).last_hidden_state
        target_ids = labels.masked_fill(labels.eq(-100), self.config.pad_token_id)
        target_mask = target_ids.ne(self.config.pad_token_id).long()
        with torch.no_grad():
            target = self.get_encoder()(input_ids=target_ids, attention_mask=target_mask,
                                        return_dict=True).last_hidden_state
            target_vector = self.pooled(target, target_mask)
        ce = result.loss
        lsr = ce.new_zeros(()) if source is None else (
            1 - (self.pooled(source, attention_mask) * target_vector).sum(-1)).mean()
        result.loss = ce + self.jepa_weight * lsr
        self.last_diagnostics = dict(ce=float(ce.detach()), lsr=float(lsr.detach()),
            weighted_lsr=float((self.jepa_weight * lsr).detach()), method=self.method_id)
        return result if as_dict else result.to_tuple()

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, folder, state_dict=None, **kwargs):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(folder)
        self.generation_config.save_pretrained(folder)
        (folder / 'alignment_residual_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8')
        torch.save(self.state_dict() if state_dict is None else state_dict, folder / 'pytorch_model.bin')

    @classmethod
    def from_pretrained(cls, folder):
        folder = Path(folder)
        metadata = json.loads((folder / 'alignment_residual_config.json').read_text(encoding='utf-8'))
        config = MBartConfig.from_pretrained(folder, local_files_only=True)
        config._attn_implementation = metadata['attention_implementation']
        model = cls(MBartForConditionalGeneration(config), **{key: metadata[key] for key in (
            'method', 'rank', 'module_seed', 'distance', 'strength', 'jepa_weight')})
        model.base_model.generation_config = GenerationConfig.from_pretrained(folder, local_files_only=True)
        model.load_state_dict(torch.load(folder / 'pytorch_model.bin', map_location='cpu', weights_only=True), strict=True)
        if model.method_metadata() != metadata:
            raise ValueError('Alignment residual checkpoint metadata mismatch')
        return model
