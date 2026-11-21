"""Matched learned-donor CE control: change only A(b-u) to A(b).

Encoder loop copied from frozen v1 to leave all historical methods untouched.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.mbart.modeling_mbart import MBartEncoder
from ..v1.model import AlignmentResidualMT, _BlockForward, _constant_call

METHODS = {'alignment-learned-donor-ce-v1': 'learned-donor-capacity'}


class LearnedDonorEncoder(MBartEncoder):
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
                correction = self.residual_modules[str(index)](donor)
                hidden = hidden + self.residual_strength * (donor + correction)
            donors.append(hidden.detach())
            if output_attentions:
                attentions += (result[1],)
        hidden = call(self.layer_norm, hidden)
        if output_hidden_states:
            states += (hidden,)
        output = BaseModelOutput(last_hidden_state=hidden, hidden_states=states, attentions=attentions)
        return output if return_dict else output.to_tuple()


class LearnedDonorCE(AlignmentResidualMT):
    def __init__(self, base_model, method, rank=64, module_seed=10042,
                 distance=2, strength=.1, jepa_weight=0.):
        if method not in METHODS or jepa_weight != 0:
            raise ValueError('Learned donor control requires CE-only and its own ID')
        super().__init__(base_model, 'alignment-residual-capacity-v1',
                         rank, module_seed, distance, strength, jepa_weight)
        self.get_encoder().__class__ = LearnedDonorEncoder
        self.method_id = method

    def method_metadata(self):
        return dict(method=self.method_id, rank=self.rank, module_seed=self.module_seed,
                    distance=self.distance, strength=self.strength, jepa_weight=0.,
                    attention_implementation=self.config._attn_implementation,
                    gradient_policy=METHODS[self.method_id], target_no_grad=True,
                    source_rng_replay=True, capacity_target_dummy=True,
                    correction_input='detached_donor',
                    extra_parameters=self.extra_parameter_count())

    def auxiliary_loss(self, *args, **kwargs):
        raise ValueError('Learned donor CE control has no auxiliary objective')
