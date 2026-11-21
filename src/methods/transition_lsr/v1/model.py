"""Supervise depth changes along original CLRR edges, not routing ON/OFF effects."""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ...common.helpers import _find_stack_layers, _masked_mean_pool
from ...original.clrr import CrossLayerResidualRewire
from ...residual_effect.v1.lsr import ResidualEffectSeq2SeqLM


METHODS = {'transition-lsr-v1': False, 'transition-full-v1': True}


def transition_cosine(source, target):
    """Mean across edges/examples; zero vectors remain finite, not collapse-proof."""
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError('Expected equal [edges, batch, width] transition tensors')
    return (1 - (F.normalize(source.float(), dim=-1, eps=1e-6) *
                 F.normalize(target.detach().float(), dim=-1, eps=1e-6)).sum(-1)).mean()


class TransitionLSRSeq2SeqLM(ResidualEffectSeq2SeqLM):
    def __init__(self, base_model, *, weight=.1, distance=2):
        if not math.isfinite(weight) or weight <= 0 or distance < 1:
            raise ValueError('Positive finite weight and positive distance required')
        super().__init__(base_model, mode='ordinary', jepa_weight=0.,
                         alignment_weight=weight, temperature=1., special_token_ids=())
        self.distance = distance
        route = self.base_model if isinstance(self.base_model, CrossLayerResidualRewire) else None
        if route is not None and (route.stack != 'encoder' or route.distance != distance):
            raise ValueError('Transition endpoints must match encoder CLRR edges')
        if len(_find_stack_layers(self.base_model, 'encoder')) <= distance:
            raise ValueError('Encoder has no eligible depth transitions')

    def method_metadata(self):
        route = self.base_model if isinstance(self.base_model, CrossLayerResidualRewire) else None
        return dict(version=1, objective='paired-depth-transition-cosine',
                    weight=self.alignment_weight, distance=self.distance,
                    clrr_strength=route.strength if route is not None else 0.,
                    recipient='block output BEFORE rewire', donor='POST-rewire output',
                    donor_detached=True, target_detached=True,
                    auxiliary_dropout=False, pooling='original attention masked mean',
                    auxiliary_epsilon=1e-6, original_lsr_weight=0., extra_parameters=0)

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(Path(folder, 'transition_lsr_config.json').read_text(encoding='utf-8'))
        if saved != self.method_metadata():
            raise ValueError('Transition LSR checkpoint recipe mismatch')

    def transitions(self, ids, attention):
        """Read pre-rewire outputs without altering CLRR, including its donor cache."""
        layers = _find_stack_layers(self.base_model, 'encoder')
        route = self.base_model if isinstance(self.base_model, CrossLayerResidualRewire) else None
        strength = route.strength if route is not None else 0.
        donors, changes, handles = {}, [], []

        def observer(index):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                donor = donors.get(index - self.distance)
                if donor is not None:
                    changes.append(_masked_mean_pool(hidden, attention) -
                                   _masked_mean_pool(donor, attention))
                    post = hidden.detach() + strength * donor
                else:
                    post = hidden.detach()
                donors[index] = post
                # Observer only: None leaves the original output unchanged.
            return hook

        try:
            for index, layer in enumerate(layers):
                handles.append(layer.register_forward_hook(observer(index), prepend=True))
            # eval disables per-layer reentrant checkpointing in this auxiliary
            # pass. CE still uses the locked training/checkpointing settings.
            with self.deterministic_view(True):
                self.get_encoder()(input_ids=ids, attention_mask=attention)
            if len(changes) != len(layers) - self.distance:
                raise RuntimeError('Missing encoder depth transitions')
            return torch.stack(changes)
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        auxiliary = None
        if self.training and labels is not None:
            pad = self.config.pad_token_id
            source_attention = attention_mask if attention_mask is not None else input_ids.ne(pad).long()
            target_ids = labels.masked_fill(labels == -100, pad)
            target_attention = target_ids.ne(pad).long()
            if not source_attention.any(-1).all() or not target_attention.any(-1).all():
                raise ValueError('Transition supervision requires nonempty source and target')
            source = self.transitions(input_ids, source_attention)
            with torch.no_grad():
                target = self.transitions(target_ids, target_attention)
            auxiliary = transition_cosine(source, target)
            self.last_diagnostics = dict(auxiliary_loss=float(auxiliary.detach()),
                source_norm_min=float(source.detach().norm(dim=-1).min()),
                target_norm_min=float(target.norm(dim=-1).min()), edges=source.shape[0],
                source_requires_grad=source.requires_grad, target_requires_grad=target.requires_grad)
        kwargs.pop('num_items_in_batch', None)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                  labels=labels, **kwargs)
        if auxiliary is not None:
            outputs.loss = outputs.loss + self.alignment_weight * auxiliary.to(outputs.loss.dtype)
        return outputs

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder, 'transition_lsr_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8')
