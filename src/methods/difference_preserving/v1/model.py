"""Isolated CE + difference-loss prototype. No change to historical methods."""
import json
import math
from pathlib import Path

import torch

from ...residual_effect.v1.lsr import ResidualEffectSeq2SeqLM
from ...original.clrr import CrossLayerResidualRewire
from .loss import difference_alignment_loss


class DifferencePreservingSeq2SeqLM(ResidualEffectSeq2SeqLM):
    """Explicit peer tensors; CE still uses only the original anchor batch.

    Auxiliary encoder dropout/checkpointing is disabled through the existing
    deterministic-view context; CE retains the original training settings.
    Target anchors evolve with the shared encoder and are not EMA teachers.
    Peer selection and semantic validation belong to a separate data gate.
    """
    def __init__(self, base_model, *, weight=.1, special_token_ids=()):
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('weight must be positive and finite')
        super().__init__(base_model, mode='ordinary', jepa_weight=0.,
                         alignment_weight=weight, temperature=1.,
                         special_token_ids=special_token_ids)

    def method_metadata(self):
        rewire = self.base_model if isinstance(self.base_model, CrossLayerResidualRewire) else None
        return dict(version=1, weight=self.alignment_weight,
                    special_token_ids=sorted(set(self.special_token_ids)),
                    clrr=(dict(distance=rewire.distance, strength=rewire.strength,
                               stack=rewire.stack) if rewire is not None else None),
                    target_detached=True, auxiliary_dropout=False,
                    target_degenerate_epsilon=1e-6, pooling='content masked mean',
                    l2_before_difference=False, original_lsr_weight=0.,
                    extra_parameters=0, semantic_pair_verification='external gate required')

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(Path(folder, 'difference_preserving_config.json').read_text(encoding='utf-8'))
        if saved != self.method_metadata():
            raise ValueError('Difference checkpoint recipe mismatch (including pooling tokens and CLRR)')

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                peer_input_ids=None, peer_attention_mask=None, peer_labels=None,
                pair_eligible=None, **kwargs):
        auxiliary = None
        counts = dict(eligible_pairs=0, valid_target_differences=0, small_source_differences=0)
        if self.training and labels is not None:
            if any(v is None for v in (peer_input_ids, peer_attention_mask,
                                       peer_labels, pair_eligible)):
                raise ValueError('Training requires peers and an explicit eligibility mask')
            if pair_eligible.dtype != torch.bool or pair_eligible.shape != (input_ids.shape[0],):
                raise ValueError('Invalid pair eligibility mask')
            selected = pair_eligible.nonzero(as_tuple=True)[0]
            if selected.numel():
                pad = self.config.pad_token_id
                attention = attention_mask if attention_mask is not None else input_ids.ne(pad).long()
                source = self.representation(input_ids[selected], attention[selected], True)
                peer = self.representation(peer_input_ids[selected], peer_attention_mask[selected], True)
                with torch.no_grad():
                    target_ids = labels[selected].masked_fill(labels[selected] == -100, pad)
                    other_ids = peer_labels[selected].masked_fill(peer_labels[selected] == -100, pad)
                    target = self.representation(target_ids, target_ids.ne(pad).long(), True)
                    other = self.representation(other_ids, other_ids.ne(pad).long(), True)
                auxiliary = difference_alignment_loss(source, peer, target, other)
                counts = dict(eligible_pairs=selected.numel(),
                    valid_target_differences=int(((target-other).norm(dim=-1) > 1e-6).sum()),
                    small_source_differences=int(((source-peer).detach().norm(dim=-1) <= 1e-6).sum()))
        # Bypass the inherited InfoNCE/LSR forward. CLRR, if present, is in base_model.
        kwargs.pop('num_items_in_batch', None)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                  labels=labels, **kwargs)
        ce_loss = outputs.loss
        if auxiliary is not None:
            outputs.loss = outputs.loss + self.alignment_weight * auxiliary.to(outputs.loss.dtype)
        if self.training and labels is not None:
            self.last_diagnostics = dict(ce=float(ce_loss.detach()),
                difference_loss=float(auxiliary.detach()) if auxiliary is not None else 0.,
                weight=self.alignment_weight, anchor_batch_size=input_ids.shape[0], **counts)
        return outputs

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder, 'difference_preserving_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8')
