"""All-training bilingual relational screen; no semantic pair labels assumed."""
import json
import math
from pathlib import Path

import torch

from ...original.clrr import CrossLayerResidualRewire
from ...residual_effect.v1.lsr import ResidualEffectSeq2SeqLM
from .geometry import relational_loss


METHODS = {
    'relational-distance-v2': ('rkd-distance', False),
    'relational-direction-v2': ('direction', False),
    'relational-clrr-distance-v2': ('rkd-distance', True),
    'relational-clrr-direction-v2': ('direction', True),
}


def content_mask(ids, attention, special_ids):
    mask = attention.bool() & ids.ne(-100)
    for ident in special_ids:
        mask &= ids.ne(ident)
    return mask


def pair_eligibility(source_ids, source_mask, target_ids, target_mask):
    """Use only post-truncation content IDs; no strings or semantic labels."""
    sequences = []
    for ids, mask in ((source_ids, source_mask), (target_ids, target_mask)):
        sequences.append([tuple(row[keep].tolist()) for row, keep in zip(ids, mask)])
    size = source_ids.shape[0]
    result = torch.zeros(size, size, dtype=torch.bool, device=source_ids.device)
    for i in range(size):
        for j in range(i + 1, size):
            valid = all(view[i] and view[j] and view[i] != view[j] for view in sequences)
            result[i, j] = result[j, i] = bool(valid)
    return result


class RelationalSeq2SeqLM(ResidualEffectSeq2SeqLM):
    """CE unchanged; deterministic source/reference encoder views add the loss."""
    def __init__(self, base_model, *, objective, weight=.1, special_token_ids=()):
        if objective not in ('direction', 'rkd-distance') or not math.isfinite(weight) or weight <= 0:
            raise ValueError('Invalid relational objective/weight')
        super().__init__(base_model, mode='ordinary', jepa_weight=0.,
                         alignment_weight=weight, temperature=1.,
                         special_token_ids=special_token_ids)
        self.objective = objective
        self.coverage = dict(batches=0, eligible_pairs=0, valid_pairs=0,
                             fewer_than_two_valid_batches=0, zero_valid_batches=0)

    def method_metadata(self):
        route = self.base_model if isinstance(self.base_model, CrossLayerResidualRewire) else None
        return dict(version=2, objective=self.objective, weight=self.alignment_weight,
                    special_token_ids=sorted(set(self.special_token_ids)),
                    clrr=(dict(distance=route.distance, strength=route.strength,
                               stack=route.stack) if route is not None else None),
                    target_detached=True, target_encoder='shared evolving',
                    auxiliary_dropout=False, pooling='content masked mean',
                    l2_before_difference=False, target_degenerate_epsilon=1e-6,
                    pairing='all unordered within microbatch; distinct nonempty content IDs both sides',
                    source_normalizer_detached=False, original_lsr_weight=0., extra_parameters=0)

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(Path(folder, 'relational_config.json').read_text(encoding='utf-8'))
        if saved != self.method_metadata():
            raise ValueError('Relational checkpoint recipe mismatch')

    def pooled(self, ids, attention, mask):
        with self.deterministic_view(True):
            hidden = self.get_encoder()(input_ids=ids, attention_mask=attention).last_hidden_state
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        aux = None
        diagnostics = {}
        if self.training and labels is not None:
            if not self.special_token_ids:
                raise ValueError('Configure tokenizer special IDs before relational training')
            pad = self.config.pad_token_id
            attention = attention_mask if attention_mask is not None else input_ids.ne(pad).long()
            target_ids = labels.masked_fill(labels == -100, pad)
            target_attention = target_ids.ne(pad).long()
            sm = content_mask(input_ids, attention, self.special_token_ids)
            tm = content_mask(target_ids, target_attention, self.special_token_ids)
            eligible = pair_eligibility(input_ids, sm, target_ids, tm)
            if eligible.any():
                source = self.pooled(input_ids, attention, sm)
                with torch.no_grad():
                    target = self.pooled(target_ids, target_attention, tm)
                aux, diagnostics = relational_loss(source, target, objective=self.objective,
                                                   eligible_pairs=eligible)
            else:
                diagnostics = dict(eligible_pairs=0, valid_pairs=0, small_source_pairs=0,
                                   insufficient_distance_pairs=True,
                                   degenerate_distance_normalizer=False)
        kwargs.pop('num_items_in_batch', None)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                  labels=labels, **kwargs)
        ce = outputs.loss
        if aux is not None:
            outputs.loss = ce + self.alignment_weight * aux.to(ce.dtype)
        if self.training and labels is not None:
            self.coverage['batches'] += 1
            self.coverage['eligible_pairs'] += diagnostics['eligible_pairs']
            self.coverage['valid_pairs'] += diagnostics['valid_pairs']
            self.coverage['fewer_than_two_valid_batches'] += int(diagnostics['valid_pairs'] < 2)
            self.coverage['zero_valid_batches'] += int(diagnostics['valid_pairs'] == 0)
            self.last_diagnostics = dict(coverage_since_model_load=dict(self.coverage),
                ce=float(ce.detach()),
                relational_loss=float(aux.detach()) if aux is not None else 0.,
                objective=self.objective, weight=self.alignment_weight,
                anchor_batch_size=input_ids.shape[0], **diagnostics)
        return outputs

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder, 'relational_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8')
