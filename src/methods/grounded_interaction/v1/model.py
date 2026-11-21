"""Research prototype; four-cell interaction is a hypothesis, not an efficacy claim."""
from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import Seq2SeqLMOutput

from ...original.clrr import CrossLayerResidualRewire


def source_negatives(input_ids, attention_mask, labels):
    """First cyclic in-microbatch source with different source AND target tokens.

    Dataset rows are untouched. Token-identical pairs are excluded conservatively;
    semantic equivalence of distinct strings is not identifiable by this rule.
    """
    sources = [tuple(x[m.bool()].tolist()) for x, m in zip(input_ids, attention_mask)]
    targets = [tuple(y[y.ne(-100)].tolist()) for y in labels]
    indices, valid = [], []
    for i in range(len(sources)):
        choices = [(i + k) % len(sources) for k in range(1, len(sources))]
        other = next((j for j in choices if sources[j] != sources[i] and targets[j] != targets[i]), None)
        indices.append(i if other is None else other)
        valid.append(other is not None)
    return (torch.tensor(indices, device=input_ids.device),
            torch.tensor(valid, dtype=torch.bool, device=input_ids.device))


def contrastive_objective(scores, valid, temperature, mode):
    """All four scores stay differentiable. Scores are mean gold-token log P."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    on_gap = scores['on_match'] - scores['on_mismatch']
    off_gap = scores['off_match'] - scores['off_mismatch']
    if mode not in ('source', 'interaction'):
        raise ValueError('Unknown contrastive objective')
    margin = on_gap if mode == 'source' else on_gap - off_gap
    # Keep the graph defined in a singleton/all-duplicate microbatch.
    return F.softplus(-margin[valid] / temperature).mean() if valid.any() else margin.sum() * 0.


class GroundedInteractionSeq2SeqLM(CrossLayerResidualRewire):
    """Original CLRR plus paired CE and optional source/route contrast.

    Inference uses only route ON. No new parameters, queue, teacher or LSR.
    Training returns reduced loss without vocabulary logits. Evaluation delegates
    to ordinary ON-route CE/logits. Checkpointing recomputes a complete view so
    OFF routing/dropout flags cannot leak into another view's backward pass.
    """
    def __init__(self, base_model, *, mode='interaction', weight=.1, temperature=1.,
                 distance=2, strength=.1):
        if mode not in ('paired-ce', 'source', 'interaction'):
            raise ValueError('Unknown grounded interaction mode')
        if not math.isfinite(weight) or weight < 0 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Need finite weight >=0 and temperature >0')
        super().__init__(base_model, distance=distance, strength=strength, stack='encoder')
        self.mode, self.interaction_weight, self.temperature = mode, weight, temperature
        self._view_checkpointing = False
        # Per-layer recomputation cannot safely reuse mutable multi-view route state.
        self.base_model.gradient_checkpointing_disable()
        self.last_diagnostics = {}

    @contextmanager
    def _view(self, rewired, deterministic):
        flags = [(m, m.training) for m in self.base_model.modules()]
        strength = self.strength
        try:
            if deterministic:
                self.base_model.eval()
            if not rewired:
                self.strength = 0.
            with self._fresh_cache():
                yield
        finally:
            self.strength = strength
            for module, flag in flags:
                module.training = flag

    def _scores(self, ids, mask, labels, rewired, deterministic):
        def compute(ids, mask, labels):
            with self._view(rewired, deterministic):
                counts = labels.ne(-100).sum(1)
                if not counts.gt(0).all():
                    raise ValueError('Every example needs at least one supervised target token')
                logits = self.base_model(input_ids=ids, attention_mask=mask,
                    decoder_input_ids=self.prepare_decoder_input_ids_from_labels(labels),
                    use_cache=False, return_dict=True).logits
                nll = F.cross_entropy(logits.float().transpose(1, 2), labels,
                                      ignore_index=-100, reduction='none').sum(1)
                return -nll / counts
        if self._view_checkpointing and self.training and torch.is_grad_enabled():
            # Closures capture immutable rewired/deterministic values per view.
            return checkpoint(compute, ids, mask, labels, use_reentrant=False, preserve_rng_state=True)
        return compute(ids, mask, labels)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self._view_checkpointing = True
        self.base_model.gradient_checkpointing_disable()

    def gradient_checkpointing_disable(self):
        self._view_checkpointing = False
        self.base_model.gradient_checkpointing_disable()

    def training_loss(self, input_ids, attention_mask, labels):
        counts = labels.ne(-100).sum(1)
        # Two stochastic positive views; both use the ordinary baseline CE mask.
        on = self._scores(input_ids, attention_mask, labels, True, False)
        off = self._scores(input_ids, attention_mask, labels, False, False)
        ce = -((on + off) * counts).sum() / (2 * counts.sum())
        diagnostics = {'paired_ce': ce.detach().item(), 'valid_negatives': 0}
        if self.mode == 'paired-ce' or self.interaction_weight == 0:
            self.last_diagnostics = diagnostics
            return ce
        indices, valid = source_negatives(input_ids, attention_mask, labels)
        if not valid.any():
            self.last_diagnostics = diagnostics
            return ce
        # Deterministic auxiliary views avoid interpreting dropout as source effect.
        scores = {}
        for name, route, mismatch in [('on_match', True, False), ('on_mismatch', True, True),
                                     ('off_match', False, False), ('off_mismatch', False, True)]:
            scores[name] = self._scores(input_ids[indices] if mismatch else input_ids,
                attention_mask[indices] if mismatch else attention_mask, labels, route, True)
        aux = contrastive_objective(scores, valid, self.temperature, self.mode)
        diagnostics.update({k: v[valid].detach().mean().item() for k, v in scores.items()})
        diagnostics.update(auxiliary=aux.detach().item(), valid_negatives=valid.sum().item(),
            interaction=(scores['on_match'] - scores['on_mismatch'] - scores['off_match']
                          + scores['off_mismatch'])[valid].detach().mean().item())
        self.last_diagnostics = diagnostics
        return ce + self.interaction_weight * aux

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        if self.training and labels is not None:
            # Extra decoder inputs could silently disagree with the shifted labels.
            kwargs.pop('decoder_input_ids', None)
            if kwargs:
                raise ValueError(f'Unsupported training inputs: {sorted(kwargs)}')
            return Seq2SeqLMOutput(loss=self.training_loss(input_ids, attention_mask, labels))
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def method_metadata(self):
        return dict(version=1, mode=self.mode, weight=self.interaction_weight,
            temperature=self.temperature, distance=self.distance, strength=self.strength,
            checkpointing='whole-view non-reentrant', auxiliary_dropout=False,
            negative_pool='microbatch only', extra_parameters=0)

    def save_pretrained(self, save_directory, **kwargs):
        result = super().save_pretrained(save_directory, **kwargs)
        (Path(save_directory) / 'interaction_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2) + '\n', encoding='utf-8')
        return result
