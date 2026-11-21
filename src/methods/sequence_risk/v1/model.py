"""Matched CE / finite-candidate MRT continuation controls (not a novelty claim)."""
from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import Seq2SeqLMOutput

from ...original.clrr import CrossLayerResidualRewire
from .loss import candidate_set_risk


class SequenceRiskSeq2SeqLM(CrossLayerResidualRewire):
    """All arms share whole-view checkpointing; only route and risk weight vary."""
    def __init__(self, base_model, *, rewired=False, weight=0., alpha=.05,
                 score_chunk=2, distance=2, strength=.1):
        if not math.isfinite(weight) or weight < 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('Need finite weight >=0 and alpha >0')
        if not isinstance(score_chunk, int) or score_chunk < 1:
            raise ValueError('score_chunk must be a positive integer')
        self.rewired = bool(rewired)
        super().__init__(base_model, distance=distance, strength=strength, stack='encoder')
        self.risk_weight, self.risk_alpha, self.score_chunk = weight, alpha, score_chunk
        self._view_checkpointing = False
        self.base_model.gradient_checkpointing_disable()
        self.last_diagnostics = {}

    def _install_hooks(self):
        if self.rewired:
            super()._install_hooks()

    @contextmanager
    def _view(self, deterministic):
        flags = [(module, module.training) for module in self.base_model.modules()]
        try:
            if deterministic:
                self.base_model.eval()
            with self._fresh_cache():
                yield
        finally:
            for module, flag in flags:
                module.training = flag

    def _scores(self, ids, mask, labels, deterministic=False, mean=False):
        """Sequence SUM logP; one flat CE kernel matches the native HF reduction."""
        def compute(ids, mask, labels):
            if not labels.ne(-100).any(1).all():
                raise ValueError('Every scored candidate needs supervised labels')
            with self._view(deterministic):
                output = self.base_model(input_ids=ids, attention_mask=mask,
                    decoder_input_ids=self.prepare_decoder_input_ids_from_labels(labels),
                    labels=labels if mean else None, use_cache=False, return_dict=True)
                if mean:
                    return output.loss
                logits = output.logits
                nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                    labels.reshape(-1), ignore_index=-100, reduction='none').view_as(labels)
                return -nll.sum(1)
        if self._view_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(compute, ids, mask, labels, use_reentrant=False, preserve_rng_state=True)
        return compute(ids, mask, labels)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self._view_checkpointing = True
        self.base_model.gradient_checkpointing_disable()

    def gradient_checkpointing_disable(self):
        self._view_checkpointing = False
        self.base_model.gradient_checkpointing_disable()

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                candidate_labels=None, candidate_costs=None, candidate_valid=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        if not self.training or labels is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        kwargs.pop('decoder_input_ids', None)
        if kwargs:
            raise ValueError(f'Unsupported training fields: {sorted(kwargs)}')
        ce = self._scores(input_ids, attention_mask, labels, mean=True)
        self.last_diagnostics = {'ce': ce.detach().item(), 'risk': 0., 'valid_candidates': 0}
        if self.risk_weight == 0:
            return Seq2SeqLMOutput(loss=ce)
        if candidate_labels is None or candidate_costs is None or candidate_valid is None:
            raise ValueError('Risk training requires the locked train-only candidate pool')
        if candidate_labels.ndim != 3 or candidate_labels.shape[:2] != candidate_valid.shape:
            raise ValueError('Expected candidate labels [source,candidate,token]')
        if candidate_valid.shape != candidate_costs.shape or candidate_valid.shape[0] != len(labels):
            raise ValueError('Candidate costs/mask/source shapes differ')
        if candidate_valid.dtype != torch.bool or not candidate_valid.any(1).all():
            raise ValueError('Each source needs at least one valid candidate')
        indices = candidate_valid.nonzero(as_tuple=False)
        values = []
        for start in range(0, len(indices), self.score_chunk):
            chosen = indices[start:start + self.score_chunk]
            owners, candidates = chosen[:, 0], chosen[:, 1]
            values.append(self._scores(input_ids[owners], attention_mask[owners],
                candidate_labels[owners, candidates], deterministic=True))
        scores = ce.new_zeros(candidate_valid.shape)
        scores = scores.index_put((indices[:, 0], indices[:, 1]), torch.cat(values))
        risk = candidate_set_risk(scores, candidate_costs, candidate_valid, alpha=self.risk_alpha)
        with torch.no_grad():
            q = torch.softmax((self.risk_alpha * scores.detach()).masked_fill(~candidate_valid, -torch.inf), 1)
        self.last_diagnostics.update(risk=risk.detach().item(),
            valid_candidates=len(indices), singleton_sources=candidate_valid.sum(1).eq(1).sum().item(),
            posterior_entropy_mean=(-(q * q.clamp_min(1e-12).log()).sum(1)).mean().item(),
            posterior_max_mean=q.max(1).values.mean().item())
        return Seq2SeqLMOutput(loss=ce + self.risk_weight * risk)

    def method_metadata(self):
        return dict(version=1, rewired=self.rewired, weight=self.risk_weight, alpha=self.risk_alpha,
            score_chunk=self.score_chunk, distance=self.distance, strength=self.strength,
            score='sum logP including EOS; excluding ignore labels',
            checkpointing='whole-view non-reentrant', auxiliary_dropout=False,
            extra_parameters=0, known_control=True)

    def save_pretrained(self, save_directory, **kwargs):
        result = super().save_pretrained(save_directory, **kwargs)
        (Path(save_directory) / 'sequence_risk_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2) + '\n', encoding='utf-8')
        return result
