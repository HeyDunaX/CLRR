"""Finite-candidate expected risk, following Shen et al. (ACL 2016), Eq. 13.

This numerical prototype does not sample candidates, compute translation
metrics, or train a model. Callers supply sequence SUM log probabilities,
deduplicated candidates, and fixed reference-based costs. Candidate generation
is outside autograd; current-model candidate scoring must remain differentiable.
"""

import math

import torch


def candidate_set_risk(
    sequence_log_probs: torch.Tensor,
    costs: torch.Tensor,
    valid: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Return mean source-level expected risk over a masked [source, candidate] set.

    At least one candidate per source is required. Invalid padding may contain
    NaN; it receives zero probability and zero gradient. Costs are detached.
    Different candidate counts do not change the relative weight of sources.
    Singleton/equal-cost pools are accepted but supply no learning signal.
    """
    if sequence_log_probs.ndim != 2 or not sequence_log_probs.numel():
        raise ValueError("sequence_log_probs must be a nonempty [source, candidate] tensor")
    if costs.shape != sequence_log_probs.shape or valid.shape != costs.shape:
        raise ValueError("scores, costs and valid must have identical shapes")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    if not sequence_log_probs.is_floating_point() or not costs.is_floating_point():
        raise ValueError("scores and costs must be floating point")
    if not (sequence_log_probs.device == costs.device == valid.device):
        raise ValueError("scores, costs and valid must be on the same device")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not valid.any(dim=1).all():
        raise ValueError("every source needs at least one valid candidate")
    if not torch.isfinite(sequence_log_probs[valid]).all() or not torch.isfinite(costs[valid]).all():
        raise ValueError("valid scores and costs must be finite")
    # Softmax in at least FP32; retain FP64 for numerical/gradient verification.
    dtype = torch.float64 if sequence_log_probs.dtype == torch.float64 else torch.float32
    scores = sequence_log_probs.to(dtype).masked_fill(~valid, -torch.inf)
    # Center before scaling so large common offsets do not affect the posterior.
    centered = scores - scores.max(dim=1, keepdim=True).values
    posterior = torch.softmax(alpha * centered, dim=1)
    fixed_costs = costs.detach().to(dtype).masked_fill(~valid, 0)
    return (posterior * fixed_costs).sum(dim=1).mean()
