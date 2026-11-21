"""Ordered gold-prefix score consistency; no detached probability arguments."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ...decoupled_consistency.v1.model import consistency_loss


def gold_differences(left, right, labels, chunk_size=32):
    if left.shape != right.shape or left.shape[:-1] != labels.shape:
        raise ValueError('Expected aligned paired logits and labels')
    mask = labels.ne(-100)
    if not mask.any() or (mask & ((labels < 0) | (labels >= left.shape[-1]))).any():
        raise ValueError('Invalid or empty supervised labels')
    flat_y = labels.flatten()
    a, b = left.flatten(0, 1), right.flatten(0, 1)
    parts = []
    def gather(x, z, y):
        valid = y.ne(-100)
        safe = y.masked_fill(~valid, 0).unsqueeze(-1)
        return ((F.log_softmax(x.float(), -1).gather(-1, safe)
                 - F.log_softmax(z.float(), -1).gather(-1, safe)).squeeze(-1) * valid)
    for start in range(0, flat_y.numel(), chunk_size):
        args = a[start:start+chunk_size], b[start:start+chunk_size], flat_y[start:start+chunk_size]
        if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
            part = checkpoint(gather, *args, use_reentrant=True, preserve_rng_state=True)
        else:
            part = gather(*args)
        parts.append(part)
    return torch.cat(parts).reshape_as(labels)


def path_statistics(differences, mask):
    """Per-sentence P/D/E; valid positions are logically packed before summation."""
    if differences.shape != mask.shape or differences.ndim != 2:
        raise ValueError('Expected aligned [batch,time] scores and masks')
    lengths = mask.sum(-1)
    if (lengths == 0).any():
        raise ValueError('Each sentence must have supervised tokens')
    a = differences.float().masked_fill(~mask, 0.)
    ranks = mask.cumsum(-1).clamp_min(1).float()
    inverse = mask / ranks
    diagonal = inverse.flip(-1).cumsum(-1).flip(-1) / lengths[:, None]
    prefix = (a.cumsum(-1).square() * inverse).sum(-1) / lengths
    diag = (a.square() * diagonal).sum(-1)
    endpoint = a.sum(-1).square() / lengths
    return dict(prefix=prefix, diagonal=diag, endpoint=endpoint, cross=prefix-diag)


def prefix_loss(left, right, labels, mode='prefix', weight=.10, beta=1.25):
    if mode not in ('prefix', 'diagonal', 'endpoint', 'disabled'):
        raise ValueError('Unknown path control')
    if weight < 0 or beta < 0 or not torch.isfinite(torch.tensor([weight, beta])).all():
        raise ValueError('Weights must be finite and nonnegative')
    # Retain W's exact token reduction and full-vocabulary chain decomposition.
    base, ce, gold, rest = consistency_loss(left, right, labels, 'full', beta)
    stats = path_statistics(gold_differences(left, right, labels), labels.ne(-100))
    auxiliary = ce * 0. if mode == 'disabled' else stats[mode].mean()
    return base + weight * auxiliary, ce, gold + rest, auxiliary, stats
