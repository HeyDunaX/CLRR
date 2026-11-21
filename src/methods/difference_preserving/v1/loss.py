"""One-way difference alignment with an explicit degenerate-target policy."""
import math

import torch
import torch.nn.functional as F


def difference_alignment_loss(source, peer_source, target, peer_target,
                              eligible=None, eps=1e-6):
    """Mean over eligible nonzero target differences, with detached targets.

    Empty selections return a differentiable zero. Source collapse is NOT
    masked away or claimed to be prevented. This objective is a prototype,
    not an implementation of Park et al.'s distance/angle RKD objectives.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be positive and finite')
    values = (source, peer_source, target, peer_target)
    if source.ndim != 2 or any(v.shape != source.shape for v in values):
        raise ValueError('Expected four equal [batch, features] tensors')
    if any(not torch.isfinite(v).all() for v in values):
        raise ValueError('Non-finite representations')
    dx = source.float() - peer_source.float()
    dy = target.detach().float() - peer_target.detach().float()
    valid = dy.norm(dim=-1) > eps
    if eligible is not None:
        if eligible.shape != (source.shape[0],) or eligible.dtype != torch.bool:
            raise ValueError('eligible must be a boolean batch vector')
        valid &= eligible.to(valid.device)
    if not valid.any():
        return (source.sum() + peer_source.sum()) * 0.
    return (1. - F.cosine_similarity(dx[valid], dy[valid], dim=-1, eps=eps)).mean()
