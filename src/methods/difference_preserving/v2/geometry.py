"""All-pair bilingual geometry objectives with detached target anchors.

The distance objective adapts RKD-D, Eq.5-8, to source/target encoder views.
This is not an exact reproduction of its fixed teacher/student experiments.
"""
import torch
import torch.nn.functional as F


def relational_loss(source, target, *, objective, eligible_pairs=None, eps=1e-6):
    """Return (loss, diagnostics) for unordered within-microbatch pairs.

    Caller supplies content-pooled, unnormalized [B,D] vectors. Eligibility
    must exclude identical source/target token sequences after preprocessing.
    Singleton, empty eligibility and target-degenerate pairs yield zero.
    Source collapse is retained in the loss, not filtered as invalid.
    """
    if objective not in ('direction', 'rkd-distance'):
        raise ValueError('Unknown relational objective')
    if eps <= 0 or not torch.isfinite(torch.tensor(eps)):
        raise ValueError('eps must be positive and finite')
    if (source.ndim != 2 or source.shape != target.shape
            or source.device != target.device or source.shape[1] == 0):
        raise ValueError('Expected equal [B,D] vectors on the same device')
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError('Non-finite representation')
    batch = source.shape[0]
    if eligible_pairs is None:
        raise ValueError('Explicit pair eligibility is required')
    if (eligible_pairs.shape != (batch, batch)
            or eligible_pairs.dtype != torch.bool
            or eligible_pairs.device != source.device
            or not torch.equal(eligible_pairs, eligible_pairs.T)
            or eligible_pairs.diag().any()):
        raise ValueError('Eligibility must be symmetric boolean [B,B], diagonal false')
    i, j = torch.triu_indices(batch, batch, offset=1, device=source.device)
    dx = source.float()[i] - source.float()[j]
    dy = target.detach().float()[i] - target.detach().float()[j]
    sx, ty = dx.norm(dim=-1), dy.norm(dim=-1)
    valid = eligible_pairs[i, j] & (ty > eps)
    count = int(valid.sum())
    stats = dict(eligible_pairs=int(eligible_pairs[i, j].sum()), valid_pairs=count,
                 small_source_pairs=int(((sx <= eps) & valid).sum()),
                 insufficient_distance_pairs=count < 2,
                 degenerate_distance_normalizer=False)
    if not count:
        return source.sum() * 0., stats
    dx, dy, sx, ty = dx[valid], dy[valid], sx[valid], ty[valid]
    if objective == 'direction':
        loss = (1. - F.cosine_similarity(dx, dy, dim=-1, eps=eps)).mean()
    else:
        # RKD-D uses separate student/teacher mean distances. Do not detach
        # the source normalizer: that would change the objective's gradient.
        mean_source = sx.mean()
        stats['degenerate_distance_normalizer'] = bool(mean_source <= eps)
        loss = F.smooth_l1_loss(sx / mean_source.clamp_min(eps), ty / ty.mean())
    return loss, stats
