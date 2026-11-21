"""Exact finite-table source/dropout information; mathematical prototype only.

No training wrapper, registry entry, data loading, or optimizer is provided.
Inputs end in [dropout_view=2, source_choice=2, vocabulary]. The caller must
replay each view's random masks across the two sources for a subnetwork
interpretation. Arbitrary four distributions still satisfy the table algebra.
"""
import math

import torch
import torch.nn.functional as F


def information_components(logits):
    """Return unreduced per-history information quantities, in natural units.

    J(m,s,v) = softmax(logits[m,s,:])[v] / 4.
    B = I(M;V|S) = A + C, A = I(M;V), C = I(M;S|V).
    All four cells and all mixture weights remain differentiable. Float64 is
    retained for mathematical checks; lower precision is promoted to float32.
    No clipping hides small negative round-off or changes the derivatives.
    """
    if logits.ndim < 3 or logits.shape[-3:-1] != (2, 2) or logits.shape[-1] < 2:
        raise ValueError('Expected [..., dropout=2, source=2, vocabulary>=2]')
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError('Expected finite floating-point logits')
    values = logits if logits.dtype == torch.float64 else logits.float()
    log_joint = F.log_softmax(values, dim=-1) - math.log(4.)
    log_mv = log_joint.logsumexp(dim=-2, keepdim=True)
    log_sv = log_joint.logsumexp(dim=-3, keepdim=True)
    log_v = log_joint.logsumexp(dim=(-3, -2), keepdim=True)
    joint = log_joint.exp()
    c_by_token_type = (joint * (log_joint + log_v - log_mv - log_sv)).sum(dim=(-3, -2))
    a = (log_mv.exp() * (log_mv - log_v + math.log(2.))).sum(dim=(-3, -2, -1))
    b = (joint * (log_joint - log_sv + math.log(2.))).sum(dim=(-3, -2, -1))
    source_information = (log_sv.exp() * (log_sv - log_v + math.log(2.))).sum(dim=(-3, -2, -1))
    return dict(source_attribution=c_by_token_type.sum(dim=-1),
                marginal_dropout=a, joint_dropout=b,
                source_information=source_information,
                attribution_by_token_type=c_by_token_type)
