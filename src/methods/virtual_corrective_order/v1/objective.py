"""Explicit head-coordinate updates; no persistent adapters or backbone unroll."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _run(fn, args, recompute):
    return checkpoint(fn, *args, use_reentrant=False) if recompute else fn(*args)


def select_tokens(hidden, labels, rows):
    h, y = hidden[rows].flatten(0, 1), labels[rows].flatten()
    keep = y.ne(-100)
    if not keep.any():
        raise ValueError('No supervised tokens in partition')
    return h[keep], y[keep]


def logits(h, weight, bias, basis, projected, matrix):
    return F.linear(h, weight, bias) + F.linear((h @ basis) @ matrix.T, projected)


def gradient(hidden, labels, weight, bias, basis, projected, matrix, chunk=16, recompute=False):
    """Exact mean CE gradient w.r.t. A, retaining the outer graph into h/E."""
    def part(h, y, w, b, B, S, A):
        probabilities = logits(h, w, b, B, S, A).softmax(-1)
        return ((probabilities @ S - S[y]).T @ (h @ B))
    result = matrix * 0
    for start in range(0, len(labels), chunk):
        args = (hidden[start:start+chunk], labels[start:start+chunk], weight,
                bias, basis, projected, matrix)
        result = result + _run(part, args, recompute)
    return result / len(labels)


def endpoints(ha, hb, labels, support_rows, weight, bias, basis, eta,
              chunk=16, recompute=False):
    if ha.shape != hb.shape or ha.shape[:2] != labels.shape:
        raise ValueError('Expected matching B,T,D features and B,T labels')
    if eta <= 0:
        raise ValueError('Virtual eta must be positive')
    a, y = select_tokens(ha, labels, support_rows)
    b, _ = select_tokens(hb, labels, support_rows)
    projected = weight @ basis
    zero = torch.zeros(basis.shape[1], basis.shape[1], device=ha.device, dtype=ha.dtype)
    def step(h, A):
        return A - eta * gradient(h, y, weight, bias, basis, projected, A, chunk, recompute)
    ab, ba = step(b, step(a, zero)), step(a, step(b, zero))
    return ab, ba, projected


def probe_penalty(hidden, labels, rows, weight, bias, basis, projected, left, right,
                  chunk=16, recompute=False):
    h, y = select_tokens(hidden, labels, rows)
    def part(h, w, b, B, S, A, C):
        lp = logits(h,w,b,B,S,A).log_softmax(-1)
        lq = logits(h,w,b,B,S,C).log_softmax(-1)
        return .5*((lp.exp()-lq.exp())*(lp-lq)).sum()
    result = left.sum()*0
    for start in range(0,len(y),chunk):
        result = result + _run(part,(h[start:start+chunk],weight,bias,basis,projected,left,right),recompute)
    return result/len(y)


def order_penalty(ha, hb, labels, support_rows, probe_rows, weight, bias, basis,
                  eta, chunk=16, recompute=False):
    if set(support_rows) & set(probe_rows):
        raise ValueError('Support/probe rows must be disjoint')
    ab, ba, S = endpoints(ha,hb,labels,support_rows,weight,bias,basis,eta,chunk,recompute)
    penalty = sum(probe_penalty(h,labels,probe_rows,weight,bias,basis,S,ab,ba,chunk,recompute)
                  for h in (ha,hb))*.5
    return penalty, ab, ba


def paired_ce_kl(ha,hb,labels,weight,bias,chunk=16,recompute=False,include_kl=True):
    rows=list(range(len(labels)))
    a,y=select_tokens(ha,labels,rows)
    b,_=select_tokens(hb,labels,rows)
    def part(a,b,y,w,bias):
        za,zb=F.linear(a,w,bias),F.linear(b,w,bias)
        ce=.5*(F.cross_entropy(za,y,reduction='sum')+F.cross_entropy(zb,y,reduction='sum'))
        kl=ce*0
        if include_kl:
            la,lb=za.log_softmax(-1),zb.log_softmax(-1)
            kl=.5*((la.exp()-lb.exp())*(la-lb)).sum()
        return torch.stack((ce,kl))
    result=weight.new_zeros(2)
    for start in range(0,len(y),chunk):
        result=result+_run(part,(a[start:start+chunk],b[start:start+chunk],y[start:start+chunk],weight,bias),recompute)
    return result/len(y)
