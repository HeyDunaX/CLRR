"""Verify the KL chain-rule decomposition and its two-sided logit gradients."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    results = []
    for vocabulary in (2, 17):
        for scale in (1., 10.):
            left = (torch.randn(128, vocabulary, dtype=torch.float64)*scale).requires_grad_()
            right = (torch.randn_like(left)*scale).requires_grad_()
            labels = torch.randint(vocabulary, (128, 1))
            logp, logq = left.log_softmax(-1), right.log_softmax(-1)
            nongold = torch.ones_like(left, dtype=torch.bool).scatter_(1, labels, False)
            # Work in log-space; never compute log(1 - p_gold).
            logrp = logp.masked_fill(~nongold, -torch.inf).logsumexp(-1, keepdim=True)
            logrq = logq.masked_fill(~nongold, -torch.inf).logsumexp(-1, keepdim=True)
            binaryp = torch.cat((logp.gather(1, labels), logrp), -1)
            binaryq = torch.cat((logq.gather(1, labels), logrq), -1)
            gold = .5*((binaryp.exp()-binaryq.exp())*(binaryp-binaryq)).sum(-1)
            # Keep gradients through the remaining probability masses too.
            diff = ((logp-logrp)-(logq-logrq)).masked_fill(~nongold, 0.)
            rest = .5*((logp.exp()-logq.exp())*diff).sum(-1)
            full = .5*((logp.exp()-logq.exp())*(logp-logq)).sum(-1)
            grad_full = torch.autograd.grad(full.mean(), (left, right), retain_graph=True)
            grad_parts = torch.autograd.grad((gold+rest).mean(), (left, right))
            value_error = (full-gold-rest).abs().max().item()
            gradient_error = max((a-b).abs().max().item() for a, b in zip(grad_full, grad_parts))
            assert value_error < 1e-10 and gradient_error < 1e-10
            assert gold.min().item() >= -1e-12 and rest.min().item() >= -1e-12
            results.append(dict(vocabulary=vocabulary, scale=scale,
                                max_value_error=value_error, max_gradient_error=gradient_error))
    artifact = dict(status='PASS', device='cpu', dtype='float64', seed=42, cases=results,
                    scope='Algebra and both-argument gradients only; not model/GPU/mixed-precision gates.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()
