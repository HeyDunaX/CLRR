"""Read-only numerical diagnostics; never changes activations or optimizer state."""
from contextlib import contextmanager
import math

import numpy as np
import torch

from .model import compose_paths, symmetric_kl


def relu_counterexample():
    # A fixed lattice approximates a population with nonzero density at a kink.
    # The statement concerns these resolved finite scales, not eta below spacing.
    theta = ((torch.arange(20000, dtype=torch.float64)+.5)/20000-.5).requires_grad_()
    def loss(p, seed):
        return p['w'].relu().sum() if seed == 0 else -p['w'].sum()
    records = []
    for eta in (.002, .004, .008):
        _, ab, ba, _, _ = compose_paths({'w': theta}, loss, eta, (0, 1))
        with torch.no_grad():
            left = torch.stack((ab['w'], torch.zeros_like(theta)), -1)[:, None]
            right = torch.stack((ba['w'], torch.zeros_like(theta)), -1)[:, None]
            raw = float(symmetric_kl(left, right, torch.zeros(len(theta), 1, dtype=torch.long)))
            records.append(dict(eta=eta, parameter_order_distance=float((ab['w']-ba['w']).norm()),
                raw_probe_skl=raw, auxiliary=raw/eta**4))
    exponents = [dict(endpoint=math.log(b['parameter_order_distance']/a['parameter_order_distance'])/math.log(b['eta']/a['eta']),
        skl=math.log(b['raw_probe_skl']/a['raw_probe_skl'])/math.log(b['eta']/a['eta']))
        for a, b in zip(records, records[1:])]
    return dict(dtype='float64', optimizer_steps=0, records=records, exponents=exponents,
        scope='counterexample to mandatory smooth scaling on resolved ReLU crossings; not a universal ReLU exponent')


class ActivationAudit:
    """Count fc1 sign changes for identical masks before/after a virtual step.

    Hooks exist only for the original forward, not subsequent checkpoint replay.
    Store packed CPU signs, omit padding, and never retain activations or grads.
    """
    def __init__(self, backbone, batch):
        if backbone.config.activation_function != 'relu':
            raise ValueError('This activation-boundary audit is specific to ReLU')
        self.modules = [(n, m) for n, m in backbone.named_modules() if n.endswith('.fc1')]
        self.batch, self.baseline, self.records = batch, {}, []

    @contextmanager
    def capture(self, seed, initial):
        seen, handles = set(), []
        def hook(name):
            def record(module, inputs, output):
                if name in seen:
                    return
                seen.add(name)
                valid = (self.batch['attention_mask'].bool() if '.encoder.' in '.'+name
                         else self.batch['labels'].ne(-100))
                if output.shape[:2] != valid.shape:
                    raise ValueError('Activation audit mask shape mismatch')
                signs = output.detach()[valid].gt(0).flatten().cpu().numpy()
                packed = np.packbits(signs)
                key = (seed, name)
                if initial:
                    self.baseline[key] = (packed, len(signs))
                else:
                    reference, count = self.baseline[key]
                    if len(signs) != count:
                        raise ValueError('Activation audit changed valid positions')
                    crossed = int(np.unpackbits(np.bitwise_xor(reference, packed))[:count].sum())
                    self.records.append(dict(layer=name, seed=seed, crossings=crossed,
                        positions=count, fraction=crossed/max(count, 1)))
            return record
        try:
            for name, module in self.modules:
                handles.append(module.register_forward_hook(hook(name)))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def run(self, model, batch, eta, seeds=(101, 211)):
        state = dict(model.base_model.named_parameters())
        def loss(parameters, seed):
            with self.capture(seed, parameters is state):
                return model._evaluate(parameters, batch, seed, True, scalar=True)
        with model.offload_context():
            ce, ab, ba, pa, pb = compose_paths(state, loss, eta, seeds)
        with torch.no_grad():
            distance = float(sum((ab[k].double()-ba[k].double()).square().sum() for k in state).sqrt())
        total = sum(r['positions'] for r in self.records)
        crossings = sum(r['crossings'] for r in self.records)
        if len(self.records) != 2*len(self.modules):
            raise ValueError('Incomplete activation-boundary audit')
        return dict(eta=eta, activation='relu', dropout_seeds=list(seeds),
            layers=self.records, crossings=crossings, positions=total,
            fraction=crossings/max(total, 1), ce=float(ce.detach()),
            parameter_order_distance=distance, optimizer_steps=0, test_used=False,
            scope='same-seed base vs first virtual state; correlation, not isolated causal proof')
