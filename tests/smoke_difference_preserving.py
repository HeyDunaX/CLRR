"""Frozen CPU correctness only; scientific and GPU admission remain separate."""
import copy
import unittest

import torch
from experiment_runner import engine as worker
from test_alignment_residual import native, batch
from test_difference_preserving import DifferenceGate, training_batch


def main():
    worker.validate_environment(gpu=False)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DifferenceGate))
    if not result.wasSuccessful():
        raise RuntimeError('Difference correctness failed')
    cases = []
    for spec in worker.plan()['runs']:
        model, metadata = worker.wrap_model(copy.deepcopy(native()), spec)
        model.special_token_ids = (1, 2)
        data = training_batch()
        if 'candidate' in spec['method']:
            data['peer_input_ids'] = data['input_ids'].clone()
            data['peer_input_ids'][:, 0] += 1
        model.train()(**data).loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        cases.append(dict(id=spec['id'], passed=True, parameters=metadata,
                          scope='tiny random model and synthetic peers only'))
    worker.write(worker.OUT / 'CPU_GATE.json', dict(passed=True, cases=cases,
        scientific_tests=result.testsRun, scientific_admission=False,
        bundle_sha256=worker.sha256(worker.OUT / 'bundle_manifest.json')))


if __name__ == '__main__':
    main()
