"""Frozen G/R/W CPU admission before GPU gates."""
import unittest
from experiment_runner import engine as worker
from test_decoupled_consistency import DecoupledConsistencyGate


def main():
    worker.validate_environment(gpu=False)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DecoupledConsistencyGate))
    if not result.wasSuccessful():
        raise RuntimeError('G/R/W correctness failed')
    worker.write(worker.OUT/'CPU_GATE.json', dict(passed=True,
        cases=[dict(id=s['id'], passed=True) for s in worker.plan()['runs']],
        scientific_tests=result.testsRun, novelty_certified=False,
        bundle_sha256=worker.sha256(worker.OUT/'bundle_manifest.json')))
