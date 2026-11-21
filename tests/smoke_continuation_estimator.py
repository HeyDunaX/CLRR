"""CPU receipt for the exact frozen estimator-pilot bundle."""
import unittest
from experiment_runner import engine as worker
from test_continuation_estimator import EstimatorGate


def main():
    worker.validate_environment(gpu=False)
    result=unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(EstimatorGate))
    if not result.wasSuccessful():
        raise RuntimeError('CMC estimator CPU gate failed')
    worker.write(worker.OUT/'CPU_GATE.json',dict(passed=True,
        cases=[dict(id=s['id'],passed=True) for s in worker.plan()['runs']],
        scientific_tests=result.testsRun,novelty_certified=False,
        bundle_sha256=worker.sha256(worker.OUT/'bundle_manifest.json')))
