"""CPU admission for the exact frozen corrective-order bundle."""
import unittest
from experiment_runner import engine
from test_corrective_order import CorrectiveOrderTests


def main():
    engine.validate_environment(gpu=False)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CorrectiveOrderTests))
    if not result.wasSuccessful():
        raise RuntimeError('Corrective-order CPU gate failed')
    engine.write(engine.OUT/'CPU_GATE.json', dict(passed=True,
        cases=[dict(id=s['id'], passed=True) for s in engine.plan()['runs']],
        scientific_tests=result.testsRun,
        bundle_sha256=engine.sha256(engine.OUT/'bundle_manifest.json')))
