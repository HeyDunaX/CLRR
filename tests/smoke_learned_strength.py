"""CPU admission for the author's two approved engineering pilots."""
import unittest
from experiment_runner import engine as worker
from test_learned_strength import StrengthGate, model_for, batch


def main():
    worker.validate_environment(gpu=False)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(StrengthGate))
    if not result.wasSuccessful():
        raise RuntimeError('Learned strength correctness failed')
    cases = []
    for spec in worker.plan()['runs']:
        model = model_for(spec['method'])
        model.train()(**batch()).loss.backward()
        cases.append(dict(id=spec['id'], passed=True))
    worker.write(worker.OUT/'CPU_GATE.json', dict(passed=True, cases=cases,
        scientific_tests=result.testsRun, novelty_certified=False,
        bundle_sha256=worker.sha256(worker.OUT/'bundle_manifest.json')))
