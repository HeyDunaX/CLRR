"""CPU admission for the reviewed, frozen two-arm transition pilot."""
import unittest
from experiment_runner import engine as worker
from test_transition_lsr import TransitionGate, model_for, batch


def main():
    worker.validate_environment(gpu=False)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TransitionGate))
    if not result.wasSuccessful():
        raise RuntimeError('Transition LSR correctness failed')
    cases=[]
    for spec in worker.plan()['runs']:
        model=model_for(spec['method'])
        model.train()(**batch()).loss.backward()
        cases.append(dict(id=spec['id'],passed=True,objective=model.method_metadata()))
    worker.write(worker.OUT/'CPU_GATE.json',dict(passed=True,cases=cases,
        scientific_tests=result.testsRun,novelty_certified=False,
        bundle_sha256=worker.sha256(worker.OUT/'bundle_manifest.json')))
