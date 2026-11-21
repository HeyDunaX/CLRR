"""Frozen CPU correctness gate for the four-arm relational development screen."""
import unittest
from experiment_runner import engine as worker
from test_relational_model_v2 import RelationalModelGate, model_for, training_batch
import test_relational_geometry_v2


def main():
    worker.validate_environment(gpu=False)
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(RelationalModelGate),
        unittest.defaultTestLoader.loadTestsFromModule(test_relational_geometry_v2)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('Relational correctness failed')
    cases = []
    for spec in worker.plan()['runs']:
        model = model_for(spec['method'])
        model.train()(**training_batch()).loss.backward()
        cases.append(dict(id=spec['id'],passed=True,objective=model.method_metadata()))
    worker.write(worker.OUT / 'CPU_GATE.json',dict(passed=True,cases=cases,
        scientific_tests=result.testsRun,novelty_established=False,
        bundle_sha256=worker.sha256(worker.OUT / 'bundle_manifest.json')))


if __name__ == '__main__':
    main()
