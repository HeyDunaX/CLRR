"""CPU gate for the exact virtual-order training bundle."""
import unittest
import torch
from experiment_runner import engine as worker
from test_virtual_corrective_order import VirtualOrderTests
from test_virtual_order_training import VirtualTrainingTests


def main():
    torch.set_num_threads(2)
    worker.validate_environment(gpu=False)
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
        for cls in (VirtualOrderTests,VirtualTrainingTests))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('Virtual-order CPU gate failed')
    worker.write(worker.OUT/'CPU_GATE.json',dict(passed=True,
        cases=[dict(id=s['id'],passed=True) for s in worker.plan()['runs']],
        tests=result.testsRun,bundle_sha256=worker.sha256(worker.OUT/'bundle_manifest.json')))
