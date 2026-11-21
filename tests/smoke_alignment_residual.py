"""Bundle-level CPU receipt plus per-arm engine save/load contract."""
import tempfile
import unittest
from pathlib import Path

import torch
from test_alignment_residual import AlignmentResidualGate, native, batch
from test_alignment_residual_controls import ResidualCEControlsGate
from test_learned_donor import LearnedDonorGate
from test_matched_residual_baseline import MatchedBaselineGate
from test_checkpoint_selection import CheckpointSelectionGate
from experiment_runner import engine as worker


def main():
    worker.validate_environment(gpu=False)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AlignmentResidualGate)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ResidualCEControlsGate))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(LearnedDonorGate))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(MatchedBaselineGate))
    if worker.protocol().get('retain_secondary_chrf', False):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(CheckpointSelectionGate))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    cases = []
    for spec in worker.plan()['runs']:
        torch.manual_seed(spec['seed'])
        model, metadata = worker.wrap_model(native(), spec)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
        model(**batch()).loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.eval()
        with torch.no_grad():
            expected = model(**batch()).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, state_dict=model.state_dict())
            restored = type(model).from_pretrained(Path(directory)).eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(**batch()).logits, expected)
        cases.append(dict(id=spec['id'], passed=True, finite_gradients=True,
                          save_reload_logits_match=True, parameters=metadata))
    worker.write(worker.OUT / 'CPU_GATE.json', dict(passed=True, cases=cases,
        scientific_tests=result.testsRun, bundle_sha256=worker.sha256(worker.OUT / 'bundle_manifest.json')))


if __name__ == '__main__':
    main()
