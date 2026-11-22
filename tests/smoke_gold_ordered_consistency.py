"""CPU gate for the frozen Gold-Ordered Consistency (GOC) and matched controls bundle."""
import sys
import gc
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / 'src'), str(ROOT / 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from transformers import MBartForConditionalGeneration

from experiment_runner import engine as worker
from test_alignment_residual import native, batch
from test_gold_ordered_consistency import GoldOrderedConsistencyGate


def main():
    torch.set_num_threads(2)
    worker.validate_environment(gpu=False)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GoldOrderedConsistencyGate))
    if not result.wasSuccessful():
        raise RuntimeError('Gold-ordered consistency CPU gate failed')
    cases = []
    for spec in worker.plan()['runs']:
        torch.manual_seed(spec['seed'])
        model, metadata = worker.wrap_model(native(.15), spec)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
        model.train()
        model(**batch()).loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.eval()
        with torch.no_grad():
            expected = model(**batch()).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, state_dict=model.state_dict())
            base = MBartForConditionalGeneration.from_pretrained(Path(directory))
            restored, _ = worker.wrap_model(base, spec)
            if hasattr(restored, 'verify_checkpoint_metadata'):
                restored.verify_checkpoint_metadata(directory)
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(**batch()).logits, expected)
            # Both wrappers can retain mmap-backed checkpoint tensors on NFS.
            del restored, base
            gc.collect()
        cases.append(dict(id=spec['id'], passed=True, finite_gradients=True,
                          save_reload_logits_match=True, parameters=metadata))
    worker.write(worker.OUT / 'CPU_GATE.json', dict(
        passed=True,
        cases=cases,
        scientific_tests=result.testsRun,
        novelty_certified=False,
        bundle_sha256=worker.sha256(worker.OUT / 'bundle_manifest.json'),
    ))


if __name__ == '__main__':
    main()
