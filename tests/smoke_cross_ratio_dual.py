"""CPU gate for the frozen CR-Dual and matched controls bundle."""
import gc
import sys
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
from test_cross_ratio_dual import CrossRatioDualGate, tiny_batch, tiny_mbart


def main():
    torch.set_num_threads(2)
    worker.validate_environment(gpu=False)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CrossRatioDualGate)
    )
    if not result.wasSuccessful():
        raise RuntimeError('CR-Dual CPU gate failed')
    cases = []
    for spec in worker.plan()['runs']:
        torch.manual_seed(spec['seed'])
        model, metadata = worker.wrap_model(tiny_mbart(dropout=0.15, seed=spec['seed']), spec)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
        model.train()
        sample = tiny_batch(batch_size=4, seed=spec['seed'])
        model(**sample).loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.eval()
        with torch.no_grad():
            expected = model(**sample).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, state_dict=model.state_dict())
            base = MBartForConditionalGeneration.from_pretrained(Path(directory))
            restored, _ = worker.wrap_model(base, spec)
            if hasattr(restored, 'verify_checkpoint_metadata'):
                restored.verify_checkpoint_metadata(directory)
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(**sample).logits, expected)
            del restored, base
            gc.collect()
        cases.append(dict(
            id=spec['id'],
            passed=True,
            finite_gradients=True,
            save_reload_logits_match=True,
            parameters=metadata,
        ))
    worker.write(worker.OUT / 'CPU_GATE.json', dict(
        passed=True,
        cases=cases,
        scientific_tests=result.testsRun,
        novelty_certified=False,
        bundle_sha256=worker.sha256(worker.OUT / 'bundle_manifest.json'),
    ))


if __name__ == '__main__':
    main()
