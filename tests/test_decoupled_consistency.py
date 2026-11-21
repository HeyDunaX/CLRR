"""Loss decomposition, gradient replay and persistence for G/R/W."""
import copy
import tempfile
import unittest
import importlib.util
from pathlib import Path

import torch
from transformers import MBartForConditionalGeneration
from test_alignment_residual import native, batch
from methods.decoupled_consistency import DecoupledConsistencyMT, METHODS
from methods.decoupled_consistency.v1.model import consistency_loss
from methods.paired_route.v1.model import paired_loss


class DecoupledConsistencyGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_chain_rule_loss_both_gradients_and_padding(self):
        torch.manual_seed(12)
        a = torch.randn(2, 4, 40, requires_grad=True)
        b = torch.randn_like(a, requires_grad=True)
        y = batch()['labels']
        actual, ce, gold, rest = consistency_loss(a, b, y, 'full', 1.25, 3)
        reference, _, kl = paired_loss(a, b, y, 1.25, 3)
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(gold + rest, kl)
        reference.backward()
        expected = (a.grad.clone(), b.grad.clone())
        a.grad = b.grad = None
        actual.backward()
        gradients = (a.grad.clone(), b.grad.clone())
        for x, z in zip(gradients, expected):
            torch.testing.assert_close(x, z, rtol=3e-5, atol=3e-7)
            self.assertTrue(torch.equal(x[y.eq(-100)], torch.zeros_like(x[y.eq(-100)])))
        for component in ('gold', 'rest'):
            a.grad = b.grad = None
            value, _, _, _ = consistency_loss(a, b, y, component, 2.5, 2)
            value.backward()
            grads = (a.grad, b.grad)
            self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))

    def test_extremes_binary_and_large_vocabulary(self):
        for vocabulary in (2, 40, 250054):
            a = torch.full((1, 1, vocabulary), -100., requires_grad=True)
            b = torch.full_like(a, -99., requires_grad=True)
            with torch.no_grad():
                a[..., 0] = 100.
                b[..., 0] = 99.
            value, _, gold, rest = consistency_loss(a, b, torch.tensor([[0]]), 'full', 2.5)
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())
            if vocabulary == 2:
                self.assertEqual(rest.item(), 0.)
        with self.assertRaises(ValueError):
            consistency_loss(a, b, torch.tensor([[-100]]), 'gold', 2.5)

    def test_checkpoint_replay_dropout_accumulation(self):
        for method in METHODS:
            plain = DecoupledConsistencyMT(native(.15), method)
            replay = DecoupledConsistencyMT(copy.deepcopy(plain.base_model), method)
            replay.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
            for model in (plain, replay):
                model.train()
                torch.manual_seed(77)
                for _ in range(2):
                    (model(**batch()).loss / 2).backward()
                self.assertGreater(model.last_diagnostics['kl'], 0.)
                self.assertFalse(model._checkpoint_context)
                self.assertTrue(all(not c for c in model._cache.values()))
            for (name, p), (_, q) in zip(plain.named_parameters(), replay.named_parameters()):
                self.assertEqual(p.grad is None, q.grad is None, name)
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, q.grad, rtol=5e-4, atol=3e-6, msg=name)

    def test_native_inference_and_persistence(self):
        core = native().eval()
        inputs = {k:v for k,v in batch().items() if k != 'labels'}
        for method in METHODS:
            model = DecoupledConsistencyMT(copy.deepcopy(core), method).eval()
            self.assertEqual(model.extra_parameter_count(), 0)
            torch.testing.assert_close(model(**batch()).logits, core(**batch()).logits, rtol=0, atol=0)
            with tempfile.TemporaryDirectory() as folder:
                model.save_pretrained(folder)
                restored = DecoupledConsistencyMT(MBartForConditionalGeneration.from_pretrained(folder), method).eval()
                restored.verify_checkpoint_metadata(folder)
                torch.testing.assert_close(restored(**batch()).logits, model(**batch()).logits, rtol=0, atol=0)
                with torch.no_grad():
                    self.assertTrue(torch.equal(model.generate(**inputs, max_new_tokens=4, use_cache=True),
                                                restored.generate(**inputs, max_new_tokens=4, use_cache=False)))
                wrong = next(m for m in METHODS if m != method)
                with self.assertRaises(ValueError):
                    DecoupledConsistencyMT(native(), wrong).verify_checkpoint_metadata(folder)

    def test_frozen_probe_gradient_statistics(self):
        path = Path(__file__).resolve().parents[1]/'scripts/analysis/measure_consistency_checkpoint.py'
        spec = importlib.util.spec_from_file_location('consistency_probe', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a, b = torch.randn(2,4,40), torch.randn(2,4,40)
        measured = module.measure_logits(a, b, batch()['labels'])
        self.assertEqual(measured['tokens'], 7)
        self.assertAlmostEqual(measured['weak_full_grad_sq'], measured['full_grad_sq']/4, places=6)
        self.assertAlmostEqual(measured['weak_full_ce_dot'], measured['full_ce_dot']/2, places=6)
        for key in ('gold_grad_sq','rest_grad_sq','ce_grad_sq'):
            self.assertGreater(measured[key], 0.)


if __name__ == '__main__':
    unittest.main()
