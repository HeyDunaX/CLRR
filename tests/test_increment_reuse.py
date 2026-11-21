"""Actual mBART replay, gradients, source masks and persistence for RO/DW/RW."""
import copy
import tempfile
import unittest

import torch
from test_alignment_residual import native, batch
from methods.increment_reuse import IncrementReuseMT
from methods.decoupled_consistency import DecoupledConsistencyMT


def wrapped(base, variant):
    return IncrementReuseMT(base, variant, 0, 2, [1, 2])


class IncrementReuseGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_alpha_matches_locked_w_loss_and_backbone_gradients(self):
        for variant in ('ro', 'dw', 'rw'):
            torch.manual_seed(9)
            core = native(.15)
            control = DecoupledConsistencyMT(copy.deepcopy(core), 'weak-full-consistency-v1')
            candidate = wrapped(core, variant)
            results = []
            for model in (control, candidate):
                model.train()
                torch.manual_seed(11)
                loss = model(**batch()).loss
                loss.backward()
                results.append(loss.detach())
            torch.testing.assert_close(*results, rtol=0, atol=0)
            for name, p in control.base_model.named_parameters():
                q = dict(candidate.base_model.named_parameters())[name]
                self.assertEqual(p.grad is None, q.grad is None, name)
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)
            self.assertTrue(torch.isfinite(candidate.alpha.grad))
            self.assertGreater(candidate.alpha.grad.abs().item(), 0)

    def test_checkpoint_replay_dropout_ga32_matches_without_checkpointing(self):
        for variant in ('ro', 'dw', 'rw'):
            torch.manual_seed(14)
            core = native(.15)
            replay_core = copy.deepcopy(core)
            plain, replay = wrapped(core, variant), wrapped(replay_core, variant)
            with torch.no_grad():
                plain.alpha.fill_(.2)
                replay.alpha.fill_(.2)
            replay.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
            for model in (plain, replay):
                model.train()
                torch.manual_seed(20)
                for _ in range(32):
                    (model(**batch()).loss / 32).backward()
                self.assertTrue(model.last_diagnostics['cache_empty'])
                self.assertGreater(model.last_diagnostics['kl'], 0)
                self.assertIsNone(model._donor)
                self.assertFalse(model._inside_checkpoint)
                for layer in model.get_encoder().layers:
                    self.assertFalse(layer.final_layer_norm._forward_pre_hooks)
            for (name, p), (_, q) in zip(plain.named_parameters(), replay.named_parameters()):
                self.assertEqual(p.grad is None, q.grad is None, name)
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, q.grad, rtol=8e-4, atol=3e-6, msg=name)

    def test_formulas_and_mask_against_explicit_receiver(self):
        for variant in ('ro', 'dw', 'rw'):
            model = wrapped(native(), variant).eval()
            layer = model.get_encoder().layers[2]
            u, donor = torch.randn(2, 4, 16), torch.randn(2, 4, 16, requires_grad=True)
            mask = torch.tensor([[[1], [1], [0], [0]], [[1], [1], [1], [0]]]).bool()
            with torch.no_grad():
                model.alpha.fill_(.3)
            increment = .3 * donor * mask
            read = u + increment if variant in ('ro', 'rw') else u
            branch = layer.fc2(layer.activation_fn(layer.fc1(layer.final_layer_norm(read))))
            expected = u + branch + (increment if variant in ('dw', 'rw') else 0)
            with model._receiver(layer, donor, mask):
                actual_branch = layer.fc2(layer.activation_fn(layer.fc1(layer.final_layer_norm(u))))
                actual = model._write((u + actual_branch, None), donor, mask)[0]
            torch.testing.assert_close(actual, expected)
            actual.sum().backward()
            self.assertTrue(torch.equal(donor.grad[~mask.expand_as(donor)],
                                        torch.zeros_like(donor.grad[~mask.expand_as(donor)])))
            self.assertGreater(donor.grad[mask.expand_as(donor)].abs().sum().item(), 0)

    def test_generation_and_nonzero_alpha_roundtrip(self):
        inputs = {k: v for k, v in batch().items() if k != 'labels'}
        for variant in ('ro', 'dw', 'rw'):
            model = wrapped(native(), variant).eval()
            with torch.no_grad():
                model.alpha.fill_(.23)
            with tempfile.TemporaryDirectory() as folder:
                model.save_pretrained(folder)
                restored = IncrementReuseMT.from_pretrained(folder).eval()
                torch.testing.assert_close(model(**batch()).logits, restored(**batch()).logits,
                                           rtol=0, atol=0)
                with torch.no_grad():
                    self.assertTrue(torch.equal(model.generate(**inputs, max_new_tokens=3),
                                                restored.generate(**inputs, max_new_tokens=3)))
            self.assertIsNone(model._mask)


if __name__ == '__main__':
    unittest.main()
