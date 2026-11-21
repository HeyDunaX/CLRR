"""Matched loss, donor-gradient, checkpoint recompute and export gates for P3 controls."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
import torch
from transformers import MBartForConditionalGeneration
from test_residual_effect import backbone, batch
from methods.registry import wrap_method, CONTRASTIVE_CONTROL_METHODS
from methods.contrastive_controls.v1.clrr import AttachedResidualRewire
from methods.original.clrr import CrossLayerResidualRewire


SPEC = dict(alignment_weight=.1, temperature=.1)


class ContrastiveControlGate(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_only_declared_loss_and_route_changes(self):
        initial_base = backbone()
        initial = initial_base.state_dict()
        def model(method):
            base = backbone(); base.load_state_dict(initial)
            return wrap_method(base, dict(SPEC, method=method)).eval()
        a, b, c = [model(m) for m in ('nce-only-v1', 'clrr-nce-v1', 'attached-ordinary-full-v1')]
        p2, p3 = model('ordinary-lsr-v1'), model('ordinary-full-v1')
        self.assertNotIsInstance(a.base_model, CrossLayerResidualRewire)
        self.assertIsInstance(b.base_model, CrossLayerResidualRewire)
        self.assertIsInstance(c.base_model, AttachedResidualRewire)
        self.assertEqual((a.jepa_weight, b.jepa_weight, c.jepa_weight), (0., 0., .1))
        with torch.no_grad():
            for control, reference in ((a, p2), (b, p3), (c, p3)):
                torch.testing.assert_close(control(**batch()).logits, reference(**batch()).logits)
                torch.testing.assert_close(control.auxiliary_loss(**batch()), reference.auxiliary_loss(**batch()))
            torch.testing.assert_close(c(**batch()).loss, p3(**batch()).loss)
        # Zero LSR weight is an exact CE+NCE loss, not a renamed Full.
        for control in (a, b):
            with torch.no_grad():
                expected = control.base_model(**batch()).loss + .1 * control.auxiliary_loss(**batch())
                torch.testing.assert_close(control(**batch()).loss, expected)
        for model in (a, b, c):
            self.assertEqual(model.extra_parameter_count(), 0)
            self.assertEqual(sum(p.numel() for p in model.parameters()),
                             sum(p.numel() for p in initial_base.parameters()))

    def test_attached_donor_has_exact_extra_gradient(self):
        for cls in (CrossLayerResidualRewire, AttachedResidualRewire):
            route = cls(backbone(), distance=2, strength=.1)
            donor = torch.randn(2, 3, 16, requires_grad=True)
            hidden = torch.randn_like(donor, requires_grad=True)
            with route._fresh_cache():
                route._make_hook('encoder', 0)(None, (), (donor,))
                output = route._make_hook('encoder', 2)(None, (), (hidden,))[0]
                torch.testing.assert_close(output, hidden + .1 * donor)
            output.sum().backward()
            torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
            if cls is AttachedResidualRewire:
                torch.testing.assert_close(donor.grad, torch.full_like(donor, .1))
            else:
                self.assertIsNone(donor.grad)

    def test_checkpointing_gradients_export_and_rng(self):
        for method in sorted(CONTRASTIVE_CONTROL_METHODS):
            with self.subTest(method=method):
                model = wrap_method(backbone(), dict(SPEC, method=method))
                model.train(); model.enable_input_require_grads()
                rng = torch.random.get_rng_state().clone()
                auxiliary = model.auxiliary_loss(**batch())
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                auxiliary.backward()
                self.assertIsNone(model.get_decoder().layers[0].fc1.weight.grad)
                model.zero_grad(set_to_none=True)
                torch.manual_seed(123)
                expected_loss = model(**batch()).loss
                expected_loss.backward()
                expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
                model.zero_grad(set_to_none=True)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
                torch.manual_seed(123)
                actual = model(**batch()).loss
                actual.backward()
                torch.testing.assert_close(actual, expected_loss)
                for name, parameter in model.named_parameters():
                    if name in expected:
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        torch.testing.assert_close(parameter.grad, expected[name], atol=2e-6, rtol=2e-4)
                model.eval()
                with torch.no_grad(): logits = model(**batch()).logits
                with tempfile.TemporaryDirectory() as folder:
                    model.save_pretrained(folder)
                    loaded = MBartForConditionalGeneration.from_pretrained(folder)
                    restored = wrap_method(loaded, dict(SPEC, method=method)).eval()
                    with torch.no_grad(): torch.testing.assert_close(logits, restored(**batch()).logits)


def run_correctness_checks():
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ContrastiveControlGate))
    if not result.wasSuccessful():
        raise AssertionError('Contrastive controls correctness gate failed')
    return {'passed': True, 'tests': result.testsRun, 'checkpoint_gradient_parity': True,
            'attached_donor_gradient': True, 'old_method_forward_parity': True}


if __name__ == '__main__':
    run_correctness_checks()
