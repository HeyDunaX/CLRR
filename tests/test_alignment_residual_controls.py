"""Scientific checks of CE controls; original method definitions stay intact."""
import copy
import tempfile
import unittest
import torch
from test_alignment_residual import native, batch, nonzero_modules
from methods.alignment_residual.v1 import AlignmentResidualMT
from methods.alignment_residual.controls import ResidualCEControl, METHODS
from methods.alignment_residual.controls.probes import residual_intervention


class ResidualCEControlsGate(unittest.TestCase):
    def test_probe_identity_and_donor_off_equal_local_forward(self):
        model = AlignmentResidualMT(native(), 'alignment-residual-capacity-v1', rank=4, jepa_weight=0.).eval()
        nonzero_modules(model)
        mask = batch()['attention_mask'].bool()
        expected = model(**batch()).logits
        with residual_intervention(model, 'all-on', mask):
            torch.testing.assert_close(model(**batch()).logits, expected)
        with residual_intervention(model, 'donor-off', mask):
            off = model(**batch()).logits
        model.get_encoder().residual_q = 0
        torch.testing.assert_close(model(**batch()).logits, off, rtol=2e-4, atol=2e-5)
        model.get_encoder().residual_q = 1
        torch.testing.assert_close(model(**batch()).logits, expected)

    def test_donor_matches_zero_module_ce_gradient_and_rng(self):
        torch.manual_seed(42)
        base = native(.15)
        original = AlignmentResidualMT(copy.deepcopy(base), 'alignment-residual-capacity-v1', rank=4, jepa_weight=0.)
        control = ResidualCEControl(copy.deepcopy(base), 'alignment-donor-ce-control-v1', rank=4)
        results = []
        for model in (original, control):
            torch.manual_seed(123)
            loss = model(**batch()).loss
            loss.backward()
            results.append((loss.detach(), torch.get_rng_state().clone()))
        torch.testing.assert_close(results[0][0], results[1][0])
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        weights = dict(original.named_parameters())
        for name, parameter in control.named_parameters():
            torch.testing.assert_close(parameter.grad, weights[name].grad, rtol=2e-4, atol=5e-5)
        self.assertEqual(control.extra_parameter_count(), 0)

    def test_local_ce_has_only_ce_gradients_and_same_capacity(self):
        torch.manual_seed(42)
        model = ResidualCEControl(native(), 'alignment-local-ce-control-v1', rank=4)
        nonzero_modules(model)
        expected = model.base_model(**batch()).loss
        actual = model(**batch()).loss
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(model.get_encoder().residual_q, 0)
        self.assertEqual(model.extra_parameter_count(), 2 * 2 * 16 * 4)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
            for p in model.get_encoder().residual_modules.parameters()))
        self.assertEqual(model.last_diagnostics['weighted_lsr'], 0.)

    def test_control_checkpoint_roundtrip_and_checkpointing(self):
        for method in METHODS:
            model = ResidualCEControl(native(), method, rank=4)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
            model(**batch()).loss.backward()
            model.eval()
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                restored = ResidualCEControl.from_pretrained(directory).eval()
                torch.testing.assert_close(model(**batch()).logits, restored(**batch()).logits)
                self.assertEqual(model.method_metadata(), restored.method_metadata())


if __name__ == '__main__':
    unittest.main(verbosity=2)
