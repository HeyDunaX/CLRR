"""Verify the factorial zero cell against native CE, including dropout draws."""
import copy
import tempfile
import unittest
import torch
from test_alignment_residual import native, batch
from methods.alignment_residual.matched_baseline import MatchedBaselineCE


class MatchedBaselineGate(unittest.TestCase):
    def test_native_loss_gradient_and_target_rng_schedule(self):
        torch.manual_seed(42)
        base = native(.15)
        control = MatchedBaselineCE(copy.deepcopy(base), 'alignment-matched-ce-control-v1', rank=4)
        results = []
        for model in (base, control):
            torch.manual_seed(123)
            loss = model(**batch()).loss
            if model is base:
                ids = batch()['labels'].masked_fill(batch()['labels'].eq(-100), 1)
                with torch.no_grad():
                    model.get_encoder()(input_ids=ids, attention_mask=ids.ne(1).long())
            loss.backward()
            results.append((loss.detach(), torch.get_rng_state().clone()))
        torch.testing.assert_close(results[0][0], results[1][0])
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        parameters = dict(base.named_parameters())
        for name, parameter in control.base_model.named_parameters():
            torch.testing.assert_close(parameter.grad, parameters[name].grad, rtol=2e-4, atol=5e-5)
        self.assertEqual(control.extra_parameter_count(), 0)

    def test_checkpointing_and_roundtrip(self):
        model = MatchedBaselineCE(native(), 'alignment-matched-ce-control-v1', rank=4)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
        model(**batch()).loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = MatchedBaselineCE.from_pretrained(directory).eval()
            torch.testing.assert_close(model(**batch()).logits, restored(**batch()).logits)
            self.assertEqual(model.method_metadata(), restored.method_metadata())
