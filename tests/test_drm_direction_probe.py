"""Ensure interventions isolate the proposed direction and remain causal."""
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
from probe_drm_directions import controlled_update, intervention
from test_alignment_residual import native, batch
from methods.decision_residual_memory.v1.model import DecisionResidualMemoryMT, update_memory


class DirectionProbeGate(unittest.TestCase):
    def test_norm_matching_and_patch_restoration(self):
        torch.manual_seed(271)
        memory = torch.randn(1, 3, 7, 5)
        attention = torch.randn(1, 3, 7).softmax(-1)
        direction = torch.randn(1, 3, 5)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        active = torch.tensor([True])
        expected = update_memory(memory, attention, direction, .1, active)-memory
        for arm in ('context', 'random', 'scalar'):
            result, error, _ = controlled_update(memory, attention, direction, .1, active, arm, 42)
            torch.testing.assert_close((result-memory).norm(dim=(-2,-1)), expected.norm(dim=(-2,-1)))
            self.assertLess(error, 1e-5)
        import methods.decision_residual_memory.v1.model as module
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with intervention('random', 42):
                module.update_memory(memory, attention, direction, .1, active)
                raise RuntimeError('deliberate')
        self.assertIs(module.update_memory, update_memory)

    def test_zero_and_single_update_do_not_change_earlier_logits(self):
        torch.set_num_threads(2)
        torch.manual_seed(90)
        model = DecisionResidualMemoryMT(native()).eval()
        data = batch()
        data = {k:v[:1] for k,v in data.items()}
        with torch.no_grad():
            model.eta = 0
            reference = model(**data).logits
            model.eta = .1
            with intervention('zero', 42):
                torch.testing.assert_close(model(**data).logits, reference, rtol=1e-5, atol=2e-6)
            for arm in ('drm', 'context', 'random', 'scalar'):
                with intervention(arm, 42, at=1):
                    actual = model(**data).logits
                torch.testing.assert_close(actual[:,:2], reference[:,:2], rtol=1e-5, atol=2e-6)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


    def test_random_axis_cannot_have_empty_positive_support(self):
        memory = torch.ones(1, 4, 9, 5)
        attention = torch.ones(1, 4, 9) / 9
        direction = torch.ones(1, 4, 5) / 5**.5
        for seed in range(20):
            result, error, coefficient = controlled_update(memory, attention, direction,
                .1, torch.tensor([True]), 'random', seed)
            self.assertTrue(torch.isfinite(result).all())
            self.assertLess(error, 1e-5)
            self.assertLessEqual(coefficient, 1.)


if __name__ == '__main__':
    unittest.main()
