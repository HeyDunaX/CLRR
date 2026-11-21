"""Mathematical gates for a known MRT control; no weights, data or GPU needed."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from methods.sequence_risk.v1 import candidate_set_risk


class SequenceRiskGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_formula_and_analytic_gradient(self):
        scores = torch.tensor([[-3., -6., -9.], [-8., -4., -5.]], dtype=torch.float64, requires_grad=True)
        costs = torch.tensor([[0., .3, .8], [.7, .1, .4]], dtype=torch.float64, requires_grad=True)
        valid = torch.ones_like(scores, dtype=torch.bool)
        alpha = .2
        loss = candidate_set_risk(scores, costs, valid, alpha=alpha)
        q = torch.softmax(alpha * scores.detach(), dim=1)
        risk = (q * costs.detach()).sum(1, keepdim=True)
        torch.testing.assert_close(loss, risk.mean())
        loss.backward()
        torch.testing.assert_close(scores.grad, alpha * q * (costs.detach() - risk) / len(scores))
        self.assertIsNone(costs.grad)

    def test_finite_difference_gradcheck(self):
        scores = torch.tensor([[-1., -2., -3.], [-7., -4., -5.]], dtype=torch.float64, requires_grad=True)
        costs = torch.tensor([[.8, .3, 0.], [.2, .4, .6]], dtype=torch.float64)
        valid = torch.tensor([[True, True, False], [True, True, True]])
        self.assertTrue(torch.autograd.gradcheck(lambda s: candidate_set_risk(s, costs, valid, alpha=.3), (scores,)))

    def test_invalid_padding_nan_and_source_mean(self):
        scores = torch.tensor([[-1., float('nan'), -3.], [-2., float('nan'), float('nan')]], requires_grad=True)
        costs = torch.tensor([[.2, float('nan'), .8], [.3, float('nan'), float('nan')]])
        valid = torch.tensor([[True, False, True], [True, False, False]])
        loss = candidate_set_risk(scores, costs, valid, alpha=1.)
        expected = ((torch.softmax(torch.tensor([-1., -3.]), 0) * torch.tensor([.2, .8])).sum() + .3) / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(scores.grad[~valid].eq(0).all())

    def test_offset_and_permutation_invariance(self):
        scores = torch.tensor([[-3., -6., -9.]], dtype=torch.float64)
        costs = torch.tensor([[0., .3, .8]], dtype=torch.float64)
        valid = torch.ones_like(scores, dtype=torch.bool)
        reference = candidate_set_risk(scores, costs, valid, alpha=.005)
        torch.testing.assert_close(reference, candidate_set_risk(scores + 1e6, costs, valid, alpha=.005))
        order = [2, 0, 1]
        torch.testing.assert_close(reference, candidate_set_risk(scores[:, order], costs[:, order], valid, alpha=.005))

    def test_equal_cost_and_singleton_have_zero_gradient(self):
        for shape in [(2, 3), (2, 1)]:
            scores = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
            loss = candidate_set_risk(scores, torch.full_like(scores, .4), torch.ones_like(scores, dtype=torch.bool), alpha=.2)
            loss.backward()
            torch.testing.assert_close(scores.grad, torch.zeros_like(scores))

    def test_update_rewards_better_candidate(self):
        scores = torch.tensor([[-2., -2.]], dtype=torch.float64, requires_grad=True)
        costs = torch.tensor([[0., 1.]], dtype=torch.float64)
        valid = torch.ones_like(scores, dtype=torch.bool)
        loss = candidate_set_risk(scores, costs, valid, alpha=1.)
        grad, = torch.autograd.grad(loss, scores)
        updated = scores.detach() - .1 * grad
        self.assertGreater(updated[0, 0], updated[0, 1])
        self.assertLess(candidate_set_risk(updated, costs, valid, alpha=1.), loss)

    def test_reject_invalid_inputs(self):
        scores = torch.zeros(2, 3)
        costs = torch.zeros_like(scores)
        valid = torch.ones_like(scores, dtype=torch.bool)
        for alpha in [0., -1., math.inf, math.nan]:
            with self.assertRaises(ValueError):
                candidate_set_risk(scores, costs, valid, alpha=alpha)
        for bad_valid in [torch.zeros_like(valid), valid.float(), valid[:, :1]]:
            with self.assertRaises(ValueError):
                candidate_set_risk(scores, costs, bad_valid, alpha=1.)
        for bad_scores in [torch.full_like(scores, math.nan), scores.flatten()]:
            with self.assertRaises(ValueError):
                candidate_set_risk(bad_scores, costs, valid, alpha=1.)
        with self.assertRaises(ValueError):
            candidate_set_risk(scores, torch.full_like(costs, math.inf), valid, alpha=1.)

    def test_ce_improvement_can_lose_top1_and_ce_already_uses_margins(self):
        before = torch.tensor([.4, .35, .25], dtype=torch.float64)
        after = torch.tensor([.42, .43, .15], dtype=torch.float64)
        self.assertLess(-after[0].log(), -before[0].log())
        self.assertEqual(before.argmax().item(), 0)
        self.assertNotEqual(after.argmax().item(), 0)
        for probs in [before, after]:
            logits = probs.log()
            # CE_y = log(1 + sum_{j!=y} exp(z_j - z_y)).
            exact = torch.log1p((logits[1:] - logits[0]).exp().sum())
            torch.testing.assert_close(exact, -probs[0].log())


if __name__ == "__main__":
    unittest.main(verbosity=2)
