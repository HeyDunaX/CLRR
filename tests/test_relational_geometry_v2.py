import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from methods.difference_preserving.v2.geometry import relational_loss


class RelationalGeometryTests(unittest.TestCase):
    def setUp(self):
        self.target = torch.tensor([[0., 0.], [1., 0.], [0., 2.], [2., 3.]])
        self.mask = ~torch.eye(4, dtype=torch.bool)

    def loss(self, source, objective, target=None, mask=None):
        return relational_loss(source, self.target if target is None else target,
                               objective=objective,
                               eligible_pairs=self.mask if mask is None else mask)

    def test_source_translation_and_positive_scale_invariance(self):
        for objective in ('direction', 'rkd-distance'):
            loss, stats = self.loss(3 * self.target + 7, objective)
            self.assertLess(abs(float(loss)), 1e-6)
            self.assertEqual(stats['valid_pairs'], 6)

    def test_rotation_distinguishes_orientation_from_distance(self):
        rotated = self.target @ torch.tensor([[0., -1.], [1., 0.]])
        self.assertAlmostEqual(float(self.loss(rotated, 'direction')[0]), 1., places=6)
        self.assertLess(float(self.loss(rotated, 'rkd-distance')[0]), 1e-7)

    def test_distance_control_has_signal_with_nonuniform_distortion(self):
        distorted = self.target.clone()
        distorted[-1] += torch.tensor([8., -2.])
        self.assertGreater(float(self.loss(distorted, 'rkd-distance')[0]), .01)

    def test_single_pair_distance_control_is_degenerate(self):
        # A necessary design warning: normalizing one pair always gives 1.
        mask = torch.tensor([[False, True], [True, False]])
        loss, _ = relational_loss(torch.tensor([[0., 0.], [8., 0.]]),
                                  torch.tensor([[0., 0.], [1., 2.]]),
                                  objective='rkd-distance', eligible_pairs=mask)
        self.assertEqual(float(loss), 0.)

    def test_target_detached_and_source_gradient_finite(self):
        for objective in ('direction', 'rkd-distance'):
            source = (self.target + torch.tensor([[0., .5], [1., 0.], [2., 1.], [0., 2.]])).requires_grad_()
            target = self.target.clone().requires_grad_()
            loss, _ = self.loss(source, objective, target=target)
            loss.backward()
            self.assertIsNone(target.grad)
            self.assertTrue(torch.isfinite(source.grad).all())
            self.assertGreater(float(source.grad.norm()), 0.)
            # Scale invariance detects a detached source normalizer.
            self.assertLess(abs(float((source.grad * source.detach()).sum())), 1e-5)

    def test_empty_and_collapsed_cases_are_explicit(self):
        source = torch.zeros_like(self.target, requires_grad=True)
        for objective in ('direction', 'rkd-distance'):
            loss, stats = self.loss(source, objective)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(stats['small_source_pairs'], 6)
            self.assertGreater(float(loss), 0.)
        zero, stats = self.loss(source, 'direction', target=torch.zeros_like(source))
        self.assertEqual(float(zero), 0.)
        self.assertEqual(stats['valid_pairs'], 0)
        zero.backward()
        self.assertTrue(torch.equal(source.grad, torch.zeros_like(source)))

    def test_joint_rank_one_shrinkage_is_not_prevented(self):
        # Both evolving views can share an impoverished geometry at zero loss.
        rank_one = torch.arange(4.)[:, None] * torch.tensor([[1., 0.]]) * .001
        for objective in ('direction', 'rkd-distance'):
            loss, _ = self.loss(rank_one, objective, target=rank_one + 3.)
            self.assertLess(abs(float(loss)), 1e-6)

    def test_reject_invalid_mask_and_nonfinite_input(self):
        with self.assertRaises(ValueError):
            self.loss(self.target, 'direction', mask=torch.ones(4, 4, dtype=torch.bool))
        bad = self.target.clone()
        bad[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.loss(bad, 'direction')


if __name__ == '__main__':
    unittest.main()
