"""Numerical gates for frozen intervention search; no pretrained weights."""
import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location('probe', Path(__file__).parents[1] / 'scripts/analysis/probe_read_recoverability.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class RecoverabilityGate(unittest.TestCase):
    def test_simplex_projection(self):
        x = torch.tensor([[[2., -1., .3]], [[-.1, .2, .5]]])
        p = probe.simplex(x)
        self.assertTrue((p >= 0).all())
        torch.testing.assert_close(p.sum(-1), torch.ones(2, 1))
        torch.testing.assert_close(probe.simplex(p), p)

    def test_feasible_nondecreasing_and_zero(self):
        torch.manual_seed(17)
        q, k, v = torch.randn(2,1,4), torch.randn(2,5,4), torch.randn(2,5,4)
        a = (q @ k.transpose(-1,-2) / 2).softmax(-1)
        c = (a @ v).flatten()
        w = torch.randn_like(c)
        for arm in ('query','simplex','context'):
            for budget in (0., .03, .3):
                result, stats = probe.search(q,k,v,c,a,lambda z:z@w,arm,budget,71,iterations=6)
                self.assertGreaterEqual(float(result@w), float(c@w)-1e-6)
                self.assertLessEqual(stats['relative_displacement'],budget+1e-6)
                if budget == 0:
                    torch.testing.assert_close(result,c,atol=1e-7,rtol=1e-7)
        self.assertIsNone(q.grad)
        self.assertIsNone(v.grad)


if __name__ == '__main__':
    unittest.main()
