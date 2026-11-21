import sys
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
from measure_route_token_interventions import install_switches, comparison
from methods.original.clrr import CrossLayerResidualRewire


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Identity() for _ in range(12)])

    def get_encoder(self):
        return self.encoder

    def forward(self, input_ids=None, **kwargs):
        value = input_ids
        for layer in self.encoder.layers:
            value = layer(value)
        return value


class RouteTokenTests(unittest.TestCase):
    def test_edge_switches_preserve_native_and_bare(self):
        core = Core()
        route = CrossLayerResidualRewire(core)
        x = torch.ones(1, 2, 3)
        native = route(input_ids=x).clone()
        disabled = set()
        install_switches(route, disabled)
        torch.testing.assert_close(route(input_ids=x), native, rtol=0, atol=0)
        disabled.update(range(3, 13))
        torch.testing.assert_close(route(input_ids=x), x, rtol=0, atol=0)
        disabled.clear()
        disabled.add(1)
        torch.testing.assert_close(route(input_ids=x), native, rtol=0, atol=0)
        disabled.clear()
        disabled.update(range(3, 8))
        value = route(input_ids=x)
        self.assertTrue(torch.all(value > x))
        self.assertTrue(torch.all(value < native))
        self.assertEqual(route.strength, .1)

    def test_repairs_and_damages_have_correct_direction(self):
        first = pd.DataFrame(dict(correct=[False, True, True], margin=[-1., 2., 3.], nll=[2., 1., 1.]))
        second = pd.DataFrame(dict(correct=[True, False, True], margin=[1., -2., 3.], nll=[1., 2., 1.]))
        result = comparison(first, second)
        self.assertEqual((result['repairs'], result['damages']), (1, 1))
        self.assertEqual(result['accuracy_delta_pp'], 0)
        self.assertAlmostEqual(result['mean_margin_delta'], -2/3)


if __name__ == '__main__':
    unittest.main()
