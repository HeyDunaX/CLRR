"""P1 with the locked historical SDPA backbone: native parity and restoration."""
import copy
import unittest
import torch
from test_depth_difference import DepthDifferenceGate, backbone
from test_depth_difference_v2 import ValueScaledGate
from methods.depth_difference.v2 import ValueScaledDepthDifference


class SDPAP1Gate(ValueScaledGate):
    def setUp(self):
        torch.manual_seed(42)
        self.native = backbone()
        self.native.set_attn_implementation('sdpa')
        self.model = ValueScaledDepthDifference(copy.deepcopy(self.native), donor_layer=2,
                                                mode='ordinary')

    def test_attention_formula_detach_shared_projections_and_head_mask(self):
        module = self.model.correction_attention.eval().double()
        module.gates.data.fill_(.3)
        attn = module.attention
        top = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
        donor = torch.randn_like(top, requires_grad=True)
        hidden = torch.randn(2, 3, 16, dtype=torch.float64)
        mask = torch.zeros(2, 1, 3, 5, dtype=torch.float64)
        mask[..., -1] = torch.finfo(torch.float64).min
        heads = lambda x: x.reshape(2, -1, 2, 8).transpose(1, 2)
        q = heads(attn.q_proj(hidden))
        at = (q @ heads(attn.k_proj(top)).transpose(-1, -2)*attn.scaling+mask).softmax(-1)
        ad = (q @ heads(attn.k_proj(donor)).transpose(-1, -2)*attn.scaling+mask).softmax(-1)
        vt = heads(attn.v_proj(top))
        vd = heads(attn.v_proj(module.donor_value_input(top, donor, mask)))
        expected = attn.out_proj((at@vt+.3*(ad@vd-at@vt)).transpose(1, 2).reshape(2, 3, 16))
        actual, _ = module(hidden, torch.cat((top, donor), -1), attention_mask=mask)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        actual.square().sum().backward()
        self.assertTrue(donor.grad is None or donor.grad.eq(0).all())
        self.assertEqual(self.model.config._attn_implementation, 'sdpa')


if __name__ == '__main__':
    unittest.main(defaultTest='SDPAP1Gate', verbosity=2)
