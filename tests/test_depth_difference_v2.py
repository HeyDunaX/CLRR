"""U2 value scaling correctness, inherited integration gates and optimizer smoke."""
import copy
import tempfile
import unittest
from pathlib import Path
import torch
from test_depth_difference import DepthDifferenceGate, backbone, batch
from methods.depth_difference.v1 import DepthDifferencePrototype
from methods.depth_difference.v2 import ValueScaledDepthDifference


class ValueScaledGate(DepthDifferenceGate):
    def setUp(self):
        torch.manual_seed(42)
        self.native = backbone()
        self.model = ValueScaledDepthDifference(copy.deepcopy(self.native), donor_layer=2)

    def test_attention_formula_detach_shared_projections_and_head_mask(self):
        module = self.model.correction_attention.eval().double()
        module.gates.data.fill_(.3)
        attn = module.attention
        hidden = torch.randn(2, 3, 16, dtype=torch.float64, requires_grad=True)
        top = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
        donor = (torch.randn(2, 5, 16, dtype=torch.float64)*5).requires_grad_()
        mask = torch.zeros(2, 1, 3, 5, dtype=torch.float64); mask[..., -1] = -1e30
        scaled = module.donor_value_input(top, donor, mask)
        self.assertFalse(scaled.requires_grad)
        torch.testing.assert_close(scaled[:, :-1].norm(dim=(1, 2)), top[:, :-1].norm(dim=(1, 2)), rtol=2e-6, atol=1e-6)
        changed = donor.detach().clone(); changed[:, -1] = 1e5
        torch.testing.assert_close(module.donor_value_input(top, changed, mask)[:, :-1], scaled[:, :-1])
        shape = lambda x: x.reshape(2, -1, 2, 8).transpose(1, 2)
        q = shape(attn.q_proj(hidden))
        at = (q @ shape(attn.k_proj(top)).transpose(-1,-2)*attn.scaling+mask).softmax(-1)
        ad = (q @ shape(attn.k_proj(donor)).transpose(-1,-2)*attn.scaling+mask).softmax(-1)
        vt = shape(attn.v_proj(top)); vd = shape(attn.v_proj(scaled))
        head_mask = torch.tensor([1., 0.], dtype=torch.float64)
        expected_context = (at @ vt + .3*((ad-at)@(vd-vt)))*head_mask[None,:,None,None]
        expected = attn.out_proj(expected_context.transpose(1,2).reshape(2,3,16))
        actual, _ = module(hidden, torch.cat((top,donor),-1), attention_mask=mask, layer_head_mask=head_mask)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        actual.square().sum().backward()
        self.assertTrue(donor.grad is None or donor.grad.eq(0).all())
        for parameter in (attn.q_proj.weight,attn.k_proj.weight,attn.v_proj.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm(), 0)
        zero = module.donor_value_input(top, torch.zeros_like(donor), mask)
        self.assertTrue(torch.isfinite(zero).all())

    def test_strict_save_load_retains_gates_logits_and_generation(self):
        self.model.eval(); self.model.correction_attention.gates.data.fill_(.2)
        data = batch(); data.pop('labels')
        with torch.no_grad(): reference = self.model(**batch()).logits
        generated = self.model.generate(**data, num_beams=4, max_new_tokens=4)
        with tempfile.TemporaryDirectory() as folder:
            self.model.save_pretrained(folder)
            with self.assertRaises(ValueError): DepthDifferencePrototype.from_pretrained(folder)
            restored = ValueScaledDepthDifference.from_pretrained(folder).eval()
            with torch.no_grad(): torch.testing.assert_close(reference, restored(**batch()).logits)
            self.assertTrue(torch.equal(generated,restored.generate(**data,num_beams=4,max_new_tokens=4)))

    def test_three_optimizer_steps(self):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        initial = self.model.correction_attention.gates.detach().clone()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = self.model(**batch(), use_cache=False).loss
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for parameter in self.model.parameters():
                if parameter.grad is not None: self.assertTrue(torch.isfinite(parameter.grad).all())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
            optimizer.step()
        self.assertFalse(torch.equal(initial,self.model.correction_attention.gates))


if __name__ == '__main__':
    unittest.main(defaultTest='ValueScaledGate', verbosity=2)
