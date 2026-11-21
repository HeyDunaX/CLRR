"""Matched-capacity donor-input control, formula and RNG gates."""
import copy
import inspect
import tempfile
import unittest

import torch
from test_alignment_residual import native, batch, nonzero_modules
from methods.alignment_residual.v1 import AlignmentResidualMT
from methods.alignment_residual.v1.model import ResidualEncoder
from methods.alignment_residual.learned_donor.model import LearnedDonorCE, LearnedDonorEncoder


class LearnedDonorGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make(self, base=None):
        return LearnedDonorCE(native() if base is None else base,
                              'alignment-learned-donor-ce-v1', rank=4)

    def test_encoder_loop_changes_only_module_input(self):
        original = inspect.getsource(ResidualEncoder.forward)
        control = inspect.getsource(LearnedDonorEncoder.forward)
        self.assertEqual(control, original.replace('(donor - hidden)', '(donor)'))

    def test_zero_init_capacity_forward_and_dropout_rng_match_c0(self):
        torch.manual_seed(42)
        base = native(.15)
        original = AlignmentResidualMT(copy.deepcopy(base), 'alignment-residual-capacity-v1',
                                       rank=4, jepa_weight=0.)
        rng = torch.get_rng_state().clone()
        control = self.make(copy.deepcopy(base))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(original.extra_parameter_count(), control.extra_parameter_count())
        results = []
        for model in (original, control):
            torch.manual_seed(123)
            output = model(**batch())
            output.loss.backward()
            results.append((output.logits.detach(), torch.get_rng_state().clone()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in model.parameters()))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        self.assertTrue(any(m.up.grad.abs().sum() > 0
                            for m in control.get_encoder().residual_modules.values()))

    def test_nonzero_formula_and_detached_donor(self):
        torch.manual_seed(42)
        model = self.make().eval()
        nonzero_modules(model)
        enc = model.get_encoder()
        ids, mask = batch()['input_ids'], batch()['attention_mask']
        inputs = []
        handles = [m.register_forward_pre_hook(lambda m, a: inputs.append(a[0]))
                   for m in enc.residual_modules.values()]
        actual = enc(input_ids=ids, attention_mask=mask).last_hidden_state
        for h in handles:
            h.remove()
        self.assertTrue(all(not x.requires_grad for x in inputs))
        hidden = enc.embed_tokens(ids)
        hidden = enc.layernorm_embedding(hidden + enc.embed_positions(ids))
        full_mask = enc._update_full_mask(mask, hidden)
        donors = []
        for i, layer in enumerate(enc.layers):
            hidden = layer.forward(hidden, full_mask, layer_head_mask=None)[0]
            if i >= 2:
                donor = donors[i - 2]
                hidden = hidden + .1 * (donor + enc.residual_modules[str(i)](donor))
            donors.append(hidden.detach())
        torch.testing.assert_close(actual, enc.layer_norm(hidden))

    def test_nonzero_checkpointing_backward_and_roundtrip(self):
        torch.manual_seed(42)
        plain = self.make()
        nonzero_modules(plain)
        checked = copy.deepcopy(plain)
        checked.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
        for model in (plain, checked):
            model(**batch()).loss.backward()
        for (n, p), (other, q) in zip(plain.named_parameters(), checked.named_parameters()):
            self.assertEqual(n, other)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=5e-5)
        self.assertTrue(all(m.down.grad.abs().sum() > 0
                            for m in checked.get_encoder().residual_modules.values()))
        checked.eval()
        with tempfile.TemporaryDirectory() as folder:
            checked.save_pretrained(folder)
            restored = LearnedDonorCE.from_pretrained(folder).eval()
            torch.testing.assert_close(restored(**batch()).logits, checked(**batch()).logits)
            self.assertEqual(restored.method_metadata(), checked.method_metadata())


if __name__ == '__main__':
    unittest.main(verbosity=2)
