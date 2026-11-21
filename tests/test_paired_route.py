"""Check exact objective, both KL gradients, route replay and persistence."""
import copy
import tempfile
import unittest

import torch
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration

from test_alignment_residual import native, batch
from methods.paired_route import PairedRouteMT, METHODS
from methods.paired_route.v1.model import paired_loss


class PairedRouteGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_loss_and_both_argument_gradients(self):
        torch.manual_seed(12)
        a = torch.randn(2, 4, 40, requires_grad=True)
        b = torch.randn_like(a, requires_grad=True)
        y = batch()['labels']
        value, ce, kl = paired_loss(a, b, y, 2.5, chunk_size=3)
        la, lb = a.log_softmax(-1), b.log_softmax(-1)
        mask = y.ne(-100)
        ref_ce = .5 * (F.cross_entropy(a.flatten(0,1), y.flatten()) +
                       F.cross_entropy(b.flatten(0,1), y.flatten()))
        ref_kl = (.5 * (la.exp()*(la-lb) + lb.exp()*(lb-la)).sum(-1))[mask].mean()
        reference = ref_ce + 2.5 * ref_kl
        torch.testing.assert_close(value, reference)
        value.backward(retain_graph=True)
        actual = (a.grad.clone(), b.grad.clone())
        expected = torch.autograd.grad(reference, (a,b))
        for x,z in zip(actual, expected):
            torch.testing.assert_close(x, z, rtol=2e-5, atol=2e-7)
            self.assertTrue(torch.equal(x[~mask], torch.zeros_like(x[~mask])))
        _, _, zero = paired_loss(a, a, y, 2.5)
        self.assertEqual(zero.item(), 0.)

    def test_off_equivalence_on_matches_original_and_no_added_parameters(self):
        from methods.original.clrr import CrossLayerResidualRewire
        core = native().eval()
        for method, (rewired, _) in METHODS.items():
            model = PairedRouteMT(copy.deepcopy(core), method).eval()
            expected = (CrossLayerResidualRewire(copy.deepcopy(core)).eval()
                        if rewired else core)
            torch.testing.assert_close(model(**batch()).logits, expected(**batch()).logits, rtol=0, atol=0)
            self.assertEqual(model.extra_parameter_count(), 0)
            self.assertEqual(set(core.state_dict()), set(model.state_dict()))
        model = PairedRouteMT(copy.deepcopy(core), 'paired-native-ce-v1').train()
        torch.testing.assert_close(model(**batch()).loss, core(**batch()).loss)

    def test_checkpoint_gradients_with_dropout_and_accumulation(self):
        for method in METHODS:
            torch.manual_seed(42)
            plain = PairedRouteMT(native(.15), method)
            replay = PairedRouteMT(native(.15), method)
            replay.load_state_dict(plain.state_dict())
            replay.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
            for model in (plain, replay):
                model.train()
                torch.manual_seed(77)
                for _ in range(2):
                    output = model(**batch())
                    (output.loss / 2).backward()
                self.assertFalse(model._checkpoint_context)
                self.assertTrue(all(not c for c in model._cache.values()))
                if model.beta:
                    self.assertGreater(model.last_diagnostics['kl'], 0.)
            left, right = dict(plain.named_parameters()), dict(replay.named_parameters())
            for name in left:
                self.assertEqual(left[name].grad is None, right[name].grad is None, name)
                if left[name].grad is not None:
                    torch.testing.assert_close(left[name].grad, right[name].grad,
                        rtol=5e-4, atol=3e-6, msg=f'{method}: {name}')

    def test_donor_detach(self):
        model = PairedRouteMT(native(), 'paired-clrr-ce-v1')
        donor = torch.randn(2,3,16, requires_grad=True)
        recipient = torch.randn_like(donor, requires_grad=True)
        model._cache['encoder'][0] = donor
        model._make_hook('encoder', 2)(None, (), recipient).sum().backward()
        self.assertIsNone(donor.grad)
        self.assertIsNotNone(recipient.grad)

    def test_persistence_generation_and_cache_isolation(self):
        for method in METHODS:
            model = PairedRouteMT(native(), method).eval()
            inputs = {k:v for k,v in batch().items() if k != 'labels'}
            expected = model(**batch()).logits.detach()
            with tempfile.TemporaryDirectory() as folder:
                model.save_pretrained(folder)
                restored = PairedRouteMT(MBartForConditionalGeneration.from_pretrained(folder), method).eval()
                restored.verify_checkpoint_metadata(folder)
                torch.testing.assert_close(restored(**batch()).logits, expected, rtol=0, atol=0)
                model(**{k:v[:1] for k,v in batch().items()})
                torch.testing.assert_close(model(**batch()).logits, expected, rtol=0, atol=0)
                with torch.no_grad():
                    self.assertTrue(torch.equal(model.generate(**inputs,max_new_tokens=4,use_cache=True),
                                                restored.generate(**inputs,max_new_tokens=4,use_cache=False)))


if __name__ == '__main__':
    unittest.main()
