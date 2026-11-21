"""CMC: explicit stochastic kernel, gradients, replay, persistence and probe."""
import copy
import tempfile
import unittest
import torch
from transformers import MBartForConditionalGeneration
from test_alignment_residual import native, batch
from methods.continuation_consistency.v1.model import ContinuationConsistencyMT, METHODS
from methods.continuation_consistency.v1.model import pushforward_kl, symmetric_kl


class ContinuationGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_hybrid_kernel_and_both_gradients(self):
        torch.manual_seed(11)
        a = torch.randn(7, requires_grad=True)
        b = torch.randn(7, requires_grad=True)
        q = torch.randn(3, 7, requires_grad=True).log_softmax(-1)
        actions = torch.tensor([0, 3, 5])
        kernel = torch.zeros(7, 14)
        kernel[actions, :7] = q.detach().exp()
        tail = torch.tensor([1, 2, 4, 6])
        kernel[tail, 7+tail] = 1.
        r, s = a.softmax(-1) @ kernel, b.softmax(-1) @ kernel
        present = r.gt(0)
        reference = symmetric_kl(r[present].log(), s[present].log())
        actual = pushforward_kl(a, b, actions, q)
        torch.testing.assert_close(actual, reference)
        for x, y in zip(torch.autograd.grad(actual, (a,b), retain_graph=True),
                        torch.autograd.grad(reference, (a,b))):
            torch.testing.assert_close(x, y, atol=1e-7, rtol=2e-5)
        self.assertLessEqual(float(actual), float(symmetric_kl(a.log_softmax(-1), b.log_softmax(-1)))+1e-6)
        self.assertIsNone(torch.autograd.grad(actual, q, allow_unused=True)[0])
        identity = torch.eye(7)[actions].clamp_min(1e-30).log()
        torch.testing.assert_close(pushforward_kl(a,b,actions,identity),
            symmetric_kl(a.log_softmax(-1), b.log_softmax(-1)))

    def test_probe_equals_full_native_forward_and_restores_flags_rng(self):
        model = ContinuationConsistencyMT(native(.15), list(METHODS)[0]).train()
        data = batch()
        ids = model.prepare_decoder_input_ids_from_labels(data['labels'])
        actions = torch.tensor([4, 7])
        flags = [m.training for m in model.base_model.modules()]
        rng = torch.get_rng_state().clone()
        q = model.continuation_channel(data['input_ids'],data['attention_mask'],ids,[(0,1,actions)])[0]
        self.assertEqual(flags, [m.training for m in model.base_model.modules()])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertFalse(q.requires_grad)
        model.eval()
        for i, action in enumerate(actions):
            prefix = torch.cat((ids[:1,:2], action.reshape(1,1)),1)
            ref = model.base_model(input_ids=data['input_ids'][:1], attention_mask=data['attention_mask'][:1],
                decoder_input_ids=prefix).logits[0,-1].log_softmax(-1)
            torch.testing.assert_close(q[i],ref)

    def test_checkpoint_dropout_accumulation_and_native_inference(self):
        for method in METHODS:
            torch.manual_seed(42)
            plain = ContinuationConsistencyMT(native(.15),method)
            replay = copy.deepcopy(plain)
            replay.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
            for model in (plain,replay):
                model.train();torch.manual_seed(77)
                for _ in range(2):
                    (model(**batch()).loss/2).backward()
                self.assertFalse(model._checkpoint_context)
                self.assertGreater(model.last_diagnostics['kl'],0.)
                self.assertGreater(model.last_diagnostics['channel_separation'],0.)
            for (name,a),(_,b) in zip(plain.named_parameters(),replay.named_parameters()):
                self.assertEqual(a.grad is None,b.grad is None,name)
                if a.grad is not None:
                    torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=5e-4,msg=name)
            plain.eval()
            torch.testing.assert_close(plain(**batch()).logits, plain.base_model(**batch()).logits,atol=0,rtol=0)
            self.assertEqual(plain.extra_parameter_count(),0)

    def test_persistence_and_recipe_mismatch(self):
        model = ContinuationConsistencyMT(native(),list(METHODS)[0]).eval()
        expected = model(**batch()).logits
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            restored = ContinuationConsistencyMT(MBartForConditionalGeneration.from_pretrained(folder),model.method).eval()
            restored.verify_checkpoint_metadata(folder)
            torch.testing.assert_close(expected,restored(**batch()).logits,atol=0,rtol=0)
            restored.top_k = 3
            with self.assertRaises(ValueError):
                restored.verify_checkpoint_metadata(folder)
