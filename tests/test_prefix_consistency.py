"""Scientific gates for ordered score loss, masking, replay and persistence."""
import copy
import json
import tempfile
import unittest
import torch
from transformers import MBartForConditionalGeneration
from test_paired_route import native, batch
from methods.decoupled_consistency.v1.model import consistency_loss
from methods.prefix_consistency import PrefixConsistencyMT, METHODS
from methods.prefix_consistency.v1.losses import gold_differences, path_statistics, prefix_loss


class PrefixConsistencyGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_kernel_value_gradient_and_noncontiguous_masks(self):
        a = torch.tensor([[.2, 9., -.7, .4], [.3, -.2, 8., 8.]], requires_grad=True)
        mask = torch.tensor([[True, False, True, True], [True, True, False, False]])
        stats = path_statistics(a, mask)
        refs = []
        for row, valid in zip(a, mask):
            v = row[valid]
            n = len(v)
            inv = 1./torch.arange(1, n+1)
            k = torch.stack([torch.stack([inv[max(i,j):].sum()/n for j in range(n)]) for i in range(n)])
            self.assertAlmostEqual(k.trace().item(), 1., places=6)
            self.assertGreaterEqual(torch.linalg.eigvalsh(k).min().item(), 0.)
            refs.append(v @ k @ v)
        reference = torch.stack(refs)
        torch.testing.assert_close(stats['prefix'], reference)
        ga = torch.autograd.grad(stats['prefix'].sum(), a, retain_graph=True)[0]
        gr = torch.autograd.grad(reference.sum(), a)[0]
        torch.testing.assert_close(ga, gr)
        self.assertEqual(ga[~mask].abs().sum().item(), 0.)
        torch.testing.assert_close(stats['cross'], stats['prefix']-stats['diagonal'])

    def test_w_exact_at_zero_and_both_view_all_logit_gradients(self):
        torch.manual_seed(19)
        a = torch.randn(2,4,40,requires_grad=True)
        b = torch.randn_like(a,requires_grad=True)
        y = batch()['labels']
        value = prefix_loss(a,b,y,weight=0.)[0]
        value.backward()
        gradients = a.grad.clone(), b.grad.clone()
        a.grad = b.grad = None
        reference = consistency_loss(a,b,y,'full',1.25)[0]
        reference.backward()
        torch.testing.assert_close(value,reference,rtol=0,atol=0)
        for actual, expected in zip(gradients,(a.grad,b.grad)):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        a.grad = b.grad = None
        stats = path_statistics(gold_differences(a,b,y),y.ne(-100))
        stats['prefix'].mean().backward()
        for grad in (a.grad,b.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertTrue((grad[y.ne(-100)].abs().sum(0)>0).all())
            self.assertEqual(grad[y.eq(-100)].abs().sum().item(),0.)

    def test_reentrant_dropout_accumulation(self):
        for method in METHODS:
            weight = 0. if METHODS[method]=='disabled' else .10
            torch.manual_seed(42)
            plain = PrefixConsistencyMT(native(.15),method,weight=weight).train()
            replay = copy.deepcopy(plain)
            replay.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
            outputs=[]
            for model in (plain,replay):
                torch.manual_seed(77)
                for _ in range(2):
                    loss=model(**batch()).loss
                    (loss/2).backward()
                self.assertGreater(model.last_diagnostics['kl'],0.)
                self.assertFalse(model._checkpoint_context)
                outputs.append({n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None})
            for n in outputs[0]:
                torch.testing.assert_close(outputs[0][n],outputs[1][n],rtol=5e-4,atol=3e-6)

    def test_native_inference_persistence_and_recipe_rejection(self):
        core=native().eval()
        model=PrefixConsistencyMT(copy.deepcopy(core),'prefix-consistency-v1').eval()
        torch.testing.assert_close(model(**batch()).logits,core(**batch()).logits,rtol=0,atol=0)
        self.assertEqual(model.extra_parameter_count(),0)
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            restored=PrefixConsistencyMT(MBartForConditionalGeneration.from_pretrained(folder),'prefix-consistency-v1').eval()
            restored.verify_checkpoint_metadata(folder)
            torch.testing.assert_close(restored(**batch()).logits,core(**batch()).logits,rtol=0,atol=0)
            wrong=PrefixConsistencyMT(native(),'endpoint-consistency-v1')
            with self.assertRaises(ValueError): wrong.verify_checkpoint_metadata(folder)

    def test_extremes_and_empty_labels(self):
        a=torch.tensor([[[100.,-100.],[-100.,100.]]],requires_grad=True)
        b=-a.detach().clone().requires_grad_()
        y=torch.tensor([[0,1]])
        for mode in ('prefix','diagonal','endpoint'):
            loss=prefix_loss(a,b,y,mode=mode)[0]
            self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError): path_statistics(torch.zeros(1,2),torch.zeros(1,2,dtype=torch.bool))


if __name__=='__main__': unittest.main()
