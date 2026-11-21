"""Verify scale-matched operators, derivatives and persisted mode IDs."""
import copy
import tempfile
import unittest
import torch
from test_depth_difference import backbone, batch
from methods.depth_difference.v2.model import METHODS, ValueScaledDepthDifference


class OperatorControls(unittest.TestCase):
    def test_donor_train_zero_gate_rng_and_detached_memory(self):
        torch.set_num_threads(1); torch.manual_seed(42)
        base = backbone().train()
        model = ValueScaledDepthDifference(copy.deepcopy(base), donor_layer=2, mode='donor').train()
        torch.manual_seed(123); native = base(**batch()).logits
        torch.manual_seed(123); actual = model(**batch()).logits
        torch.testing.assert_close(actual, native)
        model(**batch()).loss.backward()
        grad = model.correction_attention.gates.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.norm(), 0)
        model.zero_grad(set_to_none=True)
        model.correction_attention.gates.data.fill_(.2)
        module = model.correction_attention.eval()
        donor = torch.randn(2,5,16,requires_grad=True)
        top = torch.randn(2,5,16,requires_grad=True)
        query = torch.randn(2,3,16,requires_grad=True)
        result, _ = module(query,torch.cat((top,donor),-1))
        result.square().sum().backward()
        self.assertTrue(donor.grad is None or donor.grad.eq(0).all())
        for projection in (module.attention.q_proj,module.attention.k_proj,module.attention.v_proj):
            self.assertTrue(torch.isfinite(projection.weight.grad).all())
            self.assertGreater(projection.weight.grad.norm(),0)

    def test_formula_gradients_identity(self):
        torch.manual_seed(42)
        model = ValueScaledDepthDifference(backbone(),donor_layer=2)
        module = model.correction_attention
        pt = torch.randn(2,2,3,5,dtype=torch.double).softmax(-1).requires_grad_()
        pd = torch.randn(2,2,3,5,dtype=torch.double).softmax(-1).requires_grad_()
        vt = torch.randn(2,2,5,8,dtype=torch.double,requires_grad=True)
        vd = torch.randn_like(vt,requires_grad=True)
        da,dv = pd-pt,vd-vt
        expected = dict(donor=pd@vd,ordinary=pd@vd-pt@vt,attention=da@vt,value=pt@dv,
            additive=da@vt+pt@dv,mixed=da@dv)
        actual={}
        for mode in METHODS.values():
            module.mode=mode
            actual[mode]=module.correction(pt,pd,vt,vd)
            torch.testing.assert_close(actual[mode],expected[mode])
            grads=torch.autograd.grad(actual[mode].square().sum(),(pt,pd,vt,vd),allow_unused=True,retain_graph=True)
            target=torch.autograd.grad(expected[mode].square().sum(),(pt,pd,vt,vd),allow_unused=True,retain_graph=True)
            for g,t in zip(grads,target):
                if t is None: self.assertIsNone(g)
                else: torch.testing.assert_close(g,t)
        torch.testing.assert_close(actual['ordinary'],actual['additive']+actual['mixed'])

    def test_all_modes_cache_and_serialization(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        base=backbone().eval()
        inputs=batch();inputs.pop('labels')
        for ident,mode in METHODS.items():
            model=ValueScaledDepthDifference(copy.deepcopy(base),donor_layer=2,mode=mode).eval()
            with torch.no_grad():
                torch.testing.assert_close(model(**batch()).logits,base(**batch()).logits)
                model.correction_attention.gates.fill_(.2)
                reference=model(**batch()).logits
            cached=model.generate(**inputs,num_beams=4,max_new_tokens=4,use_cache=True)
            uncached=model.generate(**inputs,num_beams=4,max_new_tokens=4,use_cache=False)
            self.assertTrue(torch.equal(cached,uncached),mode)
            with tempfile.TemporaryDirectory() as folder:
                model.save_pretrained(folder,state_dict=model.state_dict(),safe_serialization=False)
                restored=ValueScaledDepthDifference.from_pretrained(folder).eval()
                self.assertEqual(restored.method_id,ident)
                with torch.no_grad():torch.testing.assert_close(reference,restored(**batch()).logits)


if __name__=='__main__':unittest.main(verbosity=2)
