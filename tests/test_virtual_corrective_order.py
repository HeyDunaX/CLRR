"""CPU algebra checks against an independent inner-autograd reference."""
import unittest
import torch
import torch.nn.functional as F
from methods.virtual_corrective_order.v1.objective import endpoints,gradient,logits,order_penalty,paired_ce_kl


class VirtualOrderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        self.ha=torch.randn(4,3,5,dtype=torch.float64,requires_grad=True)
        self.hb=torch.randn(4,3,5,dtype=torch.float64,requires_grad=True)
        self.w=torch.randn(11,5,dtype=torch.float64,requires_grad=True)
        self.bias=torch.randn(11,dtype=torch.float64,requires_grad=True)
        self.B=torch.linalg.qr(torch.randn(5,2,dtype=torch.float64)).Q
        self.y=torch.tensor([[1,2,-100],[3,4,5],[2,5,6],[7,8,-100]])

    def calc(self,chunk=2,recompute=False):
        return order_penalty(self.ha,self.hb,self.y,[0,1],[2,3],self.w,self.bias,self.B,.3,chunk,recompute)

    def test_autograd_inner_and_outer_reference(self):
        w,b,B=self.w,self.bias,self.B
        S=w@B
        def step(h,A):
            z=logits(h[:2],w,b,B,S,A)
            ce=F.cross_entropy(z.flatten(0,1),self.y[:2].flatten(),ignore_index=-100)
            return A-.3*torch.autograd.grad(ce,A,create_graph=True)[0]
        zero=torch.zeros(2,2,dtype=torch.float64,requires_grad=True)
        ab,ba=step(self.hb,step(self.ha,zero)),step(self.ha,step(self.hb,zero))
        ref=0
        for h in (self.ha[2:],self.hb[2:]):
            lp=logits(h,w,b,B,S,ab).log_softmax(-1)
            lq=logits(h,w,b,B,S,ba).log_softmax(-1)
            vals=.5*((lp.exp()-lq.exp())*(lp-lq)).sum(-1)
            ref=ref+vals[self.y[2:].ne(-100)].mean()*.5
        loss,a,c=self.calc()
        torch.testing.assert_close(a,ab,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(c,ba,rtol=1e-12,atol=1e-12)
        params=(self.ha,self.hb,w,b)
        for actual,expected in zip(torch.autograd.grad(loss,params),torch.autograd.grad(ref,params)):
            torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-12)

    def test_null_and_exchange(self):
        z,a,b=order_penalty(self.ha,self.ha,self.y,[0,1],[2,3],self.w,self.bias,self.B,.3)
        self.assertEqual(z.item(),0.)
        self.assertTrue(torch.equal(a,b))
        loss,a,b=self.calc()
        reverse,c,d=order_penalty(self.hb,self.ha,self.y,[0,1],[2,3],self.w,self.bias,self.B,.3)
        torch.testing.assert_close(a,d,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(b,c,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(loss,reverse)

    def test_checkpoint_and_chunk_equivalence(self):
        ref,*_=self.calc(20)
        expected=torch.autograd.grad(ref,(self.ha,self.hb,self.w,self.bias))
        loss,*_=self.calc(2,True)
        for a,b in zip(torch.autograd.grad(loss,(self.ha,self.hb,self.w,self.bias)),expected):
            torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)

    def test_ce_kl_and_padding(self):
        vals=paired_ce_kl(self.ha,self.hb,self.y,self.w,self.bias,2,True)
        za,zb=F.linear(self.ha,self.w,self.bias),F.linear(self.hb,self.w,self.bias)
        ce=.5*sum(F.cross_entropy(z.flatten(0,1),self.y.flatten()) for z in (za,zb))
        torch.testing.assert_close(vals[0],ce)
        self.assertGreater(vals[1].item(),0)
        for h in (self.ha,self.hb):
            with torch.no_grad(): h[self.y.eq(-100)]=1e5
        after=paired_ce_kl(self.ha,self.hb,self.y,self.w,self.bias,3)
        torch.testing.assert_close(vals.detach(),after)

    def test_partition_rejection(self):
        with self.assertRaises(ValueError):
            order_penalty(self.ha,self.hb,self.y,[0,1],[1,2],self.w,self.bias,self.B,.3)

    def test_tiny_native_backbone_backward_without_updates(self):
        from transformers import MBartConfig,MBartForConditionalGeneration
        model=MBartForConditionalGeneration(MBartConfig(vocab_size=19,d_model=8,
            encoder_layers=1,decoder_layers=1,encoder_attention_heads=2,decoder_attention_heads=2,
            encoder_ffn_dim=12,decoder_ffn_dim=12,max_position_embeddings=16,
            pad_token_id=1,eos_token_id=2,decoder_start_token_id=2,use_cache=False)).train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        before=[p._version for p in model.parameters()]
        x=torch.tensor([[4,5,2],[6,7,2],[8,9,2],[10,11,2]])
        y=x.roll(1,0)
        decoder=model.prepare_decoder_input_ids_from_labels(y)
        a,b=[model.model(input_ids=x,decoder_input_ids=decoder,use_cache=False).last_hidden_state for _ in range(2)]
        B=torch.linalg.qr(torch.randn(8,2)).Q
        penalty,*_=order_penalty(a,b,y,[0,1],[2,3],model.lm_head.weight,model.final_logits_bias.squeeze(0),B,.3,2,True)
        ce,_=paired_ce_kl(a,b,y,model.lm_head.weight,model.final_logits_bias.squeeze(0),2,True)
        (ce+penalty).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(before,[p._version for p in model.parameters()])


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
