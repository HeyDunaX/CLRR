"""Scientific gates for DRB; synthetic models, no dataset optimization."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from test_alignment_residual import native, batch
from methods.decoder_residual_bridge.v1 import DecoderResidualBridge, METHODS


def wrapped(base, method='drb-query-v1'):
    return DecoderResidualBridge(copy.deepcopy(base), method, donor_layer=2, decoder_layer=2, rank=4)


def activate(model):
    with torch.no_grad():
        generator = torch.Generator().manual_seed(987)
        model.correction_attention.up.copy_(torch.randn(model.correction_attention.up.shape, generator=generator)*.02)


class BridgeGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.base = native(.1)

    def test_up_gradient_finite_difference(self):
        model = wrapped(self.base).double().eval()
        result = model(**batch()); result.loss.backward()
        up = model.correction_attention.up
        index = tuple(int(x) for x in torch.nonzero(up.grad.abs() == up.grad.abs().max())[0])
        analytic = up.grad[index].item()
        epsilon = 1e-5
        with torch.no_grad():
            up[index] = epsilon
            plus = model(**batch()).loss.item()
            up[index] = -epsilon
            minus = model(**batch()).loss.item()
            up[index] = 0.
        self.assertAlmostEqual(analytic, (plus-minus)/(2*epsilon), delta=1e-7)

    def test_zero_init_native_logits_gradients_and_rng(self):
        model = wrapped(self.base)
        baseline = copy.deepcopy(self.base)
        rng = torch.get_rng_state().clone()
        wrapped(self.base)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        states = []
        for item in (baseline, model):
            torch.manual_seed(321)
            result = item(**batch()); result.loss.backward()
            states.append((result.logits.detach(), torch.get_rng_state()))
        torch.testing.assert_close(states[0][0], states[1][0], atol=2e-5, rtol=2e-4)
        self.assertTrue(torch.equal(states[0][1], states[1][1]))
        parameters = dict(model.named_parameters())
        for name, p in baseline.named_parameters():
            mapped = 'base_model.'+name
            if '.encoder_attn.' in mapped:
                mapped = mapped.replace('.encoder_attn.', '.encoder_attn.attention.')
            torch.testing.assert_close(p.grad, parameters[mapped].grad, atol=5e-5, rtol=2e-4, msg=name)
        self.assertEqual(model.extra_parameter_count(), 3*16*4)

    def test_gradient_flow_and_second_step(self):
        model = wrapped(self.base)
        model(**batch()).loss.backward()
        a = model.correction_attention
        self.assertGreater(float(a.up.grad.norm()), 0)
        self.assertEqual(float(a.down.grad.norm()), 0)
        self.assertEqual(float(a.query.grad.norm()), 0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        model(**batch()).loss.backward()
        self.assertGreater(float(a.down.grad.norm()), 0)
        self.assertGreater(float(a.query.grad.norm()), 0)

    def test_explicit_formula_and_detached_side(self):
        model = wrapped(self.base).eval().double(); activate(model)
        a = model.correction_attention
        top = torch.randn(2,4,16,dtype=torch.float64,requires_grad=True)
        low = torch.randn_like(top,requires_grad=True)
        q = torch.randn(2,3,16,dtype=torch.float64,requires_grad=True)
        content = torch.ones(2,4,dtype=torch.float64)
        e, query = a.side_features(q,top,low,content,None,None)
        self.assertFalse(e.requires_grad); self.assertFalse(query.requires_grad)
        h = lambda x:x.reshape(2,-1,2,8).transpose(1,2)
        n = lambda x:x*(x.square().mean(-1,keepdim=True)+1e-6).rsqrt()
        p = (h(a.attention.q_proj(q)) @ h(a.attention.k_proj(top)).transpose(-1,-2)*a.attention.scaling).softmax(-1)
        expected = (p @ h(a.attention.v_proj(n(low))-a.attention.v_proj(n(top)))).transpose(1,2).reshape(2,3,16)
        torch.testing.assert_close(e,n(expected),atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(query,n(q),atol=1e-10,rtol=1e-10)
        actual,_ = a(q,torch.cat((top,low,content[...,None]),-1))
        ordinary,_ = a.attention(q,key_value_states=top)
        delta = torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(e,a.down))*torch.sigmoid(torch.nn.functional.linear(query,a.query)),a.up)
        torch.testing.assert_close(actual,ordinary+delta)

    def test_equal_depth_null_and_arms_distinct(self):
        outputs = []
        for method in METHODS:
            model = wrapped(self.base,method).eval(); activate(model)
            outputs.append(model(**batch()).logits.detach())
        for i in range(4):
            for j in range(i):
                self.assertFalse(torch.allclose(outputs[i],outputs[j],atol=1e-8,rtol=0))
        model = wrapped(self.base).eval(); activate(model)
        memory = model.encode(batch()['input_ids'],batch()['attention_mask'])
        memory.donor_hidden_state = memory.last_hidden_state.detach()
        out=model(encoder_outputs=memory,attention_mask=batch()['attention_mask'],labels=batch()['labels']).logits
        baseline=copy.deepcopy(self.base).eval()(**batch()).logits
        torch.testing.assert_close(out,baseline,atol=2e-5,rtol=2e-4)

    def test_checkpointing_dropout_and_gradients(self):
        a=wrapped(self.base);activate(a);b=copy.deepcopy(a)
        b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
        for item in (a,b):
            torch.manual_seed(123);item(**batch()).loss.backward()
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,atol=5e-5,rtol=2e-4,msg=name)

    def test_cache_beams_and_batch_independence(self):
        model=wrapped(self.base).eval();activate(model)
        inputs={k:v for k,v in batch().items() if k!='labels'}
        model.generation_config.eos_token_id=None
        for beams in (1,4):
            a=model.generate(**inputs,max_new_tokens=5,num_beams=beams,use_cache=True)
            b=model.generate(**inputs,max_new_tokens=5,num_beams=beams,use_cache=False)
            self.assertTrue(torch.equal(a,b))
            for row in range(2):
                single={k:v[row:row+1] for k,v in inputs.items()}
                c=model.generate(**single,max_new_tokens=5,num_beams=beams,use_cache=True)
                self.assertTrue(torch.equal(a[row:row+1],c))

    def test_padding_and_causal_prefix(self):
        model=wrapped(self.base).eval();activate(model)
        data=batch();expected=model(**data).logits
        padded=dict(data)
        padded['input_ids']=torch.nn.functional.pad(data['input_ids'],(0,2),value=1)
        padded['attention_mask']=torch.nn.functional.pad(data['attention_mask'],(0,2),value=0)
        torch.testing.assert_close(expected,model(**padded).logits,atol=2e-5,rtol=2e-4)
        prefix={**data,'decoder_input_ids':torch.tensor([[2,9,10,2],[2,11,12,13]])}
        before=model(**prefix).logits
        changed={**prefix,'decoder_input_ids':prefix['decoder_input_ids'].clone()}
        changed['decoder_input_ids'][:,3]=15
        torch.testing.assert_close(before[:,:3],model(**changed).logits[:,:3],atol=2e-5,rtol=2e-4)

    def test_roundtrip_and_accumulation(self):
        model=wrapped(self.base).eval();activate(model)
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            reloaded=DecoderResidualBridge.from_pretrained(folder).eval()
            torch.testing.assert_close(model(**batch()).logits,reloaded(**batch()).logits)
        full=copy.deepcopy(model);accum=copy.deepcopy(model)
        # Repeating the same microbatch checks loss scaling without token-length confounds.
        full(**batch()).loss.backward()
        for _ in range(4): (accum(**batch()).loss/4).backward()
        for (name,p),(_,q) in zip(full.named_parameters(),accum.named_parameters()):
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-5,rtol=2e-4,msg=name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
