"""Scientific and replay gates for the two independent method candidates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import MBartForConditionalGeneration

from test_alignment_residual import native, batch
from methods.paired_route.v1.model import paired_loss
from methods.antisymmetric_consistency.v1.model import AntisymmetricConsistencyMT, antisymmetric_loss
from methods.decision_residual_memory.v1.model import DecisionResidualMemoryMT, update_memory


class TwoMethodGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_ac_forward_and_gradient_formula_with_padding(self):
        torch.manual_seed(120)
        a=torch.randn(2,3,9,requires_grad=True)
        b=torch.randn(2,3,9,requires_grad=True)
        labels=torch.tensor([[3,4,2],[5,2,-100]])
        actual,ce,kl=antisymmetric_loss(a,b,labels,chunk_size=2)
        expected,ce_ref,k_ref=paired_loss(a,b,labels,1.25,chunk_size=2)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(ce,ce_ref)
        k_ref.backward(retain_graph=True)
        g1,g2=a.grad.clone(),b.grad.clone()
        a.grad=b.grad=None
        kl.backward()
        h1,h2=a.grad,b.grad
        torch.testing.assert_close(h1,(g1-g2)/2,atol=1e-7,rtol=1e-5)
        torch.testing.assert_close(h2,-h1,atol=1e-7,rtol=1e-5)
        self.assertEqual(h1[1,2].abs().sum(),0)

    def test_checkpoint_replay_gradients(self):
        for wrapper in (AntisymmetricConsistencyMT,DecisionResidualMemoryMT):
            torch.manual_seed(121)
            base=native(.15)
            plain=wrapper(copy.deepcopy(base)).train()
            checked=wrapper(copy.deepcopy(base)).train()
            checked.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
            for model in (plain,checked):
                torch.manual_seed(122)
                model(**batch()).loss.backward()
                self.assertFalse(model._checkpoint_context)
            count=0
            for a,b in zip(plain.parameters(),checked.parameters()):
                if a.grad is not None or b.grad is not None:
                    torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=4e-4)
                    self.assertTrue(torch.isfinite(a.grad).all())
                    count+=1
            self.assertGreater(count,40)

    def test_drm_native_eta_zero(self):
        base=native().eval()
        reference=copy.deepcopy(base)
        model=DecisionResidualMemoryMT(base,eta=0).eval()
        data=batch()
        reference(**data).loss.backward()
        model(**data).loss.backward()
        for a,b in zip(reference.parameters(),model.parameters()):
            if a.grad is not None or b.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        with torch.no_grad():
            torch.testing.assert_close(model(**data).logits,reference(**data).logits,rtol=0,atol=0)

    def test_drm_prefix_causality(self):
        model=DecisionResidualMemoryMT(native()).eval()
        data=batch()
        decoder=model.prepare_decoder_input_ids_from_labels(data['labels'])
        original=model(**data,decoder_input_ids=decoder).logits.detach()
        modified=dict(data)
        modified['labels']=data['labels'].clone()
        modified['labels'][:,1]=15
        changed=model(**modified,decoder_input_ids=decoder).logits.detach()
        torch.testing.assert_close(changed[:,:2],original[:,:2],rtol=0,atol=0)
        self.assertGreater((changed[:,2]-original[:,2]).abs().max(),0)
        # Given the exact observed prefix, inference and teacher-forced memory
        # must agree at every next-token prediction.
        with torch.no_grad():
            for t in range(1,4):
                decoded=model(input_ids=data['input_ids'],attention_mask=data['attention_mask'],
                    decoder_input_ids=decoder[:,:t]).logits
                torch.testing.assert_close(decoded[:,-1],original[:,t-1],rtol=1e-5,atol=2e-6)

    def test_batch_padding_and_beam_local_replay(self):
        model=DecisionResidualMemoryMT(native()).eval()
        data=batch()
        with torch.no_grad():
            expected=model(**data).logits
            for i in range(2):
                one={k:v[i:i+1] for k,v in data.items()}
                torch.testing.assert_close(model(**one).logits,expected[i:i+1],rtol=1e-5,atol=2e-6)
            source={k:data[k] for k in ('input_ids','attention_mask')}
            for beams in (1,4):
                first=model.generate(**source,num_beams=beams,max_new_tokens=5,use_cache=True)
                second=model.generate(**source,num_beams=beams,max_new_tokens=5,use_cache=False)
                self.assertTrue(torch.equal(first,second))
                # Reordered batch cannot carry state from the first call.
                flipped=model.generate(**{k:v.flip(0) for k,v in source.items()},num_beams=beams,max_new_tokens=5)
                self.assertTrue(torch.equal(first,flipped.flip(0)))

    def test_drm_trainer_prediction_with_labels(self):
        from transformers import Seq2SeqTrainingArguments, default_data_collator
        from amis_rewire.evaluation import PrecisionSeq2SeqTrainer
        model=DecisionResidualMemoryMT(native()).eval()
        data=batch()
        data['decoder_input_ids']=model.prepare_decoder_input_ids_from_labels(data['labels'])
        rows=[{k:v[i] for k,v in data.items()} for i in range(2)]
        with tempfile.TemporaryDirectory() as folder:
            args=Seq2SeqTrainingArguments(output_dir=folder,use_cpu=True,
                per_device_eval_batch_size=2,predict_with_generate=True,
                report_to=[],disable_tqdm=True)
            trainer=PrecisionSeq2SeqTrainer(model=model,args=args,
                data_collator=default_data_collator)
            for beams in (1,4):
                trainer.args.generation_num_beams=beams
                trainer.args.generation_max_length=6
                result=trainer.predict(rows)
                self.assertEqual(result.predictions.shape[0],2)
                self.assertTrue(torch.isfinite(torch.tensor(result.metrics['test_loss'])))
                source={k:data[k] for k in ('input_ids','attention_mask')}
                expected=model.generate(**source,num_beams=beams,max_new_tokens=5)
                changed=data['labels'].clone();changed[:,:]=16
                actual=model.generate(**source,labels=changed,num_beams=beams,max_new_tokens=5)
                self.assertTrue(torch.equal(expected,actual))
                torch.testing.assert_close(torch.as_tensor(result.predictions)[:,:expected.shape[1]],expected)
        self.assertIsNone(model._memory_labels)
        self.assertIsNone(model._memory_decoder)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_memory_operator_and_repeated_fact_counterexample(self):
        memory=torch.tensor([[[[1.,1.],[0.,1.],[-1.,0.]]]])
        u=torch.tensor([[[1.,0.]]])
        a=torch.tensor([[[.5,.25,.25]]])
        updated=update_memory(memory,a,u,.1,torch.tensor([True]))
        self.assertTrue((updated.norm(dim=-1)<=memory.norm(dim=-1)+1e-7).all())
        torch.testing.assert_close(updated[...,1],memory[...,1])
        torch.testing.assert_close(updated[0,0,2],memory[0,0,2])
        torch.testing.assert_close(update_memory(memory,a,u,.1,torch.tensor([False])),memory)
        self.assertLess(updated[0,0,0,0],memory[0,0,0,0])

    def test_reload_and_reject_recipe_change(self):
        for wrapper in (AntisymmetricConsistencyMT,DecisionResidualMemoryMT):
            model=wrapper(native()).eval()
            with torch.no_grad():
                expected=model(**batch()).logits
            with tempfile.TemporaryDirectory() as folder:
                model.save_pretrained(folder)
                restored=wrapper(MBartForConditionalGeneration.from_pretrained(folder)).eval()
                restored.verify_checkpoint_metadata(folder)
                with torch.no_grad():
                    torch.testing.assert_close(restored(**batch()).logits,expected,rtol=0,atol=0)
                restored.beta=.5
                with self.assertRaises(ValueError):
                    restored.verify_checkpoint_metadata(folder)
            self.assertEqual(model.extra_parameter_count(),0)

    def test_accumulation_four_and_inference_has_no_grad_side_effect(self):
        for wrapper in (AntisymmetricConsistencyMT,DecisionResidualMemoryMT):
            torch.manual_seed(901)
            full=wrapper(native()).train()
            accumulated=wrapper(native()).train()
            # Copy only native weights; never copy DRM's bound layer closures.
            accumulated.load_state_dict(full.state_dict())
            data=batch()
            data['labels'][0,-1]=14
            large={k:v.repeat(2,1) for k,v in data.items()}
            full(**large).loss.backward()
            for i in range(4):
                (accumulated(**{k:v[i:i+1] for k,v in large.items()}).loss/4).backward()
            # Variable token counts intentionally use microbatch means in the
            # locked recipe; this fixture has equal supervised lengths.
            for a,b in zip(full.parameters(),accumulated.parameters()):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=2e-6)
            accumulated.zero_grad(set_to_none=True)
            accumulated.eval()
            with torch.no_grad():
                accumulated(**data)
            self.assertTrue(all(p.grad is None for p in accumulated.parameters()))


if __name__=='__main__':
    unittest.main()
