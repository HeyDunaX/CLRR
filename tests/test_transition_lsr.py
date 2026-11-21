"""Scientific gates for pre-rewire transitions and isolated gradients."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import MBartForConditionalGeneration
from methods.registry import wrap_method
from methods.transition_lsr.v1.model import METHODS, transition_cosine
from methods.common.helpers import _find_stack_layers, _masked_mean_pool
from test_alignment_residual import native, batch


def model_for(method, core=None):
    return wrap_method(native(.1) if core is None else core,
                       dict(method=method, alignment_weight=.1))


class TransitionGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_explicit_pre_and_post_route_formula_and_hook_cleanup(self):
        data = batch()
        for method, routed in METHODS.items():
            model = model_for(method).eval()
            layers = _find_stack_layers(model.base_model, 'encoder')
            before, after, handles = [], [], []
            for layer in layers:
                handles.append(layer.register_forward_hook(
                    lambda m,i,o: before.append(o[0].detach().clone()), prepend=True))
                handles.append(layer.register_forward_hook(
                    lambda m,i,o: after.append(o[0].detach().clone())))
            with model.deterministic_view(True), torch.no_grad():
                model.get_encoder()(input_ids=data['input_ids'],attention_mask=data['attention_mask'])
            for handle in handles: handle.remove()
            expected = torch.stack([_masked_mean_pool(before[i],data['attention_mask']) -
                _masked_mean_pool(after[i-2],data['attention_mask']) for i in range(2,len(layers))])
            for i in range(2,len(layers)):
                torch.testing.assert_close(after[i],before[i]+(.1 if routed else 0)*after[i-2])
            counts = [len(layer._forward_hooks) for layer in layers]
            actual = model.transitions(data['input_ids'],data['attention_mask'])
            torch.testing.assert_close(actual,expected)
            self.assertEqual(counts,[len(layer._forward_hooks) for layer in layers])

    def test_gradient_stop_geometry_padding_and_zero_vectors(self):
        u = torch.randn(3,2,7,requires_grad=True)
        donor = torch.randn_like(u,requires_grad=True)
        target = torch.randn_like(u,requires_grad=True)
        loss = transition_cosine(u-donor.detach(),target)
        loss.backward()
        self.assertGreater(u.grad.abs().sum().item(),0)
        self.assertIsNone(donor.grad)
        self.assertIsNone(target.grad)
        zero = torch.zeros_like(u,requires_grad=True)
        transition_cosine(zero,zero).backward()
        self.assertTrue(torch.isfinite(zero.grad).all())
        data = batch()
        for method in METHODS:
            model = model_for(method).eval()
            expected = model.transitions(data['input_ids'],data['attention_mask'])
            padded = torch.nn.functional.pad(data['input_ids'],(0,2),value=1)
            mask = torch.nn.functional.pad(data['attention_mask'],(0,2))
            torch.testing.assert_close(expected,model.transitions(padded,mask),atol=1e-6,rtol=1e-5)

    def test_ce_logits_rng_zero_parameters_and_active_auxiliary(self):
        data = batch()
        for method in METHODS:
            model = model_for(method).train()
            torch.manual_seed(103)
            expected = model.base_model(**data)
            rng = torch.get_rng_state()
            torch.manual_seed(103)
            actual = model(**data)
            torch.testing.assert_close(expected.logits,actual.logits,rtol=0,atol=0)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertGreater(actual.loss.item(),expected.loss.item())
            self.assertTrue(model.last_diagnostics['source_requires_grad'])
            self.assertFalse(model.last_diagnostics['target_requires_grad'])
            self.assertEqual(sum(p.numel() for p in model.parameters()),
                             sum(p.numel() for p in model.base_model.parameters()))
            grad = torch.autograd.grad(actual.loss-expected.loss,
                model.get_input_embeddings().weight,retain_graph=True)[0]
            self.assertGreater(grad.abs().sum().item(),0)
            self.assertTrue(torch.isfinite(grad).all())

    def test_checkpointing_gradients_and_save_reload_metadata(self):
        data = batch()
        for method in METHODS:
            core = native(.15)
            a,b = [model_for(method,copy.deepcopy(core)) for _ in range(2)]
            b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
            for model in (a,b):
                torch.manual_seed(89)
                model.train()(**data).loss.backward()
            for (name,left),(_,right) in zip(a.named_parameters(),b.named_parameters()):
                self.assertIsNotNone(left.grad,name)
                torch.testing.assert_close(left.grad,right.grad,rtol=2e-4,atol=5e-5)
            with tempfile.TemporaryDirectory() as folder:
                a.save_pretrained(folder,state_dict=a.state_dict(),safe_serialization=False)
                core = MBartForConditionalGeneration.from_pretrained(folder,attn_implementation='sdpa')
                restored = model_for(method,core).eval()
                restored.verify_checkpoint_metadata(folder)
                torch.testing.assert_close(a.eval()(**data).logits,restored(**data).logits)
                restored.alignment_weight = .2
                with self.assertRaisesRegex(ValueError,'recipe mismatch'):
                    restored.verify_checkpoint_metadata(folder)

    def test_accumulation_and_exact_optimizer_rng_resume(self):
        from transformers import Seq2SeqTrainingArguments, TrainerCallback
        from experiment_runner.engine import ControlTrainer

        class StopAfterTwo(TrainerCallback):
            def on_step_end(self,args,state,control,**kwargs):
                if state.global_step == 2:
                    control.should_save = control.should_training_stop = True
                return control

        data = batch()
        rows = [{k:v[i] for k,v in data.items()} for i in range(2)]*16
        def collate(features):
            return {k:torch.stack([row[k] for row in features]) for k in features[0]}
        for method in METHODS:
            with tempfile.TemporaryDirectory() as folder:
                core = native(.1)
                def trainer(name,callbacks=None):
                    args = Seq2SeqTrainingArguments(output_dir=str(Path(folder,name)),use_cpu=True,
                        max_steps=3,per_device_train_batch_size=2,gradient_accumulation_steps=4,
                        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':True},
                        learning_rate=5e-5,save_strategy='steps',save_steps=2,report_to=[],
                        disable_tqdm=True,logging_strategy='no',remove_unused_columns=False,
                        dataloader_pin_memory=False,seed=42)
                    return ControlTrainer(model=model_for(method,copy.deepcopy(core)),args=args,
                        train_dataset=rows,data_collator=collate,callbacks=callbacks)
                continuous=trainer('continuous');continuous.train()
                interrupted=trainer('interrupted',[StopAfterTwo()]);interrupted.train()
                checkpoint=Path(folder,'interrupted/checkpoint-2')
                resumed=trainer('resumed');resumed.train(resume_from_checkpoint=str(checkpoint))
                for name,value in continuous.model.state_dict().items():
                    torch.testing.assert_close(value,resumed.model.state_dict()[name],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
