"""Correctness of all-training relational objectives and original CLRR routing."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import MBartForConditionalGeneration
from methods.registry import wrap_method
from methods.difference_preserving.v2.model import METHODS, content_mask, pair_eligibility
from test_alignment_residual import native


def training_batch():
    ids = torch.tensor([[4,5,2,1], [6,7,8,2], [9,10,2,1], [11,12,13,2]])
    labels = torch.tensor([[14,15,2,-100], [16,17,18,2], [19,20,2,-100], [21,22,23,2]])
    return dict(input_ids=ids, attention_mask=ids.ne(1).long(), labels=labels)


def model_for(method, core=None):
    result = wrap_method(native(.1) if core is None else core,
                         dict(method=method, alignment_weight=.1))
    result.special_token_ids = (1,2)
    return result


class RelationalModelGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pair_eligibility_excludes_empty_and_duplicate_content(self):
        ids = torch.tensor([[4,2,1], [4,1,2], [5,2,1], [2,1,1]])
        mask = content_mask(ids, ids.ne(1), (1,2))
        eligible = pair_eligibility(ids, mask, ids, mask)
        self.assertEqual(eligible.sum().item(), 4)
        self.assertFalse(eligible[0,1])
        self.assertFalse(eligible[3].any())
        self.assertTrue(torch.equal(eligible, eligible.T))

    def test_auxiliary_preserves_ce_logits_rng_and_nonzero_gradients(self):
        data = training_batch()
        for method in METHODS:
            with self.subTest(method=method):
                model = model_for(method).train()
                torch.manual_seed(103)
                expected = model.base_model(**data)
                rng = torch.get_rng_state()
                torch.manual_seed(103)
                actual = model(**data)
                torch.testing.assert_close(expected.logits, actual.logits, rtol=0, atol=0)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertGreater(actual.loss.item(), expected.loss.item())
                self.assertEqual(model.last_diagnostics['valid_pairs'], 6)
                self.assertEqual(sum(p.numel() for p in model.parameters()),
                                 sum(p.numel() for p in model.base_model.parameters()))
                auxiliary = actual.loss - expected.loss
                gradient = torch.autograd.grad(auxiliary, model.get_input_embeddings().weight,
                                               retain_graph=True)[0]
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)
                actual.loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                    for p in model.parameters()))

    def test_checkpoint_replay_reload_and_metadata_all_four_arms(self):
        data = training_batch()
        for method in METHODS:
            with self.subTest(method=method):
                initial = native(.15)
                a, b = (model_for(method, copy.deepcopy(initial)) for _ in range(2))
                b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
                for model in (a,b):
                    torch.manual_seed(89)
                    model.train()(**data).loss.backward()
                for (name,left), (_,right) in zip(a.named_parameters(),b.named_parameters()):
                    self.assertIsNotNone(left.grad,name)
                    torch.testing.assert_close(left.grad,right.grad,rtol=2e-4,atol=5e-5)
                a.eval()
                expected = a(**data).logits
                with tempfile.TemporaryDirectory() as folder:
                    a.save_pretrained(folder,state_dict=a.state_dict(),safe_serialization=False)
                    core = MBartForConditionalGeneration.from_pretrained(folder,attn_implementation='sdpa')
                    restored = model_for(method,core).eval()
                    restored.verify_checkpoint_metadata(folder)
                    torch.testing.assert_close(expected,restored(**data).logits)
                    restored.alignment_weight = .2
                    with self.assertRaisesRegex(ValueError,'recipe mismatch'):
                        restored.verify_checkpoint_metadata(folder)

    def test_generation_matches_parent_and_empty_content_keeps_ce(self):
        data = training_batch()
        for method in METHODS:
            model = model_for(method).eval()
            generation = {k:v for k,v in data.items() if k != 'labels'}
            torch.testing.assert_close(model.generate(**generation,max_new_tokens=3,num_beams=2),
                model.base_model.generate(**generation,max_new_tokens=3,num_beams=2),rtol=0,atol=0)
            empty = {k:v.clone() for k,v in data.items()}
            empty['input_ids'].fill_(2)
            model.train()(**empty).loss.backward()
            self.assertEqual(model.last_diagnostics['valid_pairs'],0)
            self.assertEqual(model.last_diagnostics['relational_loss'],0.)

    def test_trainer_accumulation_resume_matches_uninterrupted(self):
        from transformers import Seq2SeqTrainingArguments, TrainerCallback
        from experiment_runner.engine import ControlTrainer

        class StopAfterTwo(TrainerCallback):
            def on_step_end(self,args,state,control,**kwargs):
                if state.global_step == 2:
                    control.should_save = control.should_training_stop = True
                return control

        data = training_batch()
        rows = [{k:v[i] for k,v in data.items()} for i in range(4)] * 8
        def collate(features):
            return {k:torch.stack([row[k] for row in features]) for k in features[0]}
        for method in METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                initial = native(.1)
                def trainer(name,callbacks=None):
                    args = Seq2SeqTrainingArguments(output_dir=str(Path(folder,name)),use_cpu=True,
                        max_steps=3,per_device_train_batch_size=4,gradient_accumulation_steps=4,
                        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':True},
                        learning_rate=5e-5,save_strategy='steps',save_steps=2,report_to=[],
                        disable_tqdm=True,logging_strategy='no',remove_unused_columns=False,
                        dataloader_pin_memory=False,seed=42)
                    return ControlTrainer(model=model_for(method,copy.deepcopy(initial)),args=args,
                        train_dataset=rows,data_collator=collate,callbacks=callbacks)
                continuous = trainer('continuous'); continuous.train()
                interrupted = trainer('interrupted',[StopAfterTwo()]); interrupted.train()
                checkpoint = Path(folder,'interrupted','checkpoint-2')
                interrupted.model.verify_checkpoint_metadata(checkpoint)
                resumed = trainer('resumed'); resumed.train(resume_from_checkpoint=str(checkpoint))
                for name,value in continuous.model.state_dict().items():
                    torch.testing.assert_close(value,resumed.model.state_dict()[name],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
