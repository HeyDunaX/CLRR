"""Estimator correctness before any new full-model training."""
import copy
import tempfile
import unittest
from unittest.mock import patch
import torch
from transformers import MBartForConditionalGeneration
from test_alignment_residual import native,batch
from methods.continuation_consistency.v2 import EstimatorConsistencyMT,METHODS
from methods.paired_route.v1.model import paired_loss


class EstimatorGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_channel_equals_dense_w_loss_and_parameter_gradients(self):
        base=native(.15)
        reference=copy.deepcopy(base).train()
        model=EstimatorConsistencyMT(base,'continuation-control-variate-v2').train()
        data=batch()
        def identity(input_ids,attention_mask,decoder,rows):
            return [torch.eye(40)[actions].clamp_min(1e-30).log() for _,_,actions in rows]
        torch.manual_seed(981)
        with patch.object(model,'continuation_channel',side_effect=identity):
            actual=model(**data).loss
        rng=torch.get_rng_state().clone()
        actual.backward()
        torch.manual_seed(981)
        decoder=reference.prepare_decoder_input_ids_from_labels(data['labels'])
        logits=[reference(input_ids=data['input_ids'],attention_mask=data['attention_mask'],
            decoder_input_ids=decoder,use_cache=False).logits for _ in range(2)]
        expected,_,_=paired_loss(*logits,data['labels'],1.25)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        expected.backward()
        torch.testing.assert_close(actual,expected)
        for a,b in zip(base.parameters(),reference.parameters()):
            if a.grad is not None or b.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=3e-4)

    def test_reentrant_checkpoint_gradient_replay(self):
        for method in METHODS:
            base=native(.15)
            plain=EstimatorConsistencyMT(copy.deepcopy(base),method).train()
            checked=EstimatorConsistencyMT(copy.deepcopy(base),method).train()
            checked.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
            for model in (plain,checked):
                torch.manual_seed(983)
                model(**batch()).loss.backward()
                self.assertFalse(model._checkpoint_context)
            for a,b in zip(plain.parameters(),checked.parameters()):
                if a.grad is not None or b.grad is not None:
                    torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=3e-4)

    def test_reload_native_inference_and_recipe_mismatch(self):
        for method in METHODS:
            model=EstimatorConsistencyMT(native(),method).eval()
            expected=model(**batch()).logits
            with tempfile.TemporaryDirectory() as folder:
                model.save_pretrained(folder)
                restored=EstimatorConsistencyMT(MBartForConditionalGeneration.from_pretrained(folder),method).eval()
                restored.verify_checkpoint_metadata(folder)
                torch.testing.assert_close(expected,restored(**batch()).logits,rtol=0,atol=0)
                restored.beta=.5
                with self.assertRaises(ValueError):
                    restored.verify_checkpoint_metadata(folder)
            self.assertEqual(model.extra_parameter_count(),0)


if __name__=='__main__':
    unittest.main()
