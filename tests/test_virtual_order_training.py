"""Wrapper checkpoint/RNG and mathematical integration checks."""
import copy
import tempfile
import unittest
from pathlib import Path
import torch
from transformers import MBartConfig, MBartForConditionalGeneration
from methods.virtual_corrective_order.v1.model import VirtualCorrectiveOrderMT
from methods.virtual_corrective_order.v1.objective import paired_ce_kl, order_penalty


class VirtualTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_wrapper_formula_reentrant_and_checkpoint_cursor(self):
        torch.manual_seed(12)
        base = MBartForConditionalGeneration(MBartConfig(vocab_size=19,d_model=8,
            encoder_layers=1,decoder_layers=1,encoder_attention_heads=2,decoder_attention_heads=2,
            encoder_ffn_dim=12,decoder_ffn_dim=12,max_position_embeddings=16,
            pad_token_id=1,eos_token_id=2,decoder_start_token_id=2,use_cache=False))
        ref = copy.deepcopy(base)
        model = VirtualCorrectiveOrderMT(base,rank=2,basis_seed=11,eta=.3,coefficient=2.,chunk=2)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
        ref.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
        x = torch.tensor([[4,5,2],[6,7,2],[8,9,2],[10,11,2]])
        y = x.roll(1,0)
        torch.manual_seed(99)
        actual = model(input_ids=x,labels=y).loss
        actual.backward()
        self.assertEqual(model.microbatch_calls,1)
        torch.manual_seed(99)
        hs = [ref.model(input_ids=x,decoder_input_ids=ref.prepare_decoder_input_ids_from_labels(y),
            use_cache=False).last_hidden_state for _ in range(2)]
        w,b = ref.lm_head.weight,ref.final_logits_bias.squeeze(0)
        ce = paired_ce_kl(*hs,y,w,b,2,True,False)[0]
        order,*_ = order_penalty(*hs,y,[0,1],[2,3],w,b,model.virtual_basis,.3,2,True)
        expected = ce+2*order
        expected.backward()
        torch.testing.assert_close(actual,expected)
        for a,c in zip(base.parameters(),ref.parameters()):
            torch.testing.assert_close(a.grad,c.grad)
        self.assertEqual(sum(p.numel() for p in base.parameters()),sum(p.numel() for p in model.parameters()))
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
        optimizer.step()
        model.eval()
        expected_logits = model(input_ids=x,labels=y).logits.detach()
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder,safe_serialization=False)
            restored = VirtualCorrectiveOrderMT(MBartForConditionalGeneration.from_pretrained(folder),
                rank=2,basis_seed=11,eta=.3,coefficient=2.,chunk=2)
            restored.verify_checkpoint_metadata(folder)
            self.assertEqual(restored.microbatch_calls,1)
            self.assertNotIn('virtual_basis',model.state_dict())
            restored.eval()
            torch.testing.assert_close(restored(input_ids=x,labels=y).logits,expected_logits)
            restored.train()
            restored(input_ids=x,labels=y)
            self.assertEqual(restored.last_diagnostics['support_rows'],[2,3])
            restored.coefficient=3.
            with self.assertRaises(ValueError): restored.verify_checkpoint_metadata(folder)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
