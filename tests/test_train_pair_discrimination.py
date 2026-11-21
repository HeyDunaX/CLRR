"""Check diagnostic score units, masks and target-prior cancellation."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec=importlib.util.spec_from_file_location('pair_diagnostic',Path(__file__).resolve().parents[1]/'scripts/analysis/measure_train_pair_discrimination.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class PairDiagnosticGate(unittest.TestCase):
    def test_token_scores_match_direct_logprob_and_ignore_specials_padding(self):
        logits=torch.tensor([[[1.,2.,3.,4.],[2.,4.,1.,0.],[0.,1.,2.,3.]],
                             [[2.,1.,3.,0.],[1.,0.,3.,2.],[4.,0.,1.,2.]]])
        labels=torch.tensor([[1,3,-100],[2,0,1]])
        content,all_scores,counts=module.token_scores(logits,labels,{0,1})
        lp=logits.log_softmax(-1)
        torch.testing.assert_close(content,torch.stack([lp[0,1,3],lp[1,0,2]]))
        torch.testing.assert_close(all_scores,torch.stack([(lp[0,0,1]+lp[0,1,3])/2,
            (lp[1,0,2]+lp[1,1,0]+lp[1,2,1])/3]))
        self.assertEqual(counts.tolist(),[1,1])

    def test_source_margins_cancel_additive_target_prior(self):
        base=module.margins(-1.,-3.,-4.,-2.)
        biased=module.margins(-1.+5.,-3.+8.,-4.+5.,-2.+8.)
        for key in ('source_margin_left','source_margin_right','symmetric_source_margin'):
            self.assertEqual(base[key],biased[key])
        self.assertEqual(base['symmetric_source_margin'],
            (base['reference_margin_left']+base['reference_margin_right'])/2)

    def test_empty_content_is_rejected(self):
        with self.assertRaisesRegex(AssertionError,'Empty content'):
            module.token_scores(torch.zeros(1,2,4),torch.tensor([[1,-100]]),{1})


if __name__=='__main__':unittest.main()
