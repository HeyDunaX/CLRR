import sys
import unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
import torch
from diagnose_token_decoding import token_decisions,category


class TokenDecisionGate(unittest.TestCase):
    def test_gold_competitor_rank_margin_ties_and_padding(self):
        logits=torch.tensor([[[3.,1.,0.],[1.,3.,2.],[2.,2.,0.],[0.,0.,1.]]])
        labels=torch.tensor([[0,2,1,-100]])
        d=token_decisions(logits,labels,2)
        self.assertEqual(d['correct'][0,:3].tolist(),[True,False,False])
        self.assertEqual(d['rank'][0,:3].tolist(),[1,2,1])
        torch.testing.assert_close(d['margin'][0,:3],torch.tensor([2.,-1.,0.]))
        self.assertEqual(d['nll'][0,-1].item(),0.)
        torch.testing.assert_close(d['nll'].sum()/3,torch.nn.functional.cross_entropy(logits.reshape(-1,3),labels.reshape(-1)))
        torch.testing.assert_close(d['eos_probability'][0,:3],logits.softmax(-1)[0,:3,2])

    def test_frequency_groups_and_specials(self):
        f=Counter({4:1,5:5,6:6,7:50,8:51})
        self.assertEqual([category(t,f,{0,1,2},2) for t in [2,0,3,4,5,6,7,8]],
            ['eos','other_special','content_unseen','content_rare_1_5','content_rare_1_5',
             'content_mid_6_50','content_mid_6_50','content_common_gt50'])


if __name__=='__main__':unittest.main()
