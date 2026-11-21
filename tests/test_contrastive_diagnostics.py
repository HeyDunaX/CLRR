"""Check diagnostic retrieval and gradient loss decomposition on tiny real mBART."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests'))
from test_residual_effect import backbone,batch
from methods.registry import wrap_method
from methods.common.helpers import _masked_mean_pool
from methods.original.clrr import CrossLayerResidualRewire
from methods.residual_effect.v1.lsr import paired_contrastive_loss
from contextlib import nullcontext


class DiagnosticsGate(unittest.TestCase):
    def test_retrieval_preserves_pairs(self):
        spec=importlib.util.spec_from_file_location('diagnostics',ROOT/'scripts/analysis/diagnose_contrastive_checkpoint.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        vectors=np.eye(4,dtype=np.float32);texts=['a','b','c','d']
        result=module.retrieval(vectors,vectors,texts,texts)
        for mode in result.values():
            self.assertEqual(mode['source_to_target_top1'],1.)
            self.assertEqual(mode['target_to_source_top1'],1.)

    def test_three_losses_and_gradients(self):
        torch.set_num_threads(1)
        for method in ['baseline','full','ordinary-lsr-v1','ordinary-full-v1']:
            with self.subTest(method=method):
                model=wrap_method(backbone(),dict(method=method,alignment_weight=.1,temperature=.1)).eval()
                core=model if method=='baseline' else model.base_model
                route=next((m for m in model.modules() if isinstance(m,CrossLayerResidualRewire)),None)
                fresh=lambda:route._fresh_cache() if route else nullcontext()
                b=batch();ids=b['labels'].masked_fill(b['labels']==-100,1);mask=ids!=1
                out=core(**b,use_cache=False)
                with torch.no_grad(),fresh():target=model.get_encoder()(input_ids=ids,attention_mask=mask).last_hidden_state
                s=F.normalize(_masked_mean_pool(out.encoder_last_hidden_state,b['attention_mask']),dim=-1)
                t=F.normalize(_masked_mean_pool(target,mask),dim=-1)
                lsr=(1-(s*t).sum(-1)).mean()
                with fresh():source=model.get_encoder()(input_ids=b['input_ids'],attention_mask=b['attention_mask']).last_hidden_state
                sm=b['attention_mask'].bool() & (b['input_ids']!=0) & (b['input_ids']!=2)
                tm=mask & (ids!=0) & (ids!=2)
                nce=paired_contrastive_loss(_masked_mean_pool(source,sm),_masked_mean_pool(target,tm),.1)
                a=.1 if method!='baseline' else 0.;c=.1 if method.startswith('ordinary-') else 0.
                with torch.no_grad():actual=model(**b,use_cache=False).loss
                torch.testing.assert_close(actual,out.loss+a*lsr+c*nce,atol=2e-6,rtol=1e-6)
                params=list(model.parameters())
                for loss,retain in [(out.loss,True),(.1*lsr,False),(.1*nce,False)]:
                    gradients=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
                    self.assertTrue(all(g is None or torch.isfinite(g).all() for g in gradients))


if __name__=='__main__':unittest.main()
