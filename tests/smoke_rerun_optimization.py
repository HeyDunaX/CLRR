"""Actual Trainer gradients vs explicit microbatch-mean reference, offline only."""
import copy
import json
import sys
import tempfile
from pathlib import Path
import torch
from transformers import MBartConfig, MBartForConditionalGeneration, Seq2SeqTrainingArguments, default_data_collator
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from amis_rewire.modeling import CrossLayerResidualRewire,JEPAGuidedSeq2SeqLM
from amis_rewire.evaluation import PrecisionSeq2SeqTrainer
from comparative_baselines.middle_align_acl2025.model import MiddleAlignMT5


class CheckedTrainer(PrecisionSeq2SeqTrainer):
    def training_step(self,model,inputs,num_items_in_batch=None):
        model.train()
        inputs=self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss=model(**inputs).loss
        params=list(model.parameters())
        previous=[p.grad for p in params]
        for p in params:p.grad=None
        # Reentrant checkpointing supports backward(), not autograd.grad().
        loss.backward()
        grads=[None if p.grad is None else p.grad.detach().clone() for p in params]
        for p,old in zip(params,previous):p.grad=old
        count=self.current_gradient_accumulation_steps
        if getattr(self,'reference_step',None)!=self.state.global_step:
            self.reference_step=self.state.global_step
            self.reference=[torch.zeros_like(p) for p in params]
        for expected,g in zip(self.reference,grads):
            if g is not None:expected.add_(g.detach()/count)
        result=super().training_step(model,inputs,num_items_in_batch)
        difference=0.
        for p,expected in zip(params,self.reference):
            actual=torch.zeros_like(p) if p.grad is None else p.grad
            torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-6)
            difference=max(difference,float((actual-expected).abs().max()))
        self.gradient_checks.append({'global_step':self.state.global_step,'accumulation_count':count,
                                    'max_gradient_abs_error':difference})
        return result


def main():
    torch.set_num_threads(2)
    cuda='--cuda' in sys.argv
    turkish='--turkish-pilot' in sys.argv
    if cuda: assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg=MBartConfig(vocab_size=32,d_model=16,encoder_layers=4,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,
        dropout=0.,attention_dropout=0.,activation_dropout=0.,classifier_dropout=0.,
        encoder_layerdrop=0.,decoder_layerdrop=0.,pad_token_id=1,decoder_start_token_id=2,use_cache=False)
    torch.manual_seed(123)
    initial=MBartForConditionalGeneration(cfg).state_dict()
    rows=[{'input_ids':torch.tensor([4+i,5,6,2]),'attention_mask':torch.ones(4,dtype=torch.long),
           'labels':torch.tensor([7+i,8,2,-100] if i%2 else [7+i,8,9,2])} for i in range(5)]
    if turkish:
        rows=rows+copy.deepcopy(rows)
    records=[]
    out=Path(sys.argv[sys.argv.index('--result-dir')+1]) if '--result-dir' in sys.argv else ROOT/'outputs_rebuttal/amis_pre_rerun_20261005'
    out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for family in (('baseline','middle','lsr','full') if turkish else ('baseline','clrr','lsr','full')):
            for checkpointed in (False,True):
                for optimizer_name in ('sgd','adamw_clipped'):
                    raw=MBartForConditionalGeneration(cfg)
                    raw.load_state_dict(copy.deepcopy(initial))
                    model=CrossLayerResidualRewire(raw,distance=2,strength=.1) if family in ('clrr','full') else raw
                    if family in ('lsr','full'):model=JEPAGuidedSeq2SeqLM(model,jepa_weight=.1)
                    if family=='middle':model=MiddleAlignMT5(raw,middle_layer_idx=2,pad_token_id=1)
                    args=Seq2SeqTrainingArguments(output_dir=str(Path(tmp)/f'{family}-{checkpointed}-{optimizer_name}'),
                        use_cpu=not cuda,bf16=cuda,per_device_train_batch_size=2 if turkish else 1,gradient_accumulation_steps=4,max_steps=2,
                        learning_rate=.001,lr_scheduler_type='constant',max_grad_norm=1. if optimizer_name=='adamw_clipped' else 0.,
                        gradient_checkpointing=checkpointed,save_strategy='no',logging_strategy='no',
                        report_to=[],disable_tqdm=True,seed=123,data_seed=123,remove_unused_columns=False)
                    optimizer=(torch.optim.SGD(model.parameters(),lr=.001) if optimizer_name=='sgd'
                               else torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.))
                    trainer=CheckedTrainer(model=model,args=args,train_dataset=rows,
                        data_collator=default_data_collator,optimizers=(optimizer,None))
                    trainer.gradient_checks=[]
                    trainer.train()
                    assert trainer.model_accepts_loss_kwargs is False
                    assert len(trainer.gradient_checks)==5 and trainer.gradient_checks[-1]['accumulation_count']==1
                    assert all(torch.isfinite(p).all() for p in model.parameters())
                    # Backward recomputation can repopulate detached caches.
                    # The next public forward must start and end with a fresh cache.
                    if hasattr(model,'_fresh_cache'):
                        model.eval()
                        with torch.no_grad(),trainer.compute_loss_context_manager():
                            model(**trainer._prepare_inputs(default_data_collator(rows[:1])))
                        assert all(not v for v in model._cache.values())
                    records.append({'family':family,'checkpointed':checkpointed,'optimizer':optimizer_name,
                        'unequal_label_tokens':True,'partial_last_group':True,'passed':True,'cuda_bf16':cuda,
                        'max_gradient_abs_error':max(r['max_gradient_abs_error'] for r in trainer.gradient_checks)})
    (out/'optimization_matrix.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('OPTIMIZATION_MATRIX_PASS',len(records),max(r['max_gradient_abs_error'] for r in records))


if __name__=='__main__':main()
