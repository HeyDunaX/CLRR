"""Two optimizer steps per family: real dynamic batches then length256 stress."""
import gc,hashlib,json,platform,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import SequentialSampler
from transformers import AutoTokenizer,DataCollatorForSeq2Seq,Seq2SeqTrainingArguments,set_seed
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from amis_rewire.train import load_model,load_split,tokenize_dataset,configure_mbart
from amis_rewire.evaluation import PrecisionSeq2SeqTrainer
OUT=ROOT/'outputs_rebuttal/turkish_old_method_pilot_es10_20261007/fullsize_gate'
MODEL='facebook/mbart-large-50-many-to-many-mmt'
REV='e30b6cb8eb0d43a0b73cab73c7676b9863223a30'

class MemoryTrainer(PrecisionSeq2SeqTrainer):
    def _get_train_sampler(self,train_dataset=None):
        return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
    def training_step(self,model,inputs,num_items_in_batch=None):
        step=self.state.global_step
        if self.active_step!=step:
            self.active_step=step
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        result=super().training_step(model,inputs,num_items_in_batch)
        assert torch.isfinite(result).all(), 'Nonfinite training loss'
        assert self.model_accepts_loss_kwargs is False
        self.microsteps.append({'optimizer_step':step,'source_length':inputs['input_ids'].shape[1],
            'target_length':inputs['labels'].shape[1],'loss_scaled':float(result),
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30})
        return result

def run(family):
    set_seed(42)
    model=load_model(MODEL,family,distance=2,strength=.1,stack='encoder',jepa_weight=.1,revision=REV)
    configure_mbart(tokenizer,model,'tr_TR','en_XX');model.config.use_cache=False
    collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model,pad_to_multiple_of=8)
    args=Seq2SeqTrainingArguments(output_dir=str(OUT/'temporary'/family),
        per_device_train_batch_size=32,gradient_accumulation_steps=4,max_steps=2,
        per_device_eval_batch_size=8,learning_rate=5e-5,weight_decay=0.,warmup_ratio=.06,
        bf16=True,fp16=False,tf32=True,gradient_checkpointing=True,optim='adamw_torch_fused',
        predict_with_generate=True,generation_num_beams=1,generation_max_length=256,
        save_strategy='no',logging_strategy='steps',logging_steps=1,report_to=[],
        disable_tqdm=True,dataloader_num_workers=0,remove_unused_columns=False,seed=42,data_seed=42)
    trainer=MemoryTrainer(model=model,args=args,train_dataset=rows,data_collator=collator,
        processing_class=tokenizer,generation_dtype='fp32')
    trainer.active_step=None;trainer.microsteps=[]
    started=time.time()
    trainer.train()
    assert trainer.state.global_step==2 and len(trainer.microsteps)==8
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(r['source_length']==256 and r['target_length']==256 for r in trainer.microsteps[4:])
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    prediction=trainer.predict(validation)
    assert prediction.predictions.shape[0]==8
    result={'family':family,'status':'PASS','optimizer_steps':2,'microbatches':trainer.microsteps,
        'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.time()-started,
        'eval_fp32_beams1_batch8':True,'eval_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'eval_peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    (OUT/(family+'.json')).write_text(json.dumps(result,indent=2))
    print('FULLSIZE_FAMILY_PASS',json.dumps(result),flush=True)
    del prediction,trainer,model,collator;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REV,use_fast=True)
    tokenizer.src_lang='tr_TR';tokenizer.tgt_lang='en_XX'
    prep=SimpleNamespace(source_prefix='',max_source_length=256,max_target_length=256)
    data=ROOT/'data_processed/opus100_turkish_english_20k'
    real=tokenize_dataset(load_split(data,'train').select(range(128)),tokenizer,prep)
    validation=tokenize_dataset(load_split(data,'validation').select(range(8)),tokenizer,prep)
    rows=[real[i] for i in range(128)]
    for row in list(rows):
        rows.append({'input_ids':row['input_ids']+[tokenizer.pad_token_id]*(256-len(row['input_ids'])),
                     'attention_mask':row['attention_mask']+[0]*(256-len(row['attention_mask'])),
                     'labels':row['labels']+[-100]*(256-len(row['labels']))})
    status={'revision':REV,'device':torch.cuda.get_device_name(),'python':platform.python_version(),
        'torch':torch.__version__,'batch':32,'accumulation':4,'length_limit':256,
        'data_sha256':{n:hashlib.sha256((data/(n+'.csv')).read_bytes()).hexdigest() for n in ('train','validation')},
        'scope':'smoke only; stress padding to256; no benchmark quality claim'}
    for family in ('baseline','middle-align','jepa','jepa-clrr-enc'):
        print('FULLSIZE_FAMILY_START',family,flush=True)
        try:run(family)
        except Exception as e:
            (OUT/(family+'_FAILED.json')).write_text(json.dumps({'family':family,'status':'FAIL','error_type':type(e).__name__,'error':str(e)}))
            raise
    status['status']='FULLSIZE_GPU_PASS'
    (OUT/'FULLSIZE_PASS.json').write_text(json.dumps(status,indent=2))
    print('FULLSIZE_GPU_PASS',flush=True)
