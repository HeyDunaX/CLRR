"""TRAIN-only full-backbone gradient scale, with no optimizer construction."""
import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from probe_virtual_corrective_order import write,digest


def run(path):
    import torch
    import pandas as pd
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,DataCollatorForSeq2Seq,set_seed
    from methods.virtual_corrective_order.v1.objective import paired_ce_kl,order_penalty
    plan=json.loads(Path(path).read_text());out=ROOT/plan['output']
    assert plan['split']=='train' and plan['optimizer_steps']==0 and not plan['test_used']
    for name,sha in plan['source_sha256'].items():assert digest(ROOT/name)==sha
    assert digest(ROOT/plan['train_path'])==plan['train_sha256']
    assert not (out/'COMPLETE.json').exists()
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    assert available>=24*2**30,'Need 24 GiB available host RAM for transient gradient comparisons'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.mem_get_info()[0]>=plan['minimum_free_gpu_mib']*2**20
    prior=ROOT/plan['prior_output'];receipt=json.loads((prior/'COMPLETE.json').read_text())
    assert receipt['completed'] and receipt['parameters_unchanged'] and receipt['checkpoint']==plan['checkpoint']
    # Reuse the export produced from the hash-verified archive in the prior receipt.
    archive=next((prior/'transient_restore/download').rglob('best.zip'))
    assert digest(archive)==plan['checkpoint']['archive_sha256']
    export=prior/'transient_restore/export'
    model,info=AutoModelForSeq2SeqLM.from_pretrained(export,attn_implementation='sdpa',output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.cuda().train();model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    tokenizer=AutoTokenizer.from_pretrained(export,use_fast=True);tokenizer.src_lang,tokenizer.tgt_lang='tl_XX','zh_CN'
    frame=pd.read_csv(ROOT/plan['train_path'],keep_default_na=False)
    ids=torch.randperm(len(frame),generator=torch.Generator().manual_seed(plan['data_seed'])).tolist()[:4*plan['batches']]
    examples=[tokenizer(frame.iloc[i].source,text_target=frame.iloc[i].target,max_length=256,truncation=True) for i in ids]
    collate=DataCollatorForSeq2Seq(tokenizer,model=None,pad_to_multiple_of=8,label_pad_token_id=-100,return_tensors='pt')
    params=tuple(model.parameters());versions=[p._version for p in params]
    count=sum(p.numel() for p in params)
    B=torch.linalg.qr(torch.randn(model.config.d_model,plan['rank'],generator=torch.Generator().manual_seed(plan['basis_seed']))).Q.cuda()
    def dot(a,b):
        return sum(float(torch.dot(a[k:k+1048576].double(),b[k:k+1048576].double())) for k in range(0,len(a),1048576))
    results=[]
    for mb in range(plan['batches']):
        batch={k:v.cuda() for k,v in collate(examples[4*mb:4*mb+4]).items()}
        gradients={};values={};timings={}
        for mode in ('CE2','W_AUX','ORDER_AUX'):
            model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            hs=[]
            for seed in (plan['dropout_seed']+2*mb,plan['dropout_seed']+2*mb+1):
                set_seed(seed)
                hs.append(model.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                    decoder_input_ids=model.prepare_decoder_input_ids_from_labels(batch['labels']),use_cache=False).last_hidden_state)
            w=model.lm_head.weight;bias=model.final_logits_bias.squeeze(0)
            if mode=='ORDER_AUX':
                support,probe=([0,1],[2,3]) if mb%2==0 else ([2,3],[0,1])
                loss,*_=order_penalty(*hs,batch['labels'],support,probe,w,bias,B,plan['eta'],plan['chunk'],True)
            else:
                vals=paired_ce_kl(*hs,batch['labels'],w,bias,plan['chunk'],True,include_kl=(mode=='W_AUX'))
                loss=vals[0 if mode=='CE2' else 1]
            loss.backward();torch.cuda.synchronize()
            timings[mode]=dict(seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            result=torch.empty(count,dtype=torch.float32);offset=0
            for parameter in params:
                n=parameter.numel()
                if parameter.grad is None:result[offset:offset+n].zero_()
                else:
                    assert torch.isfinite(parameter.grad).all()
                    result[offset:offset+n].copy_(parameter.grad.detach().flatten())
                offset+=n
            gradients[mode]=result;values[mode]=float(loss.detach())
            del loss,hs,result
        norms={k:dot(g,g)**.5 for k,g in gradients.items()}
        assert all(v>0 for v in norms.values())
        result=dict(batch=mb,rows=ids[4*mb:4*mb+4],norms=norms,values=values,timings=timings,
            order_cosine_CE=dot(gradients['ORDER_AUX'],gradients['CE2'])/(norms['ORDER_AUX']*norms['CE2']),
            order_cosine_W=dot(gradients['ORDER_AUX'],gradients['W_AUX'])/(norms['ORDER_AUX']*norms['W_AUX']),
            proposed_lambda=plan['calibration_gradient_ratio']*norms['CE2']/norms['ORDER_AUX'])
        results.append(result);write(out/'BACKBONE.json',results)
        write(out/'PROGRESS.json',dict(stage='BACKBONE_GRADIENTS',batch=mb+1,batches=plan['batches'],optimizer_steps=0))
        del gradients,batch;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    assert versions==[p._version for p in params]
    proposed=sorted(r['proposed_lambda'] for r in results)[len(results)//2]
    write(out/'COMPLETE.json',dict(completed=True,optimizer_steps=0,parameters_unchanged=True,test_used=False,validation_used=False,
        proposed_lambda=proposed,calibration_gradient_ratio=plan['calibration_gradient_ratio'],batches=len(results),
        training_admitted=False,scope='TRAIN-only median scale proposal. Gradient ratio does not equal Adam update ratio or prove quality.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--plan',required=True);args=parser.parse_args()
    try:run(args.plan)
    except Exception:
        write(ROOT/json.loads(Path(args.plan).read_text())['output']/'ERROR.json',dict(traceback=traceback.format_exc(),optimizer_steps=0))
        raise
