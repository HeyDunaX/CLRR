"""Pinned-W TRAIN-only probes and no-update backward resource measurement."""
import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def run(plan_path):
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,DataCollatorForSeq2Seq,set_seed
    from experiment_runner.integrity import restore_archive
    from methods.virtual_corrective_order.v1.objective import endpoints,order_penalty,paired_ce_kl,probe_penalty

    plan=json.loads(Path(plan_path).read_text())
    out=ROOT/plan['output']
    if (out/'COMPLETE.json').exists():
        raise FileExistsError('Diagnostic already completed')
    assert plan['split']=='train' and plan['optimizer_steps']==0 and not plan['test_used']
    assert digest(ROOT/plan['train_path'])==plan['train_sha256']
    for name,sha in plan['source_sha256'].items():
        assert digest(ROOT/name)==sha,('Source mismatch',name)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.mem_get_info()[0]>=plan['minimum_free_gpu_mib']*2**20
    temp=out/'transient_restore'
    write(out/'PROGRESS.json',dict(stage='RESTORE_PINNED_W',optimizer_steps=0))
    ck=plan['checkpoint']
    archive=Path(hf_hub_download(plan['repo'],ck['prefix']+'/best.zip',revision=ck['revision'],token=False,local_dir=temp/'download'))
    assert digest(archive)==ck['archive_sha256']
    export=temp/'export'
    if not export.exists(): restore_archive(archive,export)
    tokenizer=AutoTokenizer.from_pretrained(export,use_fast=True)
    tokenizer.src_lang,tokenizer.tgt_lang='tl_XX','zh_CN'
    model,loading=AutoModelForSeq2SeqLM.from_pretrained(export,attn_implementation='sdpa',output_loading_info=True)
    assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.cuda().train()
    model.config.use_cache=False
    params=tuple(model.parameters())
    versions=[p._version for p in params]
    names_before=tuple(model.state_dict())
    frame=pd.read_csv(ROOT/plan['train_path'],keep_default_na=False)
    ids=torch.randperm(len(frame),generator=torch.Generator().manual_seed(plan['data_seed'])).tolist()[:plan['batches']*4]
    examples=[tokenizer(frame.iloc[i].source,text_target=frame.iloc[i].target,max_length=256,truncation=True) for i in ids]
    write(out/'TRAIN_ROWS.json',dict(split='train',indices=ids,train_sha256=plan['train_sha256']))
    collate=DataCollatorForSeq2Seq(tokenizer,model=None,pad_to_multiple_of=8,label_pad_token_id=-100,return_tensors='pt')
    w=model.get_output_embeddings().weight
    bias=model.final_logits_bias.squeeze(0)
    def features(batch,seed):
        set_seed(seed)
        return model.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
            decoder_input_ids=model.prepare_decoder_input_ids_from_labels(batch['labels']),
            use_cache=False,return_dict=True).last_hidden_state
    def native_entropy(a,b,y):
        entropy=0.
        for h in (a,b):
            valid=h[y.ne(-100)]
            for start in range(0,len(valid),plan['chunk']):
                lp=torch.nn.functional.linear(valid[start:start+plan['chunk']],w,bias).log_softmax(-1)
                entropy+=float(-(lp.exp()*lp).sum())
        return entropy/(2*int(y.ne(-100).sum()))
    bases={seed:torch.linalg.qr(torch.randn(w.shape[1],plan['rank'],generator=torch.Generator().manual_seed(seed))).Q.cuda() for seed in plan['basis_seeds']}
    rows=[]
    started=time.time()
    for mb in range(plan['batches']):
        batch={k:v.cuda() for k,v in collate(examples[mb*4:(mb+1)*4]).items()}
        y=batch['labels']
        support,probe=([0,1],[2,3]) if mb%2==0 else ([2,3],[0,1])
        with torch.no_grad():
            ha=features(batch,plan['dropout_seed']+2*mb)
            hb=features(batch,plan['dropout_seed']+2*mb+1)
            replay=features(batch,plan['dropout_seed']+2*mb)
            replay_error=float((ha-replay).abs().max())
            assert replay_error==0,'Native dropout replay mismatch'
            del replay
            ce,static=paired_ce_kl(ha,hb,y,w,bias,plan['chunk'])
            entropy=native_entropy(ha,hb,y)
            for seed,B in bases.items():
                valid=y.ne(-100)
                energy=sum((h[valid]@B).square().sum() for h in (ha,hb))/sum(h[valid].square().sum() for h in (ha,hb))
                for eta in plan['eta_grid']:
                    value,ab,ba=order_penalty(ha,hb,y,support,probe,w,bias,B,eta,plan['chunk'])
                    high,*_=order_penalty(ha.double(),hb.double(),y,support,probe,w.double(),bias.double(),B.double(),eta,plan['chunk'])
                    null,*_=order_penalty(ha,ha,y,support,probe,w,bias,B,eta,plan['chunk'])
                    assert null.item()==0,'Nonzero identical-view null'
                    midpoint=(ab+ba)/2
                    random=torch.randn(ab.shape,generator=torch.Generator().manual_seed(plan['random_seed']+mb),dtype=ab.dtype).cuda()
                    random=random/random.norm()*(ab-ba).norm()
                    S=w@B
                    rand=.5*sum(probe_penalty(h,y,probe,w,bias,B,S,midpoint+random/2,midpoint-random/2,plan['chunk']) for h in (ha,hb))
                    record=dict(batch=mb,basis_seed=seed,eta=eta,order_skl=float(value),order_skl_fp64=float(high),
                        null_skl=float(null),random_displacement_matched_skl=float(rand),endpoint_norm=float((ab-ba).norm()),
                        native_static_skl=float(static),native_ce=float(ce),projected_energy_fraction=float(energy),
                        replay_error=replay_error,native_entropy=entropy,endpoint_mean_norm=float(((ab+ba)/2).norm()),
                        valid_support_tokens=int(y[support].ne(-100).sum()),valid_probe_tokens=int(y[probe].ne(-100).sum()))
                    rows.append(record)
                    write(out/'MEASUREMENTS.json',rows)
                    write(out/'PROGRESS.json',dict(stage='FROZEN_TRAIN_PROBES',batch=mb+1,batches=plan['batches'],basis_seed=seed,eta=eta,records=len(rows),elapsed_seconds=time.time()-started,optimizer_steps=0))
        del ha,hb,batch
        gc.collect();torch.cuda.empty_cache()

    # All weights remain fixed. Backpropagation populates grads, never constructs an optimizer.
    write(out/'PROGRESS.json',dict(stage='NO_UPDATE_BACKWARD_BENCHMARK',optimizer_steps=0))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    benchmark=[]
    batch={k:v.cuda() for k,v in collate(examples[:4]).items()}
    B=bases[plan['basis_seeds'][0]]
    for method in ('CE2','W','ORDER'):
        for repeat in range(plan['benchmark_repeats']):
            model.zero_grad(set_to_none=True)
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
            baseline_alloc=torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            t=time.perf_counter()
            ha=features(batch,plan['dropout_seed'])
            hb=features(batch,plan['dropout_seed']+1)
            vals=paired_ce_kl(ha,hb,batch['labels'],w,bias,plan['chunk'],True,include_kl=(method=='W'))
            ce,static=vals
            loss=ce
            aux=static if method=='W' else ce*0
            if method=='ORDER':
                aux,*_=order_penalty(ha,hb,batch['labels'],[0,1],[2,3],w,bias,B,plan['benchmark_eta'],plan['chunk'],True)
            if method=='W': loss=loss+1.25*aux
            if method=='ORDER': loss=loss+aux  # coefficient1 is timing-only, not a chosen training weight
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            torch.cuda.synchronize()
            record=dict(method=method,repeat=repeat,seconds=time.perf_counter()-t,loss=float(loss.detach()),
                auxiliary=float(aux.detach()),baseline_allocated_mib=baseline_alloc/2**20,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                scope='microbatch4 forward/backward; no optimizer/state/step; not complete training VRAM')
            benchmark.append(record)
            write(out/'BENCHMARK.json',benchmark)
            del ha,hb,loss,aux,ce,static,vals
            model.zero_grad(set_to_none=True)
            write(out/'PROGRESS.json',dict(stage='NO_UPDATE_BACKWARD_BENCHMARK',method=method,repeat=repeat,optimizer_steps=0))
    assert versions==[p._version for p in params], 'Parameter mutated'
    assert names_before==tuple(model.state_dict()),'Persistent parameters added'
    write(out/'COMPLETE.json',dict(completed=True,optimizer_steps=0,test_used=False,validation_used=False,
        checkpoint=ck,parameters_unchanged=True,measurements=len(rows),benchmark_records=len(benchmark),
        elapsed_seconds=time.time()-started,scope='Exploratory frozen checkpoint diagnostics; no trained candidate.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan',required=True)
    args=parser.parse_args()
    try:
        run(args.plan)
    except Exception:
        out=ROOT/json.loads(Path(args.plan).read_text())['output']
        write(out/'ERROR.json',dict(traceback=traceback.format_exc(),optimizer_steps=0))
        raise
