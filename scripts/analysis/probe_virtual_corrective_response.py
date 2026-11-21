"""Follow-up head-gradient audit on fixed W; zero optimizer steps."""
import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from probe_virtual_corrective_order import digest,write


def run(path):
    import torch
    import pandas as pd
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,DataCollatorForSeq2Seq,set_seed
    from experiment_runner.integrity import restore_archive
    from methods.virtual_corrective_order.v1.objective import order_penalty,paired_ce_kl,endpoints,select_tokens,logits

    plan=json.loads(Path(path).read_text())
    assert plan['optimizer_steps']==0 and plan['split']=='train' and not plan['test_used']
    for name,sha in plan['source_sha256'].items():assert digest(ROOT/name)==sha
    assert digest(ROOT/plan['train_path'])==plan['train_sha256']
    out=ROOT/plan['output']
    assert not (out/'COMPLETE.json').exists()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.mem_get_info()[0]>=plan['minimum_free_gpu_mib']*2**20
    prior=ROOT/plan['prior_output']
    receipt=json.loads((prior/'COMPLETE.json').read_text())
    assert receipt['completed'] and receipt['parameters_unchanged'] and receipt['checkpoint']==plan['checkpoint']
    archives=list((prior/'transient_restore/download').rglob('best.zip'))
    assert len(archives)==1 and digest(archives[0])==plan['checkpoint']['archive_sha256']
    export=out/'transient_restore/export'
    write(out/'PROGRESS.json',dict(stage='RESTORE_VERIFIED_LOCAL_ARCHIVE',optimizer_steps=0))
    restore_archive(archives[0],export)
    tokenizer=AutoTokenizer.from_pretrained(export,use_fast=True)
    tokenizer.src_lang,tokenizer.tgt_lang='tl_XX','zh_CN'
    model,info=AutoModelForSeq2SeqLM.from_pretrained(export,attn_implementation='sdpa',output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.cuda().train();model.config.use_cache=False
    versions=[p._version for p in model.parameters()]
    model.requires_grad_(False)
    w=model.lm_head.weight.detach();bias=model.final_logits_bias.squeeze(0).detach()
    B=torch.linalg.qr(torch.randn(w.shape[1],plan['rank'],generator=torch.Generator().manual_seed(plan['basis_seed']))).Q.cuda()
    w64,bias64,B64=w.double(),bias.double(),B.double()
    frame=pd.read_csv(ROOT/plan['train_path'],keep_default_na=False)
    ids=torch.randperm(len(frame),generator=torch.Generator().manual_seed(plan['data_seed'])).tolist()[:4*plan['batches']]
    examples=[tokenizer(frame.iloc[i].source,text_target=frame.iloc[i].target,max_length=256,truncation=True) for i in ids]
    collator=DataCollatorForSeq2Seq(tokenizer,model=None,pad_to_multiple_of=8,label_pad_token_id=-100,return_tensors='pt')
    def cosine(a,b):
        a,b=[torch.cat([v.detach().double().flatten() for v in x]) for x in (a,b)]
        return float(torch.dot(a,b)/(a.norm()*b.norm()))
    def norm(gs):return float(sum(g.detach().double().square().sum() for g in gs).sqrt())
    results=[]
    start=time.time()
    for mb in range(plan['batches']):
        batch={k:v.cuda() for k,v in collator(examples[4*mb:4*mb+4]).items()}
        y=batch['labels'];support,probe=([0,1],[2,3]) if mb%2==0 else ([2,3],[0,1])
        features=[]
        with torch.no_grad():
            for seed in (plan['dropout_seed']+2*mb,plan['dropout_seed']+2*mb+1):
                set_seed(seed)
                features.append(model.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                    decoder_input_ids=model.prepare_decoder_input_ids_from_labels(y),use_cache=False).last_hidden_state)
        a,b=[h.detach().requires_grad_() for h in features]
        a64,b64=[h.detach().double().requires_grad_() for h in features]
        ce,kl=paired_ce_kl(a,b,y,w,bias,plan['chunk'],True)
        g_ce=torch.autograd.grad(ce,(a,b),retain_graph=True)
        g_w=torch.autograd.grad(kl,(a,b))
        penalty,ab,ba=order_penalty(a,b,y,support,probe,w,bias,B,plan['eta'],plan['chunk'],True)
        g_order=torch.autograd.grad(penalty,(a,b))
        high,ab64,ba64=order_penalty(a64,b64,y,support,probe,w64,bias64,B64,plan['eta'],plan['chunk'],True)
        g_high=torch.autograd.grad(high,(a64,b64))
        err=norm(tuple(x.double()-ref for x,ref in zip(g_order,g_high)))/norm(g_high)
        with torch.no_grad():
            shuffled=y.clone()
            support_y=y[support].clone()
            valid=support_y.ne(-100)
            support_y[valid]=support_y[valid].roll(1)
            shuffled[support]=support_y
            label_change=float((support_y[valid]!=y[support][valid]).float().mean())
            shuffled_penalty,*_=order_penalty(a64,b64,shuffled,support,probe,w64,bias64,B64,plan['eta'],plan['chunk'])
            # Train probe gold NLL before/after the two virtual updates, not generation quality.
            S=w64@B64
            nll=[]
            for A in (torch.zeros_like(ab64),(ab64+ba64)/2):
                total=0.;count=0
                for h in (a64,b64):
                    q,labels=select_tokens(h,y,probe)
                    for pos in range(0,len(labels),plan['chunk']):
                        z=logits(q[pos:pos+plan['chunk']],w64,bias64,B64,S,A)
                        total+=float(torch.nn.functional.cross_entropy(z,labels[pos:pos+plan['chunk']],reduction='sum'))
                    count+=len(labels)
                nll.append(total/count)
        results.append(dict(batch=mb,rows=ids[4*mb:4*mb+4],eta=plan['eta'],order32=float(penalty.detach()),order64=float(high.detach()),
            feature_gradient_relative_error=err,feature_gradient_precision_cosine=cosine(g_order,g_high),
            feature_gradient_cosine_with_W=cosine(g_order,g_w),feature_gradient_cosine_with_CE=cosine(g_order,g_ce),
            order_feature_gradient_norm=norm(g_order),W_feature_gradient_norm=norm(g_w),CE_feature_gradient_norm=norm(g_ce),
            shuffled_support_order64=float(shuffled_penalty),shuffled_support_changed_fraction=label_change,
            probe_nll_before=nll[0],probe_nll_after_mean_endpoint=nll[1],probe_nll_delta=nll[1]-nll[0]))
        write(out/'RESPONSE.json',results)
        write(out/'PROGRESS.json',dict(stage='FROZEN_FEATURE_GRADIENTS',batch=mb+1,batches=plan['batches'],optimizer_steps=0,elapsed_seconds=time.time()-start))
        del features,a,b,a64,b64,g_ce,g_w,g_order,g_high,ce,kl,penalty,high,ab,ba,ab64,ba64,batch
        gc.collect();torch.cuda.empty_cache()
    assert versions==[p._version for p in model.parameters()]
    write(out/'COMPLETE.json',dict(completed=True,optimizer_steps=0,test_used=False,validation_used=False,
        parameters_unchanged=True,batches=len(results),elapsed_seconds=time.time()-start,
        scope='Feature gradients only, not full-backbone gradient calibration or causal quality evidence. TRAIN label permutation preserves support unigram counts; no CSV modification.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan',required=True)
    args=parser.parse_args()
    try:run(args.plan)
    except Exception:
        out=ROOT/json.loads(Path(args.plan).read_text())['output']
        write(out/'ERROR.json',dict(traceback=traceback.format_exc(),optimizer_steps=0))
        raise
