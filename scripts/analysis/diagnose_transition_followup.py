"""Frozen checkpoint geometry and eval-mode gradients, zero optimizer updates."""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from experiment_runner import engine
from experiment_runner.integrity import restore_archive, sha256
from methods.common.helpers import _find_stack_layers, _masked_mean_pool
from methods.original.clrr import CrossLayerResidualRewire
from methods.transition_lsr.v1.model import TransitionLSRSeq2SeqLM, transition_cosine


def route_of(model):
    if isinstance(model, CrossLayerResidualRewire):
        return model
    base = getattr(model, 'base_model', None)
    return base if isinstance(base, CrossLayerResidualRewire) else None


def states(model, ids, mask):
    before, after, handles = [], [], []
    layers = _find_stack_layers(model, 'encoder')
    for layer in layers:
        handles.append(layer.register_forward_hook(
            lambda m,i,o: before.append(_masked_mean_pool(o[0],mask)),prepend=True))
        handles.append(layer.register_forward_hook(
            lambda m,i,o: after.append(_masked_mean_pool(o[0],mask))))
    route = route_of(model)
    try:
        with route._fresh_cache() if route else nullcontext():
            output = model.get_encoder()(input_ids=ids,attention_mask=mask)
        return torch.stack(before),torch.stack(after),_masked_mean_pool(output.last_hidden_state,mask)
    finally:
        for handle in handles: handle.remove()


def geometry(source, target):
    source,target=source.double(),target.double()
    s,t=F.normalize(source,dim=-1),F.normalize(target,dim=-1)
    rng=torch.Generator().manual_seed(42)
    shuffled=[float((s*t[torch.randperm(len(t),generator=rng)]).sum(-1).mean()) for _ in range(100)]
    centered=F.cosine_similarity(source-source.mean(0),target-target.mean(0),dim=-1)
    singular=torch.linalg.svdvals(source-source.mean(0));p=singular/singular.sum().clamp_min(1e-30)
    return dict(source_norm_mean=float(source.norm(dim=-1).mean()),
        source_norm_min=float(source.norm(dim=-1).min()),target_norm_mean=float(target.norm(dim=-1).mean()),
        target_norm_min=float(target.norm(dim=-1).min()),matched_cosine=float((s*t).sum(-1).mean()),
        shuffled_cosine_mean=float(np.mean(shuffled)),centered_matched_cosine=float(centered.mean()),
        centered_source_effective_rank=float(torch.exp(-(p*p.clamp_min(1e-30).log()).sum())),
        centered_source_variance=float((source-source.mean(0)).square().sum()/max(len(source)-1,1)))


def gradient_groups(names, left, right):
    groups={}
    for name,a,b in zip(names,left,right):
        match=re.search(r'encoder\.layers\.(\d+)\.',name)
        key='encoder.layer.'+match[1] if match else 'other'
        sums=groups.setdefault(key,[0.,0.,0.])
        if a is not None:sums[0]+=float(a.double().square().sum())
        if b is not None:sums[1]+=float(b.double().square().sum())
        if a is not None and b is not None:sums[2]+=float((a.double()*b.double()).sum())
    total=[sum(v[i] for v in groups.values()) for i in range(3)];groups['all']=total
    return {key:dict(ce_norm=a**.5,weighted_aux_norm=b**.5,
        ratio=(b/a)**.5 if a else None,cosine=c/(a*b)**.5 if a*b else None)
        for key,(a,b,c) in groups.items()}


def measure(args, cohort, ident, out, indices, batches):
    tag=cohort.name+'__'+ident;folder=out/tag
    if (folder/'RESULT.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    engine.configure(cohort)
    marker=json.loads((cohort/'runs'/ident/'VAL_COMPLETE.json').read_text())
    assert marker['backup']['verified'] and marker['status']=='VALIDATION_COMPLETE_VERIFIED'
    spec=marker['spec'];plan=engine.plan()
    assert plan['dataset']=='amis_mandarin' and plan['protocol']['hardware_profile']=='rtx6000-fp32'
    data=ROOT/'data_processed/amis_mandarin/validation.csv'
    assert sha256(data)==plan['data']['validation']['sha256']
    manifest=json.loads((cohort/'bundle_manifest.json').read_text())
    # The registry has gained unrelated candidates since historical training.
    # Verify numerical dependencies used by this cohort, not registry text.
    dependencies=['src/methods/original/clrr.py','src/methods/original/lsr.py',
                  'src/methods/common/helpers.py']
    if spec['method'].startswith('transition-'):
        dependencies+=['src/methods/transition_lsr/v1/model.py',
                       'src/methods/residual_effect/v1/lsr.py']
    for name in dependencies:
        assert name in manifest and sha256(ROOT/name)==manifest[name],('Frozen numerical source changed',name)
    free=torch.cuda.mem_get_info()[0]//2**20
    print('GPU_BEFORE',tag,free,flush=True)
    assert free>=args.minimum_free_gpu_mib,'Leave other users processes untouched'
    archive=Path(hf_hub_download(plan['repo'],spec['remote_prefix']+'/best.zip',
        revision=marker['backup']['revision'],token=False,local_dir=folder/'download'))
    assert sha256(archive)==marker['archive_sha256']
    restore_archive(archive,folder/'model')
    model,tok,loading=engine.load_model(spec,folder/'model')
    model.gradient_checkpointing_disable();model.cuda().eval();model.config.use_cache=False
    versions={name:p._version for name,p in model.named_parameters()}
    rows=pd.read_csv(data,dtype=str,keep_default_na=False)

    def tensors(selected):
        part=rows.iloc[selected]
        options=dict(padding=True,pad_to_multiple_of=8,truncation=True,max_length=256,return_tensors='pt')
        x=tok(part.source.tolist(),**options).to('cuda')
        y=tok(text_target=part.target.tolist(),**options).to('cuda')
        labels=y.input_ids.masked_fill(y.attention_mask.eq(0),-100)
        return x,y,labels

    accumulated={k:[] for k in ('source_pre','source_post','target_pre','target_post','source_end','target_end')}
    with torch.no_grad():
        for start in range(0,len(indices),4):
            x,y,_=tensors(indices[start:start+4])
            for prefix,inputs in [('source',x),('target',y)]:
                pre,post,end=states(model,inputs.input_ids,inputs.attention_mask)
                for key,value in [('pre',pre),('post',post),('end',end)]:
                    accumulated[prefix+'_'+key].append(value.detach().cpu())
    rep={key:torch.cat(value,dim=0 if key.endswith('end') else 1) for key,value in accumulated.items()}
    edges=[]
    for layer in range(2,12):
        s=rep['source_pre'][layer]-rep['source_post'][layer-2]
        t=rep['target_pre'][layer]-rep['target_post'][layer-2]
        edges.append(dict(recipient_layer=layer+1,donor_layer=layer-1,**geometry(s,t),
            transition_to_recipient_norm=float((s.norm(dim=-1)/rep['source_pre'][layer].norm(dim=-1).clamp_min(1e-6)).mean()),
            injected_to_recipient_norm=float(((rep['source_post'][layer]-rep['source_pre'][layer]).norm(dim=-1)/rep['source_pre'][layer].norm(dim=-1).clamp_min(1e-6)).mean())))
    print('GEOMETRY_COMPLETE',tag,flush=True)
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    names,params=zip(*parameters);gradients=[]
    base=model.base_model if hasattr(model,'jepa_weight') else model
    for batch in batches:
        x,y,labels=tensors(batch)
        ce=base(**x,labels=labels).loss
        raw=torch.autograd.grad(ce,params,allow_unused=True)
        ce_grads=[g.detach().cpu() if g is not None else None for g in raw]
        assert all(g is None or torch.isfinite(g).all() for g in ce_grads)
        ce_value=float(ce.detach());del raw,ce
        if isinstance(model,TransitionLSRSeq2SeqLM):
            source=model.transitions(x.input_ids,x.attention_mask)
            with torch.no_grad():target=model.transitions(y.input_ids,y.attention_mask)
            aux=model.alignment_weight*transition_cosine(source,target)
        elif getattr(model,'jepa_weight',0)>0:
            _,_,source=states(model,x.input_ids,x.attention_mask)
            with torch.no_grad():_,_,target=states(model,y.input_ids,y.attention_mask)
            aux=model.jepa_weight*(1-F.cosine_similarity(source,target,dim=-1)).mean()
        else:
            aux=None
        if aux is not None:
            raw=torch.autograd.grad(aux,params,allow_unused=True)
            aux_grads=[g.detach().cpu() if g is not None else None for g in raw]
            del raw
            assert all(g is None or torch.isfinite(g).all() for g in ce_grads+aux_grads)
            groups=gradient_groups(names,ce_grads,aux_grads);aux_value=float(aux.detach())
            del aux_grads,aux,source,target
        else:groups=None;aux_value=0.
        del ce_grads
        route=route_of(model);off=None
        if route:
            strength=route.strength
            try:
                route.strength=0.
                with torch.no_grad():off=float(base(**x,labels=labels).loss)
            finally:route.strength=strength
        gradients.append(dict(rows=batch,eval_ce=ce_value,weighted_auxiliary=aux_value,
            route_off_eval_ce=off,groups=groups))
        print('GRADIENT_BATCH',tag,batch,groups['all'] if groups else None,flush=True)
        del x,y,labels
    assert all(p._version==versions[n] for n,p in model.named_parameters()),'Weights mutated'
    result=dict(status='PASS',checkpoint=tag,archive_sha256=marker['archive_sha256'],
        revision=marker['backup']['revision'],worker_sha256=sha256(Path(__file__)),
        geometry_rows=indices,gradient_batches=batches,edges=edges,
        endpoint=geometry(rep['source_end'],rep['target_end']),gradients=gradients,
        optimizer_steps=0,parameters_unchanged=True,actual_backend=model.config._attn_implementation,
        numerical_sources_verified={name:manifest[name] for name in dependencies},
        scope='Post-hoc eval-mode checkpoint measurements, not training-time Adam updates, causal harm or retrained ablations. Rank is limited by 64 examples.',
        validation_only=True,pooling='attention mask including special tokens, matching original objective')
    del model,base,params,parameters,route;gc.collect();torch.cuda.empty_cache()
    for temporary in (folder/'model',folder/'download'):
        assert temporary.resolve().is_relative_to(out.resolve());shutil.rmtree(temporary)
    result['transient_weights_removed']=True
    engine.write(folder/'RESULT.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',action='append',required=True,help='cohort:run')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--minimum-free-gpu-mib',type=int,default=20000)
    args=parser.parse_args()
    assert ROOT==Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT/'conda')
    out=args.output.resolve();assert out.is_relative_to(ROOT/'outputs/experiments')
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    indices=np.random.default_rng(42).permutation(576)[:64].tolist()
    batches=[indices[i:i+4] for i in range(0,16,4)]
    design=dict(checkpoints=args.checkpoint,geometry_rows=indices,gradient_batches=batches,
        worker_sha256=sha256(Path(__file__)),optimizer_steps=0,test_consulted=False)
    if (out/'DESIGN.json').exists():assert json.loads((out/'DESIGN.json').read_text())==design
    else:engine.write(out/'DESIGN.json',design)
    for item in args.checkpoint:
        cohort,ident=item.split(':');measure(args,ROOT/'outputs/experiments'/cohort,ident,out,indices,batches)
    engine.write(out/'COMPLETE.json',dict(status='COMPLETE',optimizer_steps=0,completed_unix=time.time()))


if __name__=='__main__':main()
