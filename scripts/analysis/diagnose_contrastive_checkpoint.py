"""Frozen-checkpoint validation retrieval, geometry and CE/LSR/InfoNCE gradients; zero optimizer steps."""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(1, str(ROOT / 'scripts/revalidation'))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from experiment_runner import engine
from experiment_runner.integrity import sha256, restore_archive
from methods.original.clrr import CrossLayerResidualRewire
from methods.common.helpers import _masked_mean_pool
from methods.residual_effect.v1.lsr import paired_contrastive_loss
from amis_v4_measurements import spectrum, gradient_group


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def retrieval(source, target, source_text, target_text):
    def normalize(a):
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    result = {}
    n = len(source)
    for name, s, t in [('raw', source, target), ('centered', source-source.mean(0), target-target.mean(0))]:
        sims = normalize(s) @ normalize(t).T
        matched = sims.diagonal()
        s_top, t_top = sims.argmax(1), sims.argmax(0)
        wrong = sims.copy(); np.fill_diagonal(wrong, -np.inf)
        rng = np.random.default_rng(42)
        shuffled = np.asarray([sims[np.arange(n), rng.permutation(n)].mean() for _ in range(100)])
        result[name] = {'source_to_target_top1': float(np.mean(s_top == np.arange(n))),
            'target_to_source_top1': float(np.mean(t_top == np.arange(n))),
            'source_to_target_duplicate_aware_top1': float(np.mean(np.asarray(target_text)[s_top] == np.asarray(target_text))),
            'target_to_source_duplicate_aware_top1': float(np.mean(np.asarray(source_text)[t_top] == np.asarray(source_text))),
            'matched_cosine': float(matched.mean()), 'shuffled_cosine_mean': float(shuffled.mean()),
            'matched_minus_shuffled': float(matched.mean()-shuffled.mean()),
            'matched_minus_hardest_index_negative': float(np.mean(matched-wrong.max(1))),
            'random_top1_chance': 1/n}
    return result


def measure(spec, selected, rows, indices, batches, folder, minimum_free):
    free = torch.cuda.mem_get_info()[0] // 2**20
    print('GPU_BEFORE', spec['id'], free, flush=True)
    assert free >= minimum_free
    archive = Path(hf_hub_download(engine.REPO, selected['archive_path'], revision=selected['revision'],
                                  token=False, local_dir=folder/'restore'))
    assert sha256(archive) == selected['archive_sha256']
    export = folder/'model'; restore_archive(archive, export)
    model, tok, loading = engine.load_model(spec, export)
    model.gradient_checkpointing_disable(); model.config.use_cache = False
    model.cuda().eval()
    versions = {n:p._version for n,p in model.named_parameters()}
    route = next((m for m in model.modules() if isinstance(m, CrossLayerResidualRewire)), None)
    core = model if spec['method']=='baseline' else model.base_model
    fresh = lambda: route._fresh_cache() if route else nullcontext()
    def tokens(ids, target=False):
        texts = rows.iloc[ids]['target' if target else 'source'].tolist()
        opts = dict(padding=True, pad_to_multiple_of=8, truncation=True, max_length=256, return_tensors='pt')
        batch = tok(text_target=texts, **opts) if target else tok(texts, **opts)
        return {k:v.cuda() for k,v in batch.items() if k in ('input_ids','attention_mask')}
    def content(batch):
        mask = batch['attention_mask'].bool().clone()
        for tid in tok.all_special_ids: mask &= batch['input_ids'] != tid
        assert mask.any(1).all()
        return mask
    def encode(batch):
        with fresh():
            return model.get_encoder()(**batch).last_hidden_state
    holder, pre, residuals, handles = {}, {}, {}, []
    for i, layer in enumerate(model.get_encoder().layers):
        def before(_m, _inp, output, i=i):
            pre[i]=(output[0] if isinstance(output,(tuple,list)) else output).detach()
        def after(_m, _inp, output, i=i):
            post=(output[0] if isinstance(output,(tuple,list)) else output).detach(); raw=pre.pop(i)
            mask=holder['mask'].float()
            def rms(h):return ((h.float().square()*mask[...,None]).sum((1,2))/(mask.sum(1)*h.shape[-1])).sqrt()
            ratio=(rms(post-raw)/rms(raw).clamp_min(1e-12)).cpu().tolist()
            residuals.setdefault(holder['phase']+'_'+str(i+1), []).extend(ratio)
        handles.extend([layer.register_forward_hook(before, prepend=True),layer.register_forward_hook(after)])
    reps={}
    with torch.no_grad():
        for target in (False,True):
            phase='target' if target else 'source'; values=[]
            for start in range(0,len(indices),4):
                batch=tokens(indices[start:start+4],target);mask=content(batch)
                holder.update(phase=phase,mask=mask)
                values.append(_masked_mean_pool(encode(batch),mask).float().cpu().numpy())
            reps[phase]=np.concatenate(values)
            print('REPRESENTATIONS',spec['id'],phase,len(indices),flush=True)
    for handle in handles:handle.remove()
    np.savez_compressed(folder/'representations.npz',source=reps['source'],target=reps['target'],indices=np.asarray(indices))
    geometry={'source':spectrum(reps['source']),'target':spectrum(reps['target']),
              'retrieval':retrieval(reps['source'],reps['target'],rows.iloc[indices].source.str.strip().tolist(),rows.iloc[indices].target.str.strip().tolist()),
              'source_norm_mean':float(np.linalg.norm(reps['source'],axis=1).mean()),
              'target_norm_mean':float(np.linalg.norm(reps['target'],axis=1).mean())}
    named=list(model.named_parameters());params=[p for _,p in named]
    groups=[]
    for name,_ in named:
        if 'encoder.layers.' in name:group='encoder_layer'+name.split('encoder.layers.')[1].split('.')[0]
        elif 'encoder' in name or 'shared' in name:group='encoder_other'
        else:group='decoder'
        groups.append(group)
    actual_lsr = float(getattr(model, 'jepa_weight', 0.))
    actual_nce = float(getattr(model, 'alignment_weight', 0.))
    gradients=[]
    for ids in batches:
        sb,tb=tokens(ids),tokens(ids,True)
        labels=tb['input_ids'].masked_fill(tb['input_ids']==tok.pad_token_id,-100)
        output=core(**sb,labels=labels,use_cache=False)
        with torch.no_grad():th=encode(tb).detach()
        src=F.normalize(_masked_mean_pool(output.encoder_last_hidden_state,sb['attention_mask']),dim=-1)
        tgt=F.normalize(_masked_mean_pool(th,tb['attention_mask']),dim=-1)
        lsr=(1-(src*tgt).sum(-1)).mean()
        source=_masked_mean_pool(encode(sb),content(sb))
        target=_masked_mean_pool(th,content(tb))
        nce=paired_contrastive_loss(source,target,.1)
        with torch.no_grad():actual=model(**sb,labels=labels,use_cache=False).loss
        expected=output.loss+actual_lsr*lsr+actual_nce*nce
        error=float((actual-expected).detach().abs());assert error<2e-6,(spec['id'],error)
        ce_g=torch.autograd.grad(output.loss,params,retain_graph=True,allow_unused=True)
        lsr_g=torch.autograd.grad(.1*lsr,params,allow_unused=True)
        nce_g=torch.autograd.grad(.1*nce,params,allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in ce_g+lsr_g+nce_g)
        assert all(g is None or torch.count_nonzero(g)==0 for group,g in zip(groups,nce_g) if group=='decoder')
        pairs={}
        for pair,a,b in [('CE_LSR',ce_g,lsr_g),('CE_InfoNCE',ce_g,nce_g),('LSR_InfoNCE',lsr_g,nce_g)]:
            pairs[pair]={group:gradient_group(groups,a,b,group) for group in ['all_parameters']+sorted(set(groups))}
        gradients.append({'indices':ids,'ce':float(output.loss.detach()),'lsr':float(lsr.detach()),
                          'infonce':float(nce.detach()),'actual_wrapper_loss_error':error,'pairs':pairs})
        del ce_g,lsr_g,nce_g,a,b,output,actual,expected,lsr,nce,source,target,src,tgt,th
        gc.collect();torch.cuda.empty_cache()
        print('GRADIENT',spec['id'],len(gradients),flush=True)
    assert versions=={n:p._version for n,p in model.named_parameters()}
    result={'id':spec['id'],'method':spec['method'],'selection':selected,'geometry':geometry,
            'residual_ratio_mean_by_layer':{k:float(np.mean(v)) for k,v in residuals.items()},
            'gradients':gradients,'actual_objective_weights':{'LSR':actual_lsr,'InfoNCE':actual_nce},
            'gradient_probe_weights':{'LSR':.1,'InfoNCE':.1},
            'counterfactual_probes':[n for n,w in [('LSR',actual_lsr),('InfoNCE',actual_nce)] if w==0.],
            'optimizer_steps':0,'parameter_versions_unchanged':True,
            'peak_allocated_MiB':torch.cuda.max_memory_allocated()//2**20,'loading':loading}
    write(folder/'RESULT.json',result)
    if route:
        for h in route._handles:h.remove()
        route._handles=[]
    del model,core,params,named,route;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    for p in [export,folder/'restore']:
        assert p.resolve().is_relative_to(folder.resolve());shutil.rmtree(p)
    print('MEASURED_CLEANED',spec['id'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--ids',nargs='+',required=True)
    parser.add_argument('--rows',type=int,default=576)
    parser.add_argument('--gradient-batches',type=int,default=4)
    parser.add_argument('--minimum-free-gpu-mib',type=int,default=16000)
    args=parser.parse_args()
    assert ROOT==Path('/workspace/ura/thanhdat/projects/CLRR') and Path(sys.prefix).resolve().is_relative_to(ROOT/'conda')
    out=(ROOT/args.output).resolve();assert out.is_relative_to(ROOT/'outputs/experiments')
    workspace=ROOT/args.workspace;engine.configure(workspace);engine.validate_environment()
    plan=read(workspace/'PLAN.json');lock=read(workspace/'selection_lock.json')
    raw=ROOT/'data_processed'/plan['dataset']/'validation.csv'
    assert sha256(raw)==plan['data']['validation']['sha256']
    rows=pd.read_csv(raw,dtype=str,keep_default_na=False)
    indices=np.random.default_rng(42).permutation(len(rows))[:args.rows].tolist()
    batches=np.random.default_rng(43).permutation(len(rows))[:args.gradient_batches*4].reshape(-1,4).tolist()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out.mkdir(parents=True,exist_ok=True)
    frozen={'ids':args.ids,'indices':indices,'gradient_batches':batches,'selection':{i:lock['runs'][i] for i in args.ids},
            'validation_sha256':sha256(raw),'worker_sha256':sha256(Path(__file__)),'precision':'FP32',
            'dropout':False,'optimizer_steps':0,'geometry_pooling':'content tokens excluding all tokenizer special IDs',
            'gradient_LSR_pooling':'original attention mask, includes special tokens',
            'limits':'Post-hoc checkpoint probes on validation; gradients are instantaneous, not accumulated Adam updates; not causal training evidence.'}
    if (out/'PLAN.json').exists():assert read(out/'PLAN.json')==frozen
    else:write(out/'PLAN.json',frozen)
    for ident in args.ids:
        folder=out/ident;folder.mkdir(exist_ok=True)
        if (folder/'RESULT.json').exists():continue
        spec=next(s for s in plan['runs'] if s['id']==ident)
        measure(spec,lock['runs'][ident],rows,indices,batches,folder,args.minimum_free_gpu_mib)
    write(out/'COMPLETE.json',{'status':'COMPLETE','ids':args.ids,'optimizer_steps':0,'worker_sha256':frozen['worker_sha256'],
                             'completed_unix':time.time(),'temporary_weights_removed':True})
    print('DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':main()
