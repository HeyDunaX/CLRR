"""Validation-only checks of fresh cohort. No training or test selection."""
import sys, json, hashlib, gc, csv, contextlib, types, time, platform
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import transformers, sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/revalidation'))
from amis_rewire import train
from amis_rewire.modeling import CrossLayerResidualRewire, _masked_mean_pool, _find_stack_layers
from amis_rewire.metrics import generation_metrics
from fresh_confirmation_integrity import restore_archive
from run_fresh_seed_confirmation import weight_digest
OUT=ROOT/'outputs_rebuttal/amis_small_validation_checks_20261007'
COHORT=ROOT/'outputs_rebuttal/amis_fresh_confirmation_20261005'

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def geometry(s,t):
    def norm(x): return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    su,tu=norm(s),norm(t)
    sc,tc=s-s.mean(0),t-t.mean(0)
    scu,tcu=norm(sc),norm(tc)
    idx=np.arange(len(s));rng=np.random.default_rng(20261007)
    permutations=[rng.permutation(idx) for _ in range(100)]
    matched=float((su*tu).sum(1).mean())
    matchedc=float((scu*tcu).sum(1).mean())
    shuffled=np.array([(su*tu[p]).sum(1).mean() for p in permutations])
    shuffledc=np.array([(scu*tcu[p]).sum(1).mean() for p in permutations])
    result={'rows':len(s),'matched_cosine':matched,'shuffled_cosine_mean':float(shuffled.mean()),
        'shuffled_cosine_sd':float(shuffled.std(ddof=1)),'matched_minus_shuffled':matched-float(shuffled.mean()),
        'centered_matched_cosine':matchedc,'centered_shuffled_cosine_mean':float(shuffledc.mean()),
        'centered_matched_minus_shuffled':matchedc-float(shuffledc.mean()),'shuffle_seed':20261007,'shuffle_repetitions':100}
    spectra={}
    for name,x,u,center in [('source',s,su,sc),('target',t,tu,tc)]:
        sv=torch.linalg.svdvals(torch.from_numpy(center).cuda()).cpu().numpy()
        spectra[name]=sv
        energy=sv**2
        def erank(z):
            p=z[z>0]/z.sum();return float(np.exp(-(p*np.log(p)).sum()))
        n=len(x)
        result[name]={'singular_entropy_rank':erank(sv),'energy_entropy_rank':erank(energy),
            'centered_rank_limit':min(n-1,x.shape[1]),'mean_pairwise_cosine':float(((u.sum(0)**2).sum()-n)/(n*(n-1))),
            'mean_vector_norm':float(np.linalg.norm(x.mean(0))),
            'mean_squared_norm':float((x*x).sum(1).mean()),
            'common_mean_energy_fraction':float((x.mean(0)**2).sum()/(x*x).sum(1).mean())}
    return result,spectra

def set_layers(model,enabled):
    for h in model._handles:h.remove()
    model._handles=[]
    for i,layer in enumerate(_find_stack_layers(model.base_model,'encoder')):
        original=model._make_hook('encoder',i)
        def hook(module,inputs,output,original=original,i=i):
            old=model.strength
            if i+1 not in enabled:model.strength=0.0
            try:return original(module,inputs,output)
            finally:model.strength=old
        model._handles.append(layer.register_forward_hook(hook))

def run(family,seed):
    folder=OUT/family/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'DONE.json').exists():return
    val=json.loads((COHORT/family/f'seed{seed}/VAL_COMPLETE.json').read_text())
    receipt=json.loads((COHORT/family/f'seed{seed}/collection_receipt.json').read_text())
    assert receipt['verified']
    for n,d in val['bundle_files'].items():
        if n.startswith('src/'): assert sha(ROOT/n)==d,n
    archive=val['archive_path'];repo=receipt['repo'];revision=receipt['revision']
    tmp=OUT/'temporary_weights'/f'{family}_{seed}';tmp.mkdir(parents=True,exist_ok=True)
    dl=Path(hf_hub_download(repo_id=repo,filename=archive,revision=revision,local_dir=tmp/'hf'))
    assert sha(dl)==val['archive_sha256'] and dl.stat().st_size==val['archive_size_bytes']
    restore_archive(dl,tmp/'extracted')
    choices=list((tmp/'extracted').rglob('config.json'));assert len(choices)==1
    ckpt=choices[0].parent
    wf=next((ckpt/n for n in ['pytorch_model.bin','model.safetensors'] if (ckpt/n).exists()))
    assert weight_digest(wf)==val['selected_and_exported_weight_digest']
    tokenizer=AutoTokenizer.from_pretrained(ckpt,use_fast=True)
    raw,info=AutoModelForSeq2SeqLM.from_pretrained(ckpt,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    train.configure_mbart(tokenizer,raw,'tl_XX','zh_CN')
    model=CrossLayerResidualRewire(raw,distance=2,strength=.1,stack='encoder') if family in ['full','clrr'] else raw
    model=model.cuda().eval();model.config.use_cache=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=pd.read_csv(ROOT/'data_processed/amis_mandarin/validation.csv',dtype=str,keep_default_na=False)
    assert len(rows)==576
    reps={}
    with torch.no_grad():
        for column in ['source','target']:
            parts=[]
            for start in range(0,len(rows),8):
                texts=rows[column].iloc[start:start+8].tolist()
                encoded=tokenizer(text_target=texts,padding=True,truncation=True,max_length=256,return_tensors='pt') if column=='target' else tokenizer(texts,padding=True,truncation=True,max_length=256,return_tensors='pt')
                x={k:v.cuda() for k,v in encoded.items() if k in ['input_ids','attention_mask']}
                ctx=model._fresh_cache() if isinstance(model,CrossLayerResidualRewire) else contextlib.nullcontext()
                with ctx:out=model.get_encoder()(**x,return_dict=True)
                parts.append(_masked_mean_pool(out.last_hidden_state.float(),x['attention_mask']).cpu().numpy())
            reps[column]=np.concatenate(parts).astype('float64')
    geo,spectra=geometry(reps['source'],reps['target'])
    np.savez_compressed(folder/'singular_spectra.npz',**spectra)
    (folder/'geometry.json').write_text(json.dumps(geo,indent=2))
    interventions={}
    if family in ['clrr','full']:
        saved=pd.read_csv(COHORT/family/f'seed{seed}'/'runs'/val['run_name']/'validation_predictions.csv',dtype=str,keep_default_na=False)
        assert saved[['source','target']].equals(rows[['source','target']])
        masks={'all_on':set(range(3,13)),'drop_10_11':set(range(3,13))-{10,11}}
        if family=='full':masks.update(drop_10=set(range(3,13))-{10},drop_11=set(range(3,13))-{11})
        for condition,enabled in masks.items():
            set_layers(model,enabled);preds=[]
            with torch.no_grad():
                for start in range(0,len(rows),8):
                    texts=rows['source'].iloc[start:start+8].tolist()
                    inp=tokenizer(texts,padding=True,truncation=True,max_length=256,return_tensors='pt')
                    inp={k:v.cuda() for k,v in inp.items() if k in ['input_ids','attention_mask']}
                    gen=model.generate(**inp,num_beams=1,max_length=256)
                    preds.extend(s.strip() for s in tokenizer.batch_decode(gen,skip_special_tokens=True))
            if condition=='all_on':assert preds==saved['prediction'].str.strip().tolist(),'Route-on predictions failed exact replay'
            pd.DataFrame({'source':rows['source'],'target':rows['target'],'prediction':preds}).to_csv(folder/f'{condition}_predictions.csv',index=False)
            score=generation_metrics(preds,rows['target'].tolist(),tokenize='zh')
            from sacrebleu.metrics import CHRF
            score['chrf0']=CHRF(word_order=0).corpus_score(preds,[rows['target'].str.strip().tolist()]).score
            interventions[condition]=score
        (folder/'interventions.json').write_text(json.dumps(interventions,indent=2))
    done={'family':family,'seed':seed,'scope':'validation_only_no_optimizer_steps',
        'rows':576,'device':torch.cuda.get_device_name(),'compute_capability':torch.cuda.get_device_capability(),
        'torch':torch.__version__,'transformers':transformers.__version__,'sacrebleu':sacrebleu.__version__,
        'python':platform.python_version(),'precision':'FP32_TF32_off','hf_repo':repo,'hf_revision':revision,
        'archive_sha256':sha(dl),'weight_digest':val['selected_and_exported_weight_digest'],
        'source_hashes':{n:sha(ROOT/n) for n in val['bundle_files'] if n.startswith('src/')},
        'worker_sha256':sha(__file__),'validation_sha256':sha(ROOT/'data_processed/amis_mandarin/validation.csv'),
        'all_on_exact_prediction_replay':True if interventions else None,
        'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (folder/'DONE.json').write_text(json.dumps(done,indent=2))
    del model,raw,reps,tokenizer
    gc.collect();torch.cuda.empty_cache()
    import shutil
    assert tmp.resolve().is_relative_to((OUT/'temporary_weights').resolve())
    shutil.rmtree(tmp)
    print('DONE',family,seed,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--families',nargs='+',required=True);a=p.parse_args()
    assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name()
    for fam in a.families:
        for seed in range(45,53):run(fam,seed)
    (OUT/('COMPLETE_'+'_'.join(a.families)+'.json')).write_text(json.dumps({'status':'COMPLETE','models':len(a.families)*8}))
