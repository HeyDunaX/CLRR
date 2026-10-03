"""Measure trained checkpoints without optimizer steps or changing historical scores."""
from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from amis_rewire.modeling import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM, _find_stack_layers, _masked_mean_pool


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def fresh(model):
    return model._fresh_cache() if hasattr(model,'_fresh_cache') else contextlib.nullcontext()


def save_json(path,value):
    path=Path(path)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def extract_checkpoint(archive,destination,baseline_archive):
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path=Path(info.filename)
            name=path.name
            selected=(name in ['config.json','generation_config.json','trainer_state.json','tokenizer_config.json',
                      'special_tokens_map.json','added_tokens.json','sentencepiece.bpe.model','spiece.model','tokenizer.json',
                      'vocab.json','merges.txt','pytorch_model.bin','model.safetensors','pytorch_model.bin.index.json',
                      'model.safetensors.index.json','metrics.json','test_predictions.csv'] or name.startswith(('pytorch_model-','model-')) and name.endswith(('.bin','.safetensors')))
            if not selected or info.is_dir(): continue
            target=(destination/path).resolve()
            if not target.is_relative_to(destination.resolve()): raise ValueError('Unsafe zip member')
            z.extract(info,destination)
    candidates=list({p.parent for p in destination.rglob('*') if p.name.startswith('pytorch_model') and p.suffix=='.bin' or p.suffix=='.safetensors'})
    if len(candidates)!=1: raise ValueError(f'Expected one complete checkpoint, found {candidates}')
    folder=candidates[0]
    reconstruction={}
    if not (folder/'config.json').exists():
        with zipfile.ZipFile(baseline_archive) as z:
            for filename in ['config.json','generation_config.json']:
                members=[n for n in z.namelist() if Path(n).name==filename]
                if filename=='config.json' and len(members)!=1: raise ValueError('No unique paired baseline config')
                if len(members)==1:
                    data=z.read(members[0])
                    (folder/filename).write_bytes(data)
                    reconstruction[filename]={'source':'paired trained baseline archive','archive':str(baseline_archive),
                                              'member':members[0],'sha256':hashlib.sha256(data).hexdigest()}
    return folder,reconstruction


class LayerMeasurements:
    def __init__(self,model,rewired):
        self.rewired=rewired
        self.enabled=False
        self.phase=''
        self.row_start=0
        self.mask=None
        self.pre={}
        self.rows=[]
        self.pooled={}
        self.handles=[]
        for index,layer in enumerate(_find_stack_layers(model,'encoder')):
            self.handles.append(layer.register_forward_hook(self.before(index),prepend=True))
            self.handles.append(layer.register_forward_hook(self.after(index)))

    def before(self,index):
        def hook(_module,_inputs,output):
            if self.enabled:
                self.pre[index]=(output[0] if isinstance(output,(tuple,list)) else output).detach()
        return hook

    def after(self,index):
        def hook(_module,_inputs,output):
            if not self.enabled: return
            post=(output[0] if isinstance(output,(tuple,list)) else output).detach().float()
            pre=self.pre.pop(index).float()
            mask=self.mask.to(pre.device).float().unsqueeze(-1)
            denom=(mask.sum(dim=1).squeeze(-1)*pre.shape[-1]).clamp_min(1)
            rms=lambda t: ((t.square()*mask).sum((1,2))/denom).sqrt()
            pre_rms,post_rms=rms(pre),rms(post)
            ratio=torch.zeros_like(pre_rms)
            skip_rms=torch.zeros_like(pre_rms)
            cosine=None
            if self.rewired is not None and index>=self.rewired.distance:
                skip=self.rewired._cache['encoder'][index-self.rewired.distance].float()
                residual=self.rewired.strength*skip
                skip_rms=rms(skip)
                ratio=rms(residual)/pre_rms.clamp_min(1e-12)
                dot=(pre*skip*mask).sum((1,2))
                cosine=dot/((pre.square()*mask).sum((1,2)).sqrt()*(skip.square()*mask).sum((1,2)).sqrt()).clamp_min(1e-12)
                assert torch.allclose(post,pre+residual,atol=2e-5,rtol=2e-5)
            for i in range(len(pre_rms)):
                self.rows.append({'phase':self.phase,'row_index':self.row_start+i,'layer':index+1,'pre_rms':float(pre_rms[i]),
                    'post_rms':float(post_rms[i]),'skip_rms':float(skip_rms[i]),
                    'residual_to_current_rms':float(ratio[i]),'skip_current_cosine':float(cosine[i]) if cosine is not None else None})
            pooled=_masked_mean_pool(post,self.mask).cpu().numpy()
            self.pooled.setdefault((self.phase,index+1),[]).append(pooled)
        return hook

    def close(self):
        for h in self.handles: h.remove()


def spectrum_diagnostics(representations):
    raw=np.asarray(representations,dtype='float64')
    centered=raw-raw.mean(axis=0,keepdims=True)
    singular=torch.linalg.svdvals(torch.from_numpy(centered).cuda()).cpu().numpy()
    numerical_tolerance=singular[0]*max(centered.shape)*np.finfo('float64').eps
    def erank(values):
        if values.sum()==0: return 0.0
        p=values[values>0]/values.sum()
        return float(np.exp(-(p*np.log(p)).sum()))
    energy=singular**2
    norm=np.linalg.norm(raw,axis=1)
    unit=raw/np.maximum(norm[:,None],1e-12)
    pairs=(unit@unit.T)[np.triu_indices(len(raw),1)]
    return dict(rows=len(raw),hidden_dim=raw.shape[1],maximum_centered_rank=min(len(raw)-1,raw.shape[1]),
        numerical_rank=int((singular>numerical_tolerance).sum()),numerical_tolerance=float(numerical_tolerance),
        effective_rank_singular_values=erank(singular),effective_rank_covariance_energy=erank(energy),
        participation_ratio=float(energy.sum()**2/(energy**2).sum()) if energy.sum()>0 else 0.0,
        covariance_trace=float(energy.sum()/(len(raw)-1)),top1_variance_fraction=float(energy[0]/energy.sum()) if energy.sum()>0 else 0.0,
        top10_variance_fraction=float(energy[:10].sum()/energy.sum()) if energy.sum()>0 else 0.0,
        mean_pairwise_cosine=float(pairs.mean()),std_pairwise_cosine=float(pairs.std()),
        pooled_norm_mean=float(norm.mean()),pooled_norm_std=float(norm.std()),singular_values=singular.tolist())


def extract_representations(model,tokenizer,rows,max_length,collector):
    result={}
    model.eval()
    for phase,column in [('source','source'),('target','target')]:
        values=[]
        for start in range(0,len(rows),16):
            texts=[r[column] for r in rows[start:start+16]]
            if phase=='target':
                ids=tokenizer(text_target=texts,padding=True,truncation=True,max_length=max_length,return_tensors='pt')
            else:
                ids=tokenizer(texts,padding=True,truncation=True,max_length=max_length,return_tensors='pt')
            ids={k:v.cuda() for k,v in ids.items() if k in ['input_ids','attention_mask']}
            collector.enabled=True
            collector.phase=phase
            collector.row_start=start
            collector.mask=ids['attention_mask']
            with torch.no_grad(),fresh(model):
                outputs=model.get_encoder()(**ids,return_dict=True)
                values.append(_masked_mean_pool(outputs.last_hidden_state.float(),ids['attention_mask']).cpu().numpy())
            collector.enabled=False
        result[phase]=np.concatenate(values)
        print('[representations]',phase,result[phase].shape,flush=True)
    return result


def gradient_group(name,parameter,embedding):
    if parameter is embedding: return 'shared_input_embeddings'
    if 'encoder' in name: return 'encoder_without_shared_embeddings'
    if 'decoder' in name: return 'decoder_without_shared_embeddings'
    return 'other_parameters'


def measure_gradients(model,tokenizer,rows,max_length,method):
    indices=sorted(np.random.default_rng(20261002).choice(len(rows),size=32,replace=False).tolist())
    model.train()
    model.gradient_checkpointing_disable()
    model.config.use_cache=False
    names,parameters=zip(*model.named_parameters())
    embedding=model.get_input_embeddings().weight
    groups=[gradient_group(n,p,embedding) for n,p in zip(names,parameters)]
    results=[]
    for batch_id,start in enumerate(range(0,len(indices),4)):
        selected=indices[start:start+4]
        batch_rows=[rows[i] for i in selected]
        x=tokenizer([r['source'] for r in batch_rows],padding=True,truncation=True,max_length=max_length,return_tensors='pt')
        y=tokenizer(text_target=[r['target'] for r in batch_rows],padding=True,truncation=True,max_length=max_length,return_tensors='pt')['input_ids']
        y[y==tokenizer.pad_token_id]=-100
        inputs={k:v.cuda() for k,v in x.items() if k in ['input_ids','attention_mask']}
        labels=y.cuda()
        target_captures=[]
        native_ce=[]
        def capture(_module,_inputs,output): target_captures.append(output.last_hidden_state)
        handle=model.get_encoder().register_forward_hook(capture)
        underlying=model
        while isinstance(underlying,(CrossLayerResidualRewire,JEPAGuidedSeq2SeqLM)): underlying=underlying.base_model
        def capture_ce(_module,_inputs,output): native_ce.append(output.loss)
        loss_handle=underlying.register_forward_hook(capture_ce)
        torch.manual_seed(1000+batch_id)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            outputs=model(**inputs,labels=labels,output_hidden_states=True,return_dict=True)
            reconstructed_ce=F.cross_entropy(outputs.logits.float().reshape(-1,model.config.vocab_size),labels.reshape(-1),ignore_index=-100)
            assert len(native_ce)==1
            ce=native_ce[0]
            if method in ['baseline','clrr']:
                target_ids=labels.clone()
                target_ids[target_ids==-100]=model.config.pad_token_id
                target_mask=target_ids.ne(model.config.pad_token_id).long()
                with torch.no_grad(),fresh(model):
                    target=model.get_encoder()(input_ids=target_ids,attention_mask=target_mask,return_dict=True).last_hidden_state
            else:
                assert len(target_captures)==2
                target=target_captures[-1]
                target_mask=labels.ne(-100).long()
            assert not target.requires_grad
            src=F.normalize(_masked_mean_pool(outputs.encoder_last_hidden_state,inputs['attention_mask']),dim=-1)
            tgt=F.normalize(_masked_mean_pool(target,target_mask),dim=-1)
            lsr=(1-(src*tgt).sum(-1)).mean()
            # Match the production wrapper: cast auxiliary loss before weighting.
            reconstructed_weighted=0.1*lsr.to(ce.dtype)
            weighted=reconstructed_weighted
            if method in ['full','lsr']:
                # Use the actual objective graph; do not approximate CE from BF16 returned logits.
                weighted=outputs.loss-ce
                lsr=weighted/0.1
        handle.remove()
        loss_handle.remove()
        ce_grads=torch.autograd.grad(ce,parameters,retain_graph=True,allow_unused=True)
        lsr_grads=torch.autograd.grad(weighted,parameters,allow_unused=True)
        sums={key:[0.,0.,0.] for key in ['all_parameters',*sorted(set(groups))]}
        for group,gce,glsr in zip(groups,ce_grads,lsr_grads):
            nce=float(gce.float().norm())**2 if gce is not None else 0.
            nlsr=float(glsr.float().norm())**2 if glsr is not None else 0.
            dot=float((gce.float()*glsr.float()).sum()) if gce is not None and glsr is not None else 0.
            for key in ['all_parameters',group]:
                sums[key][0]+=nce; sums[key][1]+=nlsr; sums[key][2]+=dot
        item={'batch_id':batch_id,'validation_indices':selected,'ce_loss':float(ce.detach()),
              'lsr_loss':float(lsr.detach()),'weighted_lsr_loss':float(weighted.detach()),
              'ce_reconstructed_difference':float((reconstructed_ce-ce).detach()),
              'weighted_lsr_reconstructed_difference':float((reconstructed_weighted-weighted).detach()),
              'gradient_objective':'native CE tensor and native auxiliary graph; baseline auxiliary is counterfactual',
              'lsr_active_in_training_method':method in ['full','lsr'],'target_stop_gradient':True,'groups':{}}
        for key,(a,b,c) in sums.items():
            item['groups'][key]={'ce_gradient_norm':a**0.5,'weighted_lsr_gradient_norm':b**0.5,
                'lsr_to_ce_norm_ratio':(b/a)**0.5 if a>0 else None,
                'gradient_cosine':c/(a*b)**0.5 if a>0 and b>0 else None,
                'combined_gradient_norm':max(0,a+b+2*c)**0.5}
        results.append(item)
        print('[gradients]',batch_id,'CE',item['ce_loss'],'LSR',item['lsr_loss'],item['groups']['all_parameters'],flush=True)
        del outputs,target,src,tgt,ce,lsr,weighted,ce_grads,lsr_grads,target_captures,native_ce,reconstructed_ce,reconstructed_weighted
        model.zero_grad(set_to_none=True)
    model.eval()
    return results


def run_model(item,manifest,out):
    root=out.parent
    ready=root/'downloads'/(item['id']+'.json')
    while not ready.exists():
        print('[waiting archive]',item['id'],flush=True)
        time.sleep(10)
    record=json.loads(ready.read_text())
    assert record['sha256']==item['sha256']
    baseline=next(m for m in manifest['models'] if m['dataset']==item['dataset'] and m['family']==item['family'] and m['method']=='baseline')
    baseline_archive=json.loads((root/'downloads'/(baseline['id']+'.json')).read_text())['path']
    folder,reconstruction=extract_checkpoint(record['path'],root/'checkpoints'/item['id'],baseline_archive)
    raw,loading=AutoModelForSeq2SeqLM.from_pretrained(folder,output_loading_info=True,local_files_only=True)
    for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']:
        if loading.get(key): raise ValueError(f'{item["id"]}: {key}={loading[key]}')
    model_type={'mBART':'mbart','NLLB':'m2m_100','mT5':'mt5','ByT5':'t5'}[item['family']]
    assert raw.config.model_type==model_type
    tokenizer_config=json.loads((folder/'tokenizer_config.json').read_text())
    overrides={}
    if isinstance(tokenizer_config.get('extra_special_tokens'),list):
        overrides={'extra_special_tokens':{},'additional_special_tokens':tokenizer_config['extra_special_tokens']}
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,use_fast=True,**overrides)
    vocabulary_check=None
    if (folder/'tokenizer.json').exists():
        from tokenizers import Tokenizer
        saved_vocab=Tokenizer.from_file(str(folder/'tokenizer.json')).get_vocab()
        assert tokenizer.get_vocab()==saved_vocab,'Tokenizer vocabulary/IDs changed'
        vocabulary_check=True
    saved_language={key:getattr(tokenizer,key,None) for key in ['src_lang','tgt_lang']}
    if item['src_lang']:
        tokenizer.src_lang=tokenizer_config.get('src_lang') or item['src_lang']
        tokenizer.tgt_lang=tokenizer_config.get('tgt_lang') or item['tgt_lang']
        raw.generation_config.forced_bos_token_id=tokenizer.convert_tokens_to_ids(tokenizer.tgt_lang)
    rewired=None
    if item['method'] in ['full','clrr']:
        rewired=CrossLayerResidualRewire(raw,distance=2,strength=0.1,stack='encoder')
        model=rewired
    else: model=raw
    if item['method'] in ['full','lsr']: model=JEPAGuidedSeq2SeqLM(model,jepa_weight=0.1)
    model.cuda().eval()
    model.config.use_cache=False
    source_rows=read_csv(ROOT/'data_processed'/item['dataset']/'test.csv')
    validation_rows=read_csv(ROOT/'data_processed'/item['dataset']/'validation.csv')
    model_out=out/item['id']; model_out.mkdir(exist_ok=True)
    provenance={'checkpoint':item,'actual_folder':str(folder),'loading_info':loading,'config_reconstruction':reconstruction,
                'tokenizer_legacy_extra_tokens_migrated':bool(overrides),'tokenizer_vocabulary_unchanged':vocabulary_check,
                'measurement_script_sha256':digest(__file__),
                'parameter_count':sum(p.numel() for p in model.parameters()),'saved_tokenizer_language':saved_language,
                'configured_tokenizer_language':{k:getattr(tokenizer,k,None) for k in ['src_lang','tgt_lang']},
                'weight_files':{p.name:digest(p) for p in list(folder.glob('*.bin'))+list(folder.glob('*.safetensors'))},
                'config':raw.config.to_dict(),'tokenizer_class':type(tokenizer).__name__}
    save_json(model_out/'provenance.json',provenance)
    collector=LayerMeasurements(model,rewired)
    representations=extract_representations(model,tokenizer,source_rows,item['max_length'],collector)
    collector.close()
    np.savez_compressed(model_out/'pooled_representations.npz',**representations,
                       **{phase+'_layer_'+str(layer):np.concatenate(v) for (phase,layer),v in collector.pooled.items()})
    with (model_out/'residual_per_example.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(collector.rows[0])); writer.writeheader(); writer.writerows(collector.rows)
    collapse={phase:spectrum_diagnostics(reps) for phase,reps in representations.items()}
    s,t=representations['source'],representations['target']
    alignment=(s*t).sum(1)/(np.linalg.norm(s,axis=1)*np.linalg.norm(t,axis=1)).clip(1e-12)
    collapse['source_target_alignment_cosine_mean']=float(alignment.mean())
    save_json(model_out/'collapse.json',collapse)
    print('[collapse]',item['id'],{k:{n:v[n] for n in ['effective_rank_singular_values','mean_pairwise_cosine','top1_variance_fraction']} for k,v in collapse.items() if isinstance(v,dict)},flush=True)
    gradients=measure_gradients(model,tokenizer,validation_rows,item['max_length'],item['method'])
    save_json(model_out/'gradients.json',gradients)
    sentinel_indices=[0,1,65,170,208,352,468,574]
    model.eval()
    model.config.use_cache=True
    inputs=tokenizer([source_rows[i]['source'] for i in sentinel_indices],padding=True,truncation=True,
                     max_length=item['max_length'],return_tensors='pt')
    inputs={k:v.cuda() for k,v in inputs.items() if k in ['input_ids','attention_mask']}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        generated=model.generate(**inputs,num_beams=4,max_length=item['max_length'])
    predictions=tokenizer.batch_decode(generated,skip_special_tokens=True)
    save_json(model_out/'sentinel_predictions.json',[{'index':i,'prediction':p.strip()} for i,p in zip(sentinel_indices,predictions)])
    with torch.no_grad():
        generated_fp32=model.generate(**inputs,num_beams=4,max_length=item['max_length'])
    predictions_fp32=tokenizer.batch_decode(generated_fp32,skip_special_tokens=True)
    save_json(model_out/'sentinel_predictions_fp32.json',[{'index':i,'prediction':p.strip()} for i,p in zip(sentinel_indices,predictions_fp32)])
    save_json(model_out/'DONE.json',{'id':item['id'],'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    del model,raw,collector,representations,gradients,rewired
    gc.collect(); torch.cuda.empty_cache()


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((ROOT/'checkpoint_manifest_amis.json').read_text())
    out=ROOT.parent/'outputs'; out.mkdir(exist_ok=True)
    for rel,expected in manifest['public_data_sha256'].items():
        path=ROOT/rel
        while not path.exists(): time.sleep(2)
        assert digest(path)==expected
    assert digest(ROOT/'src/amis_rewire/modeling.py')==manifest['modeling_sha256']
    if (out/'protocol.json').exists():
        previous=json.loads((out/'protocol.json').read_text())
        save_json(out/('protocol_'+previous['script_sha256']+'.json'),previous)
    save_json(out/'protocol.json',{'torch':torch.__version__,'transformers':transformers.__version__,
        'python':sys.version,'device':torch.cuda.get_device_name(0),'representation_precision':'float32, TF32 disabled',
        'gradient_precision':'BF16 autocast, FP32 parameters/unscaled gradients, no clipping',
        'representation_rows':'all test rows, source and target, mean pooling including valid special tokens',
        'gradient_rows':'32 validation rows, RNG 20261002, 8 batches of 4, same rows across methods',
        'dropout':'eval for representation; train with seed 1000+batch_id for gradients; target retains train-mode dropout',
        'rank':'SVD of float64 centered pooled representations, all singular values; maximum rank min(n-1,d)',
        'gradient_scope':'all parameters plus disjoint shared-embedding/encoder/decoder/other groups',
        'counterfactual':'Baseline and CLRR-only auxiliary gradient is hypothetical lambda=0.1, inactive during original training',
        'optimizer_steps':0,'manifest':manifest,'script_sha256':digest(__file__)})
    for item in manifest['models']:
        previous_gradients=out/item['id']/'gradients.json'
        current_objective=previous_gradients.exists() and 'gradient_objective' in json.loads(previous_gradients.read_text())[0]
        if (out/item['id']/'DONE.json').exists() and current_objective:
            print('[already completed]',item['id'],flush=True)
            continue
        if previous_gradients.exists() and not current_objective:
            save_json(previous_gradients.with_name('gradients_initial_reconstruction.json'),json.loads(previous_gradients.read_text()))
        print('[model start]',item['id'],time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),flush=True)
        try:
            run_model(item,manifest,out)
        except Exception as exc:
            save_json(out/(item['id']+'_ERROR.json'),{'type':type(exc).__name__,'message':str(exc)})
            raise
        print('[model done]',item['id'],flush=True)
    save_json(out/'COMPLETE.json',{'models':len(manifest['models']),'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})


if __name__=='__main__': main()
