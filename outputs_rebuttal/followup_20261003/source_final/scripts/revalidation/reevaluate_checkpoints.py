"""Reevaluate pinned trained weights with matched FP32 generation; no training."""
from pathlib import Path
import csv
import gc
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from amis_rewire.modeling import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM
from amis_rewire.evaluation import generation_precision
from amis_rewire.metrics import generation_metrics, metric_protocol
from measure_checkpoint_mechanisms import extract_checkpoint, digest, read_csv, save_json

OUT=ROOT/'outputs_rebuttal/followup_20261003/reevaluation'


def write_predictions(path,rows,predictions):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['index','source','target','prediction'])
        writer.writeheader()
        writer.writerows({'index':i,**row,'prediction':p} for i,(row,p) in enumerate(zip(rows,predictions)))


def predict(model,tokenizer,rows,max_length,batch_size=8):
    predictions=[]
    for start in range(0,len(rows),batch_size):
        batch=tokenizer([r['source'] for r in rows[start:start+batch_size]],padding=True,
                       truncation=True,max_length=max_length,pad_to_multiple_of=8,return_tensors='pt')
        batch={k:v.cuda() for k,v in batch.items() if k in ['input_ids','attention_mask']}
        with torch.no_grad(),generation_precision(model,'fp32'):
            ids=model.generate(**batch,num_beams=4,max_length=max_length)
        predictions.extend(p.strip() for p in tokenizer.batch_decode(ids,skip_special_tokens=True))
        if start%80==0:
            print('GENERATED',start+len(ids),'/',len(rows),flush=True)
    return predictions


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'checkpoint_manifest.json').read_text())
    for relative,url in manifest['public_data_urls'].items():
        path=ROOT/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(url,timeout=60) as response:
            data=response.read()
        assert hashlib.sha256(data).hexdigest()==manifest['public_data_sha256'][relative]
        path.write_bytes(data)
    assert digest(ROOT/'src/amis_rewire/modeling.py')==manifest['modeling_sha256']
    save_json(OUT/'protocol.json',{'manifest':manifest,'generation_precision':'FP32, TF32 disabled',
              'generation_batch_size':8,'pad_to_multiple_of':8,'beams':4,
              'max_length':'per-model historical max_length from manifest',
              'reference':'all original CSV targets; outer whitespace stripped; empty predictions retained',
              'optimizer_steps':0,'route_off':'same trained Full weights, inference alpha=0 only; not a CLRR-only/LSR-only training ablation',
              'script_sha256':digest(__file__),'torch':torch.__version__})
    archives={}
    def archive(item):
        if item['id'] not in archives:
            path=Path(hf_hub_download(repo_id=manifest['repo_id'],filename=item['archive'],
                                    revision=manifest['revision'],token=False))
            assert digest(path)==item['sha256']
            archives[item['id']]=path
            print('VERIFIED_ARCHIVE',item['id'],flush=True)
        return archives[item['id']]
    for item in manifest['models']:
        folder=OUT/item['id'];folder.mkdir(exist_ok=True)
        if (folder/'DONE.json').exists():
            print('ALREADY_DONE',item['id'],flush=True);continue
        print('MODEL_START',item['id'],time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),flush=True)
        baseline=next(m for m in manifest['models'] if m['dataset']==item['dataset'] and m['family']==item['family'] and m['method']=='baseline')
        extraction=ROOT.parent/'checkpoints'/item['id']
        extracted,reconstruction=extract_checkpoint(archive(item),extraction,archive(baseline))
        raw,loading=AutoModelForSeq2SeqLM.from_pretrained(extracted,output_loading_info=True,local_files_only=True)
        for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']:
            assert not loading.get(key),(item['id'],key)
        config=json.loads((extracted/'tokenizer_config.json').read_text())
        overrides={}
        if isinstance(config.get('extra_special_tokens'),list):
            overrides={'extra_special_tokens':{},'additional_special_tokens':config['extra_special_tokens']}
        tokenizer=AutoTokenizer.from_pretrained(extracted,local_files_only=True,use_fast=True,**overrides)
        if (extracted/'tokenizer.json').exists():
            from tokenizers import Tokenizer
            assert tokenizer.get_vocab()==Tokenizer.from_file(str(extracted/'tokenizer.json')).get_vocab()
        if item['src_lang']:
            tokenizer.src_lang=config.get('src_lang') or item['src_lang']
            tokenizer.tgt_lang=config.get('tgt_lang') or item['tgt_lang']
            raw.generation_config.forced_bos_token_id=tokenizer.convert_tokens_to_ids(tokenizer.tgt_lang)
        rewired=None
        if item['method'] in ['full','clrr']:
            rewired=CrossLayerResidualRewire(raw,distance=2,strength=.1,stack='encoder');model=rewired
        else:
            model=raw
        if item['method'] in ['full','lsr']:
            model=JEPAGuidedSeq2SeqLM(model,jepa_weight=.1)
        model.cuda().eval();model.config.use_cache=True
        rows=read_csv(ROOT/'data_processed'/item['dataset']/'test.csv')
        started=time.time()
        predictions=predict(model,tokenizer,rows,item['max_length'])
        write_predictions(folder/'test_predictions.csv',rows,predictions)
        scores=generation_metrics(predictions,[r['target'] for r in rows],tokenize='13a' if item['id'].startswith('ash-') else 'zh')
        metadata={'id':item['id'],'rows':len(rows),'scores':scores,'checkpoint':item,
                  'loading_info':loading,'config_reconstruction':reconstruction,
                  'evaluation_protocol':metric_protocol('13a' if item['id'].startswith('ash-') else 'zh'),
                  'generation_precision':'FP32','elapsed_seconds':time.time()-started,
                  'prediction_sha256':digest(folder/'test_predictions.csv'),
                  'reference_sha256':digest(ROOT/'data_processed'/item['dataset']/'test.csv')}
        save_json(folder/'metrics.json',metadata)
        print('SCORES',item['id'],scores,flush=True)
        if item['method']=='full':
            rewired.strength=0.
            route_off=predict(model,tokenizer,rows,item['max_length'])
            write_predictions(folder/'route_off_predictions.csv',rows,route_off)
            save_json(folder/'route_off_metrics.json',{'scores':generation_metrics(route_off,[r['target'] for r in rows],
                      tokenize='13a' if item['id'].startswith('ash-') else 'zh'),
                      'scope':'same trained weights; inference rewiring alpha=0; not a training ablation',
                      'prediction_sha256':digest(folder/'route_off_predictions.csv')})
            rewired.strength=.1
        if item['id'] in ['mbart-lsr','nllb-baseline','ash-mbart-full']:
            selected=[0,1,65,170,208,352,468,574]
            sample=[rows[i] for i in selected]
            one=predict(model,tokenizer,sample,item['max_length'],batch_size=1)
            eight=predict(model,tokenizer,sample,item['max_length'],batch_size=8)
            save_json(folder/'sentinel_batch_sensitivity.json',[{'index':i,'batch1':a,'batch8':b} for i,a,b in zip(selected,one,eight)])
        save_json(folder/'DONE.json',{'id':item['id'],'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        del model,raw,rewired;gc.collect();torch.cuda.empty_cache()
        shutil.rmtree(extraction)
    save_json(OUT/'COMPLETE.json',{'models':len(manifest['models'])})


if __name__=='__main__':
    main()
