"""Frozen W prefix-drift diagnostic and training-only initial norm calibration."""
import argparse
import gc
import json
import math
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import set_seed
from experiment_runner import engine
from experiment_runner.config import verify_frozen
from experiment_runner.integrity import restore_archive, sha256
from methods.prefix_consistency.v1.losses import gold_differences, path_statistics


def sentence_interval(values, seed=20261114, samples=10000):
    """Resample sentences after averaging their dropout pairs, never pseudo-replicate pairs."""
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Finite sentence means required')
    rng=np.random.default_rng(seed)
    means=[]
    for start in range(0,samples,200):
        idx=rng.integers(0,len(values),size=(min(200,samples-start),len(values)))
        means.extend(values[idx].mean(-1).tolist())
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
                bootstrap_samples=samples,resampling_unit='sentence; dropout pairs averaged within sentence')


def batches(frame,tokenizer,spec,size):
    for start in range(0,len(frame),size):
        rows=frame.iloc[start:start+size]
        inputs=tokenizer(rows.source.tolist(),padding=True,pad_to_multiple_of=8,
            truncation=True,max_length=spec['max_source_length'],return_tensors='pt')
        target=tokenizer(text_target=rows.target.tolist(),padding=True,pad_to_multiple_of=8,
            truncation=True,max_length=spec['max_target_length'],return_tensors='pt')
        inputs={k:v.cuda() for k,v in inputs.items() if k in ('input_ids','attention_mask')}
        inputs['labels']=target.input_ids.cuda().masked_fill(target.attention_mask.cuda().eq(0),-100)
        yield start,inputs


def diagnostic(out,plan):
    prep=plan['prefix_preparation']; checkpoint=prep['checkpoint']
    dest=out/'prefix_preparation';dest.mkdir(exist_ok=True)
    receipt=dest/'DIAGNOSTIC.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text());assert saved['manifest_sha256']==sha256(out/'bundle_manifest.json')
        return saved
    temporary=dest/'w_restore';temporary.mkdir(exist_ok=False)
    archive=Path(hf_hub_download(plan['repo'],checkpoint['prefix'].rstrip('/')+'/best.zip',
        revision=checkpoint['revision'],token=False,local_dir=temporary/'download'))
    assert sha256(archive)==checkpoint['archive_sha256']
    export=temporary/'export';restore_archive(archive,export)
    model,tokenizer,_=engine.load_model(checkpoint['spec'],export)
    model.gradient_checkpointing_disable();model.cuda().train()
    core=model.base_model
    frame=pd.read_csv(engine.DATA/'validation.csv',keep_default_na=False)
    assert len(frame)==576
    records=[];replay_checked=False
    for pair,seed in enumerate(prep['dropout_seeds']):
        set_seed(seed)
        for start,inputs in batches(frame,tokenizer,checkpoint['spec'],2):
            labels=inputs.pop('labels');decoder=model.prepare_decoder_input_ids_from_labels(labels)
            state=torch.cuda.get_rng_state();cpu_state=torch.get_rng_state()
            with torch.no_grad(),model._route(False):
                a=core(**inputs,decoder_input_ids=decoder,use_cache=False).logits
                b=core(**inputs,decoder_input_ids=decoder,use_cache=False).logits
                stats=path_statistics(gold_differences(a,b,labels),labels.ne(-100))
                if not replay_checked:
                    after=torch.cuda.get_rng_state();cpu_after=torch.get_rng_state()
                    torch.cuda.set_rng_state(state);torch.set_rng_state(cpu_state)
                    repeat=core(**inputs,decoder_input_ids=decoder,use_cache=False).logits
                    torch.testing.assert_close(repeat,a,rtol=0,atol=0)
                    torch.cuda.set_rng_state(after);torch.set_rng_state(cpu_after)
                    replay_checked=True;del repeat
                for j in range(len(labels)):
                    records.append(dict(row=start+j,pair=pair,dropout_seed=seed,
                        tokens=int(labels[j].ne(-100).sum()),**{k:float(v[j]) for k,v in stats.items()}))
            del a,b
            if start%32==0:
                engine.write(dest/'PROGRESS.json',dict(stage='W_DIAGNOSTIC',pair=pair+1,pairs=8,
                    rows_done=start+len(labels),rows=576,optimizer_steps=0))
    table=pd.DataFrame(records);table.to_csv(dest/'prefix_drift.csv',index=False)
    means=table.groupby('row')[['prefix','diagonal','endpoint','cross']].mean()
    summaries={k:sentence_interval(means[k].to_numpy()) for k in means}
    passed=summaries['cross']['ci95'][0]>0
    result=dict(status='PASS' if passed else 'HOLD',passed=passed,
        checkpoint=checkpoint,validation_rows=576,pairs=8,summaries=summaries,
        rng_replay_exact=replay_checked,optimizer_steps=0,test_used=False,
        manifest_sha256=sha256(out/'bundle_manifest.json'),
        csv_sha256=sha256(dest/'prefix_drift.csv'),
        interpretation='Positive cross-position drift admission only; no translation gain or novelty certification')
    engine.write(receipt,result)
    del model,core,tokenizer;gc.collect();torch.cuda.empty_cache()
    assert temporary.resolve().is_relative_to(dest.resolve());shutil.rmtree(temporary)
    print(json.dumps(result),flush=True)
    return result


def gradient_square(model):
    return sum(float(p.grad.float().norm().double().square())
               for p in model.parameters() if p.requires_grad and p.grad is not None)


def calibrate(out,plan):
    dest=out/'prefix_preparation';spec=plan['runs'][0]
    result_path=dest/'CALIBRATION.json'
    if result_path.exists():
        result=json.loads(result_path.read_text());assert result['manifest_sha256']==sha256(out/'bundle_manifest.json')
        return result
    set_seed(42)
    model,tokenizer,_=engine.load_model(spec)
    model.cuda().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
    frame=pd.read_csv(engine.DATA/'train.csv',keep_default_na=False).iloc[:64]
    records=[]
    for start,inputs in batches(frame,tokenizer,spec,4):
        state=torch.cuda.get_rng_state();cpu_state=torch.get_rng_state()
        # beta=1 gives KL gradients; remove CE by computing the native paired logits directly.
        values=[]
        for objective in ('kl','combined'):
            torch.cuda.set_rng_state(state);torch.set_rng_state(cpu_state)
            model.zero_grad(set_to_none=True)
            decoder=model.prepare_decoder_input_ids_from_labels(inputs['labels'])
            logits=[]
            for _ in range(2):
                with model._route(False):
                    logits.append(model.base_model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],decoder_input_ids=decoder,
                        use_cache=False,return_dict=True).logits)
            from methods.prefix_consistency.v1.losses import prefix_loss
            _,_,kl,path,_=prefix_loss(*logits,inputs['labels'])
            value=kl if objective=='kl' else 1.25*kl+.10*path
            value.backward();values.append(gradient_square(model))
            del logits,value,kl,path
        records.append(dict(train_start=start,kl_grad_sq=values[0],combined_grad_sq=values[1]))
        engine.write(dest/'PROGRESS.json',dict(stage='TRAIN_ONLY_CALIBRATION',rows_done=start+4,
            rows=64,optimizer_steps=0))
    numerator=sum(r['combined_grad_sq'] for r in records)
    denominator=sum(r['kl_grad_sq'] for r in records)
    assert denominator>0 and math.isfinite(numerator) and math.isfinite(denominator)
    beta=math.sqrt(numerator/denominator)
    assert math.isfinite(beta) and beta>0
    result=dict(passed=True,beta_cal=beta,records=records,train_rows=list(range(64)),
        microbatch=4,seed=42,optimizer_steps=0,validation_used=False,test_used=False,
        manifest_sha256=sha256(out/'bundle_manifest.json'),
        formula='sqrt(sum_b ||1.25 grad KL_b + .10 grad P_b||^2 / sum_b ||grad KL_b||^2)',
        limitation='Initial aggregate RMS norm matching; not directions or accumulated Adam updates')
    engine.write(result_path,result)
    model.zero_grad(set_to_none=True);del model,tokenizer;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();out,plan=verify_frozen(args.config)
    engine.configure(out);engine.wait_for_gpu();engine.validate_environment(gpu=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    admission=diagnostic(out,plan)
    if not admission['passed']:
        print('SCIENTIFIC_HOLD: no positive cross-drift admission; no calibration or training',flush=True)
        return
    calibration=calibrate(out,plan)
    engine.write(out/'prefix_preparation/COMPLETE.json',dict(passed=True,
        beta_cal=calibration['beta_cal'],manifest_sha256=sha256(out/'bundle_manifest.json'),
        training_started=False,test_used=False))


if __name__=='__main__':main()
