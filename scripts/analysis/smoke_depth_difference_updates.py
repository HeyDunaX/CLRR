"""Bounded U2-v2 pretrained optimizer smoke; never a quality experiment."""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
import transformers
from transformers import AutoTokenizer, MBartForConditionalGeneration
from methods.depth_difference.v2 import ValueScaledDepthDifference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3,choices=range(1,4))
    p.add_argument('--accumulation',type=int,default=4,choices=range(1,5))
    a=p.parse_args()
    import fcntl
    if not Path(sys.prefix).resolve().is_relative_to(ROOT/'conda'):
        raise RuntimeError('Use project-local conda')
    lock=(ROOT/'tmp/training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    free,total=torch.cuda.mem_get_info()
    if free<16000*2**20: raise RuntimeError('Not enough free VRAM; leave other users untouched')
    torch.manual_seed(42);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    model_id='facebook/mbart-large-50-many-to-many-mmt'
    revision='e30b6cb8eb0d43a0b73cab73c7676b9863223a30'
    base=MBartForConditionalGeneration.from_pretrained(model_id,revision=revision,
        cache_dir=str(a.cache_dir),local_files_only=True,attn_implementation='eager',torch_dtype=torch.float32)
    model=ValueScaledDepthDifference(base,donor_layer=6).cuda().train()
    model.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    tok=AutoTokenizer.from_pretrained(model_id,revision=revision,cache_dir=str(a.cache_dir),
        local_files_only=True,src_lang='tl_XX',tgt_lang='zh_CN')
    with a.data.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    indices=random.Random(42).sample(range(len(rows)),a.steps*a.accumulation)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0.)
    logs=[]
    reference_weight=model.base_model.model.encoder.layers[0].fc1.weight.detach().clone()
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for index in indices[step*a.accumulation:(step+1)*a.accumulation]:
            row=rows[index]
            batch=tok([row['source']],text_target=[row['target']],padding=True,truncation=True,
                max_length=256,return_tensors='pt').to('cuda')
            batch['labels']=batch['labels'].masked_fill(batch['labels']==tok.pad_token_id,-100)
            loss=model(**batch,use_cache=False).loss
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            (loss/a.accumulation).backward();losses.append(float(loss.detach()))
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError('Nonfinite gradient')
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        gate_gradient=model.correction_attention.gates.grad.detach().cpu().tolist()
        optimizer.step()
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all(): raise RuntimeError('Nonfinite weights')
        record=dict(step=step+1,microbatch_ce=losses,preclip_gradient_norm=float(grad_norm),
            gate_gradient=gate_gradient,gates=model.correction_attention.gates.detach().cpu().tolist())
        logs.append(record);print(json.dumps(record),flush=True)
    delta=float((model.base_model.model.encoder.layers[0].fc1.weight-reference_weight).norm().detach())
    if delta==0 or not model.correction_attention.gates.detach().ne(0).any():
        raise RuntimeError('Expected backbone and gate updates')
    paths=[Path(__file__),ROOT/'src/methods/depth_difference/v1/model.py',ROOT/'src/methods/depth_difference/v2/model.py',a.data]
    report=dict(status='PASS',scope='Three-step optimizer correctness only; not matched training or quality evidence',
        model=model_id,revision=revision,method=model.method_id,donor_layer=6,decoder_layer=12,
        seed=42,row_indices=indices,steps=a.steps,microbatch_size=1,accumulation=a.accumulation,
        optimizer='AdamW',lr=5e-5,weight_decay=0.,clip_norm=1.,scheduler=None,
        dtype='FP32',tf32=False,gradient_checkpointing='non-reentrant',train_mode=True,
        torch=torch.__version__,transformers=transformers.__version__,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        encoder_fc1_update_norm=delta,logs=logs,
        hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('PASS: no checkpoint saved, temporary updated weights discarded at process exit',flush=True)


if __name__=='__main__': main()
