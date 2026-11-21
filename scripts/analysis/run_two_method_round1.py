"""Frozen TRAIN probes, then independent gates and authorized validation pilots.

This is a project-locked queue worker, not a standalone GPU launcher. No test
predictions, parameter updates in the probes, or automatic follow-up controls.
"""
import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from experiment_runner import engine as w
from experiment_runner.config import verify_frozen
from experiment_runner.integrity import restore_archive, sha256


def gradient(parameters, loss, retain):
    import torch
    for p in parameters:
        p.grad = None
    loss.backward(retain_graph=retain)
    result = torch.empty(sum(p.numel() for p in parameters), dtype=torch.float32)
    offset = 0
    for p in parameters:
        size = p.numel()
        if p.grad is None:
            result[offset:offset+size].zero_()
        else:
            result[offset:offset+size].copy_(p.grad.detach().flatten())
        p.grad = None
        offset += size
    return result


def dot(a,b):
    import torch
    return sum(float(torch.dot(a[i:i+1048576].double(),b[i:i+1048576].double()))
               for i in range(0,a.numel(),1048576))


def frozen_probe(config, ident):
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, set_seed
    from methods.antisymmetric_consistency import AntisymmetricConsistencyMT
    from methods.antisymmetric_consistency.v1.model import antisymmetric_loss
    from methods.decision_residual_memory import DecisionResidualMemoryMT
    from methods.paired_route.v1.model import paired_loss
    out, plan = verify_frozen(config)
    w.configure(out)
    w.wait_for_gpu()
    w.validate_environment()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = next(s for s in plan['runs'] if s['id'] == ident)
    folder = out/'two_method_probe'/ident
    folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'ADMISSION.json').exists(), 'Probe already completed'
    record = plan['reused_w42']
    temp = folder/'transient_restore'
    started = time.time()
    w.write(out/'PROBE_PROGRESS.json',dict(id=ident,stage='RESTORE_PINNED_W',optimizer_steps=0))
    try:
        archive = Path(hf_hub_download(plan['repo'],record['prefix']+'/best.zip',
            revision=record['revision'],token=False,local_dir=temp/'download'))
        assert sha256(archive) == record['archive_sha256']
        restore_archive(archive,temp/'export')
        tokenizer = AutoTokenizer.from_pretrained(temp/'export',use_fast=True)
        tokenizer.src_lang,tokenizer.tgt_lang = spec['src_lang'],spec['tgt_lang']
        frame = pd.read_csv(w.DATA/'train.csv',keep_default_na=False)
        ids = torch.randperm(len(frame),generator=torch.Generator().manual_seed(1729)).tolist()[:4]
        rows = [tokenizer(frame.iloc[i].source,text_target=frame.iloc[i].target,
            truncation=True,max_length=256) for i in ids]
        collate = DataCollatorForSeq2Seq(tokenizer,model=None,pad_to_multiple_of=8,return_tensors='pt')
        batch = {k:v.cuda() for k,v in collate(rows).items()}
        base, info = AutoModelForSeq2SeqLM.from_pretrained(temp/'export',
            attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        cls = AntisymmetricConsistencyMT if spec['method'] == 'antisymmetric-consistency-v1' else DecisionResidualMemoryMT
        model = cls(base).cuda()
        parameters = list(model.parameters())
        versions = [p._version for p in parameters]
        set_seed(42)
        w.write(out/'PROBE_PROGRESS.json',dict(id=ident,stage='FROZEN_TRAIN_MEASUREMENT',optimizer_steps=0))
        if spec['method'] == 'antisymmetric-consistency-v1':
            available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                                 if x.startswith('MemAvailable:')))*1024
            assert available > 24*2**30, 'Full-parameter probe needs 24 GiB host RAM'
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
            decoder = model.prepare_decoder_input_ids_from_labels(batch['labels'])
            logits = []
            for _ in range(2):
                with model._route(False):
                    logits.append(base(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                        decoder_input_ids=decoder,use_cache=False).logits)
            _,_,ordinary = paired_loss(*logits,batch['labels'],1.25)
            _,_,projected = antisymmetric_loss(*logits,batch['labels'])
            before = gradient(parameters,ordinary,True)
            after = gradient(parameters,projected,False)
            aa,bb,ab = dot(before,before),dot(after,after),dot(before,after)
            scale = ab/max(aa,1e-30)
            residual = max(0.,bb-2*scale*ab+scale*scale*aa)**.5/max(bb**.5,1e-30)
            metrics = dict(w_aux_norm=aa**.5,ac_aux_norm=bb**.5,
                cosine=ab/max((aa*bb)**.5,1e-30),best_scalar_w=scale,
                relative_residual_after_best_scalar=residual,
                ordinary_kl=float(ordinary.detach()),ac_forward_kl=float(projected.detach()),
                parameters=sum(p.numel() for p in parameters),scope='one fixed B4 TRAIN batch; not GA32 or population inference')
            passed = aa>1e-20 and bb>1e-20 and residual >= .01
            reason = 'Distinct parameter update beyond scalar W' if passed else 'Degenerate or effectively scalar W on probe'
            del before,after,logits,ordinary,projected
        else:
            model.eval()
            with torch.no_grad():
                model.eta=0.
                model(**batch)  # Warm up native kernels before timing.
                torch.cuda.synchronize(); timer=time.time()
                reference=model(**batch).logits
                torch.cuda.synchronize(); native_seconds=time.time()-timer
                model.eta=.1
                model(**batch)  # Warm up the sequential derivative path too.
                torch.cuda.synchronize(); timer=time.time()
                actual=model(**batch).logits
                torch.cuda.synchronize(); memory_seconds=time.time()-timer
                labels=batch['labels']; valid=labels.ne(-100)
                valid[:,0]=False  # No earlier decision exists at the first output.
                safe=labels.clamp_min(0).unsqueeze(-1)
                native_nll=-reference.log_softmax(-1).gather(-1,safe).squeeze(-1)
                actual_nll=-actual.log_softmax(-1).gather(-1,safe).squeeze(-1)
                delta=(actual_nll-native_nll)[valid]
                stats=dict(model._last_memory_stats)
                source={k:v[:1] for k,v in batch.items() if k in ('input_ids','attention_mask')}
                torch.cuda.synchronize(); timer=time.time()
                generated=model.generate(**source,num_beams=1,max_new_tokens=16,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(spec['tgt_lang']))
                torch.cuda.synchronize(); generation_seconds=time.time()-timer
                # Prefix replay is quadratic. This is a conservative scheduling
                # estimate from TRAIN lengths, not a measured validation runtime.
                target_length=sum(len(r['labels']) for r in rows)/len(rows)
                generated_steps=generated.shape[1]-1
                projected_validation_seconds=generation_seconds*576*max(1.,target_length/max(generated_steps,1))**2
            metrics=dict(stats,mean_future_gold_nll_delta=float(delta.mean()),
                fraction_tokens_harmed_over_point1=float((delta>.1).float().mean()),
                native_forward_seconds=native_seconds,drm_forward_seconds=memory_seconds,
                forward_ratio=memory_seconds/max(native_seconds,1e-8),
                greedy16_seconds=generation_seconds,generated_steps=generated_steps,
                scheduling_estimate_validation_seconds=projected_validation_seconds,
                scope='four TRAIN references; gold NLL preservation proxy, not semantic alignment or causal proof')
            passed=(torch.isfinite(actual).all().item() and stats['relative_memory_change']>1e-8
                and float(delta.mean())<=.05 and float((delta>.1).float().mean())<=.25
                and metrics['forward_ratio']<=10 and projected_validation_seconds<=3600)
            reason='Nondegenerate preservation/latency screen' if passed else 'Preservation or resource screen failed; inspect metrics'
            del reference,actual,delta,generated
        unchanged = versions == [p._version for p in parameters]
        assert unchanged and all(p.grad is None for p in parameters)
        receipt=dict(passed=bool(passed),id=ident,reason=reason,metrics=metrics,
            train_rows=ids,optimizer_steps=0,test_used=False,parameters_unchanged=unchanged,
            checkpoint_revision=record['revision'],archive_sha256=record['archive_sha256'],
            bundle_sha256=sha256(out/'bundle_manifest.json'),seconds=time.time()-started,
            interpretation='Exploratory small-sample admission; neither efficacy nor novelty established')
        w.write(folder/'ADMISSION.json',receipt)
        print('FROZEN_PROBE_COMPLETE',json.dumps(receipt),flush=True)
    finally:
        if temp.exists():
            assert temp.resolve().is_relative_to(folder.resolve())
            shutil.rmtree(temp)


def queue(config):
    out, plan=verify_frozen(config)
    w.configure(out)
    state=dict(status='PROBES_AND_GATES',admitted=[],rejected=[],completed=[],
        bundle_sha256=sha256(out/'bundle_manifest.json'))
    assert not (out/'TWO_METHOD_GATES_COMPLETE.json').exists()
    def command(*args,timeout=None):
        subprocess.run([sys.executable,'-u',*args],check=True,cwd=ROOT,timeout=timeout)
    for spec in plan['runs']:
        ident=spec['id']
        state.update(current=ident,stage='FROZEN_TRAIN_PROBE')
        w.write(out/'TWO_METHOD_QUEUE.json',state)
        try:
            command('scripts/analysis/run_two_method_round1.py','--config',str(config),'--probe',ident,timeout=1200)
            receipt=json.loads((out/'two_method_probe'/ident/'ADMISSION.json').read_text())
            if not receipt['passed']:
                state['rejected'].append(dict(id=ident,reason=receipt['reason']))
                continue
            state['stage']='GPU_GATE_2_OPTIMIZER_STEPS_NOT_LONG_TRAIN'
            w.write(out/'TWO_METHOD_QUEUE.json',state)
            command('scripts/experiment.py','run','--config',str(config),'--id',ident,'--stage','gate',timeout=3600)
            state['admitted'].append(ident)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            state['rejected'].append(dict(id=ident,reason=str(error)))
            w.write(out/'two_method_probe'/ident/'QUEUE_FAILURE.json',dict(error=str(error),training_started=False))
    w.write(out/'TWO_METHOD_GATES_COMPLETE.json',state)
    for spec in plan['runs']:
        if spec['id'] not in state['admitted']:
            continue
        state.update(current=spec['id'],stage='TRAIN_VALIDATION',status='TRAINING')
        w.write(out/'TWO_METHOD_QUEUE.json',state)
        # Training/upload failures stop the queue and preserve local checkpoints.
        try:
            command('scripts/experiment.py','run','--config',str(config),'--id',spec['id'],'--stage','train')
        except Exception as error:
            state.update(status='FAILED',error=str(error))
            w.write(out/'TWO_METHOD_QUEUE.json',state)
            raise
        state['completed'].append(spec['id'])
    state.update(status='COMPLETE',stage='NO_AUTOMATIC_TEST_OR_REPLICATION')
    w.write(out/'TWO_METHOD_QUEUE.json',state)
    print('TWO_METHOD_QUEUE_COMPLETE',json.dumps(state),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--probe', help='Run ID declared in the frozen configuration')
    args=parser.parse_args()
    if args.probe:
        frozen_probe(args.config,args.probe)
    else:
        queue(args.config)
