"""Frozen TRAIN-only CMC update geometry; never constructs an optimizer.

Exact position sum and Monte Carlo replay share inputs and dropout masks.
Logit VJPs bound channel memory without changing the first derivative.
"""
import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temp.replace(path)


def cpu_gate():
    import torch
    from methods.continuation_consistency.v1.model import pushforward_kl
    from analyze_cmc_gradient_estimator import cpu_gate as expectation_gate
    torch.manual_seed(891)
    features = torch.randn(2, 3, 4)
    weight = torch.randn(4, 7, requires_grad=True)
    left, right = features@weight, (features+.2)@weight
    channels = torch.randn(7, 7).log_softmax(-1)
    actions = torch.arange(7)
    dense = sum(pushforward_kl(left[b,t], right[b,t], actions, channels)
                for b in range(2) for t in range(b+2))/5
    expected = torch.autograd.grad(dense, weight, retain_graph=True)[0]
    cotangents = [torch.zeros_like(left), torch.zeros_like(right)]
    for b in range(2):
        for t in range(b+2):
            x, y = [v[b,t].detach().requires_grad_() for v in (left,right)]
            grads = torch.autograd.grad(pushforward_kl(x,y,actions,channels)/5,(x,y))
            for dest, grad in zip(cotangents, grads):
                dest[b,t] = grad
    actual = torch.autograd.grad((left,right), weight, grad_outputs=cotangents)[0]
    error = float((actual-expected).abs().max())
    assert error < 2e-6
    vectors = torch.tensor([[4.,0.],[-2.,2.]], dtype=torch.float64)
    def clip(g):
        return g/max(1.,float(g.norm()))
    average_clipped = torch.stack([clip(g) for g in vectors]).mean(0)
    clipped_average = clip(vectors.mean(0))
    assert (average_clipped-clipped_average).norm() > .1
    assert torch.allclose(clip(vectors[1])/clip(vectors[1]).norm(), vectors[1]/vectors[1].norm())
    return dict(passed=True, vjp_gradient_max_error=error,
                expectation_identity=expectation_gate(),
                clipping_noncommutation_norm=float((average_clipped-clipped_average).norm()),
                optimizer_steps=0)


def dot(a,b):
    import torch
    value = 0.
    for start in range(0,a.numel(),1048576):
        value += float(torch.dot(a[start:start+1048576].double(), b[start:start+1048576].double()))
    return value


def capture(parameters, outputs, cotangents=None, retain=True):
    import torch
    for parameter in parameters:
        parameter.grad = None
    if cotangents is None:
        outputs.backward(retain_graph=retain)
    else:
        torch.autograd.backward(outputs, cotangents, retain_graph=retain)
    result = torch.empty(sum(p.numel() for p in parameters), dtype=torch.float32)
    offset = 0
    for parameter in parameters:
        n = parameter.numel()
        if parameter.grad is None:
            result[offset:offset+n].zero_()
        else:
            result[offset:offset+n].copy_(parameter.grad.detach().flatten())
        parameter.grad = None
        offset += n
    return result


def run(plan_path):
    import numpy as np
    import pandas as pd
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, set_seed
    from experiment_runner.integrity import restore_archive, sha256
    from methods.continuation_consistency.v1.model import ContinuationConsistencyMT, pushforward_kl, symmetric_kl
    from methods.paired_route.v1.model import paired_loss
    import subprocess

    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    assert plan['optimizer_steps'] == 0 and plan['split'] == 'train'
    assert ROOT.resolve() == Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT/'conda')
    out = ROOT/plan['output']
    assert not (out/'COMPLETE.json').exists()
    for name, digest in plan['source_sha256'].items():
        assert sha256(ROOT/name) == digest, ('Source mismatch',name)
    assert sha256(ROOT/plan['train_path']) == plan['train_sha256']
    assert torch.__version__.split('+')[0] == plan['runtime']['torch']
    assert transformers.__version__ == plan['runtime']['transformers']
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],text=True).strip()
    assert gpu.startswith('Quadro RTX 6000,') and int(gpu.rsplit(',',1)[1]) >= 20000, gpu
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    assert available > 24*2**30, ('Insufficient host RAM',available)
    assert shutil.disk_usage(ROOT).free > 15*2**30
    write(out/'CPU_GATE.json', cpu_gate())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    progress = dict(stage='RESTORE_PINNED_W',optimizer_steps=0)
    write(out/'PROGRESS.json',progress)
    temp = out/'transient_restore'
    try:
        receipt = plan['checkpoint']
        archive = Path(hf_hub_download(plan['repo'],receipt['prefix']+'/best.zip',revision=receipt['revision'],token=False,local_dir=temp/'download'))
        assert sha256(archive) == receipt['archive_sha256']
        export = temp/'export'
        restore_archive(archive,export)
        optimizer_files = [str(p.relative_to(export)) for p in export.rglob('*') if p.is_file() and ('optimizer' in p.name or p.name in ('trainer_state.json','scheduler.pt'))]
        tokenizer = AutoTokenizer.from_pretrained(export,use_fast=True)
        tokenizer.src_lang,tokenizer.tgt_lang = 'tl_XX','zh_CN'
        frame = pd.read_csv(ROOT/plan['train_path'],keep_default_na=False)
        ids = torch.randperm(len(frame),generator=torch.Generator().manual_seed(plan['data_seed'])).tolist()[:128]
        examples = [tokenizer(frame.iloc[i].source,text_target=frame.iloc[i].target,max_length=256,truncation=True) for i in ids]
        data = [dict(row=i,source_tokens=len(e['input_ids']),target_tokens=len(e['labels'])) for i,e in zip(ids,examples)]
        write(out/'TRAIN_ROWS.json',dict(split='train',rows=data,tokenizer_src_lang=tokenizer.src_lang,tokenizer_tgt_lang=tokenizer.tgt_lang))
        base, loading = AutoModelForSeq2SeqLM.from_pretrained(export,attn_implementation='sdpa',output_loading_info=True)
        assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model = ContinuationConsistencyMT(base,'continuation-consistency-v1').cuda().train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
        collator = DataCollatorForSeq2Seq(tokenizer,model=None,pad_to_multiple_of=8,label_pad_token_id=-100,return_tensors='pt')
        parameters = [p for p in model.parameters() if p.requires_grad]
        versions = [p._version for p in parameters]
        n = sum(p.numel() for p in parameters)
        exact = {k:torch.zeros(n) for k in ('CE','W','CMC')}
        means = {k:torch.zeros(n) for k in ('sampled_CMC','CV','sampled_direct')}
        fingerprints = []
        dense_losses = []
        trials = []

        def fingerprint(logits):
            return [hashlib.sha256(v.detach().flatten()[::1009].cpu().numpy().tobytes()).hexdigest() for v in logits]

        def cotangent(logits, labels, decoder, inputs, picks, need_direct):
            gradients_c = [torch.zeros_like(v) for v in logits]
            gradients_d = [torch.zeros_like(v) for v in logits] if need_direct else None
            count = int(labels.ne(-100).sum())
            lengths = labels.ne(-100).sum(-1)
            rows = []
            for row,pos in picks:
                a,b = [v[row,pos].detach() for v in logits]
                actions = torch.cat((a.topk(4).indices,b.topk(4).indices,labels[row,pos:pos+1])).unique(sorted=True)
                actions = actions[actions.ne(base.config.eos_token_id)]
                rows.append((row,pos,actions))
            cmc_value = direct_value = 0.
            # Bound channel storage; no gradients through the same detached K.
            for start in range(0,len(rows),8):
                part = rows[start:start+8]
                channels = model.continuation_channel(inputs['input_ids'],inputs['attention_mask'],decoder,part)
                for (row,pos,actions),channel in zip(part,channels):
                    x,y = [v[row,pos].detach().requires_grad_() for v in logits]
                    coefficient = (float(lengths[row]) if need_direct else 1.)/count
                    c = coefficient*pushforward_kl(x,y,actions,channel)
                    gs = torch.autograd.grad(c,(x,y))
                    for dest,g in zip(gradients_c,gs):
                        dest[row,pos].add_(g)
                    cmc_value += float(c.detach())
                    if need_direct:
                        d = coefficient*symmetric_kl(x.log_softmax(-1),y.log_softmax(-1))
                        gs = torch.autograd.grad(d,(x,y))
                        for dest,g in zip(gradients_d,gs):
                            dest[row,pos].add_(g)
                        direct_value += float(d.detach())
                del channels
            return gradients_c,gradients_d,cmc_value,direct_value

        for repeat in range(-1,plan['sampling_repeats']):
            sampled = None if repeat == -1 else {k:torch.zeros(n) for k in ('C','D')}
            for mb in range(32):
                set_seed(plan['dropout_seed']+mb)
                inputs = {k:v.cuda() for k,v in collator(examples[mb*4:(mb+1)*4]).items()}
                labels = inputs['labels']
                decoder = model.prepare_decoder_input_ids_from_labels(labels)
                logits = []
                for _ in range(2):
                    with model._route(False):
                        logits.append(base(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],decoder_input_ids=decoder,use_cache=False,return_dict=True).logits)
                fp = fingerprint(logits)
                if repeat == -1:
                    fingerprints.append(fp)
                else:
                    assert fp == fingerprints[mb], ('Dropout replay mismatch',repeat,mb)
                _,ce,d = paired_loss(*logits,labels,1.)
                assert torch.isfinite(ce) and torch.isfinite(d), 'Nonfinite native loss'
                if repeat == -1:
                    if mb == 0:
                        # Native reentrant-checkpoint VJP versus ordinary backward.
                        a,b = [v[0,0].detach() for v in logits]
                        actions = torch.cat((a.topk(4).indices,b.topk(4).indices,labels[0,:1])).unique(sorted=True)
                        actions = actions[actions.ne(base.config.eos_token_id)]
                        channel = model.continuation_channel(inputs['input_ids'],inputs['attention_mask'],decoder,[(0,0,actions)])[0]
                        reference = capture(parameters,pushforward_kl(logits[0][0,0],logits[1][0,0],actions,channel))
                        x,y = [v[0,0].detach().requires_grad_() for v in logits]
                        local = torch.autograd.grad(pushforward_kl(x,y,actions,channel),(x,y))
                        cot = [torch.zeros_like(v) for v in logits]
                        for dest,g in zip(cot,local):
                            dest[0,0] = g
                        actual = capture(parameters,logits,cot)
                        denominator = math.sqrt(max(dot(reference,reference),1e-30))
                        actual.sub_(reference)
                        relative_error = math.sqrt(max(dot(actual,actual),0))/denominator
                        assert relative_error < 1e-4, ('Native VJP gate failed',relative_error)
                        write(out/'GPU_GATE.json',dict(passed=True,native_vjp_relative_error=relative_error,optimizer_steps=0,parameters_unchanged=[p._version for p in parameters]==versions,precision='fp32',attention='sdpa',checkpointing='reentrant'))
                        del reference,actual,cot,channel,x,y,local,a,b
                    for name,loss in (('CE',ce),('W',d)):
                        gradient = capture(parameters,loss)
                        exact[name].add_(gradient,alpha=1/32)
                        del gradient
                    picks = [(row,int(pos)) for row in range(4) for pos in labels[row].ne(-100).nonzero().flatten()]
                else:
                    generator = torch.Generator().manual_seed(plan['sampling_seed']+repeat*1000+mb)
                    picks = []
                    for row in range(4):
                        positions = labels[row].ne(-100).nonzero().flatten().tolist()
                        picks.append((row,positions[int(torch.randint(len(positions),(),generator=generator))]))
                cg,dg,cv,dv = cotangent(logits,labels,decoder,inputs,picks,repeat != -1)
                assert math.isfinite(cv) and math.isfinite(dv), 'Nonfinite channel loss'
                gradient = capture(parameters,logits,cg,retain=repeat != -1)
                (exact['CMC'] if repeat == -1 else sampled['C']).add_(gradient,alpha=1/32)
                del gradient,cg
                if repeat != -1:
                    gradient = capture(parameters,logits,dg,retain=False)
                    sampled['D'].add_(gradient,alpha=1/32)
                    del gradient,dg
                else:
                    dense_losses.append(dict(microbatch=mb,ce=float(ce.detach()),direct=float(d.detach()),cmc=cv))
                del logits,inputs,decoder,labels,ce,d
                gc.collect()
                progress = dict(stage='DENSE' if repeat == -1 else 'MC_REPLAY',repeat=repeat,repeats=plan['sampling_repeats'],microbatches_done=mb+1,microbatches=32,elapsed_seconds=time.time()-started,optimizer_steps=0,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
                write(out/'PROGRESS.json',progress)
            if repeat == -1:
                gram = [[dot(exact[a],exact[b]) for b in ('CE','W','CMC')] for a in ('CE','W','CMC')]
                assert np.isfinite(gram).all(), 'Nonfinite dense gradient'
                write(out/'DENSE.json',dict(gram_order=['CE','W','CMC'],gram=gram,losses=dense_losses,rows_sha256=sha256(out/'TRAIN_ROWS.json'),dropout_fingerprints=fingerprints))
                continue
            arrays = list(exact.values())+[sampled['C'],sampled['D']]
            full_gram = np.zeros((5,5))
            full_gram[:3,:3] = gram
            for i in range(3,5):
                for j in range(i+1):
                    full_gram[i,j] = full_gram[j,i] = dot(arrays[i],arrays[j])
            assert np.isfinite(full_gram).all(), 'Nonfinite sampled gradient'
            beta = 1.25
            coefficients = {'CE':[1,0,0,0,0], 'W':[1,beta,0,0,0], 'dense_CMC':[1,0,beta,0,0],
                            'sampled_CMC':[1,0,0,beta,0], 'CV':[1,beta,0,beta,-beta],
                            'sampled_direct':[1,0,0,0,beta]}
            record = dict(repeat=repeat,gram_order=['CE','W','CMC','sampled_C','sampled_D'],gram=full_gram.tolist(),methods={})
            ref = np.array(coefficients['dense_CMC'])
            for method,weights in coefficients.items():
                weights = np.array(weights,dtype=float)
                norm = math.sqrt(max(0,float(weights@full_gram@weights)))
                clip_factor = min(1.,1/max(norm,1e-30))
                ref_norm = math.sqrt(max(0,float(ref@full_gram@ref)))
                cosine = float(weights@full_gram@ref)/max(norm*ref_norm,1e-30)
                record['methods'][method] = dict(preclip_norm=norm,clip_factor=clip_factor,postclip_norm=norm*clip_factor,cosine_to_dense_CMC=cosine)
                if method in means:
                    for vector,weight in zip(arrays,weights):
                        if weight:
                            means[method].add_(vector,alpha=float(weight)*clip_factor/plan['sampling_repeats'])
            trials.append(record)
            write(out/'TRIALS.json',trials)
            del sampled,arrays
            gc.collect()
        summaries = {}
        for method,mean in means.items():
            ref_method = 'W' if method == 'sampled_direct' else 'dense_CMC'
            # All references and sample statistics use CE-inclusive total gradients.
            aux = exact['W'] if ref_method == 'W' else exact['CMC']
            ref_norm2 = dot(exact['CE'],exact['CE'])+2*1.25*dot(exact['CE'],aux)+1.25**2*dot(aux,aux)
            f = min(1.,1/math.sqrt(max(ref_norm2,1e-30)))
            mean2 = dot(mean,mean)
            mean_dot_ref = f*(dot(mean,exact['CE'])+1.25*dot(mean,aux))
            ref2 = f*f*ref_norm2
            bias2 = max(0,mean2+ref2-2*mean_dot_ref)
            average2 = sum(t['methods'][method]['postclip_norm']**2 for t in trials)/len(trials)
            se2 = max(0,(average2-mean2)/(len(trials)-1))
            summaries[method] = dict(reference=ref_method,mean_postclip_distance=math.sqrt(bias2),trace_standard_error_mean=math.sqrt(se2),mean_cosine_to_reference=mean_dot_ref/max(math.sqrt(mean2*ref2),1e-30),clip_frequency=sum(t['methods'][method]['clip_factor']<1 for t in trials)/len(trials),scope='Conditional Monte Carlo estimate, not a seed test or a causal translation result')
        assert [p._version for p in parameters] == versions
        assert all(p.grad is None for p in parameters)
        result = dict(status='COMPLETE',optimizer_steps=0,parameters_unchanged=True,split='train',test_used=False,validation_used=False,checkpoint=receipt,plan_sha256=sha256(plan_path),microbatch=4,accumulation=32,examples=128,sampling_repeats=len(trials),summaries=summaries,optimizer_state_files=optimizer_files,historical_adam_updates_measured=False,elapsed_seconds=time.time()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,source_verified=True,dropout_replay_verified=True)
        del model,base,parameters,exact,means
        gc.collect();torch.cuda.empty_cache()
    finally:
        assert temp.resolve().is_relative_to(out.resolve())
        if temp.exists():
            shutil.rmtree(temp)
    result['transient_restore_removed'] = not temp.exists()
    write(out/'COMPLETE.json',result)
    write(out/'PROGRESS.json',dict(stage='COMPLETE_NO_TRAINING',optimizer_steps=0))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--plan')
    args = parser.parse_args()
    if args.cpu:
        print(json.dumps(cpu_gate(),indent=2))
    else:
        try:
            run(args.plan)
        except Exception as exc:
            if args.plan:
                plan = json.loads(Path(args.plan).read_text())
                write(ROOT/plan['output']/'ERROR.json',dict(type=type(exc).__name__,error=str(exc),optimizer_steps=0))
            raise
