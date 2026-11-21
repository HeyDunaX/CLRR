"""Frozen TRAIN-only CMC estimator audit; no optimizer, training, or test access.

Enumerates positions on short sentences, with identical two-dropout graphs for
all objectives. Parameter gradients include every trainable parameter. This is
a conditional position-sampling variance, not variance over Adam or training.
"""
import argparse
import gc
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from methods.continuation_consistency.v1.model import (
    ContinuationConsistencyMT, pushforward_kl, symmetric_kl)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def cpu_gate():
    """Unequal sentence lengths: enumerate every sampling outcome and gradient."""
    torch.manual_seed(731)
    left = torch.randn(2, 3, 7, requires_grad=True)
    right = torch.randn(2, 3, 7, requires_grad=True)
    lengths = (2, 3)
    actions = torch.arange(7)
    channels = torch.randn(2, 3, 7, 7).log_softmax(-1)
    d = symmetric_kl(left.log_softmax(-1), right.log_softmax(-1))
    c = torch.stack([torch.stack([pushforward_kl(left[b,t],right[b,t],actions,
        channels[b,t]) for t in range(lengths[b])]) .sum() for b in range(2)]).sum()/5
    dense = sum(d[b,:n].sum() for b,n in enumerate(lengths))/5
    def grad(value):
        return torch.cat([x.flatten() for x in torch.autograd.grad(
            value, (left,right), retain_graph=True)])
    gradients, values, identity_errors = [], [], []
    eye = torch.eye(7).log()
    for positions in itertools.product(range(2), range(3)):
        correction, identity_correction = 0., 0.
        for b,t in enumerate(positions):
            correction += lengths[b]*(pushforward_kl(left[b,t],right[b,t],actions,
                channels[b,t])-d[b,t])/5
            identity_correction += lengths[b]*(pushforward_kl(left[b,t],right[b,t],
                actions,eye)-d[b,t])/5
        estimate = dense+correction
        values.append(estimate.detach())
        gradients.append(grad(estimate))
        identity_errors.append(float((grad(dense+identity_correction)-grad(dense)).abs().max()))
    value_error = float((torch.stack(values).mean()-c).abs().detach())
    gradient_error = float((torch.stack(gradients).mean(0)-grad(c)).abs().max())
    assert value_error < 2e-6 and gradient_error < 2e-6
    assert max(identity_errors) < 2e-6
    # Counterexample: unbiased does not mean smaller variance.
    gd = torch.tensor([[2.,0.],[0.,2.],[-2.,0.],[0.,-2.]])
    gc_ = -gd
    old = float(gc_.square().sum(-1).mean())
    new = float((gc_-gd).square().sum(-1).mean())
    assert new > old
    return dict(passed=True,enumerated_outcomes=6,unequal_lengths=list(lengths),
        value_expectation_max_error=value_error,gradient_expectation_max_error=gradient_error,
        identity_channel_gradient_max_error=max(identity_errors),
        variance_counterexample=dict(old=old,new=new),optimizer_steps=0)


def dot(a,b):
    # Double precision reduction without a full-size FP64 copy.
    return sum(float((a[i:i+1048576].double()*b[i:i+1048576].double()).sum())
        for i in range(0,a.numel(),1048576))


def parameter_gradient(loss, parameters):
    for p in parameters:
        p.grad = None
    loss.backward(retain_graph=True)
    result = torch.empty(sum(p.numel() for p in parameters),dtype=torch.float32)
    offset = 0
    for p in parameters:
        dest = result[offset:offset+p.numel()]
        if p.grad is None:
            dest.zero_()
        else:
            dest.copy_(p.grad.detach().flatten(),non_blocking=False)
        offset += p.numel()
        p.grad = None
    return result


def gpu_audit(args):
    import shutil
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    from experiment_runner.integrity import sha256, restore_archive

    assert ROOT.resolve() == Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT/'conda')
    out = ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'COMPLETE.json').exists(), 'Never overwrite a completed audit'
    assert torch.cuda.get_device_name(0) == 'Quadro RTX 6000'
    free = torch.cuda.mem_get_info()[0]/2**20
    assert free >= 20000, ('Insufficient free VRAM',free)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config_path = ROOT/'configs/experiments/amis_continuation_consistency_rtx.json'
    config = json.loads(config_path.read_text())
    expected = json.loads((out/'SOURCE_MANIFEST.json').read_text())
    for name,digest in expected.items():
        assert sha256(ROOT/name) == digest, ('Source mismatch',name)
    assert torch.__version__.split('+')[0] == config['runtime']['torch']
    train_path = ROOT/'data_processed/amis_mandarin/train.csv'
    assert sha256(train_path) == config['data']['train']['sha256']
    receipt = config['reused_w42']
    temp = out/'restore'
    temp.mkdir(exist_ok=True)
    write(out/'PROGRESS.json',dict(stage='RESTORE_PINNED_W',optimizer_steps=0))
    archive = Path(hf_hub_download(config['repo'],receipt['prefix']+'/best.zip',
        revision=receipt['revision'],token=False,local_dir=temp/'download'))
    assert sha256(archive) == receipt['archive_sha256']
    export = temp/'export'
    restore_archive(archive,export)
    metadata = json.loads((export/'decoupled_consistency_config.json').read_text())
    assert metadata['method'] == 'weak-full-consistency-v1' and metadata['beta'] == 1.25
    tokenizer = AutoTokenizer.from_pretrained(export,use_fast=True)
    tokenizer.src_lang,tokenizer.tgt_lang = 'tl_XX','zh_CN'
    frame = pd.read_csv(train_path,keep_default_na=False)
    order = torch.randperm(len(frame),generator=torch.Generator().manual_seed(271828)).tolist()
    examples=[]
    for index in order:
        row=frame.iloc[index]
        inputs=tokenizer(row.source,text_target=row.target,return_tensors='pt',
            max_length=256,truncation=True)
        length=inputs['labels'].shape[1]
        if 5 <= length <= 10 and inputs['input_ids'].shape[1] <= 64:
            examples.append((index,inputs))
        if len(examples)==4:
            break
    assert len(examples)==4
    base,info=AutoModelForSeq2SeqLM.from_pretrained(export,
        attn_implementation='sdpa',output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model=ContinuationConsistencyMT(base,'continuation-consistency-v1').cuda().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
    parameters=[p for p in model.parameters() if p.requires_grad]
    versions=[p._version for p in parameters]
    results=[]
    for index,inputs_cpu in examples:
        for seed in (19421,19422):
            set_seed(seed)
            inputs={k:v.cuda() for k,v in inputs_cpu.items()}
            labels=inputs['labels']
            decoder=model.prepare_decoder_input_ids_from_labels(labels)
            logits=[]
            for _ in range(2):
                with model._route(False):
                    logits.append(base(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],
                        decoder_input_ids=decoder,use_cache=False,return_dict=True).logits)
            positions=labels[0].ne(-100).nonzero().flatten().tolist()
            rows=[]
            for pos in positions:
                a,b=[v[0,pos].detach() for v in logits]
                actions=torch.cat((a.topk(4).indices,b.topk(4).indices,labels[0,pos:pos+1])).unique(sorted=True)
                actions=actions[actions.ne(base.config.eos_token_id)]
                rows.append((0,pos,actions))
            channels=model.continuation_channel(inputs['input_ids'],inputs['attention_mask'],decoder,rows)
            n=sum(p.numel() for p in parameters)
            mean_c=torch.zeros(n);mean_r=torch.zeros(n)
            second_c=second_r=second_d=c_dot_d=0.
            values=[]
            for j,((_,pos,actions),channel) in enumerate(zip(rows,channels)):
                a,b=[v[0,pos] for v in logits]
                c=pushforward_kl(a,b,actions,channel)
                d=symmetric_kl(a.float().log_softmax(-1),b.float().log_softmax(-1))
                cg=parameter_gradient(c,parameters)
                dg=parameter_gradient(d,parameters)
                cn,dn,cd=dot(cg,cg),dot(dg,dg),dot(cg,dg)
                second_c+=cn/len(rows);second_d+=dn/len(rows);c_dot_d+=cd/len(rows)
                mean_c.add_(cg,alpha=1/len(rows))
                dg.neg_().add_(cg)
                second_r+=dot(dg,dg)/len(rows)
                mean_r.add_(dg,alpha=1/len(rows))
                values.append(dict(position=pos,cmc=float(c.detach()),direct=float(d.detach()),
                    gradient_c_squared=cn,gradient_d_squared=dn,gradient_c_dot_d=cd))
                del cg,dg
                write(out/'PROGRESS.json',dict(stage='ENUMERATE_ALL_PARAMETER_GRADIENTS',
                    row=index,dropout_seed=seed,positions_done=j+1,positions=len(rows),
                    conditions_done=len(results),conditions=8,optimizer_steps=0))
            vc=second_c-dot(mean_c,mean_c)
            vr=second_r-dot(mean_r,mean_r)
            assert vc>0 and vr>=-1e-6*max(second_r,1e-30)
            result=dict(train_row=index,dropout_seed=seed,supervised_tokens=len(rows),
                trainable_parameter_count=n,old_trace_variance=vc,
                cv_trace_variance=max(vr,0.),variance_ratio=max(vr,0.)/vc,
                positions=values,second_moments=dict(c=second_c,d=second_d,c_dot_d=c_dot_d))
            results.append(result)
            write(out/'RESULTS_PARTIAL.json',results)
            del mean_c,mean_r,logits,channels,c,d,a,b,inputs,decoder,labels
            gc.collect();torch.cuda.empty_cache()
    assert [p._version for p in parameters]==versions
    assert all(p.grad is None for p in parameters)
    old=sum(r['old_trace_variance'] for r in results)
    new=sum(r['cv_trace_variance'] for r in results)
    ratio=new/old
    # Locked operational screen, not a scientific significance threshold.
    improved=sum(r['variance_ratio']<1 for r in results)
    passed=ratio<=0.8 and improved>=6
    summary=dict(status='PASS' if passed else 'HOLD',passed=passed,
        scope='Conditional position-sampling variance, all trainable parameter gradients; short TRAIN sentences only',
        checkpoint=receipt,source_manifest_sha256=sha256(out/'SOURCE_MANIFEST.json'),
        cpu_gate=cpu_gate(),optimizer_steps=0,test_used=False,validation_used=False,
        parameters_unchanged=True,attention='sdpa',precision='fp32',tf32=False,
        gradient_checkpointing='reentrant',conditions=len(results),
        variance_ratio_pooled=ratio,improved_conditions=improved,
        admission='pooled variance ratio <=0.8 AND improved in >=6/8 fixed conditions',
        interpretation='Exploratory estimator screen, not proof of historical failure cause or generation gains',
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,results=results)
    del model,base,parameters;gc.collect();torch.cuda.empty_cache()
    assert temp.resolve().is_relative_to(out.resolve())
    shutil.rmtree(temp)
    summary['transient_restore_removed']=not temp.exists()
    write(out/'COMPLETE.json',summary)
    write(out/'PROGRESS.json',dict(stage='COMPLETE_NO_TRAINING',status=summary['status'],optimizer_steps=0))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('results','checkpoint')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--output',default='outputs/experiments/amis_cmc_estimator_gradient_audit')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.cpu:
        print(json.dumps(cpu_gate(),indent=2))
    else:
        assert args.output=='outputs/experiments/amis_cmc_estimator_gradient_audit'
        gpu_audit(args)
