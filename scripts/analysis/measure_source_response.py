"""Frozen source-response/dropout diagnostics. No optimizer or test access."""
import argparse
from collections import Counter
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from experiment_runner.integrity import sha256, restore_archive


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def negatives(sources, seed):
    """Fixed derangement, rejecting equal normalized source strings."""
    normalized = [' '.join(s.casefold().split()) for s in sources]
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        order = rng.permutation(len(sources))
        if all(normalized[i] != normalized[j] for i, j in enumerate(order)):
            return order.tolist()
    raise ValueError('No token-distinct source derangement found')


def symkl(a, b):
    return .5 * ((a.exp() - b.exp()) * (a - b)).sum(-1)


def token_metrics(a, b, u, v, labels, tail):
    """Four [tokens,vocab] logit views, reduced without parameter gradients."""
    la, lb, lu, lv = [F.log_softmax(t.float(), -1) for t in (a, b, u, v)]
    y = labels[:, None]
    qa, qb = F.log_softmax(a.float()-u.float(), -1), F.log_softmax(b.float()-v.float(), -1)
    pmean = .5*(la.exp()+lb.exp())
    result = dict(
        ce=-.5*(la.gather(1,y)+lb.gather(1,y)).squeeze(1),
        native_dropout_kl=symkl(la,lb),
        source_gold_gap=.5*((la-lu).gather(1,y)+(lb-lv).gather(1,y)).squeeze(1),
        response_dropout_kl=symkl(qa,qb),
        response_entropy=.5*(-(qa.exp()*qa).sum(-1)-(qb.exp()*qb).sum(-1)),
        response_tail_mass=.5*(qa.exp()[:,tail].sum(-1)+qb.exp()[:,tail].sum(-1)),
        response_low_match_probability_mass=(.5*(qa.exp()+qb.exp())*(pmean < 1e-5)).sum(-1),
        gold_probability=pmean.gather(1,y).squeeze(1),
        top1_accuracy=.5*(a.argmax(-1).eq(labels).float()+b.argmax(-1).eq(labels).float()))
    margins=[]
    for logits in (a,b):
        gold=logits.gather(1,y).squeeze(1)
        other=logits.clone().scatter_(1,y,-torch.inf).amax(-1)
        margins.append(gold-other)
    result['gold_margin']=.5*(margins[0]+margins[1])
    delta1,delta2=a.float()-u.float(),b.float()-v.float()
    delta1=delta1-delta1.mean(-1,keepdim=True)
    delta2=delta2-delta2.mean(-1,keepdim=True)
    result['response_signal_rms']=torch.sqrt(.5*(delta1.square().mean(-1)+delta2.square().mean(-1)))
    result['response_mask_difference_rms']=(delta1-delta2).square().mean(-1).sqrt()
    assert all(torch.isfinite(t).all() for t in result.values())
    return result


def cpu_gate():
    generator=torch.Generator().manual_seed(7)
    a,b,u,v=[torch.randn(5,19,generator=generator) for _ in range(4)]
    y=torch.tensor([0,1,2,3,4]);tail=torch.arange(19)>10
    la,lu=F.log_softmax(a,-1),F.log_softmax(u,-1)
    torch.testing.assert_close(F.softmax(la-lu,-1),F.softmax(a-u,-1))
    zero=token_metrics(a,b,a,b,y,tail)
    torch.testing.assert_close(zero['source_gold_gap'],torch.zeros(5))
    torch.testing.assert_close(zero['response_dropout_kl'],torch.zeros(5))
    torch.testing.assert_close(zero['response_entropy'],torch.full((5,),math.log(19)))
    whole=token_metrics(a,b,u,v,y,tail)
    parts=[token_metrics(a[i:i+1],b[i:i+1],u[i:i+1],v[i:i+1],y[i:i+1],tail) for i in range(5)]
    for key in whole:
        torch.testing.assert_close(whole[key],torch.cat([p[key] for p in parts]))
    order=negatives(['a','b','c','d'],42)
    assert order==negatives(['a','b','c','d'],42) and all(i!=j for i,j in enumerate(order))
    # A rare matched token can dominate the ratio distribution without being probable.
    pm=torch.tensor([[.6,.399999,.000001]])
    pn=torch.tensor([[.05,.94999999,.00000001]])
    assert (pm/pn).argmax(-1).item()==2 and pm.argmax(-1).item()==0
    from transformers import MBartConfig, MBartForConditionalGeneration
    config=MBartConfig(vocab_size=19,d_model=8,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=16,
        decoder_ffn_dim=16,max_position_embeddings=16,pad_token_id=1,
        decoder_start_token_id=2,dropout=.3,attention_dropout=.1,activation_dropout=.1)
    tiny=MBartForConditionalGeneration(config).train()
    inputs=dict(input_ids=torch.tensor([[3,4,2,1]]),attention_mask=torch.tensor([[1,1,1,0]]),
                decoder_input_ids=torch.tensor([[2,5,6]]),use_cache=False)
    with torch.inference_mode():
        torch.manual_seed(1);first=tiny(**inputs).logits
        torch.manual_seed(1);replay=tiny(**inputs).logits
        torch.manual_seed(2);independent=tiny(**inputs).logits
    torch.testing.assert_close(first,replay,rtol=0,atol=0)
    assert not torch.equal(first,independent) and all(p.grad is None for p in tiny.parameters())
    return dict(passed=True,checks=['log_partition_cancellation','source_insensitive_zero',
        'chunk_replay','deterministic_derangement','tail_domination_counterexample','tiny_mbart_rng_replay'])


def run(args):
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    plan=json.loads(Path(args.plan).read_text(encoding='utf-8'))
    metric_function=token_metrics
    sac=plan.get('metric_family')=='source_attribution_v1'
    if sac:
        from measure_source_attribution import sac_metrics, cpu_gate as sac_cpu_gate
        metric_function=sac_metrics
    out=Path(args.output).resolve()
    assert ROOT==Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT/'conda')
    assert out.is_relative_to(ROOT/'outputs/experiments')
    assert plan['optimizer_steps']==0 and plan['split']=='validation'
    for name,version in plan['runtime'].items():
        if name!='python': assert importlib.metadata.version(name).split('+')[0]==version
    for name,digest in plan['source_sha256'].items(): assert sha256(ROOT/name)==digest,name
    for split in ('train','validation'):
        assert sha256(ROOT/'data_processed'/plan['dataset']/(split+'.csv'))==plan['data'][split]['sha256']
    out.mkdir(parents=True,exist_ok=True)
    write(out/'CPU_GATE.json',sac_cpu_gate() if sac else cpu_gate())
    raw=pd.read_csv(ROOT/'data_processed'/plan['dataset']/'validation.csv',dtype=str,keep_default_na=False)
    rows=raw.iloc[plan['validation_rows']].reset_index(drop=True)
    orders=[negatives(rows.source.tolist(),seed) for seed in plan['negative_seeds']]
    write(out/'NEGATIVE_PAIRS.json',dict(validation_rows=plan['validation_rows'],orders=orders))
    train=pd.read_csv(ROOT/'data_processed'/plan['dataset']/'train.csv',dtype=str,keep_default_na=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    summaries={}
    for checkpoint in plan['checkpoints']:
        ident=checkpoint['id'];folder=out/ident;folder.mkdir(exist_ok=True)
        if (folder/'RESULT.json').exists():
            result=json.loads((folder/'RESULT.json').read_text())
            assert result['plan_sha256']==sha256(args.plan)
            assert result['csv_sha256']==sha256(folder/'per_sentence.csv')
            summaries[ident]=result;continue
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],text=True).strip()
        assert gpu.startswith('Quadro RTX 6000,') and int(gpu.rsplit(',',1)[1])>=plan['minimum_free_gpu_mib'],gpu
        assert shutil.disk_usage(out).free>=40*2**30
        write(out/'RESOURCE_ADMISSION.json',dict(id=ident,gpu=gpu,free_disk_gib=shutil.disk_usage(out).free/2**30))
        temporary=folder/'restore'
        assert not temporary.exists(),'Existing restore requires explicit inspection before reuse'
        temporary.mkdir()
        archive=Path(hf_hub_download(plan['repo'],checkpoint['prefix'].rstrip('/')+'/best.zip',
            revision=checkpoint['revision'],token=False,local_dir=temporary/'download'))
        assert sha256(archive)==checkpoint['archive_sha256']
        export=temporary/'export';restore_archive(archive,export)
        spec=checkpoint['spec']
        assert spec['method'] in ('baseline','paired-native-rdrop-v1','weak-full-consistency-v1')
        assert spec['strength']==0 and spec['jepa_weight']==0 and spec['source_prefix']==''
        if spec['method']!='baseline':
            name='paired_route_config.json' if spec['method']=='paired-native-rdrop-v1' else 'decoupled_consistency_config.json'
            saved=json.loads((export/name).read_text())
            assert saved['method']==spec['method'] and saved['beta']==spec['consistency_weight']
            assert saved['inference_route'] in ('OFF','native')
        model,info=AutoModelForSeq2SeqLM.from_pretrained(export,attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),info
        model.gradient_checkpointing_disable();model.cuda().train()
        assert model.config._attn_implementation=='sdpa' and next(model.parameters()).dtype==torch.float32
        tokenizer=AutoTokenizer.from_pretrained(export)
        tokenizer.src_lang=spec['src_lang'];tokenizer.tgt_lang=spec['tgt_lang']
        counts=Counter()
        for start in range(0,0 if sac else len(train),128):
            for ids in tokenizer(text_target=train.target.iloc[start:start+128].tolist(),
                truncation=True,max_length=spec['max_target_length'])['input_ids']: counts.update(ids)
        special=set(tokenizer.all_special_ids)
        tail=torch.tensor([counts[i]<=1 and i not in special for i in range(model.config.vocab_size)],device='cuda')
        records=[]
        for pair,order in enumerate(orders):
            for seed in plan['dropout_seeds']:
                for start in range(0,len(rows),plan['batch_size']):
                    batch=rows.iloc[start:start+plan['batch_size']]
                    wrong=rows.iloc[order[start:start+len(batch)]]
                    # Equal padded source shapes preserve RNG draw positions across conditions.
                    source=tokenizer(batch.source.tolist()+wrong.source.tolist(),padding=True,pad_to_multiple_of=8,
                        truncation=True,max_length=spec['max_source_length'],return_tensors='pt')
                    target=tokenizer(text_target=batch.target.tolist(),padding=True,pad_to_multiple_of=8,
                        truncation=True,max_length=spec['max_target_length'],return_tensors='pt')
                    labels=target.input_ids.cuda().masked_fill(target.attention_mask.cuda().eq(0),-100)
                    decoder=model.prepare_decoder_input_ids_from_labels(labels)
                    outputs=[]
                    with torch.inference_mode():
                        for view in range(2):
                            rng=seed+100000*view+start
                            for condition in range(2):
                                torch.manual_seed(rng);torch.cuda.manual_seed_all(rng)
                                offset=condition*len(batch)
                                outputs.append(model(input_ids=source.input_ids[offset:offset+len(batch)].cuda(),
                                    attention_mask=source.attention_mask[offset:offset+len(batch)].cuda(),
                                    decoder_input_ids=decoder,use_cache=False).logits)
                        a,u,b,v=outputs
                        if pair==0 and seed==plan['dropout_seeds'][0] and start==0:
                            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                            replay=model(input_ids=source.input_ids[:len(batch)].cuda(),
                                attention_mask=source.attention_mask[:len(batch)].cuda(),
                                decoder_input_ids=decoder,use_cache=False).logits
                            torch.testing.assert_close(a,replay,rtol=0,atol=0)
                            assert not torch.equal(a,b),'Dropout views unexpectedly identical'
                            assert all(p.grad is None for p in model.parameters())
                            write(folder/'GPU_GATE.json',dict(passed=True,exact_rng_replay=True,
                                independent_views=True,strict_loading=True,optimizer_steps=0))
                            del replay
                        for i in range(len(batch)):
                            valid=labels[i].ne(-100)
                            positions=valid.nonzero().flatten()
                            pieces=[]
                            for chunk in positions.split(8):
                                stat=metric_function(a[i,chunk],b[i,chunk],u[i,chunk],v[i,chunk],labels[i,chunk],tail)
                                pieces.append({k:t.cpu().numpy() for k,t in stat.items()})
                            arrays={k:np.concatenate([part[k] for part in pieces]) for k in pieces[0]}
                            special_mask=np.asarray([int(t) in special for t in labels[i,valid]])
                            for label,mask in [('lexical',~special_mask),('special',special_mask)]:
                                if not mask.any():continue
                                records.append(dict(row=plan['validation_rows'][start+i],negative_row=plan['validation_rows'][order[start+i]],
                                    pair=pair,dropout_seed=seed,category=label,tokens=int(mask.sum()),
                                    source_characters=len(batch.source.iloc[i]),target_characters=len(batch.target.iloc[i]),
                                    **{k:float(value[mask].mean()) for k,value in arrays.items()}))
                    del outputs,a,b,u,v
                    print(json.dumps(dict(id=ident,pair=pair,dropout_seed=seed,rows_done=start+len(batch))),flush=True)
        frame=pd.DataFrame(records);frame.to_csv(folder/'per_sentence.csv',index=False)
        metrics=[k for k in arrays]
        summary={category:group.groupby('row')[metrics].mean().mean().to_dict() for category,group in frame.groupby('category')}
        result=dict(id=ident,checkpoint=checkpoint,plan_sha256=sha256(args.plan),vocabulary_size=model.config.vocab_size,
            csv_sha256=sha256(folder/'per_sentence.csv'),summary=summary,optimizer_steps=0,test_used=False,
            aggregation='token mean within category/row; mean over fixed masks and negative pairings; equal sentence mean',
            scope='Frozen final-checkpoint descriptive probe; masks/pairings are not training seeds; no causal or confirmatory inference')
        write(folder/'RESULT.json',result);summaries[ident]=result
        del model,tokenizer,tail;gc.collect();torch.cuda.empty_cache()
        assert temporary.resolve().is_relative_to(folder.resolve())
        # Only a hash-verified copy from a pinned public checkpoint is removed.
        shutil.rmtree(temporary)
    write(out/'COMPLETE.json',dict(passed=True,ids=list(summaries),optimizer_steps=0,test_used=False,plan_sha256=sha256(args.plan)))
    write(out/'SUMMARY.json',summaries)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan');parser.add_argument('--output');parser.add_argument('--cpu-gate',action='store_true')
    args=parser.parse_args()
    if args.cpu_gate: print(json.dumps(cpu_gate()))
    else:
        assert args.plan and args.output
        run(args)
