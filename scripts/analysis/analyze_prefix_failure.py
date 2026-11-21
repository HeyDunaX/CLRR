"""Post-hoc PLC diagnostics. No optimizer, training, test data or recipe changes."""
import argparse
import csv
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'outputs/experiments/amis_prefix_consistency_failure_analysis'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def inventory():
    status = json.loads((ROOT / 'outputs/experiments/amis_prefix_consistency_pilot_rtx_suite/LATEST_COMPLETE_STATUS.json').read_text())
    runs = dict(status['runs'])
    w = json.loads((ROOT / 'outputs/experiments/amis_decoupled_consistency_rtx/runs/W_S42/VAL_COMPLETE.json').read_text())
    runs['W_S42'] = dict(w, remote_prefix=w['spec']['remote_prefix'])
    return runs


def download(record, name, folder):
    key = record['remote_prefix'] + '/' + name
    info = record['backup']['files'][key]
    path = folder / name
    if not path.exists():
        url = ('https://huggingface.co/FiveC/amis-rewire-checkpoints/resolve/'
               + record['backup']['revision'] + '/' + key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(url, timeout=90).read())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256'], name
    return path


def cpu():
    import numpy as np
    import pandas as pd
    from sacrebleu.metrics import BLEU, CHRF
    refs = pd.read_csv(ROOT / 'data_processed/amis_mandarin/validation.csv', keep_default_na=False)
    metrics = {'bleu': BLEU(tokenize='zh'), 'chrf++': CHRF(word_order=2)}
    stats, report, predictions = {}, {}, {}
    rng = np.random.default_rng(20261115)
    indices = rng.integers(0, len(refs), (10000, len(refs)))
    for name, record in inventory().items():
        folder = OUT / 'artifacts' / name
        pred = pd.read_csv(download(record, 'validation_predictions.csv', folder), keep_default_na=False)
        assert pred[['source', 'target']].equals(refs[['source', 'target']])
        predictions[name] = pred.prediction.tolist()
        stat = {key: np.array(metric._extract_corpus_statistics(pred.prediction.tolist(), [refs.target.tolist()]))
                for key, metric in metrics.items()}
        stats[name] = stat
        scores = {key: metric._compute_score_from_stats(stat[key].sum(0).tolist()).score
                  for key, metric in metrics.items()}
        assert all(abs(scores[k] - record['scores'][k]) < 1e-10 for k in scores)
        state = json.loads(download(record, 'final_trainer_state.json', folder).read_text())
        evaluations = [r for r in state['log_history'] if 'eval_bleu' in r]
        best = max(evaluations, key=lambda x: x['eval_bleu'])
        assert abs(best['eval_bleu'] - scores['bleu']) < 1e-10
        # Generation proxies only: neither repetition nor length establishes adequacy.
        def repeated(text):
            chars = ''.join(text.split())
            grams = [chars[i:i+3] for i in range(len(chars)-2)]
            return len(grams) >= 3 and any(grams.count(g) >= 3 for g in set(grams))
        sizes = np.array([len(''.join(x.split())) for x in pred.prediction])
        reference_sizes = np.array([len(''.join(x.split())) for x in refs.target])
        report[name] = dict(scores=scores, best_epoch=best['epoch'], final_epoch=state['epoch'],
            eval_history=evaluations, empty=int((sizes == 0).sum()),
            char_length_ratio=float(sizes.sum()/reference_sizes.sum()),
            repeated_char_trigram_ge3=sum(map(repeated, pred.prediction)))
    for name in report:
        if name == 'W_S42':
            continue
        paired = {}
        for key, metric in metrics.items():
            deltas = []
            for idx in indices:
                a = metric._compute_score_from_stats(stats[name][key][idx].sum(0).tolist()).score
                b = metric._compute_score_from_stats(stats['W_S42'][key][idx].sum(0).tolist()).score
                deltas.append(a-b)
            paired[key] = dict(delta=report[name]['scores'][key]-report['W_S42']['scores'][key],
                ci95=np.quantile(deltas, [.025, .975]).tolist(),
                bootstrap_positive_fraction=float(np.mean(np.array(deltas)>0)))
        report[name]['paired_vs_W'] = paired
        buckets = {}
        lengths = refs.target.map(lambda x: len(''.join(x.split()))).to_numpy()
        cuts = np.quantile(lengths, [.25, .5, .75])
        for label, mask in [('Q1', lengths <= cuts[0]), ('Q2', (lengths > cuts[0]) & (lengths <= cuts[1])),
                            ('Q3', (lengths > cuts[1]) & (lengths <= cuts[2])), ('Q4', lengths > cuts[2])]:
            buckets[label] = {'rows': int(mask.sum())}
            for key, metric in metrics.items():
                buckets[label][key+'_delta_vs_W'] = (metric._compute_score_from_stats(stats[name][key][mask].sum(0).tolist()).score
                    - metric._compute_score_from_stats(stats['W_S42'][key][mask].sum(0).tolist()).score)
        report[name]['reference_character_length_quartiles'] = dict(cuts=cuts.tolist(), groups=buckets)
    examples = []
    chrf = metrics['chrf++']
    for i, row in refs.iterrows():
        a, b = predictions['P_PREFIX_S42'][i], predictions['W_S42'][i]
        examples.append(dict(row=int(i), source=row.source, target=row.target, PLC=a, W=b,
            delta_sentence_chrf=chrf.sentence_score(a, [row.target]).score-chrf.sentence_score(b, [row.target]).score))
    pd.DataFrame(examples).sort_values('delta_sentence_chrf').to_csv(OUT/'sentence_differences.csv',index=False)
    write(OUT/'CPU_ANALYSIS.json', dict(runs=report, bootstrap_samples=10000,
        resampling_unit='validation sentence; seed42 checkpoints fixed after selection',
        exploratory=True, test_used=False,
        limitation='Post-selection validation intervals are descriptive, not confirmatory evidence across training seeds.'))
    print(json.dumps(report, ensure_ascii=False), flush=True)


def coefficients(a, mask):
    import torch
    lengths = mask.sum(-1, keepdim=True)
    inverse = mask / mask.cumsum(-1).clamp_min(1).float()
    diagonal = inverse.flip(-1).cumsum(-1).flip(-1)/lengths
    p = 2 * (a.cumsum(-1)*inverse).flip(-1).cumsum(-1).flip(-1)/lengths/a.shape[0]
    d = 2*a*diagonal/a.shape[0]
    return p*mask, d*mask


def gradients(left, right, labels):
    """Exact gradients wrt the TWO INDEPENDENT output-logit tensors, not weights."""
    import torch
    lp, lq = left.float().log_softmax(-1), right.float().log_softmax(-1)
    p, q = lp.exp(), lq.exp()
    valid = labels.ne(-100); safe = labels.masked_fill(~valid, 0).unsqueeze(-1)
    one = torch.zeros_like(p).scatter_(-1, safe, 1.)
    a = (lp-lq).gather(-1, safe).squeeze(-1)*valid
    pc, dc = coefficients(a, valid)
    n = valid.sum(); scale = valid.unsqueeze(-1)/n
    ce = ((p-one)*scale*.5, (q-one)*scale*.5)
    k1 = (p*(lp-lq)).sum(-1,keepdim=True)
    k2 = (q*(lq-lp)).sum(-1,keepdim=True)
    kl = ((p*(lp-lq-k1)+p-q)*scale*.5, (q*(lq-lp-k2)+q-p)*scale*.5)
    path = ((one-p)*pc.unsqueeze(-1), -(one-q)*pc.unsqueeze(-1))
    diag = ((one-p)*dc.unsqueeze(-1), -(one-q)*dc.unsqueeze(-1))
    return {'ce': ce, 'kl': kl, 'prefix': path, 'diagonal': diag}, a, p.gather(-1,safe).squeeze(-1), q.gather(-1,safe).squeeze(-1)


def check():
    import torch
    from methods.prefix_consistency.v1.losses import prefix_loss
    torch.manual_seed(15)
    left,right=[torch.randn(2,5,7,requires_grad=True) for _ in range(2)]
    labels=torch.tensor([[1,-100,3,2,-100],[2,3,4,1,0]])
    with torch.no_grad(): expected,_,_,_=gradients(left,right,labels)
    for key in expected:
        left.grad=right.grad=None
        _,ce,kl,path,_=prefix_loss(left,right,labels,mode='diagonal' if key=='diagonal' else 'prefix')
        {'ce':ce,'kl':kl,'prefix':path,'diagonal':path}[key].backward()
        for actual,want in zip((left.grad,right.grad),expected[key]):
            torch.testing.assert_close(actual,want,rtol=3e-5,atol=2e-7)
    import copy
    sys.path.insert(0,str(ROOT/'tests'))
    from test_paired_route import native,batch
    from methods.paired_route import PairedRouteMT
    torch.set_num_threads(1)
    full=PairedRouteMT(native(.15),'paired-native-rdrop-v1').train()
    partial=copy.deepcopy(full)
    selected=['model.encoder.layers.1.fc2.weight','model.decoder.layers.1.fc2.weight']
    for k,p in partial.base_model.named_parameters():
        p.requires_grad_(k in selected)
    hooks=[stack.embed_tokens.register_forward_hook(lambda module,args,output:output.requires_grad_(True))
           for stack in (partial.base_model.model.encoder,partial.base_model.model.decoder)]
    grads=[]
    for model in (full,partial):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
        torch.manual_seed(51); model(**batch()).loss.backward()
        grads.append({k:p.grad.clone() for k,p in model.base_model.named_parameters() if k in selected})
    assert set(grads[0])==set(selected)
    for k in selected:
        torch.testing.assert_close(grads[0][k],grads[1][k],rtol=3e-5,atol=2e-7)
    for hook in hooks:hook.remove()
    print('ANALYTIC_LOGIT_GRADIENT_CHECK_PASS',flush=True)


def parameter_probe(model, tokenizer, frame, indices, name):
    """Actual partial parameter gradients on three named blocks, never Adam updates."""
    import torch
    from transformers import set_seed
    from methods.prefix_consistency.v1.losses import prefix_loss
    keys = ['model.encoder.layers.11.fc2.weight', 'model.decoder.layers.11.fc2.weight',
            'model.decoder.layers.11.encoder_attn.q_proj.weight']
    selected = {k:p for k,p in model.base_model.named_parameters() if k in keys}
    assert set(selected)==set(keys)
    for k,p in model.base_model.named_parameters():
        p.requires_grad_(k in selected)
    # mBART's encoder/decoder embedding modules share weights with `shared`,
    # but the module returned by get_input_embeddings need not be called.
    # Attach to the two modules actually invoked, for frozen-backbone replay.
    hooks=[stack.embed_tokens.register_forward_hook(lambda module, args, output: output.requires_grad_(True))
           for stack in (model.base_model.model.encoder,model.base_model.model.decoder)]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
    records=[]
    for row in indices[:4]:
        inp=tokenizer([frame.iloc[row].source],return_tensors='pt',truncation=True,max_length=256)
        inp={k:v.cuda() for k,v in inp.items() if k in ('input_ids','attention_mask')}
        labels=tokenizer(text_target=[frame.iloc[row].target],return_tensors='pt',truncation=True,max_length=256).input_ids.cuda()
        decoder=model.prepare_decoder_input_ids_from_labels(labels)
        saved={}; objective_values=[]
        for objective in ('ce','kl','prefix','diagonal'):
            set_seed(20261115+row);model.zero_grad(set_to_none=True)
            logits=[]
            for _ in range(2):
                with model._route(False):
                    logits.append(model.base_model(**inp,decoder_input_ids=decoder,use_cache=False).logits)
            _,ce,kl,path,_=prefix_loss(*logits,labels,mode='diagonal' if objective=='diagonal' else 'prefix')
            values=[float(ce.detach()),float(kl.detach())]
            if objective_values:
                assert values==objective_values, 'Dropout replay differs across objectives'
            objective_values=values
            {'ce':ce,'kl':kl,'prefix':path,'diagonal':path}[objective].backward()
            saved[objective]={k:p.grad.detach().float().cpu().clone() for k,p in selected.items()}
            del logits,ce,kl,path
        for block in keys:
            gs={k:v[block].double() for k,v in saved.items()}
            ns={k:float(v.square().sum()) for k,v in gs.items()}
            records.append(dict(checkpoint=name,row=row,block=block,norms_sq=ns,
                cosine_ce={k:float((gs['ce']*gs[k]).sum())/max(math.sqrt(ns['ce']*ns[k]),1e-30)
                           for k in ('kl','prefix','diagonal')},
                weighted_norm_ratio_to_ce={k:w*math.sqrt(ns[k]/max(ns['ce'],1e-30))
                    for k,w in [('kl',1.25),('prefix',.1),('diagonal',.1)]}))
        del saved,gs
        print('PARAMETER_PROBE_ROW',name,row,flush=True)
    model.zero_grad(set_to_none=True)
    for hook in hooks:
        hook.remove()
    return records


def probe(args):
    import gc
    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import set_seed
    from experiment_runner import engine
    from experiment_runner.integrity import restore_archive, sha256
    from methods.prefix_consistency.v1.losses import path_statistics
    import shutil
    assert ROOT.resolve()==Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT/'conda')
    config_out=ROOT/'outputs/experiments/amis_prefix_consistency_pilot_rtx'
    engine.configure(config_out);engine.validate_environment(gpu=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    check()
    manifest=json.loads(Path(args.manifest).read_text())
    frame=pd.read_csv(engine.DATA/'validation.csv',keep_default_na=False)
    indices=np.linspace(0,len(frame)-1,args.rows,dtype=int).tolist()
    records=[];tokens=[]
    for name,record in manifest.items():
        done=OUT/'probe'/name/'COMPLETE.json'
        if done.exists():
            continue
        temporary=OUT/'probe'/name/'restore';temporary.mkdir(parents=True,exist_ok=True)
        archive=Path(hf_hub_download('FiveC/amis-rewire-checkpoints',record['prefix']+'/best.zip',
            revision=record['revision'],token=False,local_dir=temporary/'download'))
        assert sha256(archive)==record['sha256']
        export=temporary/'export'
        if not export.exists():
            restore_archive(archive,export)
        model,tokenizer,metadata=engine.load_model(record['spec'],export)
        model.gradient_checkpointing_disable();model.cuda().train()
        local_records=[];local_tokens=[];replay_checked=False
        for pair in range(args.pairs):
            for row in indices:
                seed=20261115+pair*10000+row;set_seed(seed)
                inp=tokenizer([frame.iloc[row].source],return_tensors='pt',truncation=True,max_length=256)
                target=tokenizer(text_target=[frame.iloc[row].target],return_tensors='pt',truncation=True,max_length=256)
                inp={k:v.cuda() for k,v in inp.items() if k in ('input_ids','attention_mask')}
                labels=target.input_ids.cuda();decoder=model.prepare_decoder_input_ids_from_labels(labels)
                with torch.no_grad(),model._route(False):
                    state=torch.cuda.get_rng_state();cpu_state=torch.get_rng_state()
                    left=model.base_model(**inp,decoder_input_ids=decoder,use_cache=False).logits
                    right=model.base_model(**inp,decoder_input_ids=decoder,use_cache=False).logits
                    if not replay_checked:
                        after=torch.cuda.get_rng_state();cpu_after=torch.get_rng_state()
                        torch.cuda.set_rng_state(state);torch.set_rng_state(cpu_state)
                        replay=model.base_model(**inp,decoder_input_ids=decoder,use_cache=False).logits
                        torch.testing.assert_close(replay,left,rtol=0,atol=0)
                        assert not torch.equal(left,right), 'Paired views must differ'
                        torch.cuda.set_rng_state(after);torch.set_rng_state(cpu_after)
                        replay_checked=True;del replay
                    # Trim storage immediately: exact gradients need at most token chunks.
                    lp=left.log_softmax(-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1)
                    lq=right.log_softmax(-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1)
                    a=lp-lq; mask=labels.ne(-100); pc,dc=coefficients(a,mask)
                    path=path_statistics(a,mask)
                    # Full-prefix coefficients cannot be reconstructed independently per chunk.
                    dot={}; norms={k:0. for k in ('ce','kl','prefix','diagonal')}
                    for start in range(0,labels.shape[1],8):
                        stop=start+8;y=labels[:,start:stop]
                        # CE/KL token reductions are rescaled back to full sentence denominator.
                        gs,_,pg,qg=gradients(left[:,start:stop],right[:,start:stop],y)
                        ratio=y.numel()/labels.numel()
                        gs['ce']=tuple(g*ratio for g in gs['ce']);gs['kl']=tuple(g*ratio for g in gs['kl'])
                        p=left[:,start:stop].softmax(-1);q=right[:,start:stop].softmax(-1)
                        one=torch.zeros_like(p).scatter_(-1,y.unsqueeze(-1),1.)
                        gs['prefix']=((one-p)*pc[:,start:stop,None],-(one-q)*pc[:,start:stop,None])
                        gs['diagonal']=((one-p)*dc[:,start:stop,None],-(one-q)*dc[:,start:stop,None])
                        for key,parts in gs.items():
                            norms[key]+=sum(float(g.double().square().sum()) for g in parts)
                        for other in ('kl','prefix','diagonal'):
                            dot[other]=dot.get(other,0.)+sum(float((g.double()*h.double()).sum()) for g,h in zip(gs['ce'],gs[other]))
                        for j in range(y.shape[1]):
                            local_tokens.append(dict(checkpoint=name,row=row,pair=pair,position=start+j,
                                target_id=int(y[0,j]),special=int(y[0,j]) in tokenizer.all_special_ids,
                                gold_p=float(pg[0,j]),gold_q=float(qg[0,j]),logp_delta=float(a[0,start+j]),
                                prefix_coefficient=float(pc[0,start+j]),diagonal_coefficient=float(dc[0,start+j]),
                                prefix_grad_sq=sum(float(g[0,j].double().square().sum()) for g in gs['prefix']),
                                diagonal_grad_sq=sum(float(g[0,j].double().square().sum()) for g in gs['diagonal'])))
                        del gs,p,q,one
                    result=dict(checkpoint=name,row=row,pair=pair,seed=seed,tokens=labels.numel(),
                        **{k:float(v[0]) for k,v in path.items()},norms_sq=norms,
                        cosine_ce={k:dot[k]/max(math.sqrt(norms['ce']*norms[k]),1e-30) for k in dot},
                        weighted_norm_ratio_to_ce={k:w*math.sqrt(norms[k]/max(norms['ce'],1e-30))
                            for k,w in [('kl',1.25),('prefix',.1),('diagonal',.1)]})
                    local_records.append(result)
                del left,right
                write(OUT/'probe/PROGRESS.json',dict(checkpoint=name,pair=pair,row=row,completed=len(local_records),total=args.rows*args.pairs,optimizer_steps=0))
        pd.DataFrame(local_tokens).to_csv(done.parent/'tokens.csv',index=False)
        write(done.parent/'records.json',local_records)
        param=parameter_probe(model,tokenizer,frame,indices,name)
        write(done.parent/'parameter_gradients.json',param)
        write(done,dict(checkpoint=record,rows=indices,pairs=args.pairs,optimizer_steps=0,test_used=False,
            gradients='exact wrt independent paired logits; NOT parameter gradients or Adam updates',
            parameter_probe='partial gradients of three named blocks, four fixed rows; same RNG across objectives',
            rng_replay_exact=replay_checked,microbatch=1,training_microbatch=4,
            limitation='post-hoc diagnostic; not training trajectory replay or Adam update reconstruction',metadata=metadata))
        del model,tokenizer;gc.collect();torch.cuda.empty_cache()
        assert temporary.resolve().is_relative_to((OUT/'probe').resolve());shutil.rmtree(temporary)
        print('PROBE_COMPLETE',name,flush=True)
    print('ALL_PROBES_COMPLETE',flush=True)


def summarize():
    import numpy as np
    import pandas as pd
    result={}
    for name in ('W_S42','P_PREFIX_S42','D_DIAGONAL_S42'):
        folder=OUT/'probe'/name
        done=json.loads((folder/'COMPLETE.json').read_text())
        assert done['optimizer_steps']==0 and done['rng_replay_exact'] and not done['test_used']
        rows=json.loads((folder/'records.json').read_text())
        params=json.loads((folder/'parameter_gradients.json').read_text())
        tokens=pd.read_csv(folder/'tokens.csv')
        assert len(rows)==32 and len(params)==12 and tokens.shape[0]==358
        # Independently replay the exact temporal kernel and coefficients in NumPy.
        permutation=[]
        rng=np.random.default_rng(15)
        for row in rows:
            table=tokens[(tokens.row==row['row']) & (tokens.pair==row['pair'])].sort_values('position')
            a=table.logp_delta.to_numpy();n=len(a);k=np.arange(1,n+1)
            diag=np.cumsum((1/k)[::-1])[::-1]/n
            p=float(np.mean(np.cumsum(a)**2/k));d=float(np.sum(diag*a*a))
            np.testing.assert_allclose([p,d],[row['prefix'],row['diagonal']],rtol=2e-5,atol=2e-6)
            coeff=2*np.cumsum((np.cumsum(a)/k)[::-1])[::-1]/n
            np.testing.assert_allclose(coeff,table.prefix_coefficient,rtol=2e-5,atol=2e-6)
            for key in ('prefix','diagonal'):
                np.testing.assert_allclose(table[key+'_grad_sq'].sum(),row['norms_sq'][key],rtol=1e-10,atol=1e-12)
            permutation.append(np.mean([np.mean(np.cumsum(rng.permutation(a))**2/k) for _ in range(100)]))
        summary=dict(prefix_mean=float(np.mean([x['prefix'] for x in rows])),
            diagonal_mean=float(np.mean([x['diagonal'] for x in rows])),
            cross_mean=float(np.mean([x['cross'] for x in rows])),
            shuffled_position_prefix_mean=float(np.mean(permutation)),
            token_count_with_repeated_masks=len(tokens),unique_sentences=len(done['rows']),
            coefficient_opposes_own_logp_difference_fraction=float(np.mean(tokens.prefix_coefficient*tokens.logp_delta<0)),
            low_probability_token_fraction=float(np.mean((tokens.gold_p+tokens.gold_q)/2<.01)))
        for key in ('prefix','diagonal'):
            energy=tokens[key+'_grad_sq'];low=(tokens.gold_p+tokens.gold_q)/2<.01
            summary[key+'_energy_fraction_low_probability']=float(energy[low].sum()/energy.sum())
            summary[key+'_energy_fraction_special_tokens']=float(energy[tokens.special].sum()/energy.sum())
        for scope,records in [('logits',rows),('three_parameter_blocks',params)]:
            summary[scope]={}
            for key in ('kl','prefix','diagonal'):
                cos=np.array([x['cosine_ce'][key] for x in records])
                ratio=np.array([x['weighted_norm_ratio_to_ce'][key] for x in records])
                assert np.isfinite(cos).all() and np.isfinite(ratio).all() and (np.abs(cos)<=1.00001).all()
                summary[scope][key]=dict(mean_cosine_ce=float(cos.mean()),negative_count=int((cos<0).sum()),
                    comparisons=len(cos),median_weighted_norm_ratio_to_ce=float(np.median(ratio)),
                    max_weighted_norm_ratio_to_ce=float(ratio.max()))
        result[name]=summary
    write(OUT/'MECHANISM_SUMMARY.json',dict(runs=result,independent_replay='PASS',optimizer_steps=0,
        caveats=['Post-hoc selected checkpoint states; no early/mid-training reconstruction.',
                 'Logit gradients and gradients of three named blocks are separate spaces.',
                 'Four rows x three blocks are correlated descriptive measurements, not twelve independent seeds.',
                 'Microbatch1 probes do not reproduce training microbatch4/accumulation32 or Adam history.',
                 'Position shuffling changes a diagnostic value; it is not a trained shuffled control.']))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['cpu','check','probe','summarize'])
    parser.add_argument('--manifest');parser.add_argument('--rows',type=int,default=16)
    parser.add_argument('--pairs',type=int,default=2)
    args=parser.parse_args()
    {'cpu':cpu,'check':check,'probe':lambda:probe(args),'summarize':summarize}[args.stage]()
