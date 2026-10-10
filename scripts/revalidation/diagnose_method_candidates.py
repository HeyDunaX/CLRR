"""Frozen seed42 validation diagnostics. No optimizer, training, or test access."""
import argparse
import contextlib
import gc
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'outputs_rebuttal/method_candidate_diagnostics'
REPO = 'FiveC/amis-rewire-checkpoints'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare():
    assert not (OUT / 'PLAN.json').exists(), 'Use the frozen existing plan'
    candidate_plan = json.loads((ROOT / 'outputs_rebuttal/amis_method_candidates/PLAN.json').read_text())
    api = HfApi()
    revision = api.model_info(REPO).sha
    old = 'checkpoints/amis_mandarin/mbart-large-50/full/seed-42/full_a010_l010-accumfixed'
    baseline = candidate_plan['baseline_reuse']['remote_archive'].rsplit('/', 1)[0]
    entries = [{'id': 'OLD', 'prefix': old, 'method': 'original-full', 'account': 1},
               {'id': 'BASE', 'prefix': baseline, 'method': 'baseline', 'account': 2}]
    for r in candidate_plan['runs']:
        entries.append({'id': r['id'], 'prefix': r['remote_prefix'], 'method': r['method'],
                        'account': r['account']})
    files = api.list_repo_files(REPO, revision=revision)
    for entry in entries:
        path = entry['prefix'] + '/best.zip'
        info = api.get_paths_info(REPO, [path], revision=revision)[0]
        entry.update(archive_path=path, archive_sha256=info.lfs.sha256, archive_bytes=info.size)
        names = [f for f in files if f.startswith(entry['prefix'] + '/') and f.endswith('validation_predictions.csv')]
        entry['saved_validation_path'] = names[0] if len(names) == 1 else None
    plan = {'repo': REPO, 'revision': revision, 'entries': entries,
            'validation_sha256': sha(ROOT / 'data_processed/amis_mandarin/validation.csv'),
            'model_revision': candidate_plan['runs'][0]['model_revision'],
            'representation_indices': np.random.default_rng(42).permutation(576)[:128].tolist(),
            'gradient_batches': np.random.default_rng(43).permutation(576)[:128].reshape(4,32).tolist(),
            'interventions': {'on': [], 'early_off': [3,4,5], 'middle_off': [6,7,8,9], 'late_off': [10,11,12]},
            'protocol': 'FP32 eval, dropout off, TF32 off, lengths256, val beam1 batch8; gradient microbatch32; zero optimizer steps',
            'limits': 'Post-hoc validation diagnostics of selected frozen checkpoints; no selection-adjusted significance or training-time causal claim.'}
    write(OUT / 'PLAN.json', plan)
    for entry in entries:
        if entry['saved_validation_path']:
            path = hf_hub_download(REPO, entry['saved_validation_path'], revision=revision)
            dest = OUT / entry['id'] / 'saved_validation.csv'
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    print('PLAN_FROZEN', revision, [(e['id'],bool(e['saved_validation_path'])) for e in entries], flush=True)


def bootstrap():
    """Shared paired sentence draws; native SacreBLEU bootstrap p-values."""
    from sacrebleu.significance import PairedTest
    plan = json.loads((OUT / 'PLAN.json').read_text())
    refs_frame = pd.read_csv(ROOT / 'data_processed/amis_mandarin/validation.csv', dtype=str, keep_default_na=False)
    refs = refs_frame.target.tolist()
    predictions = {}
    for entry in plan['entries']:
        path = OUT / entry['id'] / 'validation_on.csv'
        if not path.exists(): path = OUT / entry['id'] / 'saved_validation.csv'
        if not path.exists(): continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert df.source.tolist() == refs_frame.source.tolist() and df.target.tolist() == refs
        predictions[entry['id']] = df.prediction.tolist()
    assert 'OLD' in predictions and all(k in predictions for k in ('U1F','U2F'))
    import os
    os.environ['SACREBLEU_SEED'] = '42'
    metrics = {'BLEU': sacrebleu.metrics.BLEU(tokenize='zh'), 'chrF++': sacrebleu.metrics.CHRF(word_order=2)}
    signatures, native = PairedTest([('OLD', predictions['OLD'])] + [(k,predictions[k]) for k in ('U1F','U2F')],
        metrics=metrics, references=[refs], test_type='bs', n_samples=10000, n_jobs=1)()
    indices = np.random.default_rng(42).integers(0, len(refs), size=(10000,len(refs)))
    records = []
    for metric_name, metric in metrics.items():
        draws, observed = {}, {}
        for ident in ('OLD','U1F','U2F'):
            observed[ident] = metric.corpus_score(predictions[ident],[refs]).score
            stats = np.asarray(metric._extract_corpus_statistics(predictions[ident],[refs]), dtype=np.float64)
            draw = []
            for start in range(0,10000,100):
                sums = stats[indices[start:start+100]].sum(1)
                draw.extend(metric._compute_score_from_stats(s).score for s in sums)
            draws[ident] = np.asarray(draw)
        for position, ident in enumerate(('U1F','U2F'),1):
            delta = draws[ident] - draws['OLD']
            records.append({'candidate':ident,'metric':metric_name,'old':observed['OLD'],
                'candidate_score':observed[ident],'delta':observed[ident]-observed['OLD'],
                'ci95_delta':np.quantile(delta,[.025,.975]).tolist(),
                'native_sacrebleu_p':native['BLEU' if metric_name=='BLEU' else next(k for k in native if k not in ('System','BLEU'))][position].p_value})
    order = sorted(range(len(records)), key=lambda i: records[i]['native_sacrebleu_p'])
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(records)-rank)*records[i]['native_sacrebleu_p']))
        records[i]['holm4_p'] = running
    write(OUT / 'bootstrap.json', {'rows':576,'draws':10000,'rng':42,'family':'2 candidates x 2 metrics',
        'signatures':{k:str(v) for k,v in signatures.items()},'records':records,'limits':plan['limits']})
    print(json.dumps(records),flush=True)


def gpu_model(entry, plan):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from methods.original.clrr import CrossLayerResidualRewire
    from methods.precache.v1.clrr import PrecacheResidualRewire
    archive = Path(hf_hub_download(REPO,entry['archive_path'],revision=plan['revision']))
    assert archive.stat().st_size == entry['archive_bytes'] and sha(archive)==entry['archive_sha256']
    dest = OUT / 'extracted' / entry['id']
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            assert (dest/name).resolve().is_relative_to(dest.resolve())
        z.extractall(dest)
    folders = [p.parent for p in dest.rglob('config.json') if (p.parent/'pytorch_model.bin').exists() or (p.parent/'model.safetensors').exists()]
    assert len(folders)==1, folders
    base, info = AutoModelForSeq2SeqLM.from_pretrained(folders[0],output_loading_info=True,torch_dtype=torch.float32)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')), info
    base.gradient_checkpointing_disable()
    base.config.use_cache=True
    base.config.forced_bos_token_id=250025
    base.generation_config.forced_bos_token_id=250025
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt',revision=plan['model_revision'],src_lang='tl_XX',tgt_lang='zh_CN')
    if entry['method'] in ('clrr-precache','full-precache'):
        model = PrecacheResidualRewire(base, distance=2, strength=.1, stack='encoder')
    elif entry['method'] in ('original-full','full-centered'):
        model = CrossLayerResidualRewire(base, distance=2, strength=.1, stack='encoder')
    else: model = base
    return model.cuda().eval(), tokenizer, dest, archive, info


def run_gpu(ident):
    import torch
    import torch.nn.functional as F
    from methods.common.helpers import _masked_mean_pool, _find_stack_layers
    from methods.centered_lsr.v1.lsr import centered_cosine_loss
    sys.path.insert(0,str(ROOT/'scripts/revalidation'))
    from amis_v4_measurements import gradient_group, spectrum
    plan = json.loads((OUT/'PLAN.json').read_text())
    entry = next(e for e in plan['entries'] if e['id']==ident)
    result_path = OUT/ident/'COMPLETE.json'
    assert not result_path.exists(), 'Do not overwrite diagnostics'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    assert sha(ROOT/'data_processed/amis_mandarin/validation.csv')==plan['validation_sha256']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert 'A100' in torch.cuda.get_device_name(0)
    rows = pd.read_csv(ROOT/'data_processed/amis_mandarin/validation.csv',dtype=str,keep_default_na=False)
    model, tok, dest, archive, load_info = gpu_model(entry,plan)
    route = model if hasattr(model,'_fresh_cache') else None
    fresh = lambda: route._fresh_cache() if route else contextlib.nullcontext()
    disabled = set()
    if route:
        for h in route._handles: h.remove()
        route._handles=[]
        for index, layer in enumerate(_find_stack_layers(model,'encoder')):
            original = route._make_hook('encoder',index)
            def hook(module, inputs, output, i=index, original=original):
                strength=route.strength
                if i+1 in disabled: route.strength=0.
                try: return original(module,inputs,output)
                finally: route.strength=strength
            route._handles.append(layer.register_forward_hook(hook))
    def tokens(indices, column):
        text = rows.iloc[indices][column].tolist()
        batch = tok(text_target=text,padding=True,truncation=True,max_length=256,return_tensors='pt') if column=='target' else tok(text,padding=True,truncation=True,max_length=256,return_tensors='pt')
        return {k:v.cuda() for k,v in batch.items() if k in ('input_ids','attention_mask')}
    scores={}
    for condition, layers in plan['interventions'].items():
        if not route and condition!='on': continue
        disabled.clear(); disabled.update(layers)
        predictions=[]
        for start in range(0,576,8):
            with torch.no_grad():
                gen=model.generate(**tokens(list(range(start,min(start+8,576))),'source'),num_beams=1,max_length=256,forced_bos_token_id=250025)
            predictions.extend(tok.batch_decode(gen,skip_special_tokens=True))
        df=rows.copy(); df['prediction']=predictions
        df.to_csv(OUT/ident/f'validation_{condition}.csv',index=False)
        scores[condition]={'bleu':sacrebleu.metrics.BLEU(tokenize='zh').corpus_score(predictions,[rows.target.tolist()]).score,
            'chrfpp':sacrebleu.metrics.CHRF(word_order=2).corpus_score(predictions,[rows.target.tolist()]).score}
        print(ident,condition,json.dumps(scores[condition]),flush=True)
    disabled.clear()
    # Observe actual pre/post hook difference, independent of donor cache semantics.
    residuals=[]; pre={}; mask_holder={}; handles=[]; pooled={}
    for index, layer in enumerate(_find_stack_layers(model,'encoder')):
        def before(_m,_i,out,i=index): pre[i]=(out[0] if isinstance(out,(tuple,list)) else out).detach()
        def after(_m,_i,out,i=index):
            post=(out[0] if isinstance(out,(tuple,list)) else out).detach().float()
            current=pre.pop(i).float(); mask=mask_holder['mask'].float().unsqueeze(-1)
            rms=lambda v: ((v.square()*mask).sum((1,2))/(mask.sum((1,2))*v.shape[-1])).sqrt()
            ratio=rms(post-current)/rms(current).clamp_min(1e-12)
            residuals.append({'phase':mask_holder['phase'],'layer':i+1,'mean_ratio':float(ratio.mean()),'max_ratio':float(ratio.max())})
            pooled.setdefault((mask_holder['phase'],i+1),[]).append(_masked_mean_pool(post,mask.squeeze(-1)).cpu().numpy())
        handles.extend([layer.register_forward_hook(before,prepend=True),layer.register_forward_hook(after)])
    reps={}
    for phase in ('source','target'):
        values=[]
        idx=plan['representation_indices']
        for start in range(0,len(idx),32):
            batch=tokens(idx[start:start+32],phase)
            mask_holder.update(mask=batch['attention_mask'],phase=phase)
            with torch.no_grad(),fresh():
                h=model.get_encoder()(**batch).last_hidden_state
                values.append(_masked_mean_pool(h,batch['attention_mask']).cpu().numpy())
        reps[phase]=np.concatenate(values)
    for h in handles: h.remove()
    geometry={}
    for phase in ('source','target'):
        s=reps[phase]; geometry[phase]=spectrum(s)
    s=torch.tensor(reps['source'],device='cuda'); t=torch.tensor(reps['target'],device='cuda')
    cos=lambda a,b:float(F.cosine_similarity(a,b,dim=-1).mean())
    geometry['matched_cosine']=cos(s,t)
    geometry['centered_matched_cosine']=cos(s-s.mean(0),t-t.mean(0))
    geometry['shuffled_cosine']=float(np.mean([cos(s,t[torch.tensor(np.random.default_rng(42+i).permutation(len(t)),device='cuda')]) for i in range(100)]))
    geometry['centered_shuffled_cosine']=float(np.mean([cos(s-s.mean(0),(t-t.mean(0))[torch.tensor(np.random.default_rng(42+i).permutation(len(t)),device='cuda')]) for i in range(100)]))
    geometry['layer_spectra']={f'{phase}_layer{layer}':spectrum(np.concatenate(values)) for (phase,layer),values in pooled.items()}
    gradients=[]
    if entry['method'] not in ('baseline','clrr-precache'):
        named=list(model.named_parameters()); params=[p for _,p in named]
        groups=[]
        for name,_ in named:
            if 'encoder.layers.' in name: groups.append('layer'+name.split('encoder.layers.')[1].split('.')[0])
            elif 'encoder' in name or 'shared' in name: groups.append('encoder_other')
            else: groups.append('decoder')
        for batch_id, indices in enumerate(plan['gradient_batches']):
            inp=tokens(indices,'source'); tgt=tokens(indices,'target')
            labels=tgt['input_ids'].clone(); labels[labels==tok.pad_token_id]=-100
            out=model(**inp,labels=labels,output_hidden_states=True)
            source=_masked_mean_pool(out.encoder_last_hidden_state.float(),inp['attention_mask'])
            anchors={}
            for mode in ('routed','raw'):
                strength=route.strength if route else 0.
                if mode=='raw' and route: route.strength=0.
                with torch.no_grad(),fresh():
                    anchors[mode]=_masked_mean_pool(model.get_encoder()(**tgt).last_hidden_state.float(),tgt['attention_mask']).detach()
                if route: route.strength=strength
            ce_grads=torch.autograd.grad(out.loss,params,retain_graph=True,allow_unused=True)
            assert torch.isfinite(out.loss) and all(g is None or torch.isfinite(g).all() for g in ce_grads)
            record={'batch_id':batch_id,'indices':indices,'ce':float(out.loss.detach()),
                'anchor_cosine':cos(anchors['routed'],anchors['raw']),
                'anchor_relative_l2':float(((anchors['routed']-anchors['raw']).norm(dim=-1)/anchors['raw'].norm(dim=-1).clamp_min(1e-12)).mean()),'conditions':{}}
            for mode in ('routed','raw'):
                target=anchors[mode]
                loss=centered_cosine_loss(source,target) if 'centered' in entry['method'] else (1-F.cosine_similarity(source,target,dim=-1)).mean()
                lsr_grads=torch.autograd.grad(.1*loss,params,retain_graph=mode=='routed',allow_unused=True)
                assert torch.isfinite(loss) and all(g is None or torch.isfinite(g).all() for g in lsr_grads)
                record['conditions'][mode]={'lsr':float(loss.detach()),'groups':{g:gradient_group(groups,ce_grads,lsr_grads,g) for g in ['all_parameters']+sorted(set(groups))}}
                assert all(g is None or torch.count_nonzero(g)==0 for group,g in zip(groups,lsr_grads) if group=='decoder')
                del lsr_grads,loss
            gradients.append(record)
            print(ident,'gradient',batch_id,flush=True)
            del out,source,ce_grads,anchors
            gc.collect(); torch.cuda.empty_cache()
    write(result_path,{'id':ident,'entry':entry,'scores':scores,'residuals':residuals,'geometry':geometry,
        'gradients':gradients,'strict_loading':load_info,'validation_sha256':plan['validation_sha256'],
        'runtime':{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'worker_sha256':sha(__file__)},
        'optimizer_steps':0,'protocol':plan['protocol'],'limits':plan['limits'],'completed_unix':time.time()})
    del model; gc.collect(); torch.cuda.empty_cache()
    assert dest.resolve().is_relative_to((OUT/'extracted').resolve())
    shutil.rmtree(dest)
    print('DIAGNOSTIC_COMPLETE',ident,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=('prepare','bootstrap','gpu'),required=True)
    parser.add_argument('--id'); args=parser.parse_args()
    if args.stage=='prepare': prepare()
    elif args.stage=='bootstrap': bootstrap()
    else: run_gpu(args.id)
