"""Train-only frozen-baseline contrast diagnostics. No optimizer or generation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def margins(a, b, c, d):
    """Scores are mean log probabilities: a=sii,b=sij,c=sji,d=sjj."""
    return dict(source_margin_left=a-c, source_margin_right=d-b,
                reference_margin_left=a-b, reference_margin_right=d-c,
                symmetric_source_margin=(a+d-b-c)/2)


def token_scores(logits, labels, special_ids):
    import torch.nn.functional as F
    nll=F.cross_entropy(logits.transpose(1,2),labels,reduction='none',ignore_index=-100)
    mask=labels.ne(-100);content=mask.clone()
    for token in special_ids:content &= labels.ne(token)
    assert content.any(dim=1).all(), 'Empty content target'
    return (-(nll*content).sum(1)/content.sum(1),
            -(nll*mask).sum(1)/mask.sum(1),content.sum(1))


def prepare(out):
    from methods.difference_preserving.v1.pairs import discover_pairs, TOKENIZATION_POLICY
    inventory, metadata = read(out/'INVENTORY.json'), read(out/'BASELINE_METADATA.json')
    for dataset, record in metadata['metadata'].items():
        folder = out/dataset
        assert not folder.exists(), 'Preserve prior diagnostic design'
        train = ROOT/'data_processed'/dataset/'train.csv'
        meta, complete = record['run_metadata'], record['complete']
        assert meta['run']['method'] == 'baseline' and meta['run']['seed'] == 42
        assert meta['archive_sha256'] == complete['archive_sha256']
        assert digest(train) == meta['data']['train']['sha256']
        assert complete['backup']['verified']
        with train.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        near = discover_pairs(rows)
        neighbors = {}
        for pair in near:
            i,j = pair['left'],pair['right']
            neighbors.setdefault(i,set()).add(j); neighbors.setdefault(j,set()).add(i)
        selected = {tuple(sorted((i,min(js)))) for i,js in neighbors.items()}
        pairs = [dict(kind='lexical_near',left=p['left'],right=p['right'],
                      selected_by_min_peer=(p['left'],p['right']) in selected) for p in near]
        for anchor in sorted(neighbors):
            candidates = [j for j,r in enumerate(rows) if j != anchor and j not in neighbors[anchor]
                          and all(r[k].strip() and r[k].strip()!=rows[anchor][k].strip()
                                  for k in ('source','target'))]
            assert candidates
            peer = random.Random(10042+anchor).choice(candidates)
            pairs.append(dict(kind='generic',left=anchor,right=peer,selected_by_min_peer=False))
        folder.mkdir()
        write(folder/'PAIRS.json',pairs)
        design = dict(dataset=dataset,split='train',repo=complete['backup']['repo'],
            revision=inventory['revision'],archive_path=record['prefix']+'/best.zip',
            archive_sha256=complete['archive_sha256'],spec=meta['run'],
            train_sha256=digest(train),pairs_sha256=digest(folder/'PAIRS.json'),
            tokenization_policy=TOKENIZATION_POLICY,baseline_metadata=record,
            selection=meta['protocol']['selection'],seed=42,batch_size=4,
            precision='fp32',attention_implementation='sdpa',tf32=False,
            purpose='Exploratory training-fit/source sensitivity; NOT translation accuracy, semantic labels or pilot superiority',
            scientific_pairs_verified=0,test_or_validation_read=False,
            limitations=['Train memorization; one seed; lexical candidates unverified.',
                'Historical chrF++ selector is not matched BLEU-selection pilot baseline.',
                'Near/generic controls share anchors, not lengths, difficulty or semantic contrast.',
                'Repeated rows/graph edges are dependent; no iid p-values reported.',
                'Teacher-forced scores are not generated translations.'],
            score='Mean target-content token log probability excluding ALL tokenizer special IDs; auxiliary report includes all target labels.',
            invalid='Exclude pairs identical after tokenization/truncation on either side or with empty content; report counts.',
            formulas=dict(
                source_left='s(xi,yi)-s(xj,yi)',source_right='s(xj,yj)-s(xi,yj)',
                reference_left='s(xi,yi)-s(xi,yj)',reference_right='s(xj,yj)-s(xj,yi)',
                symmetric='(sii+sjj-sij-sji)/2'),
            provenance=dict(script_sha256=digest(__file__),
                pairs_source_sha256=digest(ROOT/'src/methods/difference_preserving/v1/pairs.py')))
        write(folder/'DESIGN.json',design)
        print('PREPARED',dataset,len(near),len(neighbors),flush=True)


def run(folder, minimum_free):
    import gc
    import subprocess
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    assert ROOT == Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT/'conda')
    assert not (folder/'RESULT.json').exists(), 'Already completed'
    design=read(folder/'DESIGN.json');spec=design['spec']
    assert digest(__file__)==design['provenance']['script_sha256']
    assert digest(folder/'PAIRS.json')==design['pairs_sha256']
    train=ROOT/'data_processed'/design['dataset']/'train.csv'
    assert digest(train)==design['train_sha256']
    assert torch.__version__.split('+')[0]=='2.6.0' and transformers.__version__=='4.57.6'
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=minimum_free, f'Insufficient free VRAM: {free}'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.time()
    download=folder/'restore';export=folder/'model'
    archive=Path(hf_hub_download(design['repo'],design['archive_path'],revision=design['revision'],
                                token=False,local_dir=download))
    assert digest(archive)==design['archive_sha256']
    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            assert (export/name).resolve().is_relative_to(export.resolve())
        zipped.extractall(export)
    model,info=AutoModelForSeq2SeqLM.from_pretrained(export,attn_implementation='sdpa',output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),info
    assert model.config._attn_implementation=='sdpa'
    model.float().eval().to('cuda');model.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(export,use_fast=True)
    tok.src_lang=spec['src_lang'];tok.tgt_lang=spec['tgt_lang']
    model.config.forced_bos_token_id=tok.convert_tokens_to_ids(spec['tgt_lang'])
    with train.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    pairs=read(folder/'PAIRS.json');ids=sorted({p[k] for p in pairs for k in ('left','right')})
    sources={};targets={};special=set(tok.all_special_ids)
    for start in range(0,len(ids),128):
        group=ids[start:start+128]
        x=tok([spec['source_prefix']+rows[i]['source'] for i in group],truncation=True,max_length=spec['max_source_length'])
        y=tok(text_target=[rows[i]['target'] for i in group],truncation=True,max_length=spec['max_target_length'])
        sources.update(zip(group,x['input_ids']));targets.update(zip(group,y['input_ids']))
    valid=[];excluded=[]
    for pair in pairs:
        i,j=pair['left'],pair['right']
        reason=('identical_source_tokens' if sources[i]==sources[j] else
                'identical_target_tokens' if targets[i]==targets[j] else
                'empty_target_content' if any(not any(t not in special for t in targets[k]) for k in (i,j)) else None)
        (excluded if reason else valid).append(dict(pair,exclusion=reason) if reason else pair)
    cells=sorted({(x,y) for p in valid for x in (p['left'],p['right']) for y in (p['left'],p['right'])})
    scores={};torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0,len(cells),design['batch_size']):
            group=cells[start:start+design['batch_size']]
            inputs=tok.pad([dict(input_ids=sources[x],attention_mask=[1]*len(sources[x])) for x,y in group],return_tensors='pt').to('cuda')
            labels=torch.full((len(group),max(len(targets[y]) for x,y in group)),-100,dtype=torch.long,device='cuda')
            for k,(x,y) in enumerate(group):labels[k,:len(targets[y])]=torch.tensor(targets[y],device='cuda')
            logits=model(**inputs,labels=labels,use_cache=False).logits
            content_score,all_score,counts=token_scores(logits,labels,special)
            for k,key in enumerate(group):
                scores[key]=dict(score=float(content_score[k]),all_score=float(all_score[k]),content_tokens=int(counts[k]))
            del logits,labels,inputs
            if start % 200==0:
                write(folder/'PROGRESS.json',dict(stage='FORWARD_ONLY',done=min(start+len(group),len(cells)),total=len(cells),elapsed_seconds=time.time()-started))
    records=[]
    for pair in valid:
        i,j=pair['left'],pair['right'];keys=((i,i),(i,j),(j,i),(j,j))
        values=[scores[k]['score'] for k in keys]
        records.append(dict(pair,**margins(*values),**{n:v for n,v in zip(('sii','sij','sji','sjj'),values)},
            all_supervised_symmetric_margin=margins(*[scores[k]['all_score'] for k in keys])['symmetric_source_margin']))
    with (folder/'PAIR_SCORES.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    groups={}
    for kind in ('lexical_near','selected_near','generic'):
        if kind=='selected_near':
            selected=[r for r in records if r['kind']=='lexical_near' and r['selected_by_min_peer']]
        else:
            selected=[r for r in records if r['kind']==kind]
        assert selected
        metric_names=list(margins(0.,0.,0.,0.))
        groups[kind]=dict(pairs=len(selected),mean={k:sum(r[k] for r in selected)/len(selected) for k in metric_names},
            symmetric_margin_nonpositive=sum(r['symmetric_source_margin']<=0 for r in selected),
            either_source_margin_nonpositive=sum(min(r['source_margin_left'],r['source_margin_right'])<=0 for r in selected),
            both_source_margins_positive_fraction=sum(min(r['source_margin_left'],r['source_margin_right'])>0 for r in selected)/len(selected))
    result=dict(status='COMPLETE_FORWARD_ONLY',dataset=design['dataset'],design_sha256=digest(folder/'DESIGN.json'),
        pairs_sha256=design['pairs_sha256'],archive_sha256=design['archive_sha256'],groups=groups,
        excluded_pairs=excluded,unique_scored_cells=len(cells),elapsed_seconds=time.time()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,gpu=torch.cuda.get_device_name(),
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__),
        semantic_pairs_verified=0,train_only=True,training_performed=False)
    # Unlabelled review aid: include counterexamples and positive controls.
    near=[r for r in records if r['kind']=='lexical_near']
    examples=sorted(near,key=lambda r:r['symmetric_source_margin'])[:20]+sorted(near,key=lambda r:r['symmetric_source_margin'],reverse=True)[:10]
    write(folder/'REVIEW_EXAMPLES.json',[dict(r,source_left=rows[r['left']]['source'],source_right=rows[r['right']]['source'],target_left=rows[r['left']]['target'],target_right=rows[r['right']]['target'],semantic_status='UNVERIFIED') for r in examples])
    del model;gc.collect();torch.cuda.empty_cache()
    assert digest(train)==design['train_sha256']
    for path in (export,download):
        assert path.resolve().is_relative_to(folder.resolve());shutil.rmtree(path)
    result['transient_weights_removed']=not export.exists() and not download.exists()
    write(folder/'RESULT.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','run'),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--minimum-free-gpu-mib',type=int,default=20000)
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    assert out.is_relative_to(ROOT/'outputs/experiments')
    if args.stage=='prepare':prepare(out)
    else:run(out,args.minimum_free_gpu_mib)
