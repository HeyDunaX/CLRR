"""G1 D3: paired routed/raw detached targets on fixed Full best weights; no training."""
import csv
import gc
import json
import shutil
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from amis_rewire.modeling import CrossLayerResidualRewire, _masked_mean_pool
from amis_v4_measurements import digest, load_checkpoint, strict_model, gradient_group, runtime

OUT = ROOT / 'outputs_rebuttal/amis_g0_g1_20261005/target_anchor'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
inventory = json.loads((ROOT / 'g1_inventory.json').read_text())
batches = json.loads((ROOT / 'g1_batches.json').read_text())['fixed_validation_indices']
assert len(batches)==8 and all(len(b)==4 for b in batches)
with (ROOT / 'data_processed/amis_mandarin/validation.csv').open(encoding='utf-8-sig',newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows)==576
assert digest(ROOT / 'data_processed/amis_mandarin/validation.csv')=='eac078ab237b55c10614f4db5ff026dc3a96a09a69c648ced29b5cdd8d4d11b4'
tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt',
    revision='e30b6cb8eb0d43a0b73cab73c7676b9863223a30',src_lang='tl_XX',tgt_lang='zh_CN')
assert tokenizer('source')['input_ids'][0]==250047
assert tokenizer(text_target='target')['input_ids'][0]==250025
(OUT / 'runtime.json').write_text(json.dumps(runtime(tokenizer,ROOT),indent=2))
for entry in inventory:
    assert entry['family']=='original_full_detached' and entry['alpha']==.1 and entry['lambda']==.1
    name = entry['id']
    dest = ROOT / 'scratch/g1_anchor_extract' / name
    path = load_checkpoint(entry,dest,OUT / 'loading_receipts')
    raw = strict_model(path).eval()
    route = CrossLayerResidualRewire(raw,distance=2,strength=.1,stack='encoder').eval()
    params = list(raw.parameters())
    groups = [('encoder' if 'encoder' in n or 'shared' in n else 'decoder') for n,_ in raw.named_parameters()]
    records = []
    for batch_id, indices in enumerate(batches):
        batch = [rows[i] for i in indices]
        inputs = {k:v.cuda() for k,v in tokenizer([r['source'] for r in batch],padding=True,
            truncation=True,max_length=256,return_tensors='pt').items() if k in ('input_ids','attention_mask')}
        labels = tokenizer(text_target=[r['target'] for r in batch],padding=True,
            truncation=True,max_length=256,return_tensors='pt')['input_ids'].cuda()
        labels[labels==tokenizer.pad_token_id]=-100
        outputs = route(**inputs,labels=labels,output_hidden_states=True,return_dict=True)
        ce = outputs.loss
        source = F.normalize(_masked_mean_pool(outputs.encoder_last_hidden_state,inputs['attention_mask']),dim=-1)
        target_ids = labels.clone()
        target_ids[target_ids==-100]=tokenizer.pad_token_id
        mask = target_ids.ne(tokenizer.pad_token_id).long()
        anchors = {}
        for mode,alpha in [('routed',.1),('raw',0.)]:
            route.strength=alpha
            with torch.no_grad(),route._fresh_cache():
                h = raw.get_encoder()(input_ids=target_ids,attention_mask=mask,return_dict=True).last_hidden_state
            anchors[mode]=F.normalize(_masked_mean_pool(h,mask),dim=-1).detach()
            assert not anchors[mode].requires_grad and not route._cache['encoder']
        route.strength=.1
        ce_grads=torch.autograd.grad(ce,params,retain_graph=True,allow_unused=True)
        record={'batch_id':batch_id,'indices':indices,'ce':float(ce.detach()),
                'label_tokens':int(labels.ne(-100).sum()),
                'routed_raw_anchor_cosine':float((anchors['raw']*anchors['routed']).sum(-1).mean()),
                'anchor_l2_mean':float((anchors['raw']-anchors['routed']).norm(dim=-1).mean()),'conditions':{}}
        previous_grads=None
        for mode in ('routed','raw'):
            lsr=(1-(source*anchors[mode]).sum(-1)).mean()
            grads=torch.autograd.grad(.1*lsr,params,retain_graph=mode=='routed',allow_unused=True)
            record['conditions'][mode]={'lsr':float(lsr.detach()),'weighted_lsr':float(.1*lsr.detach()),
                'gradient_groups':{g:gradient_group(groups,ce_grads,grads,g) for g in ('all_parameters','encoder','decoder')}}
            assert all(g is None or torch.count_nonzero(g)==0 for group,g in zip(groups,grads) if group=='decoder')
            if mode=='routed':
                previous_grads=grads
            else:
                stats=gradient_group(groups,previous_grads,grads,'all_parameters')
                record['routed_raw_lsr_gradient_cosine']=stats['gradient_cosine']
        records.append(record)
        print(json.dumps({'model':name,'batch':batch_id,'anchor_cos':record['routed_raw_anchor_cosine']}),flush=True)
        del outputs,source,ce,ce_grads,grads,previous_grads,lsr,h
        gc.collect()
    result={'status':'VERIFIED_COUNTERFACTUAL_ONLY','model':entry,'batches':records,
            'protocol':'FP32 eval dropout off, TF32 off, checkpointing disabled, source route fixed .10, same source graph/CE',
            'optimizer_steps':0,'target_detached':True,'cache_isolation':'PASS',
            'limits':'Not training gradients with dropout; not evidence of translation gain or corrected-training mechanism.'}
    (OUT / f'{name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for handle in route._handles:
        handle.remove()
    del raw,route,params
    gc.collect()
    torch.cuda.empty_cache()
    assert dest.resolve().is_relative_to((ROOT/'scratch/g1_anchor_extract').resolve())
    shutil.rmtree(dest)
(OUT / 'COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','models':len(inventory),'optimizer_steps':0}))
