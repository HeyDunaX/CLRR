"""Pinned validation checkpoints: teacher-forced token decisions and matched beam sensitivity."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu.metrics import BLEU, CHRF
from experiment_runner import engine
from experiment_runner.integrity import sha256, restore_archive
from methods.original.clrr import CrossLayerResidualRewire


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, value):
    p = Path(p); tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8'); tmp.replace(p)


def token_decisions(logits, labels, eos_id):
    """Raw teacher-forced logits. Strict rank=1+#logits greater than gold (ties explicit)."""
    logits = logits.float()
    safe = labels.masked_fill(labels.eq(-100), 0)
    gold = logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    top_values, top_ids = logits.topk(2, dim=-1)
    best_other = torch.where(top_ids[...,0].eq(safe), top_values[...,1], top_values[...,0])
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                          ignore_index=-100, reduction='none').reshape_as(labels)
    return dict(nll=nll, correct=logits.argmax(-1).eq(safe),
        rank=1+logits.gt(gold.unsqueeze(-1)).sum(-1), margin=gold-best_other,
        eos_top1=logits.argmax(-1).eq(eos_id),
        eos_probability=(logits[...,eos_id]-gold-nll).exp())


def category(token, frequency, specials, eos):
    if token == eos: return 'eos'
    if token in specials: return 'other_special'
    count = frequency[token]
    return 'content_unseen' if count == 0 else ('content_rare_1_5' if count <= 5 else
           ('content_mid_6_50' if count <= 50 else 'content_common_gt50'))


def summarize_tokens(frame):
    result = {}
    groups = {key: value for key,value in frame.groupby('category')}
    groups['content_all'] = frame[frame.category.str.startswith('content_')]
    for name, part in groups.items():
        if not len(part): continue
        result[name] = dict(tokens=len(part), top1=float(part.correct.mean()),
            strict_rank_top5=float(part['rank'].le(5).mean()), strict_rank_top10=float(part['rank'].le(10).mean()),
            mean_nll=float(part.nll.mean()), mean_margin=float(part.margin.mean()),
            median_strict_rank=float(part['rank'].median()),
            premature_eos_top1=int(part.eos_top1.sum()) if name.startswith('content_') else None,
            mean_eos_probability=float(part.eos_probability.mean()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', required=True)
    p.add_argument('--grounding', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--token-batch', type=int, default=4)
    p.add_argument('--generation-batch', type=int, default=2)
    p.add_argument('--beams', type=int, default=4)
    p.add_argument('--greedy-gate-rows', type=int, default=8)
    p.add_argument('--minimum-free-gpu-mib', type=int, default=16000)
    args = p.parse_args()
    assert ROOT == Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT/'conda')
    assert min(args.token_batch,args.generation_batch,args.greedy_gate_rows)>0 and args.beams>1
    ws=ROOT/args.workspace;plan=read(ws/'PLAN.json');lock=read(ws/'selection_lock.json')
    runtime={k:m.__version__.split('+')[0] for k,m in [('torch',torch),('transformers',engine.transformers),
             ('accelerate',engine.accelerate),('peft',engine.peft),('sacrebleu',engine.sacrebleu)]}
    assert all(runtime[k]==plan['runtime'][k] for k in runtime)
    original_manifest=read(ws/'bundle_manifest.json')
    for name in ['src/methods/original/clrr.py','src/methods/common/helpers.py']:
        assert sha256(ROOT/name)==original_manifest[name]
    data=ROOT/'data_processed'/plan['dataset']
    for split in ['train','validation']:assert sha256(data/(split+'.csv'))==plan['data'][split]['sha256']
    raw=pd.read_csv(data/'validation.csv',dtype=str,keep_default_na=False)
    train=pd.read_csv(data/'train.csv',dtype=str,keep_default_na=False)
    refs=raw.target.str.strip().tolist()
    out=(ROOT/args.output).resolve();assert out.is_relative_to(ROOT/'outputs/experiments')
    out.mkdir(parents=True,exist_ok=True)
    metadata={}
    for ident in args.ids:
        cohort='amis_token_rtx' if ident in ['R1','R2'] else 'amis_residual_effect_rtx'
        folder=ROOT/'outputs/experiments'/cohort/'runs'/ident
        val=read(folder/'VAL_COMPLETE.json')
        assert val['status']=='VALIDATION_COMPLETE_VERIFIED' and val['archive_sha256']==lock['runs'][ident]['archive_sha256']
        metadata[ident]=dict(cohort=cohort,predictions_sha256=sha256(folder/'validation_predictions.csv'))
    source_files=['scripts/analysis/diagnose_token_decoding.py','src/methods/original/clrr.py',
                  'src/methods/common/helpers.py','src/experiment_runner/engine.py','src/experiment_runner/integrity.py']
    frozen=dict(dataset=plan['dataset'],split='validation',rows=len(raw),data={s:plan['data'][s] for s in ['train','validation']},
        selections={i:lock['runs'][i] for i in args.ids},runtime=runtime,beam1_inputs=metadata,
        token_batch=args.token_batch,generation_batch=args.generation_batch,beams=args.beams,
        greedy_gate_rows=args.greedy_gate_rows,token_frequency_bins=['0','1..5','6..50','>50'],
        frequency='train target tokenizer IDs with same truncation; exclude all special IDs',
        gold_rank='strict: 1 + number of vocabulary logits > gold logit; ties do not imply top1 correct',
        margin='gold raw logit minus largest other logit; specials compete, masks define reporting groups',
        precision='FP32 TF32/dropout/checkpointing OFF',optimizer_steps=0,test_evaluated=False,
        no_checkpoint_reselection=True,source_sha256={f:sha256(ROOT/f) for f in source_files})
    if (out/'PLAN.json').exists():assert read(out/'PLAN.json')==frozen,'Frozen source/plan changed'
    else:write(out/'PLAN.json',frozen)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    metrics={'bleu':BLEU(tokenize=plan['protocol']['bleu_tokenizer']),'chrf++':CHRF(word_order=2),'chrf0':CHRF(word_order=0)}
    for ident in args.ids:
        folder=out/ident;folder.mkdir(exist_ok=True)
        if (folder/'RESULT.json').exists():
            for name,digest in read(folder/'RESULT.json')['file_sha256'].items():assert sha256(folder/name)==digest
            print('VERIFIED_SKIP',ident,flush=True);continue
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        assert free>=args.minimum_free_gpu_mib,('Insufficient free VRAM',free)
        print('RESTORING',ident,'FREE_MIB',free,flush=True)
        spec=next(s for s in plan['runs'] if s['id']==ident);assert spec['method'] in ['baseline','clrr','full','original-full']
        selected=lock['runs'][ident]
        archive=Path(hf_hub_download(plan['repo'],selected['archive_path'],revision=selected['revision'],token=False,local_dir=folder/'restore'))
        assert sha256(archive)==selected['archive_sha256']
        export=folder/'model';restore_archive(archive,export)
        core,loading=AutoModelForSeq2SeqLM.from_pretrained(export,output_loading_info=True)
        assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        tok=AutoTokenizer.from_pretrained(export,use_fast=True);engine.configure_languages(core,tok,spec)
        frequency=Counter();specials=set(tok.all_special_ids)
        for start in range(0,len(train),256):
            values=tok(text_target=train.target.iloc[start:start+256].tolist(),truncation=True,max_length=spec['max_target_length'])['input_ids']
            frequency.update(t for row in values for t in row if t not in specials)
        sb=tok([spec['source_prefix']+s for s in raw.source],padding=True,pad_to_multiple_of=8,truncation=True,
               max_length=spec['max_source_length'],return_tensors='pt')
        tb=tok(text_target=raw.target.tolist(),padding=True,pad_to_multiple_of=8,truncation=True,
               max_length=spec['max_target_length'],return_tensors='pt')
        labels=tb['input_ids'].masked_fill(tb['input_ids']==tok.pad_token_id,-100)
        token_hash=hashlib.sha256(b''.join(t.numpy().tobytes() for t in [sb['input_ids'],sb['attention_mask'],labels])).hexdigest()
        tokenization=dict(sha256=token_hash,frequency_sha256=hashlib.sha256(json.dumps(sorted(frequency.items())).encode()).hexdigest())
        if (out/'TOKENIZATION.json').exists():assert read(out/'TOKENIZATION.json')==tokenization
        else:write(out/'TOKENIZATION.json',tokenization)
        core.gradient_checkpointing_disable();core.float().cuda().eval();core.config.use_cache=False
        model=core if spec['method']=='baseline' else CrossLayerResidualRewire(core,distance=2,strength=.1,stack='encoder')
        model.eval();versions={n:v._version for n,v in core.named_parameters()}
        old_root=ROOT/'outputs/experiments'/metadata[ident]['cohort']/'runs'/ident
        old=pd.read_csv(old_root/'validation_predictions.csv',dtype=str,keep_default_na=False)
        assert old[['source','target']].equals(raw[['source','target']])
        old_predictions=old.prediction.str.strip().tolist()
        scores1={key:metric.corpus_score(old_predictions,[refs]).score for key,metric in metrics.items()}
        saved=read(old_root/'validation_metrics.json')['scores']
        assert all(abs(scores1[k]-saved[k])<1e-8 for k in scores1)
        generate_kwargs=dict(max_length=spec['max_target_length'],do_sample=False)
        write(folder/'GENERATION_CONFIG.json',dict(loaded=core.generation_config.to_dict(),overrides=generate_kwargs,
            beams=[1,args.beams],model_config_use_cache=core.config.use_cache))
        records=[];sentence=[];predictions=[];beam_lengths=[];greedy=[]
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for start in range(0,len(raw),args.token_batch):
                stop=min(start+args.token_batch,len(raw));target=labels[start:stop].cuda()
                batch={k:sb[k][start:stop].cuda() for k in ['input_ids','attention_mask']}
                outputs=model(**batch,labels=target,use_cache=False)
                values=token_decisions(outputs.logits,target,tok.eos_token_id)
                count=target.ne(-100).sum(1);nll=values['nll'].sum(1)
                assert abs(float(nll.sum()/count.sum())-float(outputs.loss))<3e-6
                cpu={k:v.cpu().numpy() for k,v in values.items()}
                for offset in range(stop-start):
                    sentence.append(dict(row=start+offset,ce_all_labels=float(nll[offset]/count[offset])))
                    for pos,t in enumerate(labels[start+offset].tolist()):
                        if t==-100:continue
                        records.append(dict(row=start+offset,position=pos,category=category(t,frequency,specials,tok.eos_token_id),
                            train_frequency=-1 if t in specials else frequency[t],
                            **{key:bool(v[offset,pos]) if key in ['correct','eos_top1'] else float(v[offset,pos]) for key,v in cpu.items()}))
                del outputs,values,cpu,nll,target,batch
                if start%100==0:print('TOKEN_ROWS',ident,stop,'/',len(raw),flush=True)
            for start in range(0,min(args.greedy_gate_rows,len(raw)),args.generation_batch):
                stop=min(start+args.generation_batch,args.greedy_gate_rows,len(raw))
                batch={k:sb[k][start:stop].cuda() for k in ['input_ids','attention_mask']}
                generated=model.generate(**batch,num_beams=1,**generate_kwargs)
                greedy.extend(s.strip() for s in tok.batch_decode(generated,skip_special_tokens=True))
            assert greedy==old_predictions[:len(greedy)],'Greedy checkpoint/protocol replay differs; stop beam comparison'
            print('GREEDY_GATE_PASS',ident,len(greedy),flush=True)
            for start in range(0,len(raw),args.generation_batch):
                stop=min(start+args.generation_batch,len(raw))
                batch={k:sb[k][start:stop].cuda() for k in ['input_ids','attention_mask']}
                generated=model.generate(**batch,num_beams=args.beams,**generate_kwargs)
                predictions.extend(s.strip() for s in tok.batch_decode(generated,skip_special_tokens=True))
                beam_lengths.extend(int(row.ne(tok.pad_token_id).sum()) for row in generated)
                if start%100==0:print('BEAM_ROWS',ident,stop,'/',len(raw),flush=True)
        frame=pd.DataFrame(records);per_sentence=pd.DataFrame(sentence)
        grounding=pd.read_csv(ROOT/args.grounding/ident/'per_sentence.csv').sort_values('row')
        branch='off' if spec['method']=='baseline' else 'on'
        ce_error=float(np.max(np.abs(grounding[branch+'_match'].values-per_sentence.ce_all_labels.values)))
        assert ce_error<1e-4,('Previous CE replay failed',ce_error)
        assert versions=={n:v._version for n,v in core.named_parameters()} and all(v.grad is None for v in core.parameters())
        frame.to_csv(folder/'per_token.csv',index=False);per_sentence.to_csv(folder/'per_sentence.csv',index=False)
        gen=raw.copy();gen['prediction']=predictions;gen['generated_nonpad_tokens']=beam_lengths
        gen.to_csv(folder/'validation_beam_predictions.csv',index=False)
        scores4={key:metric.corpus_score(predictions,[refs]).score for key,metric in metrics.items()}
        result=dict(id=ident,method=spec['method'],selection=selected,strict_loading=loading,
            token_summary=summarize_tokens(frame),beam1_scores=scores1,beam_scores=scores4,
            beam_gain={k:scores4[k]-scores1[k] for k in scores1},greedy_gate_rows=len(greedy),
            greedy_gate_exact=True,CE_replay_max_error=ce_error,optimizer_steps=0,gradients_none=True,
            parameter_versions_unchanged=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            generated_length_cap_rows=sum(n>=spec['max_target_length'] for n in beam_lengths),
            file_sha256={n:sha256(folder/n) for n in ['per_token.csv','per_sentence.csv','validation_beam_predictions.csv','GENERATION_CONFIG.json']})
        write(folder/'RESULT.json',result)
        if isinstance(model,CrossLayerResidualRewire):
            for handle in model._handles:handle.remove()
        del core,model,frame,labels,sb,tb,generated,batch
        gc.collect();torch.cuda.empty_cache()
        for path in [export,folder/'restore']:
            assert path.resolve().is_relative_to(out);shutil.rmtree(path)
        print('MEASURED_CLEANED',ident,flush=True)
    write(out/'COMPLETE.json',dict(status='COMPLETE',ids=args.ids,rows=len(raw),optimizer_steps=0,
                                  test_evaluated=False,temporary_weights_removed=True))


if __name__=='__main__':main()
