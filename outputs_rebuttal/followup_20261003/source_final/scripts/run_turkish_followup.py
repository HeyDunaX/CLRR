"""Matched four-method mBART pilot; waits for checkpoint and precision gates."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from huggingface_hub import snapshot_download
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs_rebuttal/followup_20261003'
REVISION='e30b6cb8eb0d43a0b73cab73c7676b9863223a30'


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)


def main():
    while not (OUT/'reevaluation/COMPLETE.json').exists():
        print('WAITING_FOR_REEVALUATION_GATE',flush=True);time.sleep(30)
    smoke=json.loads((OUT/'precision_training_smoke_gpu.json').read_text())
    assert len(smoke)==4 and all(r['passed'] and r['cuda_bf16_training'] for r in smoke)
    probe=json.loads((OUT/'generation_precision_probe.json').read_text())
    assert all(all(dtype=='torch.float32' for dtype in r['explicit_fp32_projection'].values()) for r in probe)
    data=ROOT/'data_processed/opus100_turkish_english_20k_v2'
    manifest=json.loads((data/'manifest.json').read_text())
    for split,record in manifest['splits'].items():
        assert hashlib.sha256((data/(split+'.csv')).read_bytes()).hexdigest()==record['sha256']
    model=Path(snapshot_download(repo_id='facebook/mbart-large-50-many-to-many-mmt',revision=REVISION,
                 allow_patterns=['config.json','generation_config.json','model.safetensors',
                                 'sentencepiece.bpe.model','tokenizer_config.json','special_tokens_map.json'],token=False))
    sys.path.insert(0,str(ROOT/'scripts/revalidation'))
    from reevaluate_checkpoints import predict, write_predictions, read_csv
    from amis_rewire.metrics import generation_metrics
    tokenizer=AutoTokenizer.from_pretrained(model,use_fast=True)
    assert 'tr_TR' in tokenizer.lang_code_to_id and 'en_XX' in tokenizer.lang_code_to_id
    tokenizer.src_lang='tr_TR';tokenizer.tgt_lang='en_XX'
    token_stats={}
    for split in ['train','validation','test']:
        rows=read_csv(data/(split+'.csv'))
        stats={}
        for phase,column in [('source','source'),('target','target')]:
            lengths=[]
            for start in range(0,len(rows),256):
                texts=[r[column] for r in rows[start:start+256]]
                encoded=tokenizer(text_target=texts,truncation=False) if phase=='target' else tokenizer(texts,truncation=False)
                lengths.extend(len(ids) for ids in encoded['input_ids'])
            stats[phase]={'mean_tokens':float(np.mean(lengths)),
                          'p50_p95_p99':np.percentile(lengths,[50,95,99]).tolist(),
                          'rows_truncated_at_256':sum(n>256 for n in lengths)}
        token_stats[split]=stats
    save(OUT/'turkish_tokenization_audit.json',token_stats)
    zero=OUT/'turkish_zero_shot';zero.mkdir(exist_ok=True)
    if not (zero/'metrics.json').exists():
        pretrained=AutoModelForSeq2SeqLM.from_pretrained(model).cuda().eval()
        pretrained.generation_config.forced_bos_token_id=tokenizer.lang_code_to_id['en_XX']
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        rows=read_csv(data/'test.csv')
        predictions=predict(pretrained,tokenizer,rows,256)
        write_predictions(zero/'test_predictions.csv',rows,predictions)
        save(zero/'metrics.json',{'scores':generation_metrics(predictions,[r['target'] for r in rows],tokenize='13a'),
             'rows':len(rows),'model_revision':REVISION,'generation_precision':'FP32/TF32 disabled',
             'scope':'pretrained zero-shot reference; not a tuned method'})
        del pretrained
        import gc
        gc.collect();torch.cuda.empty_cache()
    models=[('baseline','baseline',0.),('lsr','jepa',.1),('clrr','clrr-enc',0.),('full','jepa-clrr-enc',.1)]
    calibration=json.loads((OUT/'batch_calibration.json').read_text())
    microbatch=calibration['selected_microbatch'];accumulation=calibration['gradient_accumulation']
    assert microbatch*accumulation==128
    commands=[]
    for name,method,weight in models:
        run=f'mbart-turkish-{name}-seed42'
        cmd=[sys.executable,'-u','-m','amis_rewire.train','--model','mbart-large-50',
             '--model-name',str(model),'--method',method,'--jepa-weight',str(weight),
             '--data-dir',str(data),'--output-dir',str(OUT/'turkish'),
             '--backup-dir',str(OUT/'turkish_backups'),'--run-name',run,
             '--seed','42','--learning-rate','5e-5','--weight-decay','0',
             '--warmup-ratio','0.06','--num-train-epochs','20','--early-stopping-patience','4',
             '--per-device-train-batch-size',str(microbatch),'--gradient-accumulation-steps',str(accumulation),
             '--per-device-eval-batch-size','8','--max-source-length','256','--max-target-length','256',
             '--rewire-distance','2','--rewire-strength','0.1','--rewire-stack','encoder',
             '--src-lang','tr_TR','--tgt-lang','en_XX','--bleu-tokenizer','13a',
             '--eval-beams','1','--num-beams','4','--generation-precision','fp32',
             '--bf16','--no-fp16','--gradient-checkpointing','--no-auto-resume',
             '--save-total-limit','2','--dataloader-num-workers','2','--logging-steps','10']
        commands.append({'id':name,'run_name':run,'command':cmd})
    source_files=['src/amis_rewire/modeling.py','src/amis_rewire/train.py','src/amis_rewire/metrics.py',
                  'src/amis_rewire/evaluation.py','scripts/run_turkish_followup.py']
    save(OUT/'turkish_protocol.json',{'pretrained_model':'facebook/mbart-large-50-many-to-many-mmt',
          'pretrained_revision':REVISION,'dataset':manifest,'experiments':commands,
          'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files},
          'generation':'explicit FP32, TF32 disabled; val beam1; test beam4',
          'selection':'best validation corpus chrF++; no test selection',
          'quality_primary':'original-reference corpus BLEU13a',
          'primary_comparisons':['Full-Baseline','Full-LSR-only'],
          'scope':'one training seed pilot; no multi-seed generalization claim'})
    for experiment in commands:
        run_dir=OUT/'turkish'/experiment['run_name']
        if (run_dir/'COMPLETE.json').exists():continue
        if run_dir.exists():
            raise RuntimeError('Incomplete prior run; inspect and choose explicit resume rather than overwrite')
        save(OUT/'turkish_status.json',{'running':experiment['id'],'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        print('TRAIN_START',experiment['id'],flush=True)
        subprocess.run(experiment['command'],cwd=ROOT,check=True)
        assert (run_dir/'metrics.json').exists() and (run_dir/'best_model').is_dir()
        archive=OUT/'turkish_backups'/experiment['run_name']/(experiment['run_name']+'-best.zip')
        assert archive.exists() and archive.stat().st_size>0
        save(run_dir/'COMPLETE.json',{'id':experiment['id'],'archive':str(archive),
                                   'archive_bytes':archive.stat().st_size,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        print('TRAIN_COMPLETE',experiment['id'],flush=True)
    save(OUT/'turkish_status.json',{'completed':True,'runs':4})


if __name__=='__main__':main()
