"""Read-only provenance audit of source and historical run arguments."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
keys=['method','seed','num_train_epochs','early_stopping_patience','learning_rate','weight_decay','warmup_ratio',
      'per_device_train_batch_size','per_device_eval_batch_size','gradient_accumulation_steps','max_source_length',
      'max_target_length','num_beams','eval_beams','rewire_distance','rewire_strength','rewire_stack','jepa_weight',
      'fp16','bf16','gradient_checkpointing','generation_precision','routing_warmup_ratio','src_lang','tgt_lang',
      'bleu_tokenizer','model_revision','auto_resume','resume_from_checkpoint','validation_only']
records=[]
paths=[]
for folder in ['amis_confirmation_20261003','amis_configuration_20261003_v1','amis_configuration_confirmation_20261004']:
    paths.extend((ROOT/'outputs_rebuttal'/folder/'runs').glob('*/metrics.json'))
paths.extend((ROOT/'outputs_rebuttal/amis_routing_warmup_20261004_v2/collected_records/runs').glob('*/metrics.json'))
for p in sorted(paths):
    d=load(p);c=d.get('configuration',{})
    if not any(x in p.parent.name for x in ['baseline','clrr','lsr','full']):continue
    records.append({'run':p.parent.name,'metrics_file':str(p.relative_to(ROOT)),'metrics_sha256':sha(p),
      'model':d.get('model'),'configuration':{k:c.get(k) for k in keys},
      'missing_configuration_keys':[k for k in keys if k not in c],
      'selected_checkpoint':d.get('best_model_checkpoint'),'completed_epochs':d.get('completed_epochs'),
      'interpretation':'Historical arguments only; no claim runtime/backward normalization verified from metadata.'})
files=['src/amis_rewire/train.py','src/amis_rewire/evaluation.py','src/amis_rewire/scientific_records.py',
       'scripts/revalidation/run_amis_warmup_worker.py']
receipts={p:{'sha256':sha(ROOT/p),'ast_parse':'PASS'} for p in files}
for p in files:ast.parse((ROOT/p).read_text(encoding='utf-8-sig'))
summary={'date':'2026-10-05','source_receipts':receipts,'historical_run_configuration_count':len(records),
 'scope':'Static code and artifact audit only; no GPU, no training, no source mutation.',
 'baseline_reuse':'Not unconditional: metadata equality, runtime/gradient normalization, native selected-checkpoint protocol must all be supported.',
 'confirmed_behaviors':[
  'CheckpointBackupCallback.on_save is no-op; generic CLI has no per-epoch remote resumable backup.',
  'archive_best_model warns and returns after safe_hf_upload exhaustion; generic CLI can finish despite missing model archive on HF.',
  'auto_resume defaultTrue selects latest checkpoint from outputdir without config/source/runtime compatibility validation.',
  'scientific_records selection receipt hardcodes beam1; incorrect if CLI eval_beams differs.',
  'scientific_records snapshot epochs1/4/10 do not adapt to routing_warmup_ratio.',
  'duplicate save_safetensors keyword takes TypeError fallback on supported current versions; redundant path, not a failed training setup.',
  'warmupworker checks HF archive SHA256/size and best_reload/probes before COMPLETE and checkpoint cleanup.'
 ],
 'no_claims':['No assertion old runtime is exactly current runtime.','No assertion normalized Adam step equivalent to multiplying LR by4.','No new model-quality comparison.']}
v5=load(ROOT/'outputs_rebuttal/amis_routing_diagnostics_20261004_v5/validation_inventory_21_corrected_v5.json')
summary['baseline_selection_metadata']=[{'id':r['id'],'v5_best_step':r.get('best_step'),
 'historical_metrics_selected_checkpoint':next(x['selected_checkpoint'] for x in records if x['run']==r['id']),
 'status':'UNKNOWN' if r['seed']==42 else 'CONFLICT: metrics checkpoint suffix and V5 best_step differ; archive/trainer chronology needed.'}
 for r in v5 if r['family']=='baseline']
(OUT/'historical_configuration_inventory.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
(OUT/'provenance_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
for r in records:
    if 'baseline' in r['run']:
        print(r['run'],json.dumps(r['configuration']),r['selected_checkpoint'])
print('Recorded runs:',len(records))
