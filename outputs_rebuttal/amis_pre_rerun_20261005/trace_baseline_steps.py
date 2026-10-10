"""Trace historical baseline selection without modifying historical artifacts."""
import hashlib
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
conf=ROOT/'outputs_rebuttal/amis_confirmation_20261003'
snap=read(conf/'snapshot_sha256.json')
v5=read(ROOT/'outputs_rebuttal/amis_routing_diagnostics_20261004_v5/validation_inventory_21_corrected_v5.json')
allm=read(conf/'all_metrics.json')
records=[]
for seed in [43,44]:
    name=f'mbart-amis-baseline-seed{seed}'
    log=conf/(name+'.log'); text=log.read_text(encoding='utf-8-sig')
    pattern=r'\[eval @ step (\d+)\] eval_loss=([\d.]+) \| eval_bleu=([\d.]+) \| eval_chrf\+\+=([\d.]+)'
    evaluations=[dict(step=int(step),eval_loss=float(loss),bleu=float(bleu),chrfpp=float(chrf)) for step,loss,bleu,chrf in re.findall(pattern,text)]
    unique={e['step']:e for e in evaluations}
    # Final evaluation may use reloaded best weights at endpoint globalstep; initial occurrence is training history.
    first_unique={}
    for e in evaluations:first_unique.setdefault(e['step'],e)
    maxima=max(e['chrfpp'] for e in first_unique.values())
    maxsteps=[e['step'] for e in first_unique.values() if e['chrfpp']==maxima]
    metrics=read(conf/'runs'/name/'metrics.json')
    inv=next(r for r in v5 if r['id']==name)
    allrow=next(r for r in allm if r['seed']==seed and r['method']=='baseline')
    logselected=re.findall(r'"best_model_checkpoint":\s*"([^"]+)"',text)
    records.append({'id':name,'log_file':str(log.relative_to(ROOT)),'log_sha256':sha(log),
      'snapshot_recorded_log_sha256':snap.get(name+'.log'),'snapshot_hash_pass':snap.get(name+'.log')==sha(log),
      'metrics_selected_checkpoint':metrics['best_model_checkpoint'],'all_metrics_selected_checkpoint':allrow['best_model_checkpoint'],
      'log_selected_checkpoint_strings':logselected,'training_eval_records_rounded':list(first_unique.values()),
      'rounded_chrfpp_max_training_steps':maxsteps,'v5_inventory_best_step':inv['best_step'],
      'v5_best_step_source':'scripts/revalidation/audit_amis_followup_f0.py hardcoded model list; carried through build_corrected_amis_inventory_v5.py without verification except baseline42',
      'selection_conclusion':'Historical log and metrics agree on selected step; V5 best_step is unsupported hardcoded metadata.',
      'replayed_validation_scores':{k:inv.get(k) for k in ['eval_bleu','eval_chrf++','eval_chrf0']},
      'limits':['Rounded log metrics cannot prove exact tie-breaking.','No original optimizer/Trainer state or pinned training dependency receipt reconstructed by this trace.','Replay score validity does not prove compatibility with corrected optimization.']})
assert all(r['snapshot_hash_pass'] for r in records)
(OUT/'baseline_step_trace.json').write_text(json.dumps({'date':'2026-10-05','records':records,
 'decision':'Use logged selected steps720/612 for descriptive chronology of43/44, label V5 metadata252/504unsupported. No historical edits. Baseline reuse remains conditional on effective loss/runtime/selection protocol compatibility.'},indent=2),encoding='utf-8')
for r in records:print(r['id'],'log/metrics',r['metrics_selected_checkpoint'],'training max rounded',r['rounded_chrfpp_max_training_steps'],'snapshot',r['snapshot_hash_pass'])
