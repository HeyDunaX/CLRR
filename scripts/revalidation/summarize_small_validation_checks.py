"""Independent replay of validation checks; no training."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from amis_rewire.metrics import generation_metrics
from sacrebleu.metrics import CHRF
OUT=ROOT/'outputs_rebuttal/amis_small_validation_checks_20261007'
RAW=pd.read_csv(ROOT/'data_processed/amis_mandarin/validation.csv',dtype=str,keep_default_na=False)
COHORT=ROOT/'outputs_rebuttal/amis_fresh_confirmation_20261005'
def entropy(v):
 p=v[v>0]/v.sum();return float(np.exp(-(p*np.log(p)).sum()))
rows=[];contrasts=[];verified_predictions=0
for fam in ['baseline','clrr','lsr','full']:
 for seed in range(45,53):
  folder=OUT/fam/f'seed{seed}'
  done=json.loads((folder/'DONE.json').read_text());assert done['scope']=='validation_only_no_optimizer_steps'
  assert done['precision']=='FP32_TF32_off' and 'A100' in done['device']
  val=json.loads((COHORT/fam/f'seed{seed}/VAL_COMPLETE.json').read_text())
  assert done['archive_sha256']==val['archive_sha256'] and done['weight_digest']==val['selected_and_exported_weight_digest']
  for n,d in done['source_hashes'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==d
  assert done['validation_sha256']==hashlib.sha256((ROOT/'data_processed/amis_mandarin/validation.csv').read_bytes()).hexdigest()
  geo=json.loads((folder/'geometry.json').read_text());sv=np.load(folder/'singular_spectra.npz')
  for name in ['source','target']:
   assert abs(entropy(sv[name])-geo[name]['singular_entropy_rank'])<1e-8
   assert abs(entropy(sv[name]**2)-geo[name]['energy_entropy_rank'])<1e-8
  row={'family':fam,'seed':seed,**{k:v for k,v in geo.items() if not isinstance(v,dict)}}
  for name in ['source','target']:
   for k,v in geo[name].items():row[f'{name}_{k}']=v
  rows.append(row)
  if fam not in ['clrr','full']:continue
  scores=json.loads((folder/'interventions.json').read_text())
  saved=pd.read_csv(COHORT/fam/f'seed{seed}'/'runs'/val['run_name']/'validation_predictions.csv',dtype=str,keep_default_na=False)
  for cond,metrics in scores.items():
   df=pd.read_csv(folder/f'{cond}_predictions.csv',dtype=str,keep_default_na=False)
   assert df[['source','target']].equals(RAW[['source','target']])
   pred=df.prediction.str.strip().tolist()
   if cond=='all_on':assert pred==saved.prediction.str.strip().tolist()
   calc=generation_metrics(pred,RAW.target.tolist(),tokenize='zh')
   calc['chrf0']=CHRF(word_order=0).corpus_score(pred,[RAW.target.str.strip().tolist()]).score
   for metric in ['bleu','chrf++','chrf0']:assert abs(calc[metric]-metrics[metric])<1e-8
   verified_predictions+=1
   if cond!='all_on':contrasts.append({'family':fam,'seed':seed,'condition':cond,**{k:metrics[k]-scores['all_on'][k] for k in ['bleu','chrf++','chrf0']}})
df=pd.DataFrame(rows);dc=pd.DataFrame(contrasts)
df.to_csv(OUT/'geometry_summary.csv',index=False);dc.to_csv(OUT/'intervention_deltas.csv',index=False)
lines=['# Validation checks after fresh confirmation — 07/10/2026','',
 'Scope: 32 selected checkpoints; validation576 only; no training or new test generation. Pinned HF hashes and source hashes verified; all16routing-on predictions match saved validation exactly. Spectral ranks independently recomputed from saved singular spectra, not from raw pooled matrices.','',
 '## Alignment and common direction','',
 '|Family|Matched cos|Shuffled cos|Matched−shuffled|Centered matched|Centered matched−shuffled|Source common-mean energy|','|---|---:|---:|---:|---:|---:|---:|']
for fam,sub in df.groupby('family',sort=False):
 keys=['matched_cosine','shuffled_cosine_mean','matched_minus_shuffled','centered_matched_cosine','centered_matched_minus_shuffled','source_common_mean_energy_fraction']
 lines.append('|'+fam+'|'+'|'.join(f'{sub[k].mean():.6f}' for k in keys)+'|')
lines+=['','## Inference interventions: condition minus all-on','',
 '|Family|Condition|ΔBLEU|95% seed Student-t CI|Wins|ΔchrF++|ΔchrF0|','|---|---|---:|---|---:|---:|---:|']
summary=[]
for (fam,cond),sub in dc.groupby(['family','condition']):
 a=sub.bleu.to_numpy();n=len(a);mean=float(a.mean());se=a.std(ddof=1)/np.sqrt(n);half=t.ppf(.975,n-1)*se
 rec={'family':fam,'condition':cond,'n':n,'mean_bleu':mean,'ci95':[mean-half,mean+half],'wins':int((a>0).sum()),'mean_chrfpp':float(sub['chrf++'].mean()),'mean_chrf0':float(sub.chrf0.mean())};summary.append(rec)
 lines.append(f'|{fam}|{cond}|{mean:+.6f}|[{mean-half:+.6f}, {mean+half:+.6f}]|{rec["wins"]}/{n}|{rec["mean_chrfpp"]:+.6f}|{rec["mean_chrf0"]:+.6f}|')
lines+=['','Intervals are exploratory, unadjusted; paired training-seed descriptions on frozen weights. They do not establish gains from retraining. Matched-minus-shuffled evaluates discrimination beyond common direction; neither high raw cosine nor a low residual cosine proves collapse/conflict. Baseline/CLRR old LSR gradient probes are counterfactual, not their training objective.','']
(OUT/'VERIFICATION_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
(OUT/'independent_verification.json').write_text(json.dumps({'status':'LOCAL_REPLAY_VERIFIED','models':32,'prediction_conditions':verified_predictions,'all_on_exact_replays':16,'rank_replay_scope':'saved singular spectra','geometry_rows':576,'interventions':summary},indent=2))
print('\n'.join(lines))
