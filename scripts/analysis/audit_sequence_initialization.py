"""Compare cached, pinned initialization predictions to continuation epochs on CPU."""
import csv,json,hashlib,argparse
from pathlib import Path
import numpy as np
from sacrebleu.metrics import BLEU,CHRF

def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args(); root=a.root; out=root/'initialization_audit'
 plan=json.loads((root/'PLAN.json').read_text()); initmeta=json.loads((out/'run_metadata.json').read_text()); initmetric=json.loads((out/'validation_metrics.json').read_text()); refmeta=json.loads((root/'runs/S0/validation_metrics.json').read_text())
 assert initmeta['archive_sha256']==plan['initialization']['archive_sha256']
 for key in ['protocol','beams','generation_precision','tf32_generation','raw_reference_sha256']: assert initmetric[key]==refmeta[key],key
 assert initmeta['protocol']['eval_batch']==plan['protocol']['eval_batch']==2
 for key in ['max_source_length','max_target_length','src_lang','tgt_lang','source_prefix']: assert initmeta['run'][key]==plan['runs'][0][key],key
 for key in ['torch','transformers','accelerate','peft','sacrebleu']: assert initmeta['runtime'][key]==plan['runtime'][key],key
 assert initmeta['data']['validation']==plan['data']['validation']
 dataset=Path('data_processed')/plan['dataset']/'validation.csv'; assert hashlib.sha256(dataset.read_bytes()).hexdigest()==plan['data']['validation']['sha256']
 def rows(path): return list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))
 raw=rows(dataset); preds={'R1':rows(out/'validation_predictions.csv'),**{r:rows(root/'runs'/r/'validation_predictions.csv') for r in ['S0','S1','S2','S3']}}
 for rs in preds.values(): assert [(r['source'],r['target']) for r in rs]==[(r['source'],r['target']) for r in raw]
 metrics={'BLEU':BLEU(tokenize='zh'),'chrF++':CHRF(word_order=2),'chrF0':CHRF(word_order=0)}; n=len(raw); counts=np.random.default_rng(20261026).multinomial(n,np.full(n,1/n),size=5000); scores={}; boots={}; refs=[r['target'].strip() for r in raw]
 for rid,rs in preds.items():
  scores[rid]={}; boots[rid]={}
  for m,metric in metrics.items():
   hyp=[r['prediction'].strip() for r in rs]; stats=np.asarray(metric._extract_corpus_statistics(hyp,[refs]),dtype=float)
   scores[rid][m]=metric.corpus_score(hyp,[refs]).score
   boots[rid][m]=np.array([metric._compute_score_from_stats(s.tolist()).score for s in counts@stats])
   if rid=='R1': assert abs(scores[rid][m]-initmetric['scores'][{'BLEU':'bleu','chrF++':'chrf++','chrF0':'chrf0'}[m]])<1e-9
 changes=[]
 for rid in ['S0','S1','S2','S3']:
  for m in metrics:
   lo,hi=np.quantile(boots[rid][m]-boots['R1'][m],[.025,.975]); changes.append(dict(run=rid,metric=m,delta=scores[rid][m]-scores['R1'][m],ci95=[float(lo),float(hi)]))
 epochs=[]
 for rid in ['S0','S1','S2','S3']:
  state=json.loads((root/'runs'/rid/'final_trainer_state.json').read_text())
  for row in state['log_history']:
   if 'eval_bleu' in row: epochs.append({'run':rid,**row,**{f'delta_init_{m}':row[k]-scores['R1'][m] for m,k in [('BLEU','eval_bleu'),('chrF++','eval_chrf++'),('chrF0','eval_chrf0')]}})
 summary={'verification':'PASS_CACHED_PINNED_INIT_PROTOCOL_AND_METRICS','fresh_inference':False,'gpu_used':False,'bootstrap_samples':5000,'rng_seed':20261026,'scores':scores,'changes':changes,'epoch_comparisons':epochs,'caution':'Validation-selected checkpoints; post-hoc pointwise CIs conditional on this seed, not multi-seed inference. Separate bootstrap draws from previous report.'}
 (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Initialization audit — 26/10/2026','','## Material Passport','','Status: VERIFIED cached predictions/metric reproduction; ANALYZED exploratory comparisons.','No fresh GPU inference, optimizer, training, weights download or test access.','','## Provenance and reuse gate','','Pinned archive SHA256 and HF LFS metadata match R1. Original-reference hash, ordered 576 rows, max lengths, source/target language, beam1, eval batch2, FP32, TF32 off, sacrebleu settings and library versions match. Reuse is for inference only; the original 20-epoch R1 training and 3-epoch continuation are intentionally distinct. Training checkpointing differs but is inactive during eval.','','| Model | BLEU | chrF++ | chrF0 |','|---|---:|---:|---:|']
 for rid,s in scores.items(): lines.append(f"| {rid} | {s['BLEU']:.4f} | {s['chrF++']:.4f} | {s['chrF0']:.4f} |")
 lines+=['','## Selected continuation minus initialization','','5,000 paired sentence bootstrap resamples, seed20261026; pointwise 95% percentile CIs, exploratory.','','| Model | Metric | Delta | 95% CI |','|---|---|---:|---|']
 for r in changes: lines.append(f"| {r['run']} | {r['metric']} | {r['delta']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] |")
 lines+=['','## Epoch comparisons','','| Model | Epoch | BLEU minus R1 | chrF++ minus R1 | chrF0 minus R1 |','|---|---:|---:|---:|---:|']
 for r in epochs: lines.append(f"| {r['run']} | {r['epoch']:.0f} | {r['delta_init_BLEU']:+.4f} | {r['delta_init_chrF++']:+.4f} | {r['delta_init_chrF0']:+.4f} |")
 lines+=['','## Interpretation','','All four selected continuation checkpoints are below R1 on all three metrics; all 12 epoch evaluations are below R1 on all three metrics. This is a continuation-wide pattern, not evidence that only CLRR is responsible. It is compatible with overfitting or restart/schedule effects, but cannot identify which one without matched interventions. CIs do not establish universal harm, equivalence, or performance across training seeds.','','R1 itself was chosen on this validation set (epoch19/20); regression to the mean and winner selection affect comparisons to its peak. Thus this is a development diagnostic, not an independent confirmation test. Step-zero scores for the baseline are verified; scores immediately after inserting CLRR at identical weights are a different intervention, not measured here.','','The existing no-go gates still stand. Do not retrain the paper baseline, change its scores, replace checkpoint selection, or widen loss grids from this result. A from-pretrained comparison can test a different recipe, but is not automatically justified as a costly first action.','']
 (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8'); print(json.dumps(summary['changes'],indent=2))
if __name__=='__main__': main()
