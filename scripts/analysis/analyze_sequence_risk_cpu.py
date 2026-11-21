"""CPU-only exploratory diagnostics of the four locked sequence-risk arms."""
import argparse, csv, hashlib, json, re
from collections import Counter
from pathlib import Path
import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--samples',type=int,default=5000); p.add_argument('--seed',type=int,default=20261023); a=p.parse_args()
    out=a.root/'cpu_diagnostics'; out.mkdir(exist_ok=True)
    runs={r:list(csv.DictReader((a.root/'runs'/r/'validation_predictions.csv').open(encoding='utf-8-sig',newline=''))) for r in ['S0','S1','S2','S3']}
    base=runs['S0']; n=len(base); assert n==576
    for rows in runs.values(): assert [(x['source'],x['target']) for x in rows]==[(x['source'],x['target']) for x in base]
    refs=[x['target'] for x in base]
    metrics={'BLEU':BLEU(tokenize='zh'),'chrF++':CHRF(word_order=2),'chrF0':CHRF(word_order=0)}
    counts=np.random.default_rng(a.seed).multinomial(n,np.full(n,1/n),size=a.samples)
    scores={}; boot={}; validation=[]; epochs=[]; components={}; sentences=[]
    compact=lambda s: re.sub(r'\s+','',s)
    for rid, rows in runs.items():
        hyps=[x['prediction'] for x in rows]; scores[rid]={}; boot[rid]={}
        recorded=json.loads((a.root/'runs'/rid/'validation_metrics.json').read_text())['scores']
        for name,metric in metrics.items():
            stats=np.asarray(metric._extract_corpus_statistics(hyps,[refs]),dtype=np.float64)
            value=metric.corpus_score(hyps,[refs]).score; scores[rid][name]=value
            assert abs(value-recorded[{'BLEU':'bleu','chrF++':'chrf++','chrF0':'chrf0'}[name]])<1e-9
            # Validate additive sufficient statistics against public corpus scoring.
            assert abs(metric._compute_score_from_stats(stats.sum(0).tolist()).score-value)<1e-9
            sampled=counts@stats
            boot[rid][name]=np.array([metric._compute_score_from_stats(v.tolist()).score for v in sampled])
        charlens=[len(compact(h)) for h in hyps]; reflens=[len(compact(r)) for r in refs]
        repeated=lambda h: any(v>=3 for v in Counter(compact(h)[i:i+3] for i in range(max(0,len(compact(h))-2))).values())
        validation.append({'run':rid,**scores[rid],'char_length_ratio':sum(charlens)/sum(reflens),'short_half_ref_count':sum(h<.5*r for h,r in zip(charlens,reflens)),'long_1p5_ref_count':sum(h>1.5*r for h,r in zip(charlens,reflens)),'empty_count':sum(not h.strip() for h in hyps),'repeated_trigram_3plus_count':sum(repeated(h) for h in hyps),'exact_match_count':sum(h==r for h,r in zip(hyps,refs))})
        state=json.loads((a.root/'runs'/rid/'final_trainer_state.json').read_text())
        stop=json.loads((a.root/'runs'/rid/'stop_state.json').read_text())
        for row in state['log_history']:
            if 'eval_bleu' in row: epochs.append({'run':rid,**row,'selected':row['step']==stop['best_step']})
        logs=[json.loads(x) for x in (a.root/'runs'/rid/'loss_components.jsonl').read_text().splitlines() if x.strip()]
        components[rid]={'scope':'Logged last microbatch snapshots, NOT epoch means; scalar loss ratios are NOT gradient ratios','snapshots':len(logs),'first':logs[0],'last':logs[-1],'stop_state':stop}
        for i,(row,h,r) in enumerate(zip(rows,hyps,refs)):
            sentences.append({'run':rid,'row':i,'source':row['source'],'target':r,'prediction':h,'source_words':len(row['source'].split()),'target_chars':len(compact(r)),'prediction_chars':len(compact(h)),'sentence_chrf0':metrics['chrF0'].sentence_score(h,[r]).score})
        print(rid,'scores verified',scores[rid],flush=True)
    contrasts={'S1-S0':{'S1':1,'S0':-1},'S2-S0':{'S2':1,'S0':-1},'S3-S0':{'S3':1,'S0':-1},'S3-S1':{'S3':1,'S1':-1},'S3-S2':{'S3':1,'S2':-1},'S2-S1':{'S2':1,'S1':-1},'interaction':{'S3':1,'S2':-1,'S1':-1,'S0':1}}
    comparisons=[]
    for label,weights in contrasts.items():
        for m in metrics:
            samples=sum(w*boot[r][m] for r,w in weights.items()); lo,hi=np.quantile(samples,[.025,.975])
            comparisons.append({'contrast':label,'metric':m,'delta':sum(w*scores[r][m] for r,w in weights.items()),'ci95_low':float(lo),'ci95_high':float(hi)})
    strata=[]
    for label,indices in [('target_chars_1_10',[i for i,r in enumerate(refs) if len(compact(r))<=10]),('target_chars_11_20',[i for i,r in enumerate(refs) if 10<len(compact(r))<=20]),('target_chars_21plus',[i for i,r in enumerate(refs) if len(compact(r))>20])]:
        if not indices: continue
        for rid,rows in runs.items():
            strata.append({'stratum':label,'n':len(indices),'run':rid,**{m:metric.corpus_score([rows[i]['prediction'] for i in indices],[[refs[i] for i in indices]]).score for m,metric in metrics.items()}})
    paired=[]
    byrun={r:[s for s in sentences if s['run']==r] for r in runs}
    for label,left,right in [('S3-S1','S3','S1'),('S3-S0','S3','S0'),('S1-S0','S1','S0'),('S2-S0','S2','S0')]:
        for i in range(n):
            x=byrun[left][i]['sentence_chrf0']-byrun[right][i]['sentence_chrf0']
            paired.append({'contrast':label,'row':i,'delta_sentence_chrf0':x,'identical_prediction':runs[left][i]['prediction']==runs[right][i]['prediction'],'source':base[i]['source'],'target':refs[i],'left_prediction':runs[left][i]['prediction'],'right_prediction':runs[right][i]['prediction']})
    def writecsv(name,rows):
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    for name,rows in [('corpus_and_surface_proxies.csv',validation),('paired_bootstrap_ci.csv',comparisons),('epoch_metrics.csv',epochs),('length_strata.csv',strata),('sentence_scores.csv',sentences),('paired_sentence_changes.csv',paired)]: writecsv(name,rows)
    payload={'status':'ANALYZED_CPU_ONLY','n':n,'split':'validation','seed':42,'bootstrap_samples':a.samples,'bootstrap_rng_seed':a.seed,'sacrebleu':sacrebleu.__version__,'bootstrap':'Paired multinomial resampling of sentence sufficient statistics, percentile CI; identical draws for all arms and metrics. Pointwise exploratory intervals, not simultaneous CIs or confirmatory tests. Sentence independence assumed; document clusters unknown.','input_sha256':{r:hashlib.sha256((a.root/'runs'/r/'validation_predictions.csv').read_bytes()).hexdigest() for r in runs},'scores':validation,'comparisons':comparisons,'loss_snapshots':components}
    (out/'analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Sequence-risk continuation: CPU diagnostics','', '## Material Passport','', '- Mode: validation; status: ANALYZED (corpus scores independently reproduced).','- Date: 2026-10-23; split: validation, 576 aligned rows; one training seed (42).','- CPU only. No training, GPU, test evaluation or model-weight download.','', '## Paired bootstrap','',f'{a.samples} paired sentence resamples; RNG {a.seed}. Pointwise percentile 95% CIs, exploratory; no confirmatory p-values. These intervals do not estimate training-seed variability.','', '| Contrast | Metric | Delta | 95% CI |','|---|---|---:|---|']
    for r in comparisons: lines.append(f"| {r['contrast']} | {r['metric']} | {r['delta']:+.4f} | [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] |")
    lines+=['','## Per-epoch validation','', '| Run | Epoch | BLEU | chrF++ | chrF0 | Eval loss | Selected |','|---|---:|---:|---:|---:|---:|---|']
    for r in epochs: lines.append(f"| {r['run']} | {r['epoch']:.0f} | {r['eval_bleu']:.4f} | {r['eval_chrf++']:.4f} | {r['eval_chrf0']:.4f} | {r['eval_loss']:.4f} | {r['selected']} |")
    lines+=['','## Surface diagnostics (not semantic error annotation)','', '| Run | Output/reference chars | <half reference | >1.5 reference | Repeated trigram >=3 | Exact matches |','|---|---:|---:|---:|---:|---:|']
    for r in validation: lines.append(f"| {r['run']} | {r['char_length_ratio']:.3f} | {r['short_half_ref_count']} | {r['long_1p5_ref_count']} | {r['repeated_trigram_3plus_count']} | {r['exact_match_count']} |")
    lines+=['','Short outputs are only an omission proxy; repetition may be legitimate. Chinese character length removes whitespace and retains punctuation. Human bilingual assessment is required to identify actual omission/hallucination.','', '## Sentence changes (chrF0; not corpus BLEU)','', '| Contrast | Better | Worse | Tied | Identical outputs |','|---|---:|---:|---:|---:|']
    for label in ['S3-S1','S3-S0','S1-S0','S2-S0']:
        rows=[r for r in paired if r['contrast']==label]
        lines.append(f"| {label} | {sum(r['delta_sentence_chrf0']>1e-9 for r in rows)} | {sum(r['delta_sentence_chrf0']< -1e-9 for r in rows)} | {sum(abs(r['delta_sentence_chrf0'])<=1e-9 for r in rows)} | {sum(r['identical_prediction'] for r in rows)} |")
    lines+=['','## Loss interpretation','', 'loss_components.jsonl logs only the last microbatch at selected steps, not epoch-average CE/risk. Scalar risk/CE cannot establish gradient conflict, optimizer influence or causal failure. Trainer eval_loss is a teacher-forced diagnostic; checkpoint selection remains original-reference chrF++.','', '## Integrity / fallacy review (11/11 checked)','', '1. Simpson: descriptive length strata exported; no subgroup significance claims.','2. Ecological: sentence and corpus scores kept distinct.','3. Selection/Berkson: full validation retained; no filtering.','4. Collider: no adjustment on post-treatment outputs.','5. Base rate: counts and denominator 576 reported.','6. Regression to mean: matched continuation controls included; no generalization from one seed.','7. Survivorship: all four completed arms retained.','8. Look elsewhere: all 21 exploratory CIs exported; no significant-only reporting or familywise claims.','9. Forking paths: post-hoc diagnostics explicitly exploratory; test untouched.','10. Causality: length, losses and rank patterns are not causal mechanism proof.','11. Reverse causality: no causal inference from snapshot correlations.','', '## Scope','', 'All arms start from a previously trained baseline, reset optimizer, and continue for three epochs. This analysis rejects/diagnoses this recipe only; it cannot reject all CLRR designs or establish novelty. Predictions are from selected checkpoints, not per-epoch generations. Logs and predictions are insufficient for gradient measurements.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__': main()
