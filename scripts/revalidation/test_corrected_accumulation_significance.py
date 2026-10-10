"""Post-hoc descriptive inference, frozen predictions; never selects models."""
import hashlib,itertools,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU,CHRF
from sacrebleu.significance import PairedTest,_compute_p_value
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs_rebuttal/amis_accumulation_confirmation_20261005'
report=json.loads((OUT/'three_seed_comparison.json').read_text())
raw_path=ROOT/'data_processed/amis_mandarin/test.csv'
raw=pd.read_csv(raw_path,dtype=str,keep_default_na=False)
refs=raw.target.str.strip().tolist();n=len(raw);assert n==575
B=10000;indices=np.random.default_rng(42).choice(n,size=(B,n),replace=True)
metrics={'BLEU':BLEU(tokenize='zh'),'chrF++':CHRF(word_order=2)}
systems={};observed={};draws={};provenance=[]
for seed in (42,43,44):
    for family in ('baseline','lsr','full'):
        row=next(r for r in report['rows'] if r['seed']==seed and r['family']==family and r['regime']==('baseline_audited' if family=='baseline' else 'corrected'))
        if family=='baseline':path=ROOT/row['scores']['test']['prediction_file']
        else:
            verification=ROOT/row['verification_file'];v=json.loads(verification.read_text());assert v['status']=='VERIFIED'
            path=verification.parent/'runs'/v['hf_archive']['run']/'test_predictions.csv'
        pred=pd.read_csv(path,dtype=str,keep_default_na=False)
        assert pred[['source','target']].equals(raw[['source','target']])
        hypotheses=pred.prediction.str.strip().tolist();systems[seed,family]=hypotheses
        provenance.append({'seed':seed,'family':family,'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for name,metric in metrics.items():
            score=metric.corpus_score(hypotheses,[refs]).score
            expected=row['scores']['test']['bleu' if name=='BLEU' else 'chrfpp']
            assert abs(score-expected)<1e-8
            observed[seed,family,name]=score
            stats=np.asarray(metric._extract_corpus_statistics(hypotheses,[refs]),dtype=np.float32)
            boot=np.empty(B)
            for start in range(0,B,128):
                sums=stats[indices[start:start+128]].sum(axis=1)
                boot[start:start+len(sums)]=[metric._compute_score_from_stats(s).score for s in sums]
            draws[seed,family,name]=boot
        print('BOOTSTRAP_COMPLETE',seed,family,flush=True)
tests=[]
conditional_mean_intervals=[]
for control in ('lsr','baseline'):
    for name in metrics:
        # Same sampled sentence indices for every training seed: never pretend
        # the shared test set provides 3*575 independent observations.
        avg=np.mean([draws[s,'full',name]-draws[s,control,name] for s in (42,43,44)],axis=0)
        ci=np.quantile(avg,[.025,.975])
        conditional_mean_intervals.append({'contrast':'Full_minus_'+control,'metric':name,'mean_delta':float(np.mean([observed[s,'full',name]-observed[s,control,name] for s in (42,43,44)])),'conditional_delta_ci95':ci.tolist(),'scope':'Sentence uncertainty conditional on these three frozen training seeds; excludes training-seed uncertainty.'})
os.environ['SACREBLEU_SEED']='42'
for seed in (42,43,44):
    for control in ('lsr','baseline'):
        # Independently reproduce p-values through the public native API.
        _,native=PairedTest(named_systems=[('Control',systems[seed,control]),('Full',systems[seed,'full'])],metrics=metrics,references=[refs],test_type='bs',n_samples=B,n_jobs=1)()
        for name in metrics:
            delta=observed[seed,'full',name]-observed[seed,control,name]
            diffs=draws[seed,'full',name]-draws[seed,control,name]
            low,high=np.quantile(diffs,[.025,.975])
            absolute=np.abs(diffs);p=float(_compute_p_value(absolute-absolute.mean(),abs(delta)))
            native_name=metrics[name].corpus_score(systems[seed,control],[refs]).name
            assert abs(p-native[native_name][1].p_value)<1e-12
            tests.append({'seed':seed,'contrast':'Full_minus_'+control,'metric':name,'delta':delta,'delta_ci95':[float(low),float(high)],'p_sacrebleu_bs':p})
        print('NATIVE_P_VALUES_MATCH',seed,control,flush=True)
assert len(tests)==12
running=0
for rank,index in enumerate(np.argsort([r['p_sacrebleu_bs'] for r in tests])):
    running=max(running,min(1.,(len(tests)-rank)*tests[index]['p_sacrebleu_bs']))
    tests[index]['p_holm12']=running;tests[index]['holm12_below_005']=running<.05
seed_tests=[]
for control in ('lsr','baseline'):
    for name in metrics:
        differences=np.array([observed[s,'full',name]-observed[s,control,name] for s in (42,43,44)])
        perm=np.array([np.mean(differences*np.array(signs)) for signs in itertools.product((-1,1),repeat=3)])
        p=float(np.mean(np.abs(perm)>=abs(differences.mean())-1e-12))
        seed_tests.append({'contrast':'Full_minus_'+control,'metric':name,'n_seeds':3,'paired_differences':differences.tolist(),'mean_delta':float(differences.mean()),'exact_two_sided_sign_flip_p':p,'minimum_attainable_two_sided_p':.25,'assumption':'Symmetric exchangeability of paired seed effects under null; weakly assessable with n=3.'})
result={'status':'VERIFIED_NATIVE_P_VALUES','analysis':'Post-hoc significance characterization, no model/seed selection','bootstrap_samples':B,'bootstrap_rng':42,'test_rows':n,'test_reference_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),'multiplicity_family':'12 tests: 3 seeds x 2 contrasts x 2 metrics, Holm familywise correction','sentence_tests':tests,'seed_tests':seed_tests,'provenance':provenance,'limits':['Sentence resampling conditional on frozen checkpoints; does not cover training randomness.','Only three training seeds; exact two-sided seed sign-flip cannot attain p<0.05.','Sentences assumed exchangeable; correlated documents may invalidate sentence-level precision.','Exploratory analysis after inspecting test scores; no pre-registration claim.','P-values do not establish novelty, mechanism, cross-language generalization or ACL acceptance.']}
result['conditional_mean_intervals']=conditional_mean_intervals
(OUT/'significance_corrected.json').write_text(json.dumps(result,indent=2))
lines=['# Corrected accumulation — significance characterization','','## Material Passport','','Post-hoc analysis of frozen selected checkpoints, 575 test sentences, 10000 paired bootstrap draws, RNG42; all native SacreBLEU p-values independently matched. No training/GPU use or model selection.','','| Contrast | Seed | Metric | Delta | Delta 95% CI | p native | p Holm12 | Holm12 < .05 |','|---|---:|---|---:|---|---:|---:|---|']
for r in tests:lines.append(f"| {r['contrast']} | {r['seed']} | {r['metric']} | {r['delta']:+.4f} | [{r['delta_ci95'][0]:+.4f}, {r['delta_ci95'][1]:+.4f}] | {r['p_sacrebleu_bs']:.6f} | {r['p_holm12']:.6f} | {r['holm12_below_005']} |")
lines+=['','## Mean delta: sentence uncertainty with all three seeds fixed','']
for r in conditional_mean_intervals:lines.append(f"- {r['contrast']} {r['metric']}: mean {r['mean_delta']:+.4f}, conditional CI [{r['conditional_delta_ci95'][0]:+.4f}, {r['conditional_delta_ci95'][1]:+.4f}]. Same sentence draws across seeds; excludes training randomness.")
lines+=['','## Across training seeds','']
for r in seed_tests:lines.append(f"- {r['contrast']} {r['metric']}: exact two-sided sign-flip p={r['exact_two_sided_sign_flip_p']:.3f}, n=3, minimum attainable p=.25. Assumption: symmetric paired effects under null.")
lines+=['','## Limits','']+['- '+s for s in result['limits']]
lines+=['','Method: https://github.com/mjpost/sacrebleu and https://aclanthology.org/W04-3250/ .']
(OUT/'SIGNIFICANCE_CORRECTED.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'sentence_tests':tests,'seed_tests':seed_tests}))
