"""Analyze all prespecified seeds without selecting variants using test outcomes."""
from pathlib import Path
import csv
import hashlib
import json
import os
import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest, _compute_p_value

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs_rebuttal/amis_confirmation_20261003'


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def write(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def holm(values):
    result=np.zeros(len(values));running=0.
    for rank,index in enumerate(np.argsort(values)):
        running=max(running,min(1.,(len(values)-rank)*values[index]));result[index]=running
    return result


def main():
    assert (OUT/'COMPLETE.json').exists()
    protocol=json.loads((OUT/'protocol.json').read_text())
    original=read(ROOT/'data_processed/amis_mandarin/test.csv')
    references=[r['target'].strip() for r in original]
    scores=[];systems={};observed={};boot={};tests=[];paired=[]
    indices=np.random.default_rng(42).integers(0,len(original),size=(10000,len(original)))
    metrics={'BLEU':BLEU(tokenize='zh'),'chrF++':CHRF(word_order=2)}
    for run in protocol['runs']:
        folder=OUT/'runs'/run['run_name'];assert (folder/'COMPLETE.json').exists()
        records=read(folder/'test_predictions.csv')
        assert len(records)==575 and [(r['source'],r['target']) for r in records]==[(r['source'],r['target']) for r in original]
        predictions=[r['prediction'].strip() for r in records]
        key=(run['seed'],run['method']);systems[key]=predictions
        metadata=json.loads((folder/'metrics.json').read_text())
        values={name:metric.corpus_score(predictions,[references]).score for name,metric in metrics.items()}
        assert abs(values['BLEU']-metadata['test_bleu'])<1e-8
        assert abs(values['chrF++']-metadata['test_chrf++'])<1e-8
        scores.append({'seed':run['seed'],'method':run['method'],**values,
            'best_checkpoint':metadata['best_model_checkpoint'],'completed_epochs':metadata['completed_epochs'],
            'training_seconds_including_probes':metadata['training_time_seconds'],
            'prediction_sha256':hashlib.sha256((folder/'test_predictions.csv').read_bytes()).hexdigest()})
        for name,metric in metrics.items():
            observed[*key,name]=values[name]
            stats=np.array(metric._extract_corpus_statistics(predictions,[references]),dtype='float32')
            draws=np.empty(10000)
            for start in range(0,10000,128):
                sums=stats[indices[start:start+128]].sum(axis=1)
                draws[start:start+len(sums)]=[metric._compute_score_from_stats(x).score for x in sums]
            boot[*key,name]=draws
        print('BOOTSTRAPPED',key,flush=True)
    pairs=[('baseline','full'),('lsr','full'),('baseline','clrr'),('clrr','full')]
    for seed in [42,43,44]:
        for baseline,challenger in pairs:
            for metric in metrics:
                delta=observed[seed,challenger,metric]-observed[seed,baseline,metric]
                draws=boot[seed,challenger,metric]-boot[seed,baseline,metric]
                low,high=np.percentile(draws,[2.5,97.5]);absolute=np.abs(draws)
                tests.append({'seed':seed,'baseline':baseline,'challenger':challenger,'metric':metric,
                    'delta':delta,'ci_low':float(low),'ci_high':float(high),
                    'p_value':float(_compute_p_value(absolute-absolute.mean(),abs(delta)))})
    assert len(tests)==24
    for row,adjusted in zip(tests,holm([r['p_value'] for r in tests])):row['p_holm24']=float(adjusted)
    os.environ['SACREBLEU_SEED']='42'
    _,native=PairedTest(named_systems=[('Baseline',systems[42,'baseline']),('Full',systems[42,'full'])],
        metrics={'BLEU':BLEU(tokenize='zh')},references=[references],test_type='bs',n_samples=10000,n_jobs=1)()
    native_result=next(v for k,v in native.items() if k!='System')[1]
    assert abs(native_result.p_value-tests[0]['p_value'])<1e-12
    summary=[]
    for method in ['baseline','clrr','lsr','full']:
        for metric in metrics:
            vals=np.array([observed[seed,method,metric] for seed in [42,43,44]])
            summary.append({'method':method,'metric':metric,'mean':float(vals.mean()),'std_across_seeds':float(vals.std(ddof=1)),
                'minimum':float(vals.min()),'maximum':float(vals.max()),'seeds':3})
    for baseline,challenger in pairs:
        for metric in metrics:
            vals=[observed[s,challenger,metric]-observed[s,baseline,metric] for s in [42,43,44]]
            paired.append({'baseline':baseline,'challenger':challenger,'metric':metric,
                'seed42':vals[0],'seed43':vals[1],'seed44':vals[2],
                'mean_delta':float(np.mean(vals)),'std_paired_delta':float(np.std(vals,ddof=1)),
                'positive_seeds':sum(v>0 for v in vals),'seeds':3})
    write('seed_scores.csv',scores);write('seed_summary.csv',summary)
    write('paired_seed_deltas.csv',paired);write('paired_bootstrap24.csv',tests)
    full_lsr=next(r for r in paired if r['baseline']=='lsr' and r['metric']=='BLEU')
    full_base=next(r for r in paired if r['baseline']=='baseline' and r['challenger']=='full' and r['metric']=='BLEU')
    signal=full_lsr['positive_seeds']==3 and full_base['positive_seeds']==3 and full_lsr['mean_delta']>=.5
    decision={'status':'ANALYZED_ALL12','incremental_signal_in_tested_seeds':signal,
        'full_minus_lsr_bleu':full_lsr,'full_minus_baseline_bleu':full_base,
        'judgment':'Positive incremental signal across these3seeds; broader confirmation remains necessary.' if signal else
                   'Insufficient stable incremental CLRR evidence under this prespecified protocol; investigate or narrow claims before Turkish.',
        'limitations':['3training seeds do not establish universal superiority','per-seed sentence bootstrap is not a test over training-seed variability',
                       'no gold morphological preservation labels','observational gradient/rank probes are not causal mechanism identification'],
        'bootstrap_family':24,'bootstrap_samples':10000,'bootstrap_seed':42,
        'native_sacrebleu_p_verified':True,'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if (OUT/'artifact_losses.json').exists():
        decision['artifact_losses']=json.loads((OUT/'artifact_losses.json').read_text())
        decision['limitations'].append('Selected-best weights archives listed in artifact_losses were lost; final predictions/metrics and corresponding trajectory probes were preserved and verified.')
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2))
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':main()
