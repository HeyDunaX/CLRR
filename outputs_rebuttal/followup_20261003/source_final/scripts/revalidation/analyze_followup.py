"""Verify and analyze the matched inference results separately from historical tables."""
from pathlib import Path
import csv
import hashlib
import json
import os
import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest, _compute_p_value

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs_rebuttal/followup_20261003'
SAMPLES=10000


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def holm(p):
    result=np.zeros(len(p));running=0.
    for rank,index in enumerate(np.argsort(p)):
        running=max(running,min(1.,(len(p)-rank)*p[index]));result[index]=running
    return result


def main():
    directory=OUT/'reevaluation'
    manifest=json.loads((directory/'protocol.json').read_text())['manifest']
    assert (directory/'COMPLETE.json').exists()
    systems={};summary=[];provenance=[]
    for item in manifest['models']:
        folder=directory/item['id'];assert (folder/'DONE.json').exists()
        refs=read(ROOT/'data_processed'/item['dataset']/'test.csv')
        historical_path=ROOT/item['reference_prediction_path']
        assert hashlib.sha256(historical_path.read_bytes()).hexdigest()==item['reference_prediction_sha256']
        old=read(historical_path)
        original_refs=[r['target'].strip() for r in refs]
        tok='13a' if item['id'].startswith('ash-') else 'zh'
        for variant,file in [('normal','test_predictions.csv'),('route_off','route_off_predictions.csv')]:
            path=folder/file
            if not path.exists():continue
            rows=read(path)
            assert len(rows)==len(refs)
            assert [(r['source'],r['target']) for r in rows]==[(r['source'],r['target']) for r in refs]
            assert [int(r['index']) for r in rows]==list(range(len(rows)))
            metadata=json.loads((folder/('metrics.json' if variant=='normal' else 'route_off_metrics.json')).read_text())
            assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata['prediction_sha256']
            predictions=[r['prediction'].strip() for r in rows]
            old_predictions=[r['prediction'].strip() for r in old]
            assert len(old_predictions)==len(rows)
            key=item['id']+('::route_off' if variant=='route_off' else '')
            metrics={'bleu':BLEU(tokenize=tok),'chrfpp':CHRF(word_order=2),'chrf0':CHRF(word_order=0)}
            scores={name:metric.corpus_score(predictions,[original_refs]).score for name,metric in metrics.items()}
            assert abs(scores['bleu']-metadata['scores']['bleu'])<1e-9
            assert abs(scores['chrfpp']-metadata['scores']['chrf++'])<1e-9
            historical={name:metric.corpus_score(old_predictions,[original_refs]).score for name,metric in metrics.items()}
            summary.append({'id':key,'dataset':item['dataset'],'variant':variant,'rows':len(rows),**scores,
                'old_bleu':historical['bleu'],'old_chrfpp':historical['chrfpp'],
                'delta_old_bleu':scores['bleu']-historical['bleu'],'delta_old_chrfpp':scores['chrfpp']-historical['chrfpp'],
                'changed_predictions_vs_old':sum(a!=b for a,b in zip(predictions,old_predictions))})
            systems[key]={'predictions':predictions,'references':original_refs,'metrics':metrics,'dataset':item['dataset']}
            provenance.append({'id':key,'prediction_sha256':metadata['prediction_sha256'],'checkpoint_archive_sha256':item['sha256']})
    assert len(systems)==17
    comparisons=[('mbart-baseline','mbart-full'),('mbart-lsr','mbart-full'),
                 ('nllb-baseline','nllb-full'),('mt5-baseline','mt5-full'),('mt5-clrr','mt5-full'),
                 ('byt5-baseline','byt5-full'),('ash-mbart-baseline','ash-mbart-full')]
    comparisons.extend((item['id']+'::route_off',item['id']) for item in manifest['models'] if item['method']=='full')
    boot={};cache={}
    for dataset in sorted({s['dataset'] for s in systems.values()}):
        count=next(len(s['references']) for s in systems.values() if s['dataset']==dataset)
        indices=np.random.default_rng(42).choice(count,size=(SAMPLES,count),replace=True)
        for key,s in systems.items():
            if s['dataset']!=dataset:continue
            for name,metric in s['metrics'].items():
                stats=np.asarray(metric._extract_corpus_statistics(s['predictions'],[s['references']]),dtype='float32')
                values=np.empty(SAMPLES)
                for start in range(0,SAMPLES,128):
                    sums=stats[indices[start:start+128]].sum(axis=1)
                    values[start:start+len(sums)]=[metric._compute_score_from_stats(x).score for x in sums]
                boot[key,name]=values
                cache[key,name]=metric.corpus_score(s['predictions'],[s['references']]).score
            print('BOOTSTRAPPED',key,flush=True)
    tests=[]
    for base,challenger in comparisons:
        for metric in ['bleu','chrfpp','chrf0']:
            delta=cache[challenger,metric]-cache[base,metric]
            values=boot[challenger,metric]-boot[base,metric]
            lo,hi=np.percentile(values,[2.5,97.5]);absolute=np.abs(values)
            tests.append({'baseline':base,'challenger':challenger,'metric':metric,'delta':delta,
                          'ci_low':float(lo),'ci_high':float(hi),
                          'p_value':float(_compute_p_value(absolute-absolute.mean(),abs(delta)))})
    for row,adjusted in zip(tests,holm([r['p_value'] for r in tests])):row['p_holm_36']=float(adjusted)
    assert len(tests)==36
    # Cross-check one comparison against SacreBLEU's own paired bootstrap.
    os.environ['SACREBLEU_SEED']='42'
    a,b=systems['mbart-baseline'],systems['mbart-full']
    _,native=PairedTest(named_systems=[('Baseline',a['predictions']),('Full',b['predictions'])],
                       metrics={'BLEU':BLEU(tokenize='zh')},references=[a['references']],
                       test_type='bs',n_samples=SAMPLES,n_jobs=1)()
    result=next(v for k,v in native.items() if k!='System')[1]
    assert abs(result.p_value-tests[0]['p_value'])<1e-12
    historical_files=json.loads((ROOT/'outputs_rebuttal/metric_audit_20261002/full/restoration_manifest.json').read_text())['restored_files']
    for item in historical_files:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    write('matched_inference_scores.csv',summary);write('matched_inference_bootstrap.csv',tests)
    (OUT/'analysis_protocol.json').write_text(json.dumps({'samples':SAMPLES,'seed':42,
       'family':'36 exploratory comparisons: 12 pairs x BLEU/chrF++/char-only chrF, Holm correction',
       'ci':'paired sentence bootstrap percentile CI of corpus-score difference',
       'p':'SacreBLEU centered absolute-difference bootstrap; native BLEU p verified',
       'scope':'fixed checkpoints, single training seeds; no causal training attribution or multi-seed confirmation',
       'ashaninka_scope':'baseline and Full patience4; not epoch20/patience10',
       'unchanged_historical_files':len(historical_files),'inputs':provenance},indent=2))
    for row in tests:
        if row['metric']=='bleu':print(row,flush=True)


if __name__=='__main__':main()
