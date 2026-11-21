"""Sentence-level descriptive SAC probe analysis; no training admission by p-value."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correlation(x,y):
    a=pd.Series(x).rank().to_numpy();b=pd.Series(y).rank().to_numpy()
    if np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def mean_ci(x,rng):
    x=np.asarray(x,float)
    means=x[rng.integers(len(x),size=(5000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist())


def partial_rank(frame,outcome):
    controls=frame[['entropy','source_characters','target_characters']].rank().to_numpy()
    design=np.column_stack([np.ones(len(frame)),controls])
    ranks=frame[['C',outcome]].rank().to_numpy()
    residual=ranks-design@np.linalg.lstsq(design,ranks,rcond=None)[0]
    return float(np.corrcoef(residual.T)[0,1]) if np.all(residual.std(0)>1e-12) else None


def associations(frame):
    results={}
    for outcome in ('top1_error','ce','source_gap_mask_difference','source_gold_gap'):
        x=frame.C.to_numpy();y=frame[outcome].to_numpy()
        rng=np.random.default_rng(713)
        boots=[];partial_boots=[]
        for _ in range(2000):
            rows=rng.integers(len(frame),size=len(frame))
            value=correlation(x[rows],y[rows])
            if value is not None:boots.append(value)
            partial=partial_rank(frame.iloc[rows],outcome)
            if partial is not None:partial_boots.append(partial)
        results[outcome]=dict(spearman=correlation(x,y),
            sentence_bootstrap_ci95=np.quantile(boots,[.025,.975]).tolist() if boots else None,
            partial_rank_descriptive=partial_rank(frame,outcome),
            partial_rank_sentence_bootstrap_ci95=np.quantile(partial_boots,[.025,.975]).tolist() if partial_boots else None,
            controls=['entropy','source_characters','target_characters'])
    return results


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);plan=json.loads((out/'PLAN.json').read_text())
    complete=json.loads((out/'COMPLETE.json').read_text())
    assert complete['passed'] and complete['optimizer_steps']==0 and not complete['test_used']
    assert complete['plan_sha256']==digest(out/'PLAN.json')
    assert json.loads((out/'CPU_GATE.json').read_text())['passed']
    metrics=['C','A','B','source_information','source_gold_gap','source_gap_mask_difference','entropy','ce','top1_error','gold_probability']
    results={};frames={};rng=np.random.default_rng(918)
    for checkpoint in plan['checkpoints']:
        ident=checkpoint['id'];folder=out/ident
        receipt=json.loads((folder/'RESULT.json').read_text())
        assert receipt['checkpoint']==checkpoint and receipt['plan_sha256']==digest(out/'PLAN.json')
        assert receipt['csv_sha256']==digest(folder/'per_sentence.csv')
        assert json.loads((folder/'GPU_GATE.json').read_text())['passed']
        raw=pd.read_csv(folder/'per_sentence.csv');raw=raw[raw.category=='lexical']
        expected=len(plan['validation_rows'])*len(plan['negative_seeds'])*len(plan['dropout_seeds'])
        assert len(raw)==expected and not raw.duplicated(['row','pair','dropout_seed']).any()
        assert np.isfinite(raw.select_dtypes(include='number')).all().all()
        assert raw.chain_residual.max()<2e-6
        frame=raw.groupby('row',sort=True).mean(numeric_only=True)
        assert frame.index.tolist()==plan['validation_rows']
        for key in metrics:
            assert abs(frame[key].mean()-receipt['summary']['lexical'][key])<1e-10
        frames[ident]=frame
        results[ident]=dict(means={key:mean_ci(frame[key],rng) for key in metrics},
            ratio_mean_C_over_mean_B=float(frame.C.mean()/frame.B.mean()) if frame.B.mean()>1e-8 else None,
            associations=associations(frame),source_pairs=[dict(pair=int(i),C=float(g.C.mean()),B=float(g.B.mean()),source_gold_gap=float(g.source_gold_gap.mean())) for i,g in raw.groupby('pair')])
        frame.to_csv(folder/'sentence_averages.csv')
    left,right=frames['W_S42'],frames['BASELINE_S42']
    assert left.index.equals(right.index)
    delta={key:mean_ci(left[key]-right[key],rng) for key in metrics}
    summary=dict(passed=True,models=results,paired_W_minus_baseline=delta,sentences=len(left),
        bootstrap_repetitions_mean=5000,bootstrap_repetitions_correlation=2000,
        scope='Exploratory fixed checkpoint seed42; sentence resampling after averaging masks/pairs, conditional on fixed distractor pairings. Reused distractor sources can induce dependencies not covered by these approximate intervals. Not training-seed uncertainty, independent holdout, causal effect or significance-based admission.',
        training_admission='HOLD: nonzero C or correlation alone cannot justify benefit from minimizing C.',
        provenance=dict(plan_sha256=digest(out/'PLAN.json'),verified_checkpoint_archives=True,optimizer_steps=0,test_used=False))
    (out/'ANALYSIS.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Frozen SAC source-attribution probe','',summary['scope'],'',f"Sentences: {len(left)}; lexical tokens primary; special tokens retained separately in raw CSV.",'','| Metric | Baseline | W | W−baseline [95% sentence bootstrap CI] |','|---|---:|---:|---:|']
    for key in metrics:
        a=results['BASELINE_S42']['means'][key]['mean'];b=results['W_S42']['means'][key]['mean'];d=delta[key]
        lines.append(f"| {key} | {a:.6f} | {b:.6f} | {d['mean']:+.6f} [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] |")
    for ident,result in results.items():
        lines+=['',f'## {ident} associations','', '| Outcome vs C | Spearman [95% sentence CI] | Partial rank (entropy/source and target lengths) |','|---|---:|---:|']
        for key,value in result['associations'].items():
            lines.append(f"| {key} | {value['spearman']} {value['sentence_bootstrap_ci95']} | {value['partial_rank_descriptive']} {value['partial_rank_sentence_bootstrap_ci95']} |")
    lines+=['','## Interpretation limits','',summary['training_admission'],'','Teacher-forced error is not free-running translation quality. Wrong-source/gold-prefix contexts are off the training joint distribution. Confidence and pairing confound attribution. Conditional dependence C does not measure causal source faithfulness; source-blind predictions also yield C=0. A frozen endpoint cannot establish effects along training. No new BLEU or test score was measured.']
    (out/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,sentences=len(left),delta=delta,models=results)))


if __name__=='__main__':main()
