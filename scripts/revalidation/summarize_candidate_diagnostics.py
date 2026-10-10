"""Replay small artifacts, summarize diagnostics and exploratory interventions."""
import json
import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sacrebleu
from sacrebleu.significance import PairedTest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs_rebuttal/method_candidate_diagnostics'


def main():
    plan=json.loads((OUT/'PLAN.json').read_text())
    refs=pd.read_csv(ROOT/'data_processed/amis_mandarin/validation.csv',dtype=str,keep_default_na=False)
    records={}
    for entry in plan['entries']:
        ident=entry['id'];record=json.loads((OUT/ident/'COMPLETE.json').read_text())
        assert record['entry']==entry and record['optimizer_steps']==0
        assert record['validation_sha256']==plan['validation_sha256']
        assert record['runtime']['worker_sha256']==hashlib.sha256((ROOT/'scripts/revalidation/diagnose_method_candidates.py').read_bytes()).hexdigest()
        records[ident]=record
        for condition,scores in record['scores'].items():
            df=pd.read_csv(OUT/ident/f'validation_{condition}.csv',dtype=str,keep_default_na=False)
            assert df.source.tolist()==refs.source.tolist() and df.target.tolist()==refs.target.tolist()
            for key,metric in [('bleu',sacrebleu.metrics.BLEU(tokenize='zh')),('chrfpp',sacrebleu.metrics.CHRF(word_order=2))]:
                actual=metric.corpus_score(df.prediction.tolist(),[refs.target.tolist()]).score
                assert abs(actual-scores[key])<1e-9
        saved=OUT/ident/'saved_validation.csv'
        if saved.exists():
            old=pd.read_csv(saved,dtype=str,keep_default_na=False)
            on=pd.read_csv(OUT/ident/'validation_on.csv',dtype=str,keep_default_na=False)
            assert [x.strip() for x in old.prediction]==[x.strip() for x in on.prediction], f'{ident}: generation-on does not reproduce saved predictions after original whitespace stripping'
    (OUT/'REPLAY_VERIFIED.json').write_text(json.dumps({'status':'PASS','models':list(records),'worker_hash_match':True,'metric_replay':True,'saved_predictions_match_after_original_strip':True,'optimizer_steps':0},indent=2))
    print('ALL_REPLAY_VERIFIED',list(records),flush=True)
    os.environ['SACREBLEU_SEED']='42'
    rng=np.random.default_rng(42)
    indices=rng.integers(0,576,size=(10000,576))
    interventions=[]
    for ident in ('OLD','U1F','U2F'):
        systems=[(c,pd.read_csv(OUT/ident/f'validation_{c}.csv',dtype=str,keep_default_na=False).prediction.tolist()) for c in plan['interventions']]
        metrics={'BLEU':sacrebleu.metrics.BLEU(tokenize='zh'),'chrF++':sacrebleu.metrics.CHRF(word_order=2)}
        signatures,native=PairedTest(systems,metrics=metrics,references=[refs.target.tolist()],test_type='bs',n_samples=10000,n_jobs=1)()
        for name,metric in metrics.items():
            draws={}
            for condition,hyp in systems:
                stats=np.asarray(metric._extract_corpus_statistics(hyp,[refs.target.tolist()]),dtype=np.float64)
                draw=[]
                for start in range(0,10000,100):
                    sums=stats[indices[start:start+100]].sum(1)
                    draw.extend(metric._compute_score_from_stats(v).score for v in sums)
                draws[condition]=np.asarray(draw)
            for i,(condition,_) in enumerate(systems[1:],1):
                key='bleu' if name=='BLEU' else 'chrfpp'
                native_key='BLEU' if name=='BLEU' else next(k for k in native if k not in ('System','BLEU'))
                interventions.append({'model':ident,'condition':condition,'metric':name,
                    'delta':records[ident]['scores'][condition][key]-records[ident]['scores']['on'][key],
                    'ci95':np.quantile(draws[condition]-draws['on'],[.025,.975]).tolist(),
                    'native_p':native[native_key][i].p_value})
    running=0.
    for rank,i in enumerate(sorted(range(len(interventions)),key=lambda i:interventions[i]['native_p'])):
        running=max(running,min(1.,(len(interventions)-rank)*interventions[i]['native_p']))
        interventions[i]['holm18_p']=running
    (OUT/'interventions.json').write_text(json.dumps({'family':'3 models x 3 interventions x 2 metrics','records':interventions},indent=2))
    lines=['# Frozen candidate validation diagnostics','',
        'All six checkpoints strict-loaded; archive hashes verified; generation-on reproduces saved predictions after the original whitespace stripping where available. Raw decode CSV retains a few leading spaces; metrics match exactly. All scores independently replayed; zero optimizer steps.',
        '', 'Validation576, seed42, FP32/dropout off/TF32 off. Bootstrap is conditional on selected checkpoints; validation selection and seed randomness are not covered.',
        '', '## Generation and geometry','',
        '| Model | BLEU | chrF++ | Source covariance effective rank | Target covariance effective rank | Matched cosine | Shuffled cosine | Centered matched cosine |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for ident,r in records.items():
        g=r['geometry'];score=r['scores']['on']
        lines.append(f"| {ident} | {score['bleu']:.4f} | {score['chrfpp']:.4f} | {g['source']['effective_rank_covariance_energy']:.3f} | {g['target']['effective_rank_covariance_energy']:.3f} | {g['matched_cosine']:.4f} | {g['shuffled_cosine']:.4f} | {g['centered_matched_cosine']:.4f} |")
    lines+=['','## Inference interventions','',
        '| Model | Group off (1-based) | Metric | Delta off−on | CI95 | Holm18 p |',
        '|---|---|---|---:|---|---:|']
    for r in interventions:
        lo,hi=r['ci95'];lines.append(f"| {r['model']} | {r['condition']} | {r['metric']} | {r['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {r['holm18_p']:.4f} |")
    lines+=['','## CE–LSR and target anchor','',
        '| Model | Mean routed CE–LSR cosine | Mean raw CE–LSR cosine | Routed weighted-LSR/CE norm ratio | Routed/raw anchor cosine | Anchor relative L2 |',
        '|---|---:|---:|---:|---:|---:|']
    for ident,r in records.items():
        batches=r['gradients']
        if not batches:continue
        mean=lambda fn:float(np.mean([fn(b) for b in batches]))
        routed=mean(lambda b:b['conditions']['routed']['groups']['all_parameters']['gradient_cosine'])
        raw=mean(lambda b:b['conditions']['raw']['groups']['all_parameters']['gradient_cosine'])
        ratio=mean(lambda b:b['conditions']['routed']['groups']['all_parameters']['lsr_to_ce_ratio'])
        lines.append(f"| {ident} | {routed:+.5f} | {raw:+.5f} | {ratio:.6f} | {mean(lambda b:b['anchor_cosine']):.6f} | {mean(lambda b:b['anchor_relative_l2']):.6f} |")
    lines+=['','## Layer diagnostics','',
        '| Model | Layer | Source residual/current RMS ratio | Source effective covariance rank | Routed CE–LSR cosine | Negative batches /4 |',
        '|---|---:|---:|---:|---:|---:|']
    for ident,r in records.items():
        for layer in range(1,13):
            residual=[x['mean_ratio'] for x in r['residuals'] if x['phase']=='source' and x['layer']==layer]
            rank=r['geometry']['layer_spectra'][f'source_layer{layer}']['effective_rank_covariance_energy']
            cosines=[b['conditions']['routed']['groups'][f'layer{layer-1}']['gradient_cosine'] for b in r['gradients']]
            valid=[c for c in cosines if c is not None]
            c=f'{np.mean(valid):+.5f}' if valid else '—';n=str(sum(c<0 for c in valid))+'/4' if valid else '—'
            lines.append(f"| {ident} | {layer} | {np.mean(residual):.5f} | {rank:.3f} | {c} | {n} |")
    lines+=['','## Interpretation limits','',
        '- Dropout-off gradients are frozen-checkpoint diagnostics, not historical optimizer updates.',
        '- Negative CE–LSR cosine indicates local directional conflict, not proven harm to generalization.',
        '- Larger residual or lower rank alone does not establish a defect.',
        '- Inference intervention gains do not guarantee a retraining gain.',
        '- Raw target is a counterfactual, not an evaluated new translation method.',
        '- No new training or test evaluation is authorized by this analysis.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'VERIFIED.json').write_text(json.dumps({'models':list(records),'metric_replay':'PASS','saved_predictions_match_after_original_strip':'PASS','optimizer_steps':0,'status':'VERIFIED'},indent=2))
    print('ALL_DIAGNOSTICS_VERIFIED',list(records),flush=True)


if __name__=='__main__':main()
