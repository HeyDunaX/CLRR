"""CPU replay and descriptive paired sentence bootstrap for token/beam diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--beam1-inputs',type=Path,required=True)
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--baseline-id',default='R1')
    p.add_argument('--draws',type=int,default=10000)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();out=args.input
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    plan=read(out/'PLAN.json');complete=read(out/'COMPLETE.json');collection=read(out/'COLLECTION_MANIFEST.json')
    assert complete['status']=='COMPLETE' and complete['optimizer_steps']==0 and collection['temporary_weights_removed']
    for name,meta in collection['files'].items():assert digest(out/name)==meta['sha256']
    raw=pd.read_csv(args.references,dtype=str,keep_default_na=False);refs=raw.target.str.strip().tolist();n=len(raw)
    assert digest(args.references)==plan['data']['validation']['sha256'] and n==plan['rows']
    ids=complete['ids'];frames={};results={};systems={};scores={};configs={};checks={}
    metrics={'bleu':BLEU(tokenize='zh'),'chrf++':CHRF(word_order=2),'chrf0':CHRF(word_order=0)}
    first=None
    for ident in ids:
        folder=out/ident;r=read(folder/'RESULT.json');results[ident]=r
        assert r['selection']==plan['selections'][ident]
        assert r['greedy_gate_exact'] and r['greedy_gate_rows']==plan['greedy_gate_rows']
        assert r['CE_replay_max_error']<1e-4 and r['gradients_none'] and r['parameter_versions_unchanged']
        for name,h in r['file_sha256'].items():assert digest(folder/name)==h
        df=pd.read_csv(folder/'per_token.csv').sort_values(['row','position']).reset_index(drop=True)
        keys=df[['row','position','category','train_frequency']]
        if first is None:first=keys
        else:pd.testing.assert_frame_equal(keys,first)
        assert np.isfinite(df[['nll','rank','margin','eos_probability']].values).all()
        assert df.eos_probability.between(0,1.0001).all() and df['rank'].ge(1).all()
        groups={key:part for key,part in df.groupby('category')}
        groups['content_all']=df[df.category.str.startswith('content_')]
        for key,g in groups.items():
            expected=r['token_summary'][key];assert expected['tokens']==len(g)
            values=dict(top1=g.correct.mean(),mean_nll=g.nll.mean(),mean_margin=g.margin.mean(),
                strict_rank_top5=g['rank'].le(5).mean(),strict_rank_top10=g['rank'].le(10).mean(),
                median_strict_rank=g['rank'].median(),mean_eos_probability=g.eos_probability.mean())
            for field,value in values.items():np.testing.assert_allclose(value,expected[field],atol=1e-12,rtol=0)
        ps=pd.read_csv(folder/'per_sentence.csv').sort_values('row')
        means=df.groupby('row').nll.mean().reindex(range(n)).values
        np.testing.assert_allclose(means,ps.ce_all_labels,atol=2e-6,rtol=0)
        frames[ident]=df
        for beam,path in [(1,args.beam1_inputs/ident/'validation_predictions.csv'),
                          (plan['beams'],folder/'validation_beam_predictions.csv')]:
            pred=pd.read_csv(path,dtype=str,keep_default_na=False)
            assert pred[['source','target']].equals(raw[['source','target']])
            if beam==1:assert digest(path)==plan['beam1_inputs'][ident]['predictions_sha256']
            systems[ident,beam]=pred.prediction.str.strip().tolist()
            values={k:m.corpus_score(systems[ident,beam],[refs]).score for k,m in metrics.items()}
            expected=r['beam1_scores'] if beam==1 else r['beam_scores']
            for k,value in values.items():assert abs(value-expected[k])<1e-8
            scores[ident,beam]=values
        gen=read(folder/'GENERATION_CONFIG.json');config=gen['loaded'].copy()
        for key in list(config):
            if key.startswith('_') or key in ['transformers_version','max_length','num_beams','do_sample']:config.pop(key)
        assert gen['overrides']==dict(max_length=256,do_sample=False)
        configs[ident]=config
        checks[ident]=dict(status='PASS',token_rows=len(df),content_tokens=len(groups['content_all']),
            CE_replay_error=r['CE_replay_max_error'],greedy_gate_rows=r['greedy_gate_rows'],length_cap_rows=r['generated_length_cap_rows'])
    assert all(configs[i]==configs[ids[0]] for i in ids),'Generation settings differ outside declared overrides'
    indices=np.random.default_rng(args.seed).integers(0,n,(args.draws,n),dtype=np.int32)
    token_pairs={};contributions={};frequency_pairs=[]
    base=frames[args.baseline_id]
    for ident in ids:
        if ident==args.baseline_id:continue
        cand=frames[ident];mask=base.category.str.startswith('content_')
        a=base[mask];b=cand[mask]
        byrow_count=a.groupby('row').size().reindex(range(n),fill_value=0).values
        byrow_delta=(b.correct.astype(int)-a.correct.astype(int)).groupby(a.row).sum().reindex(range(n),fill_value=0).values
        draws=100*byrow_delta[indices].sum(1)/byrow_count[indices].sum(1)
        token_pairs[ident+'-'+args.baseline_id]=dict(top1_delta_pp=100*float(b.correct.mean()-a.correct.mean()),
            descriptive_sentence_ci95_pp=np.quantile(draws,[.025,.975]).tolist(),
            wrong_to_correct=int((~a.correct&b.correct).sum()),correct_to_wrong=int((a.correct&~b.correct).sum()),
            content_nll_benefit=float(a.nll.mean()-b.nll.mean()),mean_margin_delta=float(b.margin.mean()-a.margin.mean()))
        rows=[]
        for name,m in [('both_correct',a.correct&b.correct),('both_wrong',~a.correct&~b.correct),
                       ('fixed',~a.correct&b.correct),('lost',a.correct&~b.correct)]:
            benefit=(a.nll-b.nll)[m]
            rows.append(dict(group=name,tokens=int(m.sum()),mean_nll_benefit=float(benefit.mean()) if m.any() else None,
                             contribution_to_content_mean_benefit=float(benefit.sum()/len(a))))
        assert abs(sum(row['contribution_to_content_mean_benefit'] for row in rows)-(a.nll.mean()-b.nll.mean()))<1e-12
        contributions[ident+'-'+args.baseline_id]=rows
        for name in sorted(a.category.unique()):
            m=a.category.eq(name)
            frequency_pairs.append(dict(control=args.baseline_id,candidate=ident,category=name,tokens=int(m.sum()),
                control_top1=float(a[m].correct.mean()),candidate_top1=float(b[m].correct.mean()),
                wrong_to_correct=int((~a.correct[m]&b.correct[m]).sum()),correct_to_wrong=int((a.correct[m]&~b.correct[m]).sum())))
    # Descriptive paired sentence bootstrap, using corpus sufficient statistics.
    bootstrap={}
    for (ident,beam),hyp in systems.items():
        for name in ['bleu','chrf0']:
            metric=metrics[name];stats=np.asarray(metric._extract_corpus_statistics(hyp,[refs]),dtype=np.float64)
            sampled=[]
            for start in range(0,args.draws,100):
                sums=stats[indices[start:start+100]].sum(1)
                sampled.extend(metric._compute_score_from_stats(s).score for s in sums)
            bootstrap[ident,beam,name]=np.asarray(sampled)
    contrasts=[]
    pairs=[((ident,1),(ident,plan['beams'])) for ident in ids]
    pairs += [((args.baseline_id,beam),(ident,beam)) for beam in [1,plan['beams']] for ident in ids if ident!=args.baseline_id]
    for control,candidate in pairs:
        for name in ['bleu','chrf0']:
            diffs=bootstrap[*candidate,name]-bootstrap[*control,name]
            contrasts.append(dict(control=list(control),candidate=list(candidate),metric=name,
                delta=scores[candidate][name]-scores[control][name],descriptive_paired_ci95=np.quantile(diffs,[.025,.975]).tolist()))
    summary=dict(status='PASS',checks=checks,generation_configs_match=True,token_comparisons=token_pairs,
        nll_benefit_decomposition=contributions,frequency_comparisons=frequency_pairs,
        beam_scores={ident:{str(b):scores[ident,b] for b in [1,plan['beams']]} for ident in ids},
        descriptive_beam_contrasts=contrasts,bootstrap=dict(draws=args.draws,seed=args.seed,unit='sentence',
            exploratory_validation=True,not_training_seed_uncertainty=True,no_confirmatory_p_values=True),
        optimizer_steps=0,test_evaluated=False,temporary_weights_removed=True,gpu_after=collection['gpu_after'],
        analyzer_sha256=digest(Path(__file__)))
    (out/'ANALYSIS.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    pd.DataFrame(frequency_pairs).to_csv(out/'frequency_comparisons.csv',index=False)
    print(json.dumps(dict(status=summary['status'],token_comparisons=token_pairs,beam_scores=summary['beam_scores'],
                         beam_contrasts=contrasts,nll_decomposition=contributions),indent=2))


if __name__=='__main__':main()
