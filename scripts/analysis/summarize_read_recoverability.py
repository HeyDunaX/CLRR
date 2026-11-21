"""Summarize locally collected frozen oracle probes; never reads test data."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
out = ROOT/'outputs/experiments/w_read_recoverability_rtx'
records = json.loads((out/'measurements.json').read_text())
plan = json.loads((out/'PLAN.json').read_text())
gates = json.loads((out/'GATES.json').read_text())
complete = json.loads((out/'COMPLETE.json').read_text())
assert complete['parameter_versions_unchanged'] and complete['parameter_grads_absent']
assert not plan['test_used'] and plan['model_optimizer_steps']==0
assert len(records)==len(plan['rows'])*18
assert len({(r['row'],r['label_kind'],r['arm'],r['budget']) for r in records})==len(records)
assert max(r['relative_displacement']-r['budget'] for r in records)<1e-4
assert all(r['intervened']['margin']>=r['native']['margin']-1e-4 for r in records)
rng = np.random.default_rng(1729)
def estimate(values):
    values = np.array(values,dtype=float)
    if not len(values):
        return dict(n=0,mean=None,ci95=None)
    draws = values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return dict(n=len(values),mean=float(values.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())
summary=[]
for kind in ('gold','alternative'):
    for arm in ('query','simplex','context'):
        for budget in (.03,.1,.3):
            subset=[r for r in records if (r['label_kind'],r['arm'],r['budget'])==(kind,arm,budget)]
            wrong=[r for r in subset if not r['native']['target_top1']]
            recovered=sum(r['intervened']['target_top1'] for r in wrong)
            interval=binomtest(recovered,len(wrong)).proportion_ci(confidence_level=.95,method='exact')
            summary.append(dict(label_kind=kind,arm=arm,budget=budget,n=len(subset),
                native_wrong=len(wrong),recovered=recovered,recovery_exact_ci95=[interval.low,interval.high],
                nll_delta=estimate([r['intervened']['nll']-r['native']['nll'] for r in wrong]),
                margin_delta=estimate([r['intervened']['margin']-r['native']['margin'] for r in wrong]),
                mean_actual_displacement=float(np.mean([r['relative_displacement'] for r in subset])),
                correct_lost=sum(r['native']['target_top1'] and not r['intervened']['target_top1'] for r in subset)))
paired=[]
lookup={(r['row'],r['label_kind'],r['arm'],r['budget']):r for r in records}
for arm in ('query','simplex','context'):
    for budget in (.03,.1,.3):
        pairs=[]
        for row in plan['rows']:
            g,a=[lookup[row,kind,arm,budget] for kind in ('gold','alternative')]
            if g['native']['target_top1'] or a['native']['target_top1'] or abs(g['native_rank']-a['native_rank'])>1:
                continue
            pairs.append((g,a))
        paired.append(dict(arm=arm,budget=budget,n=len(pairs),
            gold_recovered=sum(g['intervened']['target_top1'] for g,a in pairs),
            alternative_recovered=sum(a['intervened']['target_top1'] for g,a in pairs),
            gold_minus_alternative_recovery=estimate([int(g['intervened']['target_top1'])-int(a['intervened']['target_top1']) for g,a in pairs]),
            gold_minus_alternative_margin_gain=estimate([(g['intervened']['margin']-g['native']['margin'])-(a['intervened']['margin']-a['native']['margin']) for g,a in pairs])))
native_wrong=[r for r in records if r['label_kind']=='gold' and r['arm']=='context' and r['budget']==.3 and not r['native']['target_top1']]
native_difficulty=dict(wrong=len(native_wrong),median_gold_rank=float(np.median([r['gold_rank'] for r in native_wrong])),
    gold_rank_over_5=sum(r['gold_rank']>5 for r in native_wrong),
    median_native_margin=float(np.median([r['native']['margin'] for r in native_wrong])),
    positive_fixed_rival_margin_but_wrong=sum(r['intervened']['margin']>0 and not r['intervened']['target_top1'] for r in native_wrong))
result=dict(summary=summary,paired_adjacent_rank_controls=paired,native_difficulty=native_difficulty,
    max_replay_error=max(r.get('logit_error',0) for r in gates),
    interpretation='Exploratory fixed-checkpoint teacher-forced oracle, not translation scores or training-seed significance; local search failure is inconclusive. Bootstrap unit: fixed sampled validation sentence.')
(out/'SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
lines=['# W42 native-source read recoverability','',result['interpretation'],'',
       '|Target|Read|Context budget|Native wrong|Recovered top1|Mean NLL delta on wrong|Actual displacement|',
       '|---|---|---:|---:|---:|---:|---:|']
for r in summary:
    mean=r['nll_delta']['mean']
    lines.append(f"|{r['label_kind']}|{r['arm']}|{r['budget']:.2f}|{r['native_wrong']}|{r['recovered']}|{mean:+.4f}|{r['mean_actual_displacement']:.4f}|")
lines+=['','## Paired adjacent-rank controls','',
        'Only positions where gold and alternative were both initially wrong and their ranks differ by at most one.',
        'Alternative labels are not semantic counterfactual references. This control tests arbitrary label steerability only.','',
        '|Read|Budget|Pairs|Gold recovered|Alternative recovered|Gold−alternative recovery CI95%|',
        '|---|---:|---:|---:|---:|---|']
for r in paired:
    lines.append(f"|{r['arm']}|{r['budget']:.2f}|{r['n']}|{r['gold_recovered']}|{r['alternative_recovered']}|{r['gold_minus_alternative_recovery']['ci95']}|")
lines+=['','## Interpretation','',
        f"The native-wrong subset has median gold rank {native_difficulty['median_gold_rank']:.1f}; {native_difficulty['gold_rank_over_5']}/{native_difficulty['wrong']} targets rank below the top five. Median fixed-rival margin is {native_difficulty['median_native_margin']:.4f}. This sample contains many errors far from the current decision boundary.",
        'No selective-gold repair signal sufficient to admit a new training method is established by this probe. NLL improvements also occur for arbitrary adjacent-rank labels.',
        'No new training, larger-budget search, or checkpoint changes are launched from these results.',
        '', '## Limits','',
        '- All interventions optimize a label-informed fixed-rival margin; a positive margin need not make the target top1. Actual top1 and NLL are separately reported.',
        '- Budgets bound the concatenated pre-output-projection context norm. Achieved displacements differ and are reported; no claim of matched global optima.',
        '- Query and simplex searches are local. Larger feasible sets need not give better local solutions. Two restarts do not certify unattainability.',
        '- Zero observed recoveries do not establish a zero population rate: 0/20 has an exact binomial CI95% extending to approximately 16.8%. Degenerate empirical-bootstrap intervals for all-zero paired outcomes are not evidence of equivalence.',
        '- Correct-position preservation is mechanically favored by optimizing their gold margin and is not evidence of a selective deployable algorithm.',
        '- One fixed checkpoint and 32 teacher-forced middle tokens do not establish semantic sufficiency, generation quality, multilingual generality, or novelty.',
        '- Only the final decoder cross-attention read is intervened on. Earlier encoder/decoder computation is fixed; failure here does not exclude useful earlier-layer mechanisms.',
        f"- Replay gate max absolute logit difference: {result['max_replay_error']:.8g}.",
        '- Model parameters were frozen, versions unchanged, and gradients absent. No test access. Public pinned restore removed after use.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
