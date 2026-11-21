"""CPU-only evidence snapshot and fixed validation sample, no model loading."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

ROOT = Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',default='outputs/experiments/method_search_round2')
args=parser.parse_args()
out=ROOT/args.output
out.mkdir(parents=True,exist_ok=True)
inputs={}
def read(path):
    path=ROOT/path
    inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path,keep_default_na=False)
registry=read('outputs/experiments/rtx_documentation_snapshot/RUN_REGISTRY.csv')
registry['comparison_scope']='cohort only; method/model compute controls need spec audit'
registry.loc[registry.selector.str.contains('chrF',case=False),'comparison_scope']='separate selector; do not pool with BLEU-selected'
registry.loc[registry.attention_backend.eq('eager'),'comparison_scope']='separate eager backend; do not pool with SDPA'
registry.loc[registry.dataset.ne('amis_mandarin'),'comparison_scope']='different dataset; no pooled scores'
def exclusions(row):
    reasons=['cohort-specific; compute and code matching require receipt audit']
    if 'chrf' in row.selector.lower():reasons.append('different checkpoint selector from BLEU cohort')
    if row.attention_backend == 'eager':reasons.append('different backend from SDPA cohort')
    if row.attention_backend not in ('eager','sdpa'):reasons.append('backend not explicit in snapshot')
    if row.dataset != 'amis_mandarin':reasons.append('different dataset')
    if not row.archive_sha256:reasons.append('missing archive SHA in snapshot')
    return '; '.join(reasons)
registry['comparison_scope']=registry.apply(exclusions,axis=1)
registry['configuration_source']=[
    str(candidate.relative_to(ROOT)) if candidate.exists() else row.validation_source
    for _,row in registry.iterrows()
    for candidate in [ROOT/'configs/experiments'/f'{Path(row.cohort).name}.json']]
registry['recipe_label_from_hf_prefix']=registry.hf_prefix.str.rsplit('/',n=1).str[-1]
registry.to_csv(out/'RUN_EVIDENCE.csv',index=False)
reference=read('data_processed/amis_mandarin/validation.csv')
paths={
    'baseline42':'outputs/experiments/amis_residual_seed_diagnostics/inputs/R1/validation_predictions.csv',
    'W42':'outputs/experiments/cmc_estimator_review/collected/W/validation_predictions.csv',
    'CV42':'outputs/experiments/cmc_estimator_review/collected/CV/validation_predictions.csv',
    'CMC42':'outputs/experiments/cmc_estimator_review/collected/CMC/validation_predictions.csv',
    'PLC42':'outputs/experiments/amis_prefix_consistency_failure_analysis/artifacts/P_PREFIX_S42/validation_predictions.csv',
    'Middle42':'outputs/experiments/amis_prefix_consistency_failure_analysis/artifacts/MIDDLE_S42/validation_predictions.csv',
}
predictions={}
metrics={}
for name,path in paths.items():
    frame=read(path)
    assert frame[['source','target']].equals(reference[['source','target']]),name
    predictions[name]=frame.prediction.tolist()
    metrics[name]=dict(bleu=BLEU(tokenize='zh').corpus_score(frame.prediction.tolist(),[reference.target.tolist()]).score,
        chrf=CHRF(word_order=0).corpus_score(frame.prediction.tolist(),[reference.target.tolist()]).score,
        chrfpp=CHRF(word_order=2).corpus_score(frame.prediction.tolist(),[reference.target.tolist()]).score,
        path=path)
state_path='outputs/experiments/amis_residual_seed_diagnostics/inputs/R1/final_trainer_state.json'
state_bytes=(ROOT/state_path).read_bytes()
inputs[state_path]=hashlib.sha256(state_bytes).hexdigest()
history=json.loads(state_bytes)['log_history']
evaluations=[entry for entry in history if 'eval_bleu' in entry]
baseline_selector={metric:{key:max(evaluations,key=lambda entry:entry[metric])[key]
    for key in (metric,'step','epoch')} for metric in ('eval_bleu','eval_chrf++')}
rows=np.random.default_rng(20261118).choice(len(reference),24,replace=False).tolist()
sample=reference.iloc[rows].copy()
sample.insert(0,'row',rows)
for name,values in predictions.items():sample[name]=[values[i] for i in rows]
sample.to_csv(out/'FIXED_VALIDATION_SAMPLE.csv',index=False)
# Natural one-word substitution graph, exact normalized source frame. No synthetic data.
# Tokens are whitespace-delimited orthographic units, not Amis morphemes.
train=read('data_processed/amis_mandarin/train.csv')
groups=defaultdict(list)
for row,s in enumerate(train.source):
    tokens=re.findall(r"[^\W\d_]+(?:[’'][^\W\d_]+)*",s.lower())
    for pos,token in enumerate(tokens):
        key=(len(tokens),pos,tuple(tokens[:pos]),tuple(tokens[pos+1:]))
        groups[key].append((row,token))
edges=set(); repeat_groups=0;coverage=set();triples=0
examples=[]
for key,members in groups.items():
    unique={token:row for row,token in members}
    if len(unique)<2:continue
    repeat_groups+=1
    if len(unique)>=3:triples+=1
    items=sorted(unique.items())
    for i,(a,ra) in enumerate(items):
        for b,rb in items[i+1:]:
            if train.iloc[ra].target==train.iloc[rb].target:continue
            edge=tuple(sorted((ra,rb)));edges.add(edge);coverage.update(edge)
            if len(examples)<24:examples.append(dict(row_a=ra,row_b=rb,edit_from=a,edit_to=b,source_a=train.iloc[ra].source,
                source_b=train.iloc[rb].source,target_a=train.iloc[ra].target,target_b=train.iloc[rb].target))
pd.DataFrame(examples).to_csv(out/'NATURAL_EDIT_EXAMPLES.csv',index=False)
summary=dict(registry_rows=len(registry),registry_note='Historical 97-checkpoint snapshot; AC43/44 and DRM updates are separately cited, not silently added.',
    baseline_selector_replay=baseline_selector,metrics_recomputed=metrics,sample_rows=rows,sample_scope='24 random validation sentences, fixed seed20261118, reference-relative inspection only',
    natural_edit_graph=dict(train_rows=len(train),single_word_substitution_pairs=len(edges),covered_rows=len(coverage),
        source_frames_with_multiple_words=repeat_groups,source_frames_with_three_or_more_words=triples,
        limitations='Unicode letter words, punctuation/case normalized for analysis only; no edited dataset; pairs are not verified semantic minimal pairs, triangles are source-frame counts not evidence of transferable target edit actions.'))
(out/'CPU_EVIDENCE.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(out/'INPUT_MANIFEST.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
