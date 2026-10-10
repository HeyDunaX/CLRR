"""Freeze matched seed42 historical controls for the corrected pilot."""
import csv
import hashlib
import json
from pathlib import Path
import sacrebleu
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
suite=ROOT/'outputs_rebuttal/amis_configuration_confirmation_20261004'
v5dir=ROOT/'outputs_rebuttal/amis_routing_diagnostics_20261004_v5'
v5=read(v5dir/'validation_inventory_21_corrected_v5.json')
finaldir=ROOT/'outputs_rebuttal/amis_final_evaluation_20261004'
records=[]
for name in ['mbart-amis-lsr_l010-seed42','mbart-amis-full_a010_l010-seed42-replica']:
    folder=suite/'runs'/name;metrics=read(folder/'metrics.json');done=read(folder/'COMPLETE.json')
    inv=next(r for r in v5 if r['id']==name)
    evaluated=[]
    for split,p,expected_hash in [
        ('validation',ROOT/inv['validation_prediction_file'],inv['validation_prediction_sha256']),
        ('test',finaldir/'runs'/name/'test_predictions.csv',read(finaldir/'runs'/name/'test_metrics.json')['prediction_sha256'])]:
        d=rows(p);gold=rows(ROOT/'data_processed/amis_mandarin'/f'{split}.csv')
        assert len(d)==len(gold) and [(r['source'],r['target']) for r in d]==[(r['source'],r['target']) for r in gold]
        assert sha(p)==expected_hash
        hyp=[r['prediction'].strip() for r in d];refs=[[r['target'] for r in gold]]
        scores={'bleu':sacrebleu.metrics.BLEU(tokenize='zh').corpus_score(hyp,refs).score,
                'chrfpp':sacrebleu.metrics.CHRF(word_order=2).corpus_score(hyp,refs).score,
                'chrf0':sacrebleu.metrics.CHRF(word_order=0).corpus_score(hyp,refs).score}
        evaluated.append({'split':split,'scores':scores,'prediction_file':str(p.relative_to(ROOT)),
           'prediction_sha256':sha(p),'gold_rows':len(gold),'independent_metrics':'PASS',
           'beams':1 if split=='validation' else 4,'precision':'fp32','tf32':False})
    oldname='mbart-amis-lsr-seed42' if 'lsr' in name else 'mbart-amis-full-seed42'
    oldfolder=ROOT/'outputs_rebuttal/amis_confirmation_20261003/runs'/oldname
    oldmetrics=read(oldfolder/'metrics.json')
    oldpred=rows(oldfolder/'test_predictions.csv');newpred=rows(finaldir/'runs'/name/'test_predictions.csv')
    assert [r['prediction'].strip() for r in oldpred]==[r['prediction'].strip() for r in newpred]
    records.append({'id':name,'role':'matched_lsr_lambda010' if 'lsr' in name else 'original_full_fixed_alpha010_lambda010_replica',
      'configuration':metrics['configuration'],'metrics_file':str((folder/'metrics.json').relative_to(ROOT)),
      'metrics_sha256':sha(folder/'metrics.json'),'selected_checkpoint_from_metrics':metrics['best_model_checkpoint'],
      'v5_hardcoded_best_step_unsupported':inv['best_step'],'completed_epochs':metrics['completed_epochs'],
      'HF_archive':done,'native_replayed_metrics':evaluated,
      'original_lost_archive_run':oldname,'original_selection':oldmetrics['best_model_checkpoint'],
      'original_test_hypotheses_equal_replica':True,
      'interpretation':'Replica retrained from pretrained, not recovery of lost original optimizer state. Exact hypotheses validate recorded test equivalence, not training fairness or runtime identity.'})
inputs=read(suite/'confirmation_inputs.json')
result={'date':'2026-10-05','sacrebleu_version':sacrebleu.__version__,'comparators':records,
 'historical_inputs_manifest':inputs,'historical_inputs_manifest_sha256':sha(suite/'confirmation_inputs.json'),
 'local_data_matches_historical_manifest':{k:sha(ROOT/k)==v for k,v in inputs['files'].items() if k.startswith('data_processed/')},
 'historical_runtime_status':'No standalone exact training dependency/GPU/runtime receipt found locally for these two replicas. Pinned model/source/data receipts exist; inference replay runtime is distinct.',
 'pilot_recipe_required':{'seed':42,'LSR_method':'jepa','LSR_alpha':0.0,'Full_method':'jepa-clrr-enc','Full_alpha':0.1,'lambda':0.1,'routing_warmup_ratio':0.0,'max_epochs':20,'patience':4,'lr':5e-5,'weight_decay':0.0,'lr_warmup_ratio':0.06,'microbatch':32,'accumulation':4,'eval_batch':8,'source_max_length':256,'target_max_length':256,'distance':2,'stack':'encoder','src_lang':'tl_XX','tgt_lang':'zh_CN','bf16':True,'gradient_checkpointing':True,'eval_beams':1,'test_beams':4,'generation_precision':'fp32','fresh_start':True},
 'intentional_difference':'Corrected loss-kwargs/gradient-accumulation normalization. Do not describe old-vs-corrected as changing method architecture or lambda, and do not resume old optimizer.',
 'remaining_limits':['Replica logs/full Trainer histories not found locally; selected suffix576/720 corroborated by metrics and confirmation_summary, not full chronology audit.','Old optimization scaling is implicated by root receipts; these are historical controls, not fair corrected-cohort comparators.']}
assert all(result['local_data_matches_historical_manifest'].values())
(OUT/'historical_comparators.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
for r in records:print(r['id'],r['selected_checkpoint_from_metrics'],r['native_replayed_metrics'])
