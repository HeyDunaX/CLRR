"""Read-only G0/D1/D4 audit; writes only this new audit directory."""
import csv
import ast
import hashlib
import json
import statistics
from pathlib import Path
import sacrebleu

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
W = ROOT / 'outputs_rebuttal/amis_routing_warmup_20261004_v2'
V = ROOT / 'outputs_rebuttal/amis_routing_diagnostics_20261004_v5'
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name, x):
    (OUT/name).write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')
def table(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)

gold=list(csv.DictReader((ROOT/'data_processed/amis_mandarin/validation.csv').open(encoding='utf-8-sig',newline='')))
rows=[]
for run in sorted((W/'collected_records/runs').iterdir()):
    s=read(run/'final_trainer_state.json');m=read(run/'metrics.json')
    h=[r for r in s['log_history'] if 'eval_chrf++' in r]
    best=max(h,key=lambda r:r['eval_chrf++']);end=h[-1]
    assert best['step']==s['best_global_step']
    assert abs(best['eval_chrf++']-s['best_metric'])<1e-9
    pred=list(csv.DictReader((run/'validation_predictions.csv').open(encoding='utf-8-sig',newline='')))
    assert len(pred)==len(gold)==576
    assert [r['target'] for r in pred]==[r['target'] for r in gold]
    assert [r['source'] for r in pred]==[r['source'] for r in gold]
    assert sha(run/'validation_predictions.csv')==read(run/'COMPLETE.json')['validation_prediction_sha256']
    hyp=[r['prediction'].strip() for r in pred];refs=[[r['target'] for r in gold]]
    measured={'bleu':sacrebleu.metrics.BLEU(tokenize='zh').corpus_score(hyp,refs).score,
              'chrfpp':sacrebleu.metrics.CHRF(word_order=2).corpus_score(hyp,refs).score,
              'chrf0':sacrebleu.metrics.CHRF(word_order=0).corpus_score(hyp,refs).score}
    assert abs(measured['chrfpp']-best['eval_chrf++'])<1e-8
    schedule=[json.loads(l) for l in (run/'routing_schedule.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(abs(r['alpha_train']-.1*min(r['optimizer_step_before_update']/144,1))<1e-10 for r in schedule)
    assert all(r['alpha_eval']==.1 for r in schedule)
    rows.append({'run':run.name,'family':'full' if 'full' in run.name else 'clrr_only','seed':m['seed'],
      'best_epoch':best['epoch'],'best_step':best['step'],'end_epoch':s['epoch'],'end_step':s['global_step'],
      'early_stopping_inferred':s['epoch']<20,'eval_count':len(h),
      'best_bleu':measured['bleu'],'best_chrfpp':measured['chrfpp'],'best_chrf0':measured['chrf0'],
      'endpoint_bleu':end['eval_bleu'],'endpoint_chrfpp':end['eval_chrf++'],
      'best_minus_endpoint_bleu':measured['bleu']-end['eval_bleu'],
      'best_minus_endpoint_chrfpp':measured['chrfpp']-end['eval_chrf++'],
      'best_eval_loss':best['eval_loss'],'endpoint_eval_loss':end['eval_loss'],
      'bleu_optimal_epoch':max(h,key=lambda r:r['eval_bleu'])['epoch'],
      'min_loss_epoch':min(h,key=lambda r:r['eval_loss'])['epoch'],
      'alpha_schedule_checks':'PASS','prediction_sha256':sha(run/'validation_predictions.csv')})
table('warmup_history_selection_audit.csv',rows)

inventory=read(V/'validation_inventory_21_corrected_v5.json')
inventory_rows=[]
for r in inventory:
    if r['family'] not in ['baseline','original_full_detached','matched_lsr','tuned_lsr','clrr_only_detached']:
        continue
    receipt=ROOT/r.get('validation_prediction_file','__missing__')
    if not receipt.exists():
        # Some older entries retain a remote path; retain unknown rather than guessing.
        verified='NOT_LOCAL_PATH'
    else:
        expected=r.get('validation_prediction_sha256')
        verified='PASS' if expected and sha(receipt)==expected else 'NO_HASH_OR_MISMATCH'
    inventory_rows.append({'id':r['id'],'family':r['family'],'seed':r['seed'],
       'alpha':r['alpha'],'lambda':r['lambda'],'detach':r['detach'],'best_step':r.get('best_step'),
       'bleu':r.get('eval_bleu'),'chrfpp':r.get('eval_chrf++'),'chrf0':r.get('eval_chrf0'),
       'archive_sha256':r.get('archive_sha256'),'archive_repo':r.get('archive_repo'),
       'archive_path':r.get('archive_path'),'prediction_file':r.get('validation_prediction_file'),
       'prediction_hash_check':verified,'validation_status':r.get('v5_validation_status')})
table('control_inventory_audit.csv',inventory_rows)
dump('control_configuration_receipt_copy.json',read(W/'CONTROL_TRAINING_CONFIGURATION_PASS.json'))
replay=[]
for machine in ['machine1','machine2']:
    for r in read(W/'gate_receipts'/machine/f'CONTROLS_PASS_{machine}.json')['results']:
        replay.append(dict(machine=machine,**r))
dump('control_runtime_replay_receipts.json',replay)

errors=read(V/'error_analysis.json')
error_rows=[]
def walk(x,path=''):
    if isinstance(x,dict):
        if 'rows' in x and 'BLEUzh' in x:
            error_rows.append({'path':path,'rows':x['rows'],'bleu':x['BLEUzh'],
             'chrfpp':x.get('chrF++'),'chrf0':x.get('chrF0'),
             'length_ratio':x.get('prediction_reference_nonwhitespace_character_ratio'),
             'empty_rate':x.get('empty_prediction_rate')})
        for k,v in x.items():walk(v,path+'/'+k)
walk(errors)
table('v5_error_bins_descriptive.csv',error_rows)
comparisons=[]
for split in ['test','validation']:
    for name,bins in errors[split+'_comparisons'].items():
        for binname,b in bins.items():
            if 'corpus_delta_full_minus_comparator' not in b:continue
            d=b['corpus_delta_full_minus_comparator']
            comparisons.append({'split':split,'comparison':name,'bin':binname,'rows':b['rows'],
                 'bleu_delta':d['BLEUzh'],'chrfpp_delta':d['chrF++'],'chrf0_delta':d['chrF0']})
table('v5_error_comparisons_descriptive.csv',comparisons)
mean_comparisons=[]
for split in ['test','validation']:
    for comparator in ['baseline','matched_lsr','tuned_lsr']:
        for binname in ['all','<=5','6-10','>10']:
            subset=[r for r in comparisons if r['split']==split and r['bin']==binname and ('minus_'+comparator+'_seed') in r['comparison']]
            if not subset:continue
            mean_comparisons.append({'split':split,'comparator':comparator,'bin':binname,'seeds':len(subset),'rows_per_seed':subset[0]['rows'],
                 **{k:statistics.mean(r[k] for r in subset) for k in ['bleu_delta','chrfpp_delta','chrf0_delta']}})
table('v5_error_comparison_means.csv',mean_comparisons)
error_provenance=[]
for split in ['test','validation']:
    for name,r in errors[split].items():
        p=ROOT/r['prediction_file']; expected=r.get('verified_sha256',r.get('observed_sha256'))
        error_provenance.append({'split':split,'id':name,'file':r['prediction_file'],
         'sha256':sha(p),'hash_check':'PASS' if sha(p)==expected else 'FAIL'})
assert all(r['hash_check']=='PASS' for r in error_provenance)
table('v5_error_prediction_hash_audit.csv',error_provenance)
oldcode=ROOT/'outputs_rebuttal/followup_20261003/source_final/src/amis_rewire/modeling.py'
tree=ast.parse(oldcode.read_text(encoding='utf-8-sig'))
oldforwards=[]
for cls in [n for n in ast.walk(tree) if isinstance(n,ast.ClassDef)]:
    for fn in [n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='forward']:
        calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='pop' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='num_items_in_batch']
        if calls:oldforwards.append({'class':cls.name,'forward_lineno':fn.lineno,'has_kwargs':fn.args.kwarg is not None,'pops_num_items_in_batch':True,'pop_lines':[n.lineno for n in calls]})
oldtrainer=ROOT/'outputs_rebuttal/followup_20261003/source_final/src/amis_rewire/evaluation.py'
dump('archived_wrapper_ast_receipt.json',{'source':str(oldcode.relative_to(ROOT)),'sha256':sha(oldcode),'forwards':oldforwards,
 'archived_evaluation_source':str(oldtrainer.relative_to(ROOT)), 'archived_evaluation_sha256':sha(oldtrainer),
 'archived_accepts_loss_kwargs_occurrence':'accepts_loss_kwargs' in oldtrainer.read_text(encoding='utf-8-sig'),
 'scope':'Historical 20261003 code archive corroborates wrapper pattern; not proof every training job used this exact runtime.'})
dump('audit_summary.json',{'date':'2026-10-05','sacrebleu_version':sacrebleu.__version__,
 'warmup_runs_verified':len(rows),'warmup_selected_metrics_check':'PASS',
 'routing_alpha_schedule_checks':'PASS','validation_gold_sha256':sha(ROOT/'data_processed/amis_mandarin/validation.csv'),
 'control_rows':len(inventory_rows),'control_hash_checks':{s:sum(r['prediction_hash_check']==s for r in inventory_rows) for s in set(r['prediction_hash_check'] for r in inventory_rows)},
 'selection_rule':'raw-reference validation chrF++ beam1, alpha_eval .1',
 'D2_recommendation':'Optional: only to identify warmup checkpoint dependence; not required to re-establish warmup NO_GO.',
 'error_analysis_source_sha256':sha(V/'error_analysis.json'),
 'technical_status':'TECHNICAL_HOLD: independent root accumulation test found wrapper vs baseline loss scaling mismatch; see accumulation/receipt.json.',
 'limitations':['This script performs no training or GPU forward/backward.','Control original best-step baseline42 unknown.','Configuration equality does not certify effective optimization equality.','Root accumulation correctness audit is authoritative; all causal upgrade conclusions conditional until cohort impact repaired.']})
print(json.dumps(rows,ensure_ascii=False,indent=2))
print('Control families:',sorted(set(r['family'] for r in inventory)))
print('Error analysis sections:',list(errors))
print('Outputs:',OUT)
