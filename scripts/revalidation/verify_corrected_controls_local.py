"""Recompute validation/test scores from each collected seed, without old-score assumptions."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--family',choices=('lsr003','clrr'),required=True);p.add_argument('--seed',type=int,choices=(42,43,44),required=True);args=p.parse_args()
family=args.family;seed=args.seed;root=Path(__file__).resolve().parents[2]
base=root/'outputs_rebuttal/amis_corrected_controls_20261005';out=base/family/f'seed{seed}'
archive=root/f'scratch/controls_results_{family}_seed{seed}.zip';out.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(archive) as z:
    assert all((out/n).resolve().is_relative_to(out.resolve()) for n in z.namelist());z.extractall(out)
manifest=json.loads((base/'bundle_manifest.json').read_text());assert manifest==json.loads((out/'remote_bundle_manifest.json').read_text())
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
receipt=json.loads((out/'collection_receipt.json').read_text());assert receipt['seed']==seed and receipt['family']==family
assert receipt['remote_archive_verified'] and receipt['source_hashes_match'] and len(receipt['remote_metadata_verified'])==6 and all(receipt['remote_metadata_verified'].values())
folder=out/'runs'/receipt['run'];m=json.loads((folder/'metrics.json').read_text());state=json.loads((out/'final_trainer_state.json').read_text());test=json.loads((folder/'test_metrics.json').read_text()) if receipt['status']=='COMPLETE' else None
assert not m['test_evaluated'] and (test is None or not test['test_used_for_selection'])
assert m['best_model_checkpoint']==state['best_model_checkpoint']==receipt['selected_checkpoint']
best=max((r for r in state['log_history'] if 'eval_chrf++' in r),key=lambda r:r['eval_chrf++'])
assert best['eval_chrf++']==state['best_metric'] and abs(m['eval_chrf++']-state['best_metric'])<1e-8
assert best['step']==int(state['best_model_checkpoint'].rsplit('-',1)[-1])
sys.path.insert(0,str(root/'src'));from amis_rewire.metrics import generation_metrics
scores_by_split={}
for split in (('validation','test') if test else ('validation',)):
    prediction_path=folder/(split+'_predictions.csv');pred=pd.read_csv(prediction_path,dtype=str,keep_default_na=False)
    raw=pd.read_csv(root/'data_processed/amis_mandarin'/(split+'.csv'),dtype=str,keep_default_na=False)
    assert pred[['source','target']].equals(raw[['source','target']])
    scores=generation_metrics(pred['prediction'].tolist(),raw['target'].tolist(),tokenize='zh');metadata=m if split=='validation' else test;key='eval' if split=='validation' else 'test'
    assert abs(scores['bleu']-metadata[key+'_bleu'])<1e-8 and abs(scores['chrf++']-metadata[key+'_chrf++'])<1e-8
    scores_by_split[split]={'bleu':scores['bleu'],'chrfpp':scores['chrf++'],'rows':len(raw),'prediction_sha256':hashlib.sha256(prediction_path.read_bytes()).hexdigest()}
summary={'status':'VERIFIED' if test else 'VALIDATION_VERIFIED','family':family,'seed':seed,'selected_step':best['step'],'selected_epoch':best['epoch'],'completed_epochs':m['completed_epochs'],'scores':scores_by_split,'bundle_hashes_match':True,'hf_archive':receipt,'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
(out/'local_verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
