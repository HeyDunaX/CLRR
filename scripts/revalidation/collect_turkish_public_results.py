"""Collect only public test receipts/metrics/predictions; never model weights."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0, str(ROOT / 'src'))
from experiment_runner.colab import get_hf_token
from huggingface_hub import HfApi,hf_hub_download
OUT=ROOT/'outputs_rebuttal/turkish_old_method_pilot_es10_20261007'
REPO='FiveC/amis-rewire-checkpoints'
CONFIG={'baseline':('baseline','ce'),'middle':('middle-alignment','layer6-tau0p10-weight0p10'),'lsr':('lsr','lambda0p10'),'full':('full','alpha0p10-lambda0p10-distance2')}
api=HfApi(token=get_hf_token());revision=api.model_info(REPO).sha
rows=[]
for f,(method,recipe) in CONFIG.items():
 prefix=f'checkpoints/opus100_turkish_english_20k/mbart-large-50/{method}/{recipe}-lr5e-5-bs32-ga4-bf16-es10/pilot/seed-42'
 name=prefix+'/provenance/COMPLETE.json'
 if not api.get_paths_info(REPO,[name],revision=revision):print('TEST_PENDING',f);continue
 paths=[name,prefix+'/test_metrics.json',prefix+'/test_predictions.csv']
 manifest={}
 for remote in paths:
  cached=Path(hf_hub_download(REPO,remote,revision=revision,token=api.token,local_dir=OUT/'public_results_cache'))
  data=cached.read_bytes();entry=api.get_paths_info(REPO,[remote],revision=revision)[0]
  digest=hashlib.sha256(data).hexdigest()
  assert entry.size==len(data)
  assert entry.lfs.sha256==digest if entry.lfs else entry.blob_id==hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  manifest[remote]={'sha256':digest,'size':len(data)}
  dest=OUT/f/'seed42'/('COMPLETE.json' if remote==name else Path(remote).name)
  dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 val=json.loads((OUT/f/'seed42'/'COMPLETE.json').read_text())
 metrics=json.loads((OUT/f/'seed42'/'test_metrics.json').read_text())
 assert val['status']=='COMPLETE' and val['seed']==42 and metrics['beams']==4 and metrics['generation_precision']=='fp32' and metrics['tf32_generation'] is False
 assert metrics['raw_test_sha256']==hashlib.sha256((ROOT/'data_processed/opus100_turkish_english_20k/test.csv').read_bytes()).hexdigest()
 row={'family':f,'seed':42,'test_bleu':metrics['test_bleu'],'test_chrfpp':metrics['test_chrf++'],'test_chrf0':metrics['test_chrf0'],'selected_checkpoint':val['selected_checkpoint'],'repo':REPO,'revision':revision,'prefix':prefix,'verified_files':manifest}
 rows.append(row);print('PUBLIC_TEST_VERIFIED',json.dumps({k:v for k,v in row.items() if k!='verified_files'}))
(OUT/'public_test_results.json').write_text(json.dumps(rows,indent=2))
print('COMPLETED_TESTS',len(rows))
