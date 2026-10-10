"""Evaluate the four locked selections from the public canonical store."""
import argparse,json,os,re,sys,time
from pathlib import Path
ROOT=Path('/content/CLRR_turkish');sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts/revalidation'))
from huggingface_hub import HfApi,hf_hub_download
import run_turkish_old_method_pilot as worker
TARGET='FiveC/amis-rewire-checkpoints';OUT=ROOT/'outputs_rebuttal'/worker.PREFIX
CONFIG={'baseline':('baseline','ce'),'middle':('middle-alignment','layer6-tau0p10-weight0p10'),'lsr':('lsr','lambda0p10'),'full':('full','alpha0p10-lambda0p10-distance2')}
def root(f):
 method,recipe=CONFIG[f]
 return f'checkpoints/opus100_turkish_english_20k/mbart-large-50/{method}/{recipe}-lr5e-5-bs32-ga4-bf16-es10/pilot/seed-42'
def mapped(path):
 match=re.match(re.escape(worker.PREFIX)+r'/mbart-tr-en-(baseline|middle|lsr|full)-seed42-es10/(.*)',path)
 if not match:return path
 family,suffix=match.groups()
 return root(family)+'/'+('best.zip' if suffix.endswith('-best.zip') else suffix)
p=argparse.ArgumentParser();p.add_argument('--account',choices=['1','2'],required=True);a=p.parse_args()
os.environ['HF_TOKEN']=Path('/content/.turkish_hf_token').read_text().strip()
api=HfApi(token=os.environ['HF_TOKEN']);assert not api.model_info(TARGET).private
status=OUT/f'queue_account{a.account}.json'
state=json.loads(status.read_text())
for key in ('error','error_type'):state.pop(key,None)
state.update(status='WAITING_FOUR_PUBLIC_SELECTIONS',alive=True);status.write_text(json.dumps(state,indent=2))
deadline=time.time()+1800
while True:
 revision=api.model_info(TARGET).sha;files=set(api.list_repo_files(TARGET,revision=revision))
 if all(root(f)+'/provenance/VAL_COMPLETE.json' in files and root(f)+'/best.zip' in files for f in CONFIG):break
 assert time.time()<deadline,'Four public selected checkpoints not ready'
 time.sleep(30)
records={}
for family in CONFIG:
 receipt_path=Path(hf_hub_download(TARGET,root(family)+'/provenance/VAL_COMPLETE.json',revision=revision,token=api.token,local_dir=OUT/'public_metadata_cache'))
 val=json.loads(receipt_path.read_text());assert val['seed']==42 and val['family']==family and val['status']=='VALIDATION_COMPLETE'
 archive_path=root(family)+'/best.zip';entry=api.get_paths_info(TARGET,[archive_path],revision=revision)[0]
 assert entry.lfs and entry.lfs.sha256==val['archive_sha256'] and entry.size==val['archive_size_bytes']
 val.update(hf_repo=TARGET,hf_prefix=root(family),archive_path=archive_path)
 d=OUT/family/'seed42';d.mkdir(parents=True,exist_ok=True)
 (d/'VAL_COMPLETE.json').write_text(json.dumps(val,indent=2))
 collected={'repo':TARGET,'revision':revision,'verified':True,'files':{archive_path:{'sha256':entry.lfs.sha256,'size':entry.size}}}
 (d/'collection_receipt.json').write_text(json.dumps(collected,indent=2))
 records[val['run_name']]={'weight_digest':val['selected_and_exported_weight_digest'],'validation_receipt_sha256':worker.sha256(d/'VAL_COMPLETE.json'),'repo':TARGET,'revision':revision,'archive_sha256':entry.lfs.sha256}
lock={'status':'ALL_4_VALIDATION_VERIFIED','runs':records,'scope':'four seed42 selections; public canonical storage','test_protocol':'beam4 FP32 TF32-off BLEU13a chrF++ chrF0'}
(OUT/'selection_lock.json').write_text(json.dumps(lock,sort_keys=True,indent=2))
original_upload=worker.train.safe_hf_upload
def upload(api,path,repo,name,**kwargs):return original_upload(api,path,TARGET,mapped(name),**kwargs)
worker.train.safe_hf_upload=upload
original_folder=worker.upload_metadata_folder
def upload_folder(api,repo,folder,remote,required=False):return original_folder(api,TARGET,folder,mapped(remote),required=required)
worker.upload_metadata_folder=upload_folder
original_verify=worker.verify_remote_files
def verify(api,repo,files):return original_verify(api,TARGET,{mapped(k):v for k,v in files.items()})
worker.verify_remote_files=verify
for config in worker.FAMILY_CONFIGS.values():config['repo']=TARGET
assigned=('baseline','full') if a.account=='1' else ('middle','lsr')
try:
 for family in assigned:
  state.update(status='TESTING',current={'family':family,'seed':42});status.write_text(json.dumps(state,indent=2))
  worker.run_worker(family,42,'test')
 state.update(status='TEST_QUEUE_COMPLETE',completed_utc_epoch=time.time());status.write_text(json.dumps(state,indent=2))
except Exception as e:
 state.update(status='FAILED',error_type=type(e).__name__,error=str(e));status.write_text(json.dumps(state,indent=2));raise
print('PUBLIC_TEST_QUEUE_COMPLETE',a.account,flush=True)
