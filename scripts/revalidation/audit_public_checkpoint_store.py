"""Final read-only audit after source deletion, including anonymous downloads."""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0, str(ROOT / 'src'))
from experiment_runner.colab import get_hf_token
from huggingface_hub import HfApi,get_hf_file_metadata,hf_hub_url
OUT=ROOT/'outputs_rebuttal/hf_checkpoint_consolidation';repo='FiveC/amis-rewire-checkpoints'
api=HfApi(token=get_hf_token());info=api.model_info(repo);assert not info.private
mapping=json.loads((OUT/'path_mapping.json').read_text());receipts=json.loads((OUT/'migration_receipts.json').read_text())
entries={e.path:e for e in api.list_repo_tree(repo,recursive=True,revision=info.sha) if hasattr(e,'size')}
for m in mapping:
 assert not re.search(r'20\d{6}',m['destination'])
 e=entries[m['destination']];assert e.size==m['size'] and (e.lfs.sha256 if e.lfs else e.blob_id)==(m['lfs_sha256'] or m['blob_id'])
deleted=[]
for r in receipts:
 assert r['verified'] and r['source_deleted']
 try:api.model_info(r['source_repo'])
 except Exception as e:
  assert getattr(getattr(e,'response',None),'status_code',None)==404;deleted.append(r['source_repo'])
 else:raise AssertionError('Source still exists: '+r['source_repo'])
archives=[p for p in entries if p.startswith('checkpoints/') and p.endswith('/best.zip')]
assert len(archives)==44,len(archives)
for key in ('/baseline/','/full/'):
 p=next(p for p in archives if 'opus100_turkish_english_20k' in p and key in p)
 header=get_hf_file_metadata(hf_hub_url(repo,p,revision=info.sha),token=False)
 assert header.size==entries[p].size and header.etag==entries[p].lfs.sha256
 print('ANONYMOUS_CHECKPOINT_ACCESS_VERIFIED',p,flush=True)
result={'status':'COMPLETE_VERIFIED','target':repo,'public':True,'target_revision':info.sha,'migrated_files':len(mapping),'consolidated_archives':len(archives),'deleted_sources':deleted,'verification':'All mapped sizes/Git blob IDs/LFS hashes after deletion; two representative anonymous archive HEAD checks; new paths contain no dates'}
(OUT/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
