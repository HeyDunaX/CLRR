"""Retain the existing 20k sample and restore official OPUS-100 evaluation splits."""
from pathlib import Path
import csv, hashlib, json
import pandas as pd
from huggingface_hub import hf_hub_download
ROOT=Path(__file__).resolve().parents[1]
REVISION='805090dc28bf78897da9641cdf08b61287580df9'
HASHES={'train':'a1a2979fe986891682913dba5da0888323a5d21b3b3249f69c373c9fb3a3b5ae','validation':'0c52bc5106b2e4f84ceffd79a15e913f8d4fdfb3eacc5377804c6e668f243a48','test':'a512f2d6fa5a215eeeda862fff9909b4d5b048219055f11bdef0892f76d8b143'}
def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    out=ROOT/'data_processed/opus100_turkish_english_20k'; train_path=out/'train.csv'
    if not train_path.exists(): raise FileNotFoundError('Retained 20k training CSV is required; no silent resampling.')
    train_hash=digest(train_path)
    with train_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f); assert reader.fieldnames==['source','target']; train=[(r['source'],r['target']) for r in reader]
    assert len(train)==20000
    wanted={(s.strip(),t.strip()) for s,t in train}; ids={}; official={}
    for split in ('train','validation','test'):
        path=Path(hf_hub_download(repo_id='Helsinki-NLP/opus-100',repo_type='dataset',filename=f'en-tr/{split}-00000-of-00001.parquet',revision=REVISION,token=False))
        assert digest(path)==HASHES[split],f'Upstream hash mismatch: {split}'
        data=pd.read_parquet(path)['translation']
        if split=='train':
            first={}
            for i,r in enumerate(data):
                pair=(r['tr'].strip(),r['en'].strip())
                if pair in wanted and pair not in first: first[pair]=i
            assert wanted==set(first),'Retained training pairs do not match pinned official train'
            ids[split]=[first[(s.strip(),t.strip())] for s,t in train]
        else:
            official[split]=[(r['tr'],r['en']) for r in data]; assert len(official[split])==2000; ids[split]=list(range(2000))
        del data
    for split,pairs in official.items():
        with (out/f'{split}.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f,lineterminator='\n'); writer.writerow(['source','target']); writer.writerows(pairs)
    assert digest(train_path)==train_hash
    splits={'train':train,**official}; ids_path=out/'upstream_row_ids.json'; ids_path.write_text(json.dumps(ids,separators=(',',':')),encoding='utf-8')
    records={s:{'rows':len(p),'unique_pairs':len(set(p)),'duplicate_pair_excess':len(p)-len(set(p)),'sha256':digest(out/f'{s}.csv')} for s,p in splits.items()}
    sources={s:{x.strip() for x,_ in p} for s,p in splits.items()}
    overlap={f'{a}/{b}':len(sources[a]&sources[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    manifest={'dataset':out.name,'source_language':'tr','target_language':'en','upstream_repo':'Helsinki-NLP/opus-100','upstream_revision':REVISION,'upstream_parquet_sha256':HASHES,'sampling':'Previously retained seed42 20000-pair training subset, unchanged bytes/order; membership verified against pinned official train. CSV/row IDs reproduce the sample; original RNG implementation not asserted.','validation':'All 2000 official rows, exact strings and original order; no filtering/deduplication.','test':'All 2000 official rows, exact strings and original order; no filtering/deduplication.','cross_split_stripped_source_overlap':overlap,'overlap_policy':'Audit only; retain official validation/test regardless of overlap.','upstream_row_ids_sha256':digest(ids_path),'splits':records,'scope':'OPUS-100 official evaluation splits with simulated low-resource 20k training subset, not full 1M training benchmark; no pretraining-disjointness claim.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (out/'README.md').write_text('# OPUS-100 Turkish → English — 20k training subset\n\nOfficial validation/test: 2000 rows each, exact strings/order, duplicates retained. No evaluation filtering or deduplication. Existing 20k training sample unchanged and verified against pinned official train.\n\nCSV schema: `source,target`. Manifest records revision, hashes, upstream row IDs and overlap audit. This is a low-resource subset experiment, not the full 1M benchmark.\n\nRun `conda run -n clrr python scripts/prepare_turkish_followup.py` to verify the sample and restore official evaluation rows. Preparation does not launch training.\n',encoding='utf-8')
    print(json.dumps({'status':'OFFICIAL_EVALUATION_VERIFIED','train_unchanged':True,'splits':records,'source_overlap_audit':overlap},indent=2))
if __name__=='__main__': main()
