"""Create a pinned, disjoint OPUS-100 Turkish-English 20k dataset in a new folder."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parents[1]
REVISION='805090dc28bf78897da9641cdf08b61287580df9'
HASHES={'train':'a1a2979fe986891682913dba5da0888323a5d21b3b3249f69c373c9fb3a3b5ae',
        'validation':'0c52bc5106b2e4f84ceffd79a15e913f8d4fdfb3eacc5377804c6e668f243a48',
        'test':'a512f2d6fa5a215eeeda862fff9909b4d5b048219055f11bdef0892f76d8b143'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=ROOT/'data_processed/opus100_turkish_english_20k_v2'
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text())
        for split,record in manifest['splits'].items():
            assert digest(out/(split+'.csv'))==record['sha256']
        print('EXISTING_DATASET_VERIFIED',flush=True)
        return
    upstream={}
    for split in ['test','validation','train']:
        path=Path(hf_hub_download(repo_id='Helsinki-NLP/opus-100',repo_type='dataset',
                 filename=f'en-tr/{split}-00000-of-00001.parquet',revision=REVISION,token=False))
        assert digest(path)==HASHES[split]
        translations=pd.read_parquet(path)['translation'].tolist()
        upstream[split]=[(str(row['tr']).strip(),str(row['en']).strip()) for row in translations]
        print('UPSTREAM',split,len(translations),flush=True)
    test=upstream['test']
    assert len(test)==2000 and all(s and t for s,t in test)
    forbidden={s for s,_ in test}
    validation=[];seen=set();validation_ids=[]
    for index,pair in enumerate(upstream['validation']):
        if not all(pair) or pair in seen or pair[0] in forbidden:
            continue
        seen.add(pair);validation.append(pair);validation_ids.append(index)
    forbidden.update(s for s,_ in upstream['validation'])
    train=[];train_ids=[];seen=set()
    # Fresh deterministic sample; do not reuse the previous unpinned sampling.
    for index in np.random.default_rng(42).permutation(len(upstream['train'])):
        pair=upstream['train'][int(index)]
        if not all(pair) or pair in seen or pair[0] in forbidden:
            continue
        seen.add(pair);train.append(pair);train_ids.append(int(index))
        if len(train)==20000:
            break
    assert len(train)==len(set(train))==20000
    splits={'train':train,'validation':validation,'test':test}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not ({s for s,_ in splits[a]} & {s for s,_ in splits[b]})
    out.mkdir(parents=True,exist_ok=True)
    records={}
    for split,rows in splits.items():
        path=out/(split+'.csv')
        with path.open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f,lineterminator='\n');writer.writerow(['source','target']);writer.writerows(rows)
        records[split]={'rows':len(rows),'unique_pairs':len(set(rows)),
                        'duplicate_pair_excess':len(rows)-len(set(rows)),'sha256':digest(path)}
    selected={'train':train_ids,'validation':validation_ids,'test':list(range(len(test)))}
    ids_path=out/'upstream_row_ids.json'
    ids_path.write_text(json.dumps(selected,separators=(',',':')),encoding='utf-8')
    manifest={'dataset':out.name,'source_language':'tr','target_language':'en',
              'upstream_repo':'Helsinki-NLP/opus-100','upstream_revision':REVISION,'upstream_parquet_sha256':HASHES,
              'sampling':'NumPy default_rng(42).permutation official train rows; first 20000 eligible unique stripped pairs',
              'train_exclusions':'empty strings, duplicate pairs, source present in official validation or test',
              'validation':'pair deduplicated; empty or test-source-overlapping rows removed; upstream order retained',
              'test':'official 2000 rows and order retained, including duplicates',
              'cross_split_source_overlap':{'train/validation':0,'train/test':0,'validation/test':0},
              'upstream_row_ids_sha256':digest(ids_path),'numpy_version':np.__version__,'splits':records,
              'scope':'simulated low-resource fine-tuning, 20000 pairs; no claim of pretraining disjointness'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    main()
