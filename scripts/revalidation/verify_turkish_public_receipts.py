"""Verify final local test artifacts against one public pinned revision."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0, str(ROOT / 'src'));sys.path.insert(0,str(ROOT/'src'))
from experiment_runner.colab import get_hf_token
from huggingface_hub import HfApi
import pandas as pd
from amis_rewire.metrics import generation_metrics
from sacrebleu.metrics import CHRF
OUT=ROOT/'outputs_rebuttal/turkish_old_method_pilot_es10_20261007'
rows=json.loads((OUT/'public_test_results.json').read_text());assert len(rows)==4
api=HfApi(token=get_hf_token());raw=pd.read_csv(ROOT/'data_processed/opus100_turkish_english_20k/test.csv',dtype=str,keep_default_na=False)
assert len(raw)==2000
for row in rows:
 f=row['family'];d=OUT/f/'seed42';pred=pd.read_csv(d/'test_predictions.csv',dtype=str,keep_default_na=False)
 assert pred[['source','target']].equals(raw[['source','target']])
 scores=generation_metrics(pred.prediction.tolist(),raw.target.tolist(),tokenize='13a')
 scores['chrf0']=CHRF(word_order=0).corpus_score(pred.prediction.tolist(),[raw.target.str.strip().tolist()]).score
 for name,key in [('bleu','test_bleu'),('chrf++','test_chrfpp'),('chrf0','test_chrf0')]:assert abs(scores[name]-row[key])<1e-8
 val=json.loads((d/'COMPLETE.json').read_text());archive=row['prefix']+'/best.zip'
 entry=api.get_paths_info(row['repo'],[archive],revision=row['revision'])[0]
 assert entry.lfs.sha256==val['archive_sha256'] and entry.size==val['archive_size_bytes']
 files={archive:{'sha256':entry.lfs.sha256,'size':entry.size},**row['verified_files']}
 for remote,record in row['verified_files'].items():
  p=d/('COMPLETE.json' if remote.endswith('/COMPLETE.json') else Path(remote).name)
  assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
 receipt={'repo':row['repo'],'revision':row['revision'],'verified':True,'files':files,'verification':'Public LFS archive SHA256/size; downloaded test files Git/LFS hashes; recomputed all three metrics from original 2000 ordered test rows'}
 (d/'collection_receipt.json').write_text(json.dumps(receipt,indent=2))
 print('FINAL_RECEIPT_AND_METRICS_VERIFIED',f,flush=True)
baseline=next(r for r in rows if r['family']=='baseline');full=next(r for r in rows if r['family']=='full')
lines=['# Turkish→English — mBART-50, pilot seed 42','','Official OPUS100 validation/test splits unchanged; 20,000 training pairs, 2,000 validation and 2,000 test pairs. All four selected checkpoints: epoch 2; training stopped at epoch 12 with patience 10. Test after four selections were locked: beam 4, FP32 generation, TF32 off, BLEU13a/native chrF++/chrF0. All test predictions and scores independently verified.','','| Method | Seed | Test BLEU | Test chrF++ | Test chrF0 | ΔBLEU vs Baseline |','|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['family']} | 42 | {r['test_bleu']:.6f} | {r['test_chrfpp']:.6f} | {r['test_chrf0']:.6f} | {r['test_bleu']-baseline['test_bleu']:+.6f} |")
lines+=['','## Full minus each comparator','','| Comparator | ΔBLEU | ΔchrF++ | ΔchrF0 |','|---|---:|---:|---:|']
for r in rows:
 if r['family']!='full':lines.append(f"| {r['family']} | {full['test_bleu']-r['test_bleu']:+.6f} | {full['test_chrfpp']-r['test_chrfpp']:+.6f} | {full['test_chrf0']-r['test_chrf0']:+.6f} |")
lines+=['','## Interpretation','','Full improves on LSR in all three metrics for this frozen seed. Full has almost the same chrF++ as Baseline, slightly lower BLEU, and slightly higher chrF0. The run does not show a clear Full advantage over Baseline. This is an exploratory one-seed result, not evidence of seed robustness or statistical significance. No method or checkpoint was tuned using these test results.','','Checkpoint repo: https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/opus100_turkish_english_20k .','Storage rules: docs/operations/HUGGINGFACE_CHECKPOINT_STORAGE.md .','Results: public_test_results.json; per-family final collection_receipt.json.']
(OUT/'TURKISH_PILOT_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('REPORT_WRITTEN')
