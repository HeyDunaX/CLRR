import json
from pathlib import Path
import pandas as pd
from sacrebleu.metrics import BLEU,CHRF
root=Path(__file__).resolve().parents[2]
base=root/'outputs_rebuttal/amis_lsr_l030_validation_20261005'
raw=pd.read_csv(root/'data_processed/amis_mandarin/validation.csv',dtype=str,keep_default_na=False)
decision=json.loads((base/'lsr_grid_decision.json').read_text())
for seed in (42,43,44):
    folder=base/'runs'/f'mbart-amis-lsr_l030-seed{seed}-corrected-validation'
    pred=pd.read_csv(folder/'validation_predictions.csv',dtype=str,keep_default_na=False)
    assert pred[['source','target']].equals(raw[['source','target']])
    row=decision['grid']['0.30']['per_seed'][str(seed)]
    scores={name:metric.corpus_score(pred.prediction.str.strip().tolist(),[raw.target.str.strip().tolist()]).score for name,metric in [('bleu',BLEU(tokenize='zh')),('chrfpp',CHRF(word_order=2)),('chrf0',CHRF(word_order=0))]}
    assert abs(scores['bleu']-row['bleu'])<1e-8 and abs(scores['chrfpp']-row['chrfpp'])<1e-8
    print(seed,'RAW_REPLAY_PASS',scores)
bench=json.loads((root/'outputs_rebuttal/amis_cost_benchmark_20261005/cost_benchmark_results.json').read_text())
lsr=next(r for r in bench if r['family']=='lsr'); full=next(r for r in bench if r['family']=='full')
print('FULL_VS_LSR_TRAIN_OVERHEAD_PCT',100*(full['train_step_ms']/lsr['train_step_ms']-1))
