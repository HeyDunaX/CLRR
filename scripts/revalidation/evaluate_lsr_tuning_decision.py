"""Evaluate validation metrics across LSR lambda in {0.03, 0.10, 0.30} and lock the tuned LSR baseline.
Never touches test predictions. Selects best observed lambda by validation raw chrF++.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DECISION_PATH = ROOT / 'outputs_rebuttal/amis_corrected_controls_20261005/lsr_selection_decision.json'
L030_DIR = ROOT / 'outputs_rebuttal/amis_lsr_l030_validation_20261005'
OUT_DECISION = L030_DIR / 'lsr_grid_decision.json'

def evaluate_decision():
    old_decision = json.loads(DECISION_PATH.read_text())
    
    # Read lambda=0.30 runs
    l030_runs = list((L030_DIR / 'runs').glob('mbart-amis-lsr_l030-seed*-corrected-validation'))
    scores_030 = {}
    
    for r in l030_runs:
        m_file = r / 'metrics.json'
        if not m_file.exists():
            continue
        m = json.loads(m_file.read_text())
        seed_str = [p for p in r.name.split('-') if p.startswith('seed')][0].replace('seed', '')
        seed = int(seed_str)
        # validation metrics recorded
        scores_030[seed] = {
            'bleu': m.get('eval_bleu', m.get('best_metric_bleu')),
            'chrfpp': m.get('eval_chrf++', m.get('best_metric')),
            'chrf0': m.get('eval_chrf0'),
            'best_step': m.get('best_step')
        }
        
    print(f"Loaded lambda=0.30 scores for seeds: {list(scores_030.keys())}")
    
    # Grid comparison
    mean_chrfpp_003 = old_decision.get('lambda003_mean', old_decision.get('lambda_0.03', {}).get('mean_chrfpp'))
    mean_chrfpp_010 = old_decision.get('lambda010_mean', old_decision.get('lambda_0.10', {}).get('mean_chrfpp'))
    
    seeds_003 = old_decision.get('lambda003_scores', old_decision.get('lambda_0.03', {}).get('per_seed'))
    seeds_010 = old_decision.get('lambda010_scores', old_decision.get('lambda_0.10', {}).get('per_seed'))
    
    if len(scores_030) == 3:
        mean_chrfpp_030 = float(np.mean([scores_030[s]['chrfpp'] for s in (42, 43, 44)]))
        mean_bleu_030 = float(np.mean([scores_030[s]['bleu'] for s in (42, 43, 44)]))
    else:
        mean_chrfpp_030 = None
        mean_bleu_030 = None
        
    grid = {
        '0.03': {'mean_chrfpp': mean_chrfpp_003, 'per_seed': seeds_003},
        '0.10': {'mean_chrfpp': mean_chrfpp_010, 'per_seed': seeds_010},
        '0.30': {'mean_chrfpp': mean_chrfpp_030, 'mean_bleu': mean_bleu_030, 'per_seed': scores_030}
    }
    
    best_lambda = '0.10'
    best_val = mean_chrfpp_010
    if mean_chrfpp_030 is not None and mean_chrfpp_030 > best_val:
        best_lambda = '0.30'
        best_val = mean_chrfpp_030
        
    decision = {
        'status': 'GRID_SELECTION_COMPLETE' if len(scores_030) == 3 else 'PARTIAL_RUNS',
        'grid': grid,
        'selected_best_lambda': float(best_lambda),
        'best_mean_validation_chrfpp': best_val,
        'requires_distinct_tuned_lsr_family': (best_lambda != '0.10'),
        'confirmation_budget_recommendation': 40 if best_lambda != '0.10' else 32,
        'policy': 'Locked on validation raw chrF++ before opening any test predictions'
    }
    
    OUT_DECISION.write_text(json.dumps(decision, indent=2))
    print(f"\nDecision Summary:")
    print(f"  lambda=0.03: {mean_chrfpp_003:.4f}")
    print(f"  lambda=0.10: {mean_chrfpp_010:.4f}")
    if mean_chrfpp_030 is not None:
        print(f"  lambda=0.30: {mean_chrfpp_030:.4f}")
    print(f"Selected Best: lambda={best_lambda} -> Confirmation runs: {decision['confirmation_budget_recommendation']}")
    return decision

if __name__ == '__main__':
    evaluate_decision()
