"""Verify Fresh Seed Confirmation results locally and compute statistical tests.
Replays all 32 models (4 families x 8 seeds = 32 runs).
Evaluates against pre-locked gates:
- Primary Gate 1: Full - LSR >= +0.30 BLEU, >= 6/8 wins, non-degradation of chrF++/chrF0.
- Secondary Gate 2: Full - FT >= +0.50 BLEU, >= 5/8 wins.
Calculates exact Permutation Sign-Flip test, Paired t-test, 95% CI, and Interaction delta.
"""
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys_path = ROOT / 'src'
import sys
sys.path.insert(0, str(sys_path))
from amis_rewire.metrics import generation_metrics
from sacrebleu.metrics import CHRF
from scipy.stats import t as student_t
import argparse
import hashlib

BASE_DIR = ROOT / 'outputs_rebuttal/amis_fresh_confirmation_20261005'
DATA_DIR = ROOT / 'data_processed/amis_mandarin'

def exact_permutation_sign_flip_p(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    obs_stat = abs(np.sum(diffs))
    total_assignments = 2 ** n
    count_as_or_more_extreme = 0
    for signs in itertools.product([-1, 1], repeat=n):
        s = np.array(signs)
        if abs(np.sum(s * diffs)) >= obs_stat - 1e-12:
            count_as_or_more_extreme += 1
    return count_as_or_more_extreme / total_assignments

def paired_t_test(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    sd_d = float(np.std(diffs, ddof=1))
    se_d = sd_d / np.sqrt(n)
    if n < 2 or not np.isfinite(diffs).all():
        raise ValueError('Need at least two finite paired differences')
    t_stat = mean_d / se_d if se_d > 0 else (0.0 if mean_d == 0 else math.copysign(float('inf'), mean_d))
    
    # Exact Student-t quantile for this sample size.
    t_crit = float(student_t.ppf(.975, n-1))
    ci_lower = mean_d - t_crit * se_d
    ci_upper = mean_d + t_crit * se_d
    
    # Student-t tail, not a normal approximation with only eight seeds.
    p_norm = float(2 * student_t.sf(abs(t_stat), n-1))
    return {
        'mean': mean_d,
        'std': sd_d,
        'se': se_d,
        't_stat': t_stat,
        'ci_95': [ci_lower, ci_upper],
        'p_value': p_norm,
        'df': n-1
    }

def verify_and_report(stage='test'):
    print("=== AUDITING FRESH SEED CONFIRMATION (32 RUNS) ===")
    raw_test = pd.read_csv(DATA_DIR / 'test.csv', dtype=str, keep_default_na=False)
    targets = raw_test['target'].tolist()
    assert len(targets) == 575
    
    families = ['baseline', 'clrr', 'lsr', 'full']
    seeds = range(45, 53)
    
    scores = {f: {} for f in families}
    missing = []
    
    for fam in families:
        for seed in seeds:
            seed_dir = BASE_DIR / fam / f"seed{seed}"
            receipt_path = seed_dir / ('COMPLETE.json' if stage == 'test' else 'VAL_COMPLETE.json')
            if not receipt_path.exists():
                missing.append(f"{fam} seed {seed}")
                continue
            receipt = json.loads(receipt_path.read_text())
            assert receipt['family'] == fam and receipt['seed'] == seed
            collection = json.loads((seed_dir/'collection_receipt.json').read_text())
            assert collection['verified'] and collection['files'] and collection['revision']
            manifest=json.loads((ROOT/'.cache/frozen_sources/fresh_confirmation_manifest.json').read_text())
            assert receipt['bundle_files']==manifest['files'], 'Run used a different source/data bundle'
            remote_receipt=f"{receipt['hf_prefix']}/{receipt['run_name']}/provenance/{receipt_path.name}"
            assert collection['files'][remote_receipt]['sha256']==hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            state_path=seed_dir/'final_trainer_state.json'
            remote_state=f"{receipt['hf_prefix']}/{receipt['run_name']}/provenance/final_trainer_state.json"
            assert collection['files'][remote_state]['sha256']==hashlib.sha256(state_path.read_bytes()).hexdigest()
            
            run_name = f"mbart-amis-{fam}-seed{seed}-fresh-confirmation"
            split = 'test' if stage == 'test' else 'validation'
            raw_split = pd.read_csv(DATA_DIR/(split+'.csv'), dtype=str, keep_default_na=False)
            preds_csv = seed_dir / 'runs' / run_name / (split+'_predictions.csv')
            if preds_csv.exists():
                df_preds = pd.read_csv(preds_csv, dtype=str, keep_default_na=False)
                assert df_preds[['source','target']].equals(raw_split[['source','target']]), f'Reference/order mismatch: {fam}/{seed}'
                prediction = df_preds['prediction'].str.strip().tolist()
                references = raw_split.target.str.strip().tolist()
                replayed = generation_metrics(prediction, references, tokenize='zh')
                replayed['chrf0'] = CHRF(word_order=0).corpus_score(prediction,[references]).score
                metadata_path = preds_csv.parent/('test_metrics.json' if stage=='test' else 'metrics.json')
                metadata = json.loads(metadata_path.read_text())
                prefix = 'test' if stage=='test' else 'eval'
                for key in ('bleu','chrf++'):
                    assert abs(replayed[key]-metadata[prefix+'_'+key]) < 1e-8
                remote_prediction = f"{receipt['hf_prefix']}/{run_name}/{preds_csv.name}"
                remote_metric = f"{receipt['hf_prefix']}/{run_name}/{metadata_path.name}"
                for remote, local in ((remote_prediction,preds_csv),(remote_metric,metadata_path)):
                    assert collection['files'][remote]['sha256'] == hashlib.sha256(local.read_bytes()).hexdigest()
                scores[fam][seed] = {
                    'bleu': replayed['bleu'],
                    'chrfpp': replayed['chrf++'],
                    'chrf0': replayed['chrf0']
                }
            else:
                raise FileNotFoundError(f'Cannot verify without predictions: {preds_csv}')
                
    if missing:
        print(f"Warning: Missing {len(missing)} of 32 runs: {missing}")
        raise RuntimeError('Incomplete cohort: '+', '.join(missing))

    if stage == 'validation':
        runs = {}
        for fam in families:
            for seed in seeds:
                folder = BASE_DIR/fam/f'seed{seed}'
                val = json.loads((folder/'VAL_COMPLETE.json').read_text())
                state = json.loads((folder/'final_trainer_state.json').read_text())
                best = max((r for r in state['log_history'] if 'eval_chrf++' in r), key=lambda r:r['eval_chrf++'])
                assert state['best_model_checkpoint'] == val['selected_checkpoint']
                assert best['step'] == int(val['selected_checkpoint'].rsplit('-',1)[-1])
                assert abs(scores[fam][seed]['chrfpp']-state['best_metric'])<1e-8
                runs[val['run_name']] = {'weight_digest':val['selected_and_exported_weight_digest'],
                                         'archive_sha256':val['archive_sha256'],
                                         'validation_receipt_sha256':hashlib.sha256((folder/'VAL_COMPLETE.json').read_bytes()).hexdigest()}
        lock = {'status':'ALL_32_VALIDATION_VERIFIED','runs':runs,'test_used_for_selection':False}
        path = BASE_DIR/'selection_lock.json'
        if path.exists(): assert json.loads(path.read_text())==lock, 'Existing selection lock differs'
        else: path.write_text(json.dumps(lock,indent=2))
        print('ALL_32_VALIDATION_VERIFIED; selection lock written')
        return
            
    # Compute contrasts
    bleu_deltas_lsr = []
    chrfpp_deltas_lsr = []
    chrf0_deltas_lsr = []
    
    bleu_deltas_ft = []
    chrfpp_deltas_ft = []
    chrf0_deltas_ft = []
    
    interactions = []
    
    print("\n--- Individual Seed Results (Test Beam 4) ---")
    for s in seeds:
        b_full = scores['full'][s]['bleu']
        b_lsr = scores['lsr'][s]['bleu']
        b_clrr = scores['clrr'][s]['bleu']
        b_base = scores['baseline'][s]['bleu']
        
        d_lsr = b_full - b_lsr
        d_ft = b_full - b_base
        inter = b_full - b_lsr - b_clrr + b_base
        
        bleu_deltas_lsr.append(d_lsr)
        chrfpp_deltas_lsr.append(scores['full'][s]['chrfpp'] - scores['lsr'][s]['chrfpp'])
        chrf0_deltas_lsr.append(scores['full'][s]['chrf0'] - scores['lsr'][s]['chrf0'])
        
        bleu_deltas_ft.append(d_ft)
        chrfpp_deltas_ft.append(scores['full'][s]['chrfpp'] - scores['baseline'][s]['chrfpp'])
        chrf0_deltas_ft.append(scores['full'][s]['chrf0'] - scores['baseline'][s]['chrf0'])
        
        interactions.append(inter)
        print(f"Seed {s}: Full={b_full:.2f}, LSR={b_lsr:.2f}, CLRR={b_clrr:.2f}, FT={b_base:.2f} | Full-LSR={d_lsr:+.2f}, Full-FT={d_ft:+.2f}, Inter={inter:+.2f}")
        
    t_lsr = paired_t_test(bleu_deltas_lsr)
    sf_p_lsr = exact_permutation_sign_flip_p(bleu_deltas_lsr)
    wins_lsr = sum(1 for d in bleu_deltas_lsr if d > 0)
    
    t_ft = paired_t_test(bleu_deltas_ft)
    sf_p_ft = exact_permutation_sign_flip_p(bleu_deltas_ft)
    wins_ft = sum(1 for d in bleu_deltas_ft if d > 0)
    
    # Gate Verdicts
    primary_pass = (t_lsr['mean'] >= 0.30) and (wins_lsr >= 6) and (np.mean(chrfpp_deltas_lsr) >= 0.0) and (np.mean(chrf0_deltas_lsr) >= 0.0)
    secondary_pass = (t_ft['mean'] >= 0.50) and (wins_ft >= 5)
    
    print("\n=======================================================")
    print(f"PRIMARY GATE 1 (Full vs LSR): {'PASS (GO)' if primary_pass else 'NO-GO'}")
    print(f"  Mean Delta: {t_lsr['mean']:+.4f} BLEU (Threshold: >= +0.30)")
    print(f"  Seed Wins: {wins_lsr}/8 (Investment threshold: >= 6/8)")
    print(f"  95% CI: [{t_lsr['ci_95'][0]:+.4f}, {t_lsr['ci_95'][1]:+.4f}]")
    print(f"  Exact Sign-Flip p-value: {sf_p_lsr:.5f}")
    print(f"  Mean chrF++ Delta: {np.mean(chrfpp_deltas_lsr):+.4f}")
    print(f"  Mean chrF0 Delta: {np.mean(chrf0_deltas_lsr):+.4f}")
    print("-------------------------------------------------------")
    print(f"SECONDARY GATE 2 (Full vs FT): {'PASS' if secondary_pass else 'DID NOT PASS'}")
    print(f"  Mean Delta: {t_ft['mean']:+.4f} BLEU (Threshold: >= +0.50)")
    print(f"  Seed Wins: {wins_ft}/8 (Threshold: >= 5/8)")
    print(f"  95% CI: [{t_ft['ci_95'][0]:+.4f}, {t_ft['ci_95'][1]:+.4f}]")
    print(f"  Exact Sign-Flip p-value: {sf_p_ft:.5f}")
    print("-------------------------------------------------------")
    print(f"INTERACTION (Full - LSR - CLRR + FT):")
    print(f"  Mean Delta: {np.mean(interactions):+.4f} BLEU")
    print("=======================================================")
    
    summary = {
        'cohort': 'fresh_seeds_45_to_52',
        'n_seeds': 8,
        'primary_gate': {
            'verdict': 'PASS' if primary_pass else 'NO-GO',
            'contrast': 'Full - LSR',
            'mean_bleu': t_lsr['mean'],
            'ci_95': t_lsr['ci_95'],
            'wins': f"{wins_lsr}/8",
            'exact_sign_flip_p': sf_p_lsr,
            'mean_chrfpp': float(np.mean(chrfpp_deltas_lsr)),
            'mean_chrf0': float(np.mean(chrf0_deltas_lsr))
        },
        'secondary_gate': {
            'verdict': 'PASS' if secondary_pass else 'DID NOT PASS',
            'contrast': 'Full - FT',
            'mean_bleu': t_ft['mean'],
            'ci_95': t_ft['ci_95'],
            'wins': f"{wins_ft}/8",
            'exact_sign_flip_p': sf_p_ft
        },
        'interaction': {
            'mean_bleu': float(np.mean(interactions)),
            'per_seed': interactions
        }
    }
    (BASE_DIR / 'confirmation_summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\nWritten {BASE_DIR / 'confirmation_summary.json'}")

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('validation','test'),default='test')
    verify_and_report(parser.parse_args().stage)
