"""Train LSR-only with lambda=0.30 on validation-only to compare against lambda=0.03 and 0.10.
Runs seeds 42, 43, 44 under exact corrected accumulation and fairness rules.
Test set remains strictly blind; selection is purely based on validation raw chrF++.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data_processed/amis_mandarin'
OUTPUT_DIR = ROOT / 'outputs_rebuttal/amis_lsr_l030_validation_20261005/runs'
BACKUP_DIR = ROOT / 'outputs_rebuttal/amis_lsr_l030_validation_20261005/backups'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def train_lsr_030_seed(seed):
    run_name = f'mbart-amis-lsr_l030-seed{seed}-corrected-validation'
    cmd = [
        sys.executable, '-u', '-m', 'amis_rewire.train',
        '--model', 'mbart-large-50',
        '--model-name', 'facebook/mbart-large-50-many-to-many-mmt',
        '--model-revision', 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30',
        '--method', 'jepa',
        '--rewire-stack', 'encoder',
        '--rewire-distance', '2',
        '--rewire-strength', '0.0',
        '--jepa-weight', '0.30',
        '--routing-warmup-ratio', '0.0',
        '--data-dir', str(DATA_DIR),
        '--output-dir', str(OUTPUT_DIR),
        '--backup-dir', str(BACKUP_DIR),
        '--run-name', run_name,
        '--seed', str(seed),
        '--learning-rate', '5e-5',
        '--weight-decay', '0.0',
        '--warmup-ratio', '0.06',
        '--num-train-epochs', '20',
        '--early-stopping-patience', '4',
        '--per-device-train-batch-size', '32',
        '--gradient-accumulation-steps', '4',
        '--per-device-eval-batch-size', '8',
        '--max-source-length', '256',
        '--max-target-length', '256',
        '--src-lang', 'tl_XX',
        '--tgt-lang', 'zh_CN',
        '--bleu-tokenizer', 'zh',
        '--eval-beams', '1',
        '--num-beams', '4',
        '--generation-precision', 'fp32',
        '--bf16',
        '--no-fp16',
        '--gradient-checkpointing',
        '--save-total-limit', '2',
        '--dataloader-num-workers', '2',
        '--logging-steps', '10',
        '--validation-only',
        '--scientific-records',
        '--hf-backup-repo', 'FiveC/amis-rewire-checkpoints',
        '--hf-backup-prefix', 'amis_lsr_l030_validation_20261005'
    ]
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1')
    print(f"\n=======================================================")
    print(f">>> STARTING LSR lambda=0.30 Training | Seed {seed}")
    print(f"=======================================================")
    res = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Training failed for seed {seed} with returncode {res.returncode}")
    print(f">>> COMPLETED LSR lambda=0.30 Training | Seed {seed}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    args = p.parse_args()
    train_lsr_030_seed(args.seed)
