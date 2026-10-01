"""Orchestration script for Asháninka -> Spanish (AmericasNLP 2021) Experiments.

Executes sequential training across mBART-50 and NLLB-200 backbones:
1. mBART-large-50 Baseline (Full-FT, lr=5e-5, target: es_XX)
2. mBART-large-50 CLRR-Enc + LSR (d=2, alpha=0.1, lambda=0.1, lr=5e-5)
3. NLLB-200 Baseline (Full-FT, lr=5e-5, target: spa_Latn)
4. NLLB-200 CLRR-Enc + LSR (d=2, alpha=0.1, lambda=0.1, lr=5e-5)

All runs evaluate BLEU with tokenizer='13a' and chrF++ with word_order=2.
Results are consolidated into results/ashaninka_spanish_scores.csv.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd


def run_command(cmd: list[str], cwd: Path) -> None:
    print(f"\n[exec] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data_processed/ashaninka_spanish")
    parser.add_argument("--output-dir", default="results/ashaninka_spanish")
    parser.add_argument("--num-train-epochs", type=str, default="20")
    parser.add_argument("--early-stopping-patience", type=str, default="4")
    parser.add_argument("--hf-backup-repo", default="FiveC/amis-rewire-checkpoints")
    parser.add_argument("--runs", nargs="*", default=["mbart_baseline", "mbart_clrr", "nllb_baseline", "nllb_clrr"],
                        choices=["mbart_baseline", "mbart_clrr", "nllb_baseline", "nllb_clrr", "mbart_lora"],
                        help="Subset of runs to execute.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "id": "mbart_baseline",
            "name": "mbart-ashaninka-es-baseline",
            "desc": "mBART-large-50 Standard Full Fine-Tuning (Asháninka -> Spanish)",
            "cmd": [
                sys.executable, "-u", "-m", "amis_rewire.train",
                "--model", "mbart",
                "--method", "baseline",
                "--model-name", "facebook/mbart-large-50-many-to-many-mmt",
                "--data-dir", args.data_dir,
                "--output-dir", args.output_dir,
                "--run-name", "mbart-ashaninka-es-baseline",
                "--learning-rate", "5e-5",
                "--num-train-epochs", args.num_train_epochs,
                "--early-stopping-patience", args.early_stopping_patience,
                "--per-device-train-batch-size", "4",
                "--gradient-accumulation-steps", "32",
                "--src-lang", "es_XX",
                "--tgt-lang", "es_XX",
                "--bleu-tokenizer", "13a",
                "--hf-backup-repo", args.hf_backup_repo,
                "--hf-backup-prefix", "ashaninka-es",
            ],
        },
        {
            "id": "mbart_clrr",
            "name": "mbart-ashaninka-es-clrr-enc",
            "desc": "mBART-large-50 CLRR-Enc + LSR (d=2, alpha=0.1, lambda=0.1, Asháninka -> Spanish)",
            "cmd": [
                sys.executable, "-u", "-m", "amis_rewire.train",
                "--model", "mbart",
                "--method", "jepa-clrr-enc",
                "--model-name", "facebook/mbart-large-50-many-to-many-mmt",
                "--data-dir", args.data_dir,
                "--output-dir", args.output_dir,
                "--run-name", "mbart-ashaninka-es-clrr-enc",
                "--rewire-distance", "2",
                "--rewire-strength", "0.1",
                "--rewire-stack", "encoder",
                "--jepa-weight", "0.1",
                "--learning-rate", "5e-5",
                "--num-train-epochs", args.num_train_epochs,
                "--early-stopping-patience", args.early_stopping_patience,
                "--per-device-train-batch-size", "4",
                "--gradient-accumulation-steps", "32",
                "--src-lang", "es_XX",
                "--tgt-lang", "es_XX",
                "--bleu-tokenizer", "13a",
                "--hf-backup-repo", args.hf_backup_repo,
                "--hf-backup-prefix", "ashaninka-es",
            ],
        },
        {
            "id": "nllb_baseline",
            "name": "nllb-ashaninka-es-baseline",
            "desc": "NLLB-200 Standard Full Fine-Tuning (Asháninka -> Spanish)",
            "cmd": [
                sys.executable, "-u", "-m", "src.nllb_suite.train_nllb",
                "--method", "baseline",
                "--model-name", "facebook/nllb-200-distilled-600M",
                "--data-dir", args.data_dir,
                "--output-dir", args.output_dir,
                "--run-name", "nllb-ashaninka-es-baseline",
                "--learning-rate", "5e-5",
                "--num-train-epochs", args.num_train_epochs,
                "--early-stopping-patience", args.early_stopping_patience,
                "--per-device-train-batch-size", "16",
                "--gradient-accumulation-steps", "8",
                "--src-lang", "spa_Latn",
                "--tgt-lang", "spa_Latn",
                "--bleu-tokenizer", "13a",
                "--hf-backup-repo", args.hf_backup_repo,
                "--hf-backup-prefix", "ashaninka-es",
            ],
        },
        {
            "id": "nllb_clrr",
            "name": "nllb-ashaninka-es-clrr-enc",
            "desc": "NLLB-200 CLRR-Enc + LSR (d=2, alpha=0.1, lambda=0.1, Asháninka -> Spanish)",
            "cmd": [
                sys.executable, "-u", "-m", "src.nllb_suite.train_nllb",
                "--method", "clrr_enc",
                "--model-name", "facebook/nllb-200-distilled-600M",
                "--data-dir", args.data_dir,
                "--output-dir", args.output_dir,
                "--run-name", "nllb-ashaninka-es-clrr-enc",
                "--learning-rate", "5e-5",
                "--num-train-epochs", args.num_train_epochs,
                "--early-stopping-patience", args.early_stopping_patience,
                "--per-device-train-batch-size", "16",
                "--gradient-accumulation-steps", "8",
                "--src-lang", "spa_Latn",
                "--tgt-lang", "spa_Latn",
                "--bleu-tokenizer", "13a",
                "--hf-backup-repo", args.hf_backup_repo,
                "--hf-backup-prefix", "ashaninka-es",
            ],
        },
    ]

    for exp in experiments:
        if exp["id"] not in args.runs:
            continue
        run_name = exp["name"]
        run_dir = out_dir / run_name
        metrics_file = run_dir / "metrics.json"

        print(f"\n=======================================================", flush=True)
        print(f"PIPELINE STEP: {exp['desc']}", flush=True)
        print(f"Run directory: {run_dir}", flush=True)
        print(f"=======================================================", flush=True)

        if metrics_file.exists():
            print(f"[skip] {run_name} already completed with {metrics_file}. Skipping.", flush=True)
            continue

        run_command(exp["cmd"], cwd=repo_root)

    print("\n[done] All requested Asháninka -> Spanish runs finished.", flush=True)


if __name__ == "__main__":
    main()
