"""Orchestration script for running Strong LoRA Suite on NLLB-200.

Executes sequential training for NLLB-200 PEFT baselines:
1. Run A: NLLB-200 + LoRA All-Linear (r=16, alpha=32, target: q,k,v,o,fc1,fc2, frozen embeddings, lr=2e-4)
2. Run B: NLLB-200 + LoRA All-Linear + Unfrozen Embeddings (modules_to_save: shared, embed_tokens, lm_head, lr=1e-4)

Consolidates all metrics into outputs_rebuttal/strong_lora_nllb_scores.csv.
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
    parser.add_argument("--data-dir", default="data_processed/amis_mandarin")
    parser.add_argument("--output-dir", default="results/nllb-200")
    parser.add_argument("--model-name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--num-train-epochs", type=str, default="20")
    parser.add_argument("--early-stopping-patience", type=str, default="4")
    parser.add_argument("--per-device-train-batch-size", type=str, default="16")
    parser.add_argument("--gradient-accumulation-steps", type=str, default="8")
    parser.add_argument("--src-lang", default=None)
    parser.add_argument("--tgt-lang", default=None)
    parser.add_argument("--bleu-tokenizer", default=None)
    parser.add_argument("--hf-backup-repo", default="FiveC/amis-rewire-checkpoints")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rebuttal_dir = repo_root / "outputs_rebuttal"
    rebuttal_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "method": "strong_lora_a",
            "name": "nllb-200-lora-all-linear",
            "lr": "2e-4",
            "desc": "Strong LoRA Run A on NLLB: All-Linear (r=16, alpha=32, q,k,v,o,fc1,fc2, frozen embeddings)",
        },
        {
            "method": "strong_lora_b",
            "name": "nllb-200-lora-all-linear-unfreeze-embed",
            "lr": "1e-4",
            "desc": "Strong LoRA Run B on NLLB: All-Linear + Unfrozen Embeddings (shared, lm_head)",
        },
    ]

    for exp in experiments:
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

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.nllb_suite.train_nllb",
            "--method",
            exp["method"],
            "--model-name",
            args.model_name,
            "--data-dir",
            args.data_dir,
            "--output-dir",
            args.output_dir,
            "--run-name",
            run_name,
            "--learning-rate",
            exp["lr"],
            "--num-train-epochs",
            args.num_train_epochs,
            "--early-stopping-patience",
            args.early_stopping_patience,
            "--per-device-train-batch-size",
            args.per_device_train_batch_size,
            "--gradient-accumulation-steps",
            args.gradient_accumulation_steps,
            "--hf-backup-repo",
            args.hf_backup_repo,
        ]
        if args.src_lang:
            cmd.extend(["--src-lang", args.src_lang])
        if args.tgt_lang:
            cmd.extend(["--tgt-lang", args.tgt_lang])
        if args.bleu_tokenizer:
            cmd.extend(["--bleu-tokenizer", args.bleu_tokenizer])

        run_command(cmd, cwd=repo_root)

    print("\n[done] All NLLB Strong LoRA experiments finished successfully.", flush=True)


if __name__ == "__main__":
    main()
