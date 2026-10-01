"""Orchestration script for running PEFT Baselines (LoRA & BitFit) on mBART-50.

Executes sequential training:
1. mBART-large-50 + LoRA (Hu et al., ICLR 2022) with r=8, alpha=16, lr=2e-4
2. mBART-large-50 + BitFit (Ben-Zaken et al., ACL 2022) with bias tuning, lr=1e-4

Consolidates all metrics into results/peft_scores.csv.
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
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model-name", default="facebook/mbart-large-50-many-to-many-mmt")
    parser.add_argument("--num-train-epochs", type=str, default="20")
    parser.add_argument("--early-stopping-patience", type=str, default="4")
    parser.add_argument("--per-device-train-batch-size", type=str, default="4")
    parser.add_argument("--gradient-accumulation-steps", type=str, default="32")
    parser.add_argument("--hf-backup-repo", default="FiveC/amis-rewire-checkpoints")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "method": "lora",
            "name": "mbart-large-50-lora",
            "lr": "2e-4",
            "desc": "LoRA (Hu et al., ICLR 2022, r=8, alpha=16)",
            "extra_args": ["--lora-r", "8", "--lora-alpha", "16", "--lora-dropout", "0.05"],
        },
        {
            "method": "bitfit",
            "name": "mbart-large-50-bitfit",
            "lr": "1e-4",
            "desc": "BitFit (Ben-Zaken et al., ACL 2022, all bias vectors)",
            "extra_args": [],
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
            "peft_baselines.train_peft",
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
            "--auto-resume",
        ] + exp["extra_args"]

        run_command(cmd, cwd=repo_root)

    # Consolidate results
    print("\n=======================================================", flush=True)
    print("Consolidating PEFT Comparison Table...", flush=True)
    print("=======================================================", flush=True)

    summary_rows = [
        # Reference baselines from paper
        {
            "Model": "facebook/mbart-large-50",
            "Method": "Standard Fine-Tuning",
            "Reference": "Official Baseline",
            "Trainable Params": "610,879,488 (100%)",
            "BLEU (zh)": 19.6106,
            "chrF++ (w=2)": 14.0538,
        },
        {
            "Model": "facebook/mbart-large-50",
            "Method": "Middle-Layer Alignment",
            "Reference": "Liu & Niehues (ACL 2025)",
            "Trainable Params": "610,879,488 (100%)",
            "BLEU (zh)": 19.7243,
            "chrF++ (w=2)": 15.5253,
        },
    ]

    for exp in experiments:
        run_name = exp["name"]
        metrics_file = out_dir / run_name / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            t_params = data.get("trainable_parameters", 0)
            all_params = data.get("total_parameters", 0)
            t_pct = data.get("trainable_percent", 0.0)
            summary_rows.append({
                "Model": "facebook/mbart-large-50",
                "Method": f"{exp['method'].upper()}",
                "Reference": data.get("reference_paper", ""),
                "Trainable Params": f"{t_params:,} ({t_pct:.4f}%)",
                "BLEU (zh)": data.get("test_bleu", 0.0),
                "chrF++ (w=2)": data.get("test_chrf++", 0.0),
            })

    # Add CLRR reference
    summary_rows.append({
        "Model": "facebook/mbart-large-50",
        "Method": "CLRR-Enc + LSR (Ours)",
        "Reference": "Proposed (Delta theta = 0)",
        "Trainable Params": "610,879,488 (Zero New Params)",
        "BLEU (zh)": 20.3927,
        "chrF++ (w=2)": 19.0839,
    })

    df = pd.DataFrame(summary_rows)
    csv_path = out_dir / "peft_scores.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nFinal Comparative Matrix saved to: {csv_path}\n")
    print(df.to_string(index=False), flush=True)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and args.hf_backup_repo and csv_path.exists():
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=str(csv_path),
                path_in_repo="peft_baselines/peft_scores.csv",
                repo_id=args.hf_backup_repo,
                repo_type="model",
            )
            print(f"Uploaded peft_scores.csv to {args.hf_backup_repo} successfully!", flush=True)
        except Exception as e:
            print(f"HF upload note: {e}", flush=True)


if __name__ == "__main__":

    main()
