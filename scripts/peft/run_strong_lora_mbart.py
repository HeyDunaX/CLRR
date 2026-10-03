"""Orchestration script for running Strong LoRA Suite on mBART-50.

Executes sequential training for Reviewer Rebuttal Priority 1:
1. Run A: mBART-large-50 + LoRA All-Linear (r=16, alpha=32, target: q,k,v,o,fc1,fc2, frozen embeddings, lr=2e-4)
2. Run B: mBART-large-50 + LoRA All-Linear + Unfrozen Embeddings (modules_to_save: shared, lm_head, lr=1e-4)

Consolidates all metrics into outputs_rebuttal/strong_lora_scores.csv and results/strong_lora_scores.csv.
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
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model-name", default="facebook/mbart-large-50-many-to-many-mmt")
    parser.add_argument("--num-train-epochs", type=str, default="20")
    parser.add_argument("--early-stopping-patience", type=str, default="4")
    parser.add_argument("--per-device-train-batch-size", type=str, default="4")
    parser.add_argument("--gradient-accumulation-steps", type=str, default="32")
    parser.add_argument("--hf-backup-repo", default="FiveC/amis-rewire-checkpoints")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from src.amis_rewire.metrics import score_prediction_csv, verified_reference_metrics
    reference_path = repo_root / args.data_dir / "test.csv"
    reference_scores = {method: verified_reference_metrics("mBART", method, reference_path)
                        for method in ("Baseline", "CLRR+LSR", "Middle-Layer Alignment", "BitFit", "Narrow LoRA")}
    out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rebuttal_dir = repo_root / "outputs_rebuttal"
    rebuttal_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "method": "lora",
            "name": "mbart-large-50-lora-all-linear",
            "lr": "2e-4",
            "desc": "Strong LoRA Run A: All-Linear (r=16, alpha=32, q,k,v,o,fc1,fc2, frozen embeddings)",
            "extra_args": [
                "--lora-r", "16",
                "--lora-alpha", "32",
                "--lora-dropout", "0.05",
                "--target-modules", "q_proj,k_proj,v_proj,out_proj,fc1,fc2",
            ],
        },
        {
            "method": "lora",
            "name": "mbart-large-50-lora-all-linear-unfreeze-embed",
            "lr": "1e-4",
            "desc": "Strong LoRA Run B: All-Linear + Unfrozen Embeddings (shared, lm_head)",
            "extra_args": [
                "--lora-r", "16",
                "--lora-alpha", "32",
                "--lora-dropout", "0.05",
                "--target-modules", "q_proj,k_proj,v_proj,out_proj,fc1,fc2",
                "--unfreeze-embeddings",
            ],
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
    print("Consolidating Strong LoRA Comparison Table...", flush=True)
    print("=======================================================", flush=True)

    summary_rows = [
        {
            "Backbone": "facebook/mbart-large-50",
            "Method": "Standard Fine-Tuning",
            "Target Modules": "Full Model",
            "Trainable Params": "610,879,488 (100%)",
            "BLEU (zh)": reference_scores["Baseline"]["bleu"],
            "chrF++ (w=2)": reference_scores["Baseline"]["chrf++"],
            "Note": "Official Full-Tuning Baseline",
        },
        {
            "Backbone": "facebook/mbart-large-50",
            "Method": "BitFit (Bias-only)",
            "Target Modules": "Bias vectors",
            "Trainable Params": "335,872 (0.0550%)",
            "BLEU (zh)": reference_scores["BitFit"]["bleu"],
            "chrF++ (w=2)": reference_scores["BitFit"]["chrf++"],
            "Note": "Ben-Zaken et al. (ACL 2022)",
        },
        {
            "Backbone": "facebook/mbart-large-50",
            "Method": "LoRA (r=8, narrow)",
            "Target Modules": "q_proj, v_proj",
            "Trainable Params": "1,179,648 (0.193%)",
            "BLEU (zh)": reference_scores["Narrow LoRA"]["bleu"],
            "chrF++ (w=2)": reference_scores["Narrow LoRA"]["chrf++"],
            "Note": "Hu et al. (ICLR 2022)",
        },
    ]

    for exp in experiments:
        run_name = exp["name"]
        metrics_file = out_dir / run_name / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            scores = score_prediction_csv(metrics_file.parent / "test_predictions.csv", reference_path)
            t_params = data.get("trainable_parameters", 0)
            t_pct = data.get("trainable_percent", 0.0)
            summary_rows.append({
                "Backbone": "facebook/mbart-large-50",
                "Method": run_name.replace("mbart-large-50-", ""),
                "Target Modules": "All-Linear" + (" + Embed" if "unfreeze" in run_name else ""),
                "Trainable Params": f"{t_params:,} ({t_pct:.4f}%)",
                "BLEU (zh)": scores["bleu"],
                "chrF++ (w=2)": scores["chrf++"],
                "Note": exp["desc"],
            })

    # Add audited CLRR reference
    summary_rows.append({
        "Backbone": "facebook/mbart-large-50",
        "Method": "CLRR-Enc + LSR (Ours)",
        "Target Modules": "Residual Rewiring + LSR",
        "Trainable Params": "610,879,488 (Zero New Params)",
        "BLEU (zh)": reference_scores["CLRR+LSR"]["bleu"],
        "chrF++ (w=2)": reference_scores["CLRR+LSR"]["chrf++"],
        "Note": "Proposed (Delta theta = 0)",
    })

    df = pd.DataFrame(summary_rows)
    for p in [out_dir / "strong_lora_scores.csv", rebuttal_dir / "strong_lora_scores.csv"]:
        df.to_csv(p, index=False, encoding="utf-8")
        print(f"Comparative Matrix saved to: {p}")

    print("\n" + df.to_string(index=False), flush=True)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and args.hf_backup_repo:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=str(rebuttal_dir / "strong_lora_scores.csv"),
                path_in_repo="peft_baselines/strong_lora_scores.csv",
                repo_id=args.hf_backup_repo,
                repo_type="model",
            )
            print(f"Uploaded strong_lora_scores.csv to {args.hf_backup_repo} successfully!", flush=True)
        except Exception as e:
            print(f"HF upload note: {e}", flush=True)


if __name__ == "__main__":
    main()
