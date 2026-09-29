"""Ablation studies on facebook/mbart-large-50-many-to-many-mmt.

Trains two isolated ablation models:
1. CLRR-only: Encoder rewiring (d=2, alpha=0.1) without LSR alignment loss (lambda=0).
2. LSR-only: Semantic regularization loss (lambda=0.1) without CLRR rewiring (alpha=0).

Resolves reviewer doubt regarding the origin of the +5.03 chrF++ gain.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd


def run_command(cmd: list[str], cwd: Path) -> None:
    print(f"\n[exec] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def run_mbart_ablations() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "outputs_revalidation"
    out_dir.mkdir(parents=True, exist_ok=True)

    ablations = [
        {
            "name": "mbart-large-50-ami-cmn-clrr-only",
            "method": "clrr-enc",
            "jepa_weight": 0.0,
            "desc": "CLRR-only (alpha=0.1, lambda=0)",
        },
        {
            "name": "mbart-large-50-ami-cmn-lsr-only",
            "method": "jepa",
            "jepa_weight": 0.1,
            "desc": "LSR-only (alpha=0, lambda=0.1)",
        },
    ]

    hf_repo = "FiveC/amis-rewire-checkpoints"
    results = []

    for ab in ablations:
        run_name = ab["name"]
        print(f"\n=======================================================")
        print(f"Starting Ablation: {ab['desc']}")
        print(f"Run name: {run_name}")
        print(f"=======================================================", flush=True)

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "amis_rewire.train",
            "--model",
            "mbart-large-50",
            "--method",
            ab["method"],
            "--jepa-weight",
            str(ab["jepa_weight"]),
            "--learning-rate",
            "5e-5",
            "--num-train-epochs",
            "20",
            "--early-stopping-patience",
            "4",
            "--per-device-train-batch-size",
            "4",
            "--gradient-accumulation-steps",
            "32",  # Effective batch = 128
            "--per-device-eval-batch-size",
            "8",
            "--num-beams",
            "4",
            "--eval-beams",
            "1",
            "--save-total-limit",
            "2",
            "--run-name",
            run_name,
            "--output-dir",
            str(out_dir),
            "--backup-dir",
            str(repo_root / "backups"),
        ]

        if "HF_TOKEN" in os.environ:
            cmd.extend([
                "--hf-backup-repo",
                hf_repo,
                "--hf-backup-prefix",
                "revalidation_ablations",
            ])

        run_command(cmd, cwd=repo_root)

        # Read results
        metrics_file = out_dir / run_name / "metrics.json"
        if metrics_file.exists():
            import json
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "Run": run_name,
                "Description": ab["desc"],
                "BLEU": data.get("test_bleu", "N/A"),
                "chrF++": data.get("test_chrf++", "N/A"),
            })

    # Summary table
    if results:
        res_df = pd.DataFrame(results)
        summary_csv = out_dir / "mbart_ablation_results.csv"
        res_df.to_csv(summary_csv, index=False)
        print("\nmBART-50 Ablation Results Summary:")
        print(res_df.to_string(index=False))


if __name__ == "__main__":
    run_mbart_ablations()
