"""Runner script for Comparative Baseline Experiments on Amis-Chinese.

Orchestrates sequential training and evaluation for:
1. mt5-small-layerskip-acl2024 (Elhoushi et al., ACL 2024)
2. mt5-small-middle-align-acl2025 (Liu & Niehues, ACL 2025)

Aggregates results into outputs_comparative/comparative_scores.csv and displays Table 5.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


HISTORICAL_SCORES = [
    {
        "model": "mt5-small-baseline",
        "method": "Standard Fine-Tuning",
        "reference": "Baseline",
        "extra_params": 0,
        "bleu": 2.788710,
        "chrf++": 3.812914,
        "status": "Official (Frozen)",
    },
    {
        "model": "mt5-small-clrr-enc",
        "method": "CLRR-Enc (Ours)",
        "reference": "Proposed (Ablation)",
        "extra_params": 0,
        "bleu": 4.439348,
        "chrf++": 5.176137,
        "status": "Official (Frozen)",
    },
    {
        "model": "mt5-small-jepa-clrr-enc",
        "method": "JEPA + CLRR-Enc (Ours)",
        "reference": "Proposed (Main)",
        "extra_params": 0,
        "bleu": 4.596068,
        "chrf++": 5.042505,
        "status": "Official (Frozen)",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparative baselines on mt5-small")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs_comparative")
    parser.add_argument("--hf-backup-repo", default="FiveC/amis-rewire-checkpoints")
    parser.add_argument("--num-train-epochs", type=float, default=20.0)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run_command_streaming(cmd: list[str]) -> int:
    print(f"\n[EXEC] Running: {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=str(PROJECT_ROOT),
    )
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
    process.stdout.close()
    return process.wait()


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    py_exec = sys.executable

    print("=================================================================")
    print("STARTING COMPARATIVE BASELINE PIPELINE (ACL 2024 & ACL 2025)")
    print("=================================================================")

    # 1. Run LayerSkip (ACL 2024)
    print("\n>>> STAGE 1: Training LayerSkip (Elhoushi et al., ACL 2024) <<<", flush=True)
    layerskip_dir = output_dir / "layerskip_acl2024"
    cmd_layerskip = [
        py_exec,
        "-u",
        "-m",
        "src.comparative_baselines.layerskip_acl2024.train",
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(layerskip_dir),
        "--hf-backup-repo",
        args.hf_backup_repo,
        "--num-train-epochs",
        str(args.num_train_epochs),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--per-device-train-batch-size",
        str(args.per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--p-max",
        "0.2",
    ]
    if args.bf16:
        cmd_layerskip.append("--bf16")
    ret1 = run_command_streaming(cmd_layerskip)
    if ret1 != 0:
        print(f"[ERROR] LayerSkip training failed with exit code {ret1}", flush=True)
        sys.exit(ret1)

    # 2. Run Middle-Layer Alignment (ACL 2025)
    print("\n>>> STAGE 2: Training Middle-Layer Alignment (Liu & Niehues, ACL 2025) <<<", flush=True)
    middle_align_dir = output_dir / "middle_align_acl2025"
    cmd_middle = [
        py_exec,
        "-u",
        "-m",
        "src.comparative_baselines.middle_align_acl2025.train",
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(middle_align_dir),
        "--hf-backup-repo",
        args.hf_backup_repo,
        "--num-train-epochs",
        str(args.num_train_epochs),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--per-device-train-batch-size",
        str(args.per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--middle-layer-idx",
        "4",
        "--temperature",
        "0.1",
        "--align-weight",
        "0.1",
    ]
    if args.bf16:
        cmd_middle.append("--bf16")
    ret2 = run_command_streaming(cmd_middle)
    if ret2 != 0:
        print(f"[ERROR] Middle-Layer Alignment training failed with exit code {ret2}", flush=True)
        sys.exit(ret2)

    # 3. Aggregate Table 5
    print("\n>>> STAGE 3: Compiling Final Comparative Table 5 <<<", flush=True)
    rows = list(HISTORICAL_SCORES)

    # Read LayerSkip metrics
    ls_metrics_path = layerskip_dir / "metrics.json"
    if ls_metrics_path.exists():
        with open(ls_metrics_path, encoding="utf-8") as f:
            ls_data = json.load(f)
        rows.append({
            "model": "mt5-small-layerskip-acl2024",
            "method": "LayerSkip",
            "reference": "Elhoushi et al. (ACL 2024)",
            "extra_params": 0,
            "bleu": ls_data.get("test_bleu"),
            "chrf++": ls_data.get("test_chrf++"),
            "status": "Measured",
        })

    # Read Middle-Align metrics
    ma_metrics_path = middle_align_dir / "metrics.json"
    if ma_metrics_path.exists():
        with open(ma_metrics_path, encoding="utf-8") as f:
            ma_data = json.load(f)
        rows.append({
            "model": "mt5-small-middle-align-acl2025",
            "method": "Middle-Layer Alignment",
            "reference": "Liu & Niehues (ACL 2025)",
            "extra_params": 0,
            "bleu": ma_data.get("test_bleu"),
            "chrf++": ma_data.get("test_chrf++"),
            "status": "Measured",
        })

    df = pd.DataFrame(rows)
    table_path = output_dir / "comparative_scores.csv"
    df.to_csv(table_path, index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print("TABLE 5: COMPARISON WITH RECENT ACL BASELINES ON mT5-SMALL")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"[DONE] Saved final comparative scores to {table_path}", flush=True)

    # Upload final comparative_scores.csv to Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and args.hf_backup_repo:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=str(table_path),
                path_in_repo="comparative_baselines/comparative_scores.csv",
                repo_id=args.hf_backup_repo,
                repo_type="model",
            )
            print(f"[BACKUP] Uploaded {table_path.name} to Hugging Face repo {args.hf_backup_repo}", flush=True)
        except Exception as e:
            print(f"[BACKUP] Warning: Upload of {table_path.name} failed: {e}", flush=True)


if __name__ == "__main__":
    main()
