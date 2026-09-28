"""Runner script for Comprehensive Comparative Experiments on Amis-Chinese (ACL 2024, ACL 2025 & CLRR-Dec).

Orchestrates sequential training and evaluation for the full 3x5 matrix:
Backbones:
  - google/mt5-small
  - facebook/mbart-large-50-many-to-many-mmt
  - google/byt5-small
Methods:
  - Standard Fine-Tuning (Baseline)
  - LayerSkip (Elhoushi et al., ACL 2024)
  - Middle-Layer Alignment (Liu & Niehues, ACL 2025)
  - CLRR-Enc (Ours Ablation)
  - JEPA + CLRR-Enc (Ours Main)
  - JEPA + CLRR-Dec (Ours Decoder-only)

Aggregates all results into outputs_comparative/full_comparative_matrix_acl.csv and displays Table 5.
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


OFFICIAL_BASELINE_SCORES = [
    # 1. google/mt5-small
    {"backbone": "google/mt5-small", "method": "Standard Fine-Tuning", "reference": "Baseline", "extra_params": 0, "bleu": 2.788710, "chrf++": 3.812914, "status": "Official (Frozen)"},
    {"backbone": "google/mt5-small", "method": "LayerSkip", "reference": "Elhoushi et al. (ACL 2024)", "extra_params": 0, "bleu": 3.972051, "chrf++": 5.288554, "status": "Measured (ACL 2024)"},
    {"backbone": "google/mt5-small", "method": "Middle-Layer Alignment", "reference": "Liu & Niehues (ACL 2025)", "extra_params": 0, "bleu": 4.749767, "chrf++": 5.937303, "status": "Measured (ACL 2025)"},
    {"backbone": "google/mt5-small", "method": "CLRR-Enc (Ours)", "reference": "Proposed (Ablation)", "extra_params": 0, "bleu": 4.439348, "chrf++": 5.176137, "status": "Official (Frozen)"},
    {"backbone": "google/mt5-small", "method": "JEPA + CLRR-Enc (Ours)", "reference": "Proposed (Main)", "extra_params": 0, "bleu": 4.596068, "chrf++": 5.042505, "status": "Official (Frozen)"},
    {"backbone": "google/mt5-small", "method": "JEPA + CLRR-Dec (Ours)", "reference": "Proposed (Decoder-only)", "extra_params": 0, "bleu": 4.831513, "chrf++": 5.187342, "status": "Official (Frozen)"},

    # 2. facebook/mbart-large-50
    {"backbone": "facebook/mbart-large-50", "method": "Standard Fine-Tuning", "reference": "Baseline", "extra_params": 0, "bleu": 19.610600, "chrf++": 14.053800, "status": "Official (Frozen)"},
    {"backbone": "facebook/mbart-large-50", "method": "JEPA + CLRR-Enc (Ours)", "reference": "Proposed (Main)", "extra_params": 0, "bleu": 20.392700, "chrf++": 19.083900, "status": "Official (Frozen)"},

    # 3. google/byt5-small
    {"backbone": "google/byt5-small", "method": "Standard Fine-Tuning", "reference": "Baseline", "extra_params": 0, "bleu": 7.579430, "chrf++": 8.310188, "status": "Official (Frozen)"},
    {"backbone": "google/byt5-small", "method": "JEPA + CLRR-Enc (Ours)", "reference": "Proposed (Main)", "extra_params": 0, "bleu": 7.308960, "chrf++": 8.096336, "status": "Official (Frozen)"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparative and multi-backbone experiments for ACL")
    parser.add_argument("--run-group", choices=["all_extensions", "mbart_only", "byt5_only", "all"], default="all_extensions")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs_comparative")
    parser.add_argument("--backup-dir", default="backups_comparative")
    parser.add_argument("--hf-backup-repo", default="FiveC/amis-rewire-checkpoints")
    parser.add_argument("--num-train-epochs", type=float, default=20.0)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=64)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run_command_streaming(cmd: list[str]) -> int:
    print(f"\n[EXEC] Running: {' '.join(cmd)}", flush=True)
    env = dict(os.environ)
    src_dir = str(PROJECT_ROOT / "src")
    repo_dir = str(PROJECT_ROOT)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{repo_dir}{os.pathsep}{existing_pp}"
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
    process.stdout.close()
    return process.wait()


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    backup_dir = PROJECT_ROOT / args.backup_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    py_exec = sys.executable

    print("=================================================================")
    print("STARTING ACL MULTI-BACKBONE & COMPARATIVE EXPERIMENT PIPELINE")
    print(f"Run group: {args.run_group}")
    print("=================================================================")

    # Define the 6 target extension runs
    # (tag, backbone_name, method_name, command_func)
    tasks = []

    # 1. mBART LayerSkip
    if args.run_group in ("all_extensions", "mbart_only", "all"):
        mbart_ls_dir = output_dir / "mbart-large-50-layerskip-acl2024"
        cmd = [
            py_exec, "-u", "-m", "src.comparative_baselines.layerskip_acl2024.train",
            "--model-name", "facebook/mbart-large-50-many-to-many-mmt",
            "--run-name", "mbart-large-50-layerskip-acl2024",
            "--data-dir", args.data_dir,
            "--output-dir", str(mbart_ls_dir),
            "--hf-backup-repo", args.hf_backup_repo,
            "--learning-rate", "5e-5",
            "--num-train-epochs", str(args.num_train_epochs),
            "--early-stopping-patience", str(args.early_stopping_patience),
            "--warmup-ratio", str(args.warmup_ratio),
            "--per-device-train-batch-size", str(args.per_device_train_batch_size),
            "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
            "--per-device-eval-batch-size", str(args.per_device_eval_batch_size),
            "--p-max", "0.2",
        ]
        if args.bf16:
            cmd.append("--bf16")
        tasks.append({
            "name": "mBART-50 + LayerSkip (ACL 2024)",
            "run_name": "mbart-large-50-layerskip-acl2024",
            "backbone": "facebook/mbart-large-50",
            "method": "LayerSkip",
            "reference": "Elhoushi et al. (ACL 2024)",
            "output_dir": mbart_ls_dir,
            "cmd": cmd,
        })

    # 2. mBART Middle-Layer Alignment (L6)
    if args.run_group in ("all_extensions", "mbart_only", "all"):
        mbart_ma_dir = output_dir / "mbart-large-50-middle-align-acl2025"
        cmd = [
            py_exec, "-u", "-m", "src.comparative_baselines.middle_align_acl2025.train",
            "--model-name", "facebook/mbart-large-50-many-to-many-mmt",
            "--run-name", "mbart-large-50-middle-align-acl2025",
            "--data-dir", args.data_dir,
            "--output-dir", str(mbart_ma_dir),
            "--hf-backup-repo", args.hf_backup_repo,
            "--learning-rate", "5e-5",
            "--middle-layer-idx", "6",
            "--temperature", "0.1",
            "--align-weight", "0.1",
            "--num-train-epochs", str(args.num_train_epochs),
            "--early-stopping-patience", str(args.early_stopping_patience),
            "--warmup-ratio", str(args.warmup_ratio),
            "--per-device-train-batch-size", str(args.per_device_train_batch_size),
            "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
            "--per-device-eval-batch-size", str(args.per_device_eval_batch_size),
        ]
        if args.bf16:
            cmd.append("--bf16")
        tasks.append({
            "name": "mBART-50 + Middle-Layer Alignment (ACL 2025)",
            "run_name": "mbart-large-50-middle-align-acl2025",
            "backbone": "facebook/mbart-large-50",
            "method": "Middle-Layer Alignment",
            "reference": "Liu & Niehues (ACL 2025)",
            "output_dir": mbart_ma_dir,
            "cmd": cmd,
        })

    # 3. mBART JEPA + CLRR-Dec
    if args.run_group in ("all_extensions", "mbart_only", "all"):
        mbart_dec_run = "mbart-large-50-ami-cmn-jepa-clrr-dec"
        mbart_dec_dir = output_dir / mbart_dec_run
        cmd = [
            py_exec, "-u", "-m", "src.amis_rewire.train",
            "--model", "mbart-large-50",
            "--method", "jepa-clrr",
            "--rewire-stack", "decoder",
            "--jepa-weight", "0.1",
            "--data-dir", args.data_dir,
            "--output-dir", str(output_dir),
            "--backup-dir", str(backup_dir),
            "--run-name", mbart_dec_run,
            "--seed", "42",
            "--num-train-epochs", str(args.num_train_epochs),
            "--early-stopping-patience", str(args.early_stopping_patience),
            "--learning-rate", "5e-5",
            "--warmup-ratio", str(args.warmup_ratio),
            "--per-device-train-batch-size", str(args.per_device_train_batch_size),
            "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
            "--per-device-eval-batch-size", str(args.per_device_eval_batch_size),
            "--max-source-length", "256",
            "--max-target-length", "256",
            "--num-beams", "4",
            "--eval-beams", "1",
            "--rewire-distance", "2",
            "--rewire-strength", "0.1",
            "--hf-backup-repo", args.hf_backup_repo,
            "--hf-backup-prefix", "comparative_baselines",
        ]
        if args.bf16:
            cmd.append("--bf16")
        tasks.append({
            "name": "mBART-50 + JEPA + CLRR-Dec (Ours Decoder-only)",
            "run_name": mbart_dec_run,
            "backbone": "facebook/mbart-large-50",
            "method": "JEPA + CLRR-Dec (Ours)",
            "reference": "Proposed (Decoder-only)",
            "output_dir": mbart_dec_dir,
            "cmd": cmd,
        })

    # 4. ByT5 LayerSkip
    if args.run_group in ("all_extensions", "byt5_only", "all"):
        byt5_ls_dir = output_dir / "byt5-small-layerskip-acl2024"
        cmd = [
            py_exec, "-u", "-m", "src.comparative_baselines.layerskip_acl2024.train",
            "--model-name", "google/byt5-small",
            "--run-name", "byt5-small-layerskip-acl2024",
            "--data-dir", args.data_dir,
            "--output-dir", str(byt5_ls_dir),
            "--hf-backup-repo", args.hf_backup_repo,
            "--learning-rate", "3e-4",
            "--num-train-epochs", str(args.num_train_epochs),
            "--early-stopping-patience", str(args.early_stopping_patience),
            "--warmup-ratio", str(args.warmup_ratio),
            "--per-device-train-batch-size", str(args.per_device_train_batch_size),
            "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
            "--per-device-eval-batch-size", str(args.per_device_eval_batch_size),
            "--p-max", "0.2",
        ]
        if args.bf16:
            cmd.append("--bf16")
        tasks.append({
            "name": "ByT5-small + LayerSkip (ACL 2024)",
            "run_name": "byt5-small-layerskip-acl2024",
            "backbone": "google/byt5-small",
            "method": "LayerSkip",
            "reference": "Elhoushi et al. (ACL 2024)",
            "output_dir": byt5_ls_dir,
            "cmd": cmd,
        })

    # 5. ByT5 Middle-Layer Alignment (L6)
    if args.run_group in ("all_extensions", "byt5_only", "all"):
        byt5_ma_dir = output_dir / "byt5-small-middle-align-acl2025"
        cmd = [
            py_exec, "-u", "-m", "src.comparative_baselines.middle_align_acl2025.train",
            "--model-name", "google/byt5-small",
            "--run-name", "byt5-small-middle-align-acl2025",
            "--data-dir", args.data_dir,
            "--output-dir", str(byt5_ma_dir),
            "--hf-backup-repo", args.hf_backup_repo,
            "--learning-rate", "3e-4",
            "--middle-layer-idx", "6",
            "--temperature", "0.1",
            "--align-weight", "0.1",
            "--num-train-epochs", str(args.num_train_epochs),
            "--early-stopping-patience", str(args.early_stopping_patience),
            "--warmup-ratio", str(args.warmup_ratio),
            "--per-device-train-batch-size", str(args.per_device_train_batch_size),
            "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
            "--per-device-eval-batch-size", str(args.per_device_eval_batch_size),
        ]
        if args.bf16:
            cmd.append("--bf16")
        tasks.append({
            "name": "ByT5-small + Middle-Layer Alignment (ACL 2025)",
            "run_name": "byt5-small-middle-align-acl2025",
            "backbone": "google/byt5-small",
            "method": "Middle-Layer Alignment",
            "reference": "Liu & Niehues (ACL 2025)",
            "output_dir": byt5_ma_dir,
            "cmd": cmd,
        })

    # 6. ByT5 JEPA + CLRR-Dec
    if args.run_group in ("all_extensions", "byt5_only", "all"):
        byt5_dec_run = "byt5-small-ami-cmn-jepa-clrr-dec"
        byt5_dec_dir = output_dir / byt5_dec_run
        cmd = [
            py_exec, "-u", "-m", "src.amis_rewire.train",
            "--model", "byt5-small",
            "--method", "jepa-clrr",
            "--rewire-stack", "decoder",
            "--jepa-weight", "0.1",
            "--data-dir", args.data_dir,
            "--output-dir", str(output_dir),
            "--backup-dir", str(backup_dir),
            "--run-name", byt5_dec_run,
            "--seed", "42",
            "--num-train-epochs", str(args.num_train_epochs),
            "--early-stopping-patience", str(args.early_stopping_patience),
            "--learning-rate", "3e-4",
            "--warmup-ratio", str(args.warmup_ratio),
            "--per-device-train-batch-size", str(args.per_device_train_batch_size),
            "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
            "--per-device-eval-batch-size", str(args.per_device_eval_batch_size),
            "--max-source-length", "256",
            "--max-target-length", "256",
            "--num-beams", "4",
            "--eval-beams", "1",
            "--rewire-distance", "2",
            "--rewire-strength", "0.1",
            "--hf-backup-repo", args.hf_backup_repo,
            "--hf-backup-prefix", "comparative_baselines",
        ]
        if args.bf16:
            cmd.append("--bf16")
        tasks.append({
            "name": "ByT5-small + JEPA + CLRR-Dec (Ours Decoder-only)",
            "run_name": byt5_dec_run,
            "backbone": "google/byt5-small",
            "method": "JEPA + CLRR-Dec (Ours)",
            "reference": "Proposed (Decoder-only)",
            "output_dir": byt5_dec_dir,
            "cmd": cmd,
        })

    # Sequential execution loop with skip-if-completed
    for idx, task in enumerate(tasks, start=1):
        print(f"\n=======================================================", flush=True)
        print(f">>> TASK {idx}/{len(tasks)}: {task['name']} <<<", flush=True)
        print(f"=======================================================", flush=True)

        metrics_file = task["output_dir"] / "metrics.json"
        if metrics_file.exists():
            print(f"[SKIP] Found completed metrics at {metrics_file}. Skipping training.", flush=True)
            continue

        ret = run_command_streaming(task["cmd"])
        if ret != 0:
            print(f"[ERROR] Task failed with exit code {ret}: {task['name']}", flush=True)
            sys.exit(ret)

    # Compile Table 5 (Full 3x5 Comparative Matrix)
    print("\n=================================================================", flush=True)
    print(">>> STAGE: Compiling Final Comprehensive Table 5 (3x5 Matrix) <<<", flush=True)
    print("=================================================================", flush=True)

    rows = list(OFFICIAL_BASELINE_SCORES)

    # Read all newly generated task metrics
    for task in tasks:
        metrics_file = task["output_dir"] / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, encoding="utf-8") as f:
                data = json.load(f)
            # Find test bleu and chrf
            bleu = data.get("test_bleu")
            chrf = data.get("test_chrf++") or data.get("test_chrf")
            rows.append({
                "backbone": task["backbone"],
                "method": task["method"],
                "reference": task["reference"],
                "extra_params": 0,
                "bleu": bleu,
                "chrf++": chrf,
                "status": "Measured (This Run)",
            })

    df = pd.DataFrame(rows)
    # Deduplicate keeping latest
    df = df.drop_duplicates(subset=["backbone", "method"], keep="last")

    matrix_path = output_dir / "full_comparative_matrix_acl.csv"
    df.to_csv(matrix_path, index=False, encoding="utf-8")

    print("\n" + "=" * 96)
    print("TABLE 5: COMPREHENSIVE CROSS-ARCHITECTURE COMPARATIVE MATRIX (ACL MAIN/FINDINGS)")
    print("=" * 96)
    print(df.to_string(index=False))
    print("=" * 96)
    print(f"\n[DONE] Saved complete 3x5 comparative matrix to {matrix_path}", flush=True)

    # Upload full_comparative_matrix_acl.csv to Hugging Face Hub
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and args.hf_backup_repo:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=str(matrix_path),
                path_in_repo="comparative_baselines/full_comparative_matrix_acl.csv",
                repo_id=args.hf_backup_repo,
                repo_type="model",
            )
            print(f"[BACKUP] Uploaded {matrix_path.name} to Hugging Face repo {args.hf_backup_repo}", flush=True)
        except Exception as e:
            print(f"[BACKUP] Warning: Upload of {matrix_path.name} failed: {e}", flush=True)


if __name__ == "__main__":
    main()
