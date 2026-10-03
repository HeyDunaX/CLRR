"""Runner script for Comprehensive Comparative Experiments on Amis-Chinese (ACL 2024, ACL 2025 & CLRR-Dec).

Orchestrates sequential training and evaluation for the retained multi-backbone matrix:
Backbones:
  - google/mt5-small
  - facebook/mbart-large-50-many-to-many-mmt
  - google/byt5-small
Methods:
  - Standard Fine-Tuning (Baseline)
  - Middle-Layer Alignment (Liu & Niehues, ACL 2025)
  - CLRR-Enc (Ours Ablation)
  - JEPA + CLRR-Enc (Ours Main)
  - JEPA + CLRR-Dec (Ours Decoder-only)

Aggregates all results into results/full_comparative_matrix_acl.csv and displays Table 5.
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


def official_baseline_scores(reference_path: Path) -> list[dict]:
    from src.amis_rewire.metrics import verified_reference_metrics

    rows = [
        {"backbone": "google/mt5-small", "method": "Standard Fine-Tuning", "reference": "Baseline", "extra_params": 0, "status": "Verified raw references (2026-10-02)"},
        {"backbone": "google/mt5-small", "method": "Middle-Layer Alignment", "reference": "Liu & Niehues (ACL 2025)", "extra_params": 0, "status": "Verified raw references (2026-10-02)"},
        {"backbone": "google/mt5-small", "method": "CLRR-Enc (Ours)", "reference": "Proposed (Ablation)", "extra_params": 0, "status": "Verified raw references (2026-10-02)"},
        {"backbone": "google/mt5-small", "method": "JEPA + CLRR-Enc (Ours)", "reference": "Proposed (Main)", "extra_params": 0, "status": "Verified raw references (2026-10-02)"},
        {"backbone": "google/mt5-small", "method": "JEPA + CLRR-Dec (Ours)", "reference": "Proposed (Decoder-only)", "extra_params": 0, "status": "Verified raw references (2026-10-02)"},
        {"backbone": "facebook/mbart-large-50", "method": "Standard Fine-Tuning", "reference": "Baseline", "extra_params": 0, "status": "Verified raw references (2026-10-02)"},
        {"backbone": "facebook/mbart-large-50", "method": "JEPA + CLRR-Enc (Ours)", "reference": "Proposed (Main)", "extra_params": 0, "status": "Verified raw references (2026-10-02)"},
        {"backbone": "google/byt5-small", "method": "Standard Fine-Tuning", "reference": "Baseline", "extra_params": 0, "status": "Verified raw references (2026-10-02)"},
        {"backbone": "google/byt5-small", "method": "JEPA + CLRR-Enc (Ours)", "reference": "Proposed (Main)", "extra_params": 0, "status": "Verified raw references (2026-10-02)"},
    ]
    backbones = {"google/mt5-small": "mT5", "facebook/mbart-large-50": "mBART", "google/byt5-small": "ByT5"}
    methods = {"Standard Fine-Tuning": "Baseline", "CLRR-Enc (Ours)": "CLRR-only",
               "JEPA + CLRR-Enc (Ours)": "CLRR+LSR", "JEPA + CLRR-Dec (Ours)": "CLRR-Dec+LSR"}
    for row in rows:
        row.update(verified_reference_metrics(backbones[row["backbone"]],
                   methods.get(row["method"], row["method"]), reference_path))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparative and multi-backbone experiments for ACL")
    parser.add_argument("--run-group", choices=["all_extensions", "mbart_only", "byt5_only", "all"], default="all_extensions")
    parser.add_argument("--data-dir", default="data_processed/amis_mandarin")
    parser.add_argument("--output-dir", default="results")
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
    reference_path = PROJECT_ROOT / args.data_dir / "test.csv"
    baseline_rows = official_baseline_scores(reference_path)
    from src.amis_rewire.metrics import score_prediction_csv
    output_dir = PROJECT_ROOT / args.output_dir
    backup_dir = PROJECT_ROOT / args.backup_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    py_exec = sys.executable

    print("=================================================================")
    print("STARTING ACL MULTI-BACKBONE & COMPARATIVE EXPERIMENT PIPELINE")
    print(f"Run group: {args.run_group}")
    print("=================================================================")

    # Define the 4 retained extension runs
    # (tag, backbone_name, method_name, command_func)
    tasks = []

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
            "--per-device-eval-batch-size", str(min(args.per_device_eval_batch_size, 16)),
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
            "--per-device-eval-batch-size", str(min(args.per_device_eval_batch_size, 32)),
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


    # Compile Table 5 (Full Retained Comparative Matrix)
    print("\n=================================================================", flush=True)
    print(">>> STAGE: Compiling Final Comprehensive Table 5 (Retained Matrix) <<<", flush=True)
    print("=================================================================", flush=True)

    rows = baseline_rows

    # Read all newly generated task metrics
    for task in tasks:
        metrics_file = task["output_dir"] / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, encoding="utf-8") as f:
                data = json.load(f)
            # Find test bleu and chrf
            scores = score_prediction_csv(task["output_dir"] / "test_predictions.csv", reference_path)
            bleu, chrf = scores["bleu"], scores["chrf++"]
            rows.append({
                "backbone": task["backbone"],
                "method": task["method"],
                "reference": task["reference"],
                "extra_params": 0,
                "bleu": bleu,
                "chrf++": chrf,
                "status": "Verified from saved predictions / original CSV",
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
    print(f"\n[DONE] Saved retained comparative matrix to {matrix_path}", flush=True)

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
