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
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF


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
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print commands without training.")
    parser.add_argument("--runs", nargs="*", default=["mbart_baseline", "mbart_clrr", "nllb_baseline", "nllb_clrr"],
                        choices=["mbart_baseline", "mbart_clrr", "nllb_baseline", "nllb_clrr"],
                        help="Subset of runs to execute.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = repo_root / args.data_dir
    dataset_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset_hashes = {}
    for split, expected in (("train", 3883), ("validation", 881), ("test", 1003)):
        path = data_dir / f"{split}.csv"
        frame = pd.read_csv(path, keep_default_na=False)
        if list(frame.columns) != ["source", "target"] or len(frame) != expected:
            raise ValueError(f"Unexpected schema or row count: {path}")
        if frame.apply(lambda column: column.map(lambda value: not str(value).strip())).any().any():
            raise ValueError(f"Empty aligned text: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != dataset_manifest["splits"][split]["sha256"]:
            raise ValueError(f"Dataset SHA-256 mismatch: {path}")
        dataset_hashes[path.name] = digest

    experiments = [
        {
            "id": "mbart_baseline",
            "name": "mbart-ashaninka-es-baseline",
            "desc": "mBART-large-50 Standard Full Fine-Tuning (Asháninka -> Spanish)",
            "cmd": [
                sys.executable, "-u", "-m", "amis_rewire.train",
                "--model", "mbart-large-50",
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
                "--model", "mbart-large-50",
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

    common = ["--seed", "42", "--warmup-ratio", "0.06", "--weight-decay", "0.0",
              "--bf16", "--logging-steps", "10"]
    source_files = ["src/amis_rewire/train.py", "src/amis_rewire/modeling.py",
                    "src/amis_rewire/metrics.py", "src/nllb_suite/train_nllb.py",
                    "src/nllb_suite/modeling_nllb.py", "scripts/run_ashaninka_experiments.py"]
    source_hashes = {name: hashlib.sha256((repo_root / name).read_bytes()).hexdigest()
                     for name in source_files}
    for exp in experiments:
        exp["cmd"].extend(common)
        if exp["id"].startswith("mbart"):
            exp["cmd"].extend([
                "--max-source-length", "256", "--max-target-length", "256",
                "--per-device-eval-batch-size", "8", "--eval-beams", "1", "--num-beams", "4",
                "--gradient-checkpointing", "--auto-resume", "--save-total-limit", "2",
                "--dataloader-num-workers", "4", "--test-reference", "original_csv",
                "--backup-dir", str(out_dir / "backups"),
            ])
        else:
            exp["cmd"].extend([
                "--max-source-length", "128", "--max-target-length", "128",
                "--per-device-eval-batch-size", "16", "--eval-beams", "4",
                "--save-total-limit", "1", "--dataloader-num-workers", "2",
            ])
    if args.dry_run:
        print(json.dumps({"dataset_sha256": dataset_hashes,
                          "commands": [exp["cmd"] for exp in experiments if exp["id"] in args.runs]}, indent=2))
        return

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

        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = run_dir / "run_manifest.json"
        run_manifest = {"dataset_sha256": dataset_hashes, "source_sha256": source_hashes,
                        "command": exp["cmd"][3:]}
        if manifest_file.exists():
            if json.loads(manifest_file.read_text(encoding="utf-8")) != run_manifest:
                raise ValueError(f"Refusing to reuse output with different data/code/config: {run_dir}")
        elif any(run_dir.glob("checkpoint-*")) or metrics_file.exists():
            raise ValueError(f"Existing output has no verified manifest: {run_dir}")
        manifest_file.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        completion = run_dir / "completed.json"
        if completion.exists() and metrics_file.exists():
            print(f"[skip] {run_name} already completed with {metrics_file}. Skipping.", flush=True)
            continue

        run_command(exp["cmd"], cwd=repo_root)
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        predictions = pd.read_csv(run_dir / "test_predictions.csv", keep_default_na=False)
        expected_test = pd.read_csv(data_dir / "test.csv", keep_default_na=False)
        if not predictions[["source", "target"]].equals(expected_test):
            raise ValueError(f"Prediction rows/references do not match original test CSV: {run_name}")
        preds = predictions["prediction"].str.strip().tolist()
        refs = [expected_test["target"].str.strip().tolist()]
        checked = {"test_bleu": BLEU(tokenize="13a").corpus_score(preds, refs).score,
                   "test_chrf++": CHRF(word_order=2).corpus_score(preds, refs).score}
        for key, value in checked.items():
            if abs(metrics[key] - value) > 1e-8:
                raise ValueError(f"Official metric verification failed: {run_name} {key}")
        if exp["id"].startswith("mbart"):
            archive = out_dir / "backups" / run_name / f"{run_name}-best.zip"
        else:
            archive = run_dir / f"nllb-200-{metrics['method']}-best.zip"
        if not archive.is_file() or archive.stat().st_size == 0:
            raise FileNotFoundError(f"Missing best checkpoint archive: {archive}")
        completion.write_text(json.dumps({"verified_test_rows": len(predictions), **checked,
                                         "archive": str(archive), "archive_bytes": archive.stat().st_size}, indent=2), encoding="utf-8")
        rows = []
        for item in experiments:
            folder = out_dir / item["name"]
            if (folder / "completed.json").exists():
                values = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                rows.append({"run_name": item["name"], "model": values["model"], "method": values["method"],
                             "seed": values["seed"], "test_bleu": values["test_bleu"],
                             "test_chrf++": values["test_chrf++"],
                             "trainable_parameters": values["trainable_parameters"],
                             "training_time_seconds": values["training_time_seconds"]})
        pd.DataFrame(rows).to_csv(out_dir.parent / "ashaninka_spanish_scores.csv", index=False)

    print("\n[done] All requested Asháninka -> Spanish runs finished.", flush=True)


if __name__ == "__main__":
    main()
