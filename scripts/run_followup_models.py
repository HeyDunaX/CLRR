"""Run the four additional Colab experiments with resumable HF backups."""

from __future__ import annotations

import argparse
import os
import json
import re
import subprocess
import time
import sys
import zipfile
from pathlib import Path

from huggingface_hub import HfApi

from followup_analysis import REPO, fetch_file, preflight, remote_file, token


EXPERIMENTS = {
    "byt5": (
        ("byt5-small-ami-cmn-baseline", "byt5-small", "baseline", "encoder"),
        ("byt5-small-ami-cmn-jepa-clrr-enc", "byt5-small", "jepa-clrr-enc", "encoder"),
    ),
    "ablation": (
        ("mt5-small-ami-cmn-jepa-clrr-dec", "mt5-small", "jepa-clrr", "decoder"),
        ("mt5-small-ami-cmn-jepa-clrr-both", "mt5-small", "jepa-clrr", "both"),
    ),
}


def required_remote_files(run_name: str) -> set[str]:
    return {
        remote_file(run_name, f"{run_name}-best.zip"),
        remote_file(run_name, "metrics.json"),
        remote_file(run_name, "test_predictions.csv"),
    }


def upload_completed_local(repo: str, run_name: str, output_root: Path, backup_root: Path) -> None:
    output_run = output_root / run_name
    local_files = {
        "metrics.json": output_run / "metrics.json",
        "test_predictions.csv": output_run / "test_predictions.csv",
        f"{run_name}-best.zip": backup_root / run_name / f"{run_name}-best.zip",
    }
    if not all(path.exists() for path in local_files.values()):
        return
    api = HfApi(token=token())
    remote = set(api.list_repo_files(repo, repo_type="model"))
    for filename, path in local_files.items():
        target = remote_file(run_name, filename)
        if target not in remote:
            for attempt in range(1, 4):
                try:
                    api.upload_file(
                        path_or_fileobj=str(path), path_in_repo=target, repo_id=repo, repo_type="model"
                    )
                    print(f"[upload] {target}", flush=True)
                    break
                except Exception as exc:
                    print(f"[upload] attempt {attempt}/3 failed for {target} ({type(exc).__name__}): {exc}", flush=True)
                    if attempt < 3:
                        time.sleep(5 * attempt)


def restore_latest(repo: str, run_name: str, output_root: Path, remote: set[str]) -> None:
    expression = re.compile(rf"^{re.escape('checkpoints/' + run_name)}/checkpoint-(\d+)\.zip$")
    matches = [(int(match.group(1)), path) for path in remote if (match := expression.match(path))]
    if not matches:
        return
    step, remote_archive = max(matches)

    def restore(checkpoint_step: int, archive_name: str) -> Path:
        destination = output_root / run_name / f"checkpoint-{checkpoint_step}"
        has_weights = (destination / "pytorch_model.bin").exists() or (
            destination / "model.safetensors"
        ).exists()
        if (destination / "trainer_state.json").exists() and has_weights and (
            destination / "optimizer.pt"
        ).exists():
            return destination
        archive_path = fetch_file(repo, archive_name)
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for item in archive.infolist():
                target = (destination / item.filename).resolve()
                if target != root and root not in target.parents:
                    raise ValueError(f"Unsafe path in {archive_path}: {item.filename}")
            archive.extractall(destination)
        if not (destination / "trainer_state.json").exists() or not (
            (destination / "pytorch_model.bin").exists() or (destination / "model.safetensors").exists()
        ):
            raise FileNotFoundError(f"Restored checkpoint lacks trainer state or model weights: {destination}")
        print(f"[restore] {run_name}: checkpoint-{checkpoint_step}", flush=True)
        return destination

    latest_dir = restore(step, remote_archive)
    state = json.loads((latest_dir / "trainer_state.json").read_text(encoding="utf-8"))
    best_path = state.get("best_model_checkpoint")
    if best_path:
        best_step = int(Path(best_path).name.removeprefix("checkpoint-"))
        if best_step != step:
            best_archive = remote_file(run_name, f"checkpoint-{best_step}.zip")
            if best_archive not in remote:
                raise FileNotFoundError(f"Best checkpoint archive required for resume is missing: {best_archive}")
            restore(best_step, best_archive)


def training_command(
    run_name: str, model: str, method: str, stack: str, repo: str,
    output_root: Path, backup_root: Path, data_dir: Path,
) -> list[str]:
    return [
        sys.executable, "-u", "-m", "amis_rewire.train",
        "--model", model, "--method", method, "--rewire-stack", stack,
        "--jepa-weight", "0.1", "--data-dir", str(data_dir),
        "--output-dir", str(output_root), "--backup-dir", str(backup_root),
        "--run-name", run_name, "--seed", "42",
        "--num-train-epochs", "20", "--early-stopping-patience", "4",
        "--learning-rate", "3e-4", "--warmup-ratio", "0.06",
        "--per-device-train-batch-size", "128", "--per-device-eval-batch-size", "128",
        "--gradient-accumulation-steps", "1", "--max-source-length", "256",
        "--max-target-length", "256", "--num-beams", "4", "--eval-beams", "1",
        "--rewire-distance", "2", "--rewire-strength", "0.1", "--bf16",
        "--no-gradient-checkpointing", "--dataloader-num-workers", "4",
        "--dataloader-pin-memory", "--hf-backup-repo", repo,
        "--hf-backup-prefix", "checkpoints",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("byt5", "ablation", "all"), required=True)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--backup-dir", type=Path, default=Path("backups_extra"))
    parser.add_argument("--data-dir", type=Path, default=Path("data_processed/amis_mandarin"))
    args = parser.parse_args()
    preflight(args.repo, args.output_dir, args.data_dir)
    stages = ("byt5", "ablation") if args.stage == "all" else (args.stage,)
    for stage in stages:
        for run_name, model, method, stack in EXPERIMENTS[stage]:
            remote = set(HfApi(token=token()).list_repo_files(args.repo, repo_type="model"))
            if required_remote_files(run_name) <= remote:
                print(f"[skip] {run_name}: best model, metrics, and predictions are remote", flush=True)
                continue
            upload_completed_local(args.repo, run_name, args.output_dir, args.backup_dir)
            remote = set(HfApi(token=token()).list_repo_files(args.repo, repo_type="model"))
            if required_remote_files(run_name) <= remote:
                print(f"[skip] {run_name}: completed local artifacts uploaded", flush=True)
                continue
            restore_latest(args.repo, run_name, args.output_dir, remote)
            command = training_command(
                run_name, model, method, stack, args.repo,
                args.output_dir, args.backup_dir, args.data_dir,
            )
            print("[train] " + " ".join(command), flush=True)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
            remote = set(HfApi(token=token()).list_repo_files(args.repo, repo_type="model"))
            missing = required_remote_files(run_name) - remote
            if missing:
                raise RuntimeError(f"Run finished but remote artifacts are missing: {sorted(missing)}")


if __name__ == "__main__":
    main()
