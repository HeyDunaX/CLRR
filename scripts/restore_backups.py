"""Restore the newest checkpoint ZIP for each run before resuming training."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)\.zip$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--hf-backup-repo", default=None)
    args = parser.parse_args()
    backup_root = Path(args.backup_dir)
    output_root = Path(args.output_dir)
    if args.hf_backup_repo:
        token = __import__("os").environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required when --hf-backup-repo is set.")
        api = HfApi(token=token)
        for remote_file in api.list_repo_files(args.hf_backup_repo, repo_type="model"):
            match = re.fullmatch(r"(.+)/checkpoint-(\d+)\.zip", remote_file)
            if not match:
                continue
            run_name, step_text = match.groups()
            checkpoint_dir = output_root / run_name / f"checkpoint-{step_text}"
            if (checkpoint_dir / "trainer_state.json").exists():
                continue
            archive = hf_hub_download(
                repo_id=args.hf_backup_repo,
                filename=remote_file,
                repo_type="model",
                token=token,
                local_dir=str(backup_root),
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(checkpoint_dir)
            print(f"[restore] {run_name}: restored checkpoint-{step_text} from Hugging Face")
    if not backup_root.exists():
        print(f"[restore] no backup directory: {backup_root}")
        return
    for run_dir in sorted(path for path in backup_root.iterdir() if path.is_dir()):
        candidates = []
        for archive in run_dir.glob("checkpoint-*.zip"):
            match = CHECKPOINT_PATTERN.fullmatch(archive.name)
            if match:
                candidates.append((int(match.group(1)), archive))
        if not candidates:
            continue
        step, archive = max(candidates, key=lambda item: item[0])
        checkpoint_dir = output_root / run_dir.name / f"checkpoint-{step}"
        marker = checkpoint_dir / "trainer_state.json"
        if marker.exists():
            print(f"[restore] {run_dir.name}: checkpoint-{step} already present")
            continue
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(checkpoint_dir)
        print(f"[restore] {run_dir.name}: restored checkpoint-{step}")


if __name__ == "__main__":
    main()