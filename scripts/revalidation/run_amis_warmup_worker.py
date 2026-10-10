"""Worker script for Section 19.5 pilot: Train-time routing warmup on Amis.

Dispatches:
  - machine1: Warmup Full (alpha 0->0.10 in 20% steps, lambda=0.10, seeds 42, 43, 44)
  - machine2: Warmup CLRR-only (alpha 0->0.10 in 20% steps, lambda=0.00, seeds 42, 43, 44)
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT_DEFAULT = ROOT / "outputs_rebuttal/amis_routing_warmup_20261004_v2"
DATA_DEFAULT = ROOT / "data_processed/amis_mandarin"

QUEUES = {
    "machine1": [
        ("amis-warmup-full-seed42", "jepa-clrr-enc", 0.10, 0.10, 0.20, 42),
        ("amis-warmup-full-seed43", "jepa-clrr-enc", 0.10, 0.10, 0.20, 43),
        ("amis-warmup-full-seed44", "jepa-clrr-enc", 0.10, 0.10, 0.20, 44),
    ],
    "machine2": [
        ("amis-warmup-clrr-seed42", "clrr-enc", 0.10, 0.00, 0.20, 42),
        ("amis-warmup-clrr-seed43", "clrr-enc", 0.10, 0.00, 0.20, 43),
        ("amis-warmup-clrr-seed44", "clrr-enc", 0.10, 0.00, 0.20, 44),
    ],
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["machine1", "machine2"], required=True)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DEFAULT))
    parser.add_argument("--data-dir", type=str, default=str(DATA_DEFAULT))
    args = parser.parse_args()

    out_base = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    runs_dir = out_base / "runs"
    backups_dir = out_base / "backups"
    runs_dir.mkdir(parents=True, exist_ok=True)
    backups_dir.mkdir(parents=True, exist_ok=True)
    from amis_warmup_protocol import preflight, REVISION
    preflight(out_base)
    control_pass = out_base / f"CONTROLS_PASS_{args.worker}.json"
    assert control_pass.exists(), 'Frozen controls must pass replay before training'
    assert json.loads(control_pass.read_text())['status'] == 'PASS'
    gate = json.loads((out_base / 'ALLOW_CORRECTED_WARMUP_TRAIN.json').read_text())
    assert gate['status'] == 'PASS' and gate['manifest_sha256'] == digest(ROOT / 'warmup_inputs_manifest_v2.json')

    trials = QUEUES[args.worker]
    summaries = []

    print(f"=== STARTING WORKER: {args.worker} ===", flush=True)
    print(f"Total trials in queue: {len(trials)}", flush=True)
    for run_name, method, alpha, jepa_weight, warmup_ratio, seed in trials:
        print(f"  - {run_name}: method={method} alpha={alpha} lambda={jepa_weight} warmup={warmup_ratio} seed={seed}", flush=True)

    for run_name, method, alpha, jepa_weight, warmup_ratio, seed in trials:
        folder = runs_dir / run_name
        complete_marker = folder / "COMPLETE.json"
        metrics_file = folder / "metrics.json"

        if complete_marker.exists() and metrics_file.exists():
            print(f"[skip] TRIAL_ALREADY_COMPLETE: {run_name}", flush=True)
            summaries.append(json.loads(metrics_file.read_text(encoding="utf-8")))
            continue

        if folder.exists():
            raise RuntimeError(f"Incomplete run exists; explicit recovery needed, no deletion: {folder}")

        command = [
            sys.executable, "-u", "-m", "amis_rewire.train",
            "--model", "mbart-large-50",
            "--model-name", "facebook/mbart-large-50-many-to-many-mmt",
            "--model-revision", REVISION,
            "--scientific-records",
            "--hf-backup-repo", "FiveC/amis-rewire-checkpoints",
            "--hf-backup-prefix", "amis_routing_warmup_20261004_v2",
            "--method", method,
            "--rewire-stack", "encoder",
            "--rewire-distance", "2",
            "--rewire-strength", str(alpha),
            "--jepa-weight", str(jepa_weight),
            "--routing-warmup-ratio", str(warmup_ratio),
            "--data-dir", str(data_dir),
            "--output-dir", str(runs_dir),
            "--backup-dir", str(backups_dir),
            "--run-name", run_name,
            "--seed", str(seed),
            "--learning-rate", "5e-5",
            "--weight-decay", "0.0",
            "--warmup-ratio", "0.06",
            "--num-train-epochs", "20",
            "--early-stopping-patience", "4",
            "--per-device-train-batch-size", "32",
            "--gradient-accumulation-steps", "4",
            "--per-device-eval-batch-size", "8",
            "--max-source-length", "256",
            "--max-target-length", "256",
            "--src-lang", "tl_XX",
            "--tgt-lang", "zh_CN",
            "--bleu-tokenizer", "zh",
            "--eval-beams", "1",
            "--num-beams", "4",
            "--generation-precision", "fp32",
            "--bf16",
            "--no-fp16",
            "--gradient-checkpointing",
            "--no-auto-resume",
            "--save-total-limit", "2",
            "--dataloader-num-workers", "2",
            "--logging-steps", "10",
            "--validation-only",
        ]

        save_json(out_base / "status.json", {
            "status": "TRAINING",
            "worker": args.worker,
            "run": run_name,
            "method": method,
            "alpha": alpha,
            "lambda": jepa_weight,
            "warmup_ratio": warmup_ratio,
            "seed": seed,
            "completed_runs": len(summaries),
            "total_queue_runs": len(trials),
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

        log_path = out_base / f"{run_name}.log"
        print(f"\n>>> TRIAL_START: {run_name} (method={method}, seed={seed}, alpha={alpha}, lambda={jepa_weight}, warmup={warmup_ratio})", flush=True)

        t0 = time.time()
        worker_env = os.environ.copy()
        src_path = str(ROOT / "src")
        worker_env["PYTHONPATH"] = f"{src_path}:{worker_env.get('PYTHONPATH', '')}" if worker_env.get("PYTHONPATH") else src_path
        with log_path.open("w", encoding="utf-8") as log_f:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=worker_env
            )
        elapsed = time.time() - t0

        if result.returncode != 0:
            save_json(out_base / "status.json", {
                "status": "FAILED",
                "worker": args.worker,
                "run": run_name,
                "exit_code": result.returncode,
                "elapsed_seconds": elapsed,
            })
            raise RuntimeError(f"Trial {run_name} failed with exit code {result.returncode}. See {log_path}")
        save_json(folder / 'command_receipt.json', {'command': command,
                  'manifest_sha256': digest(ROOT / 'warmup_inputs_manifest_v2.json')})
        history = json.loads((folder / 'training_history.json').read_text())
        assert json.loads((folder / 'eval_loss_trace.json').read_text()), 'Missing validation history'
        schedule = [json.loads(line) for line in (folder / 'routing_schedule.jsonl').read_text().splitlines()]
        assert len(schedule) == json.loads((folder / 'final_trainer_state.json').read_text())['global_step']
        assert [r['optimizer_step_before_update'] for r in schedule] == list(range(len(schedule)))
        for r in schedule:
            expected_alpha = alpha * min(1, r['optimizer_step_before_update'] / int(r['planned_steps'] * warmup_ratio))
            assert abs(r['alpha_train'] - expected_alpha) < 1e-12 and r['lambda'] == jepa_weight
        save_json(out_base / 'status.json', {'status': 'PROBING_AND_RELOADING', 'worker': args.worker, 'run': run_name})
        with log_path.open('a', encoding='utf-8') as log_f:
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/revalidation/probe_amis_warmup_trajectory.py'),
                            '--run-dir', str(folder)], cwd=ROOT, env=worker_env, stdout=log_f,
                           stderr=subprocess.STDOUT, check=True)

        # Post-process: compute native chrF0 on validation_predictions.csv and update metrics.json
        val_csv = folder / "validation_predictions.csv"
        if not val_csv.exists() or not metrics_file.exists():
            raise RuntimeError(f"Missing expected output files for {run_name}")

        with val_csv.open(encoding="utf-8-sig", newline="") as f:
            val_rows = list(csv.DictReader(f))

        preds = [r["prediction"] for r in val_rows]
        refs = [[r["target"] for r in val_rows]]
        chrf0_score = CHRF(word_order=0).corpus_score(preds, refs).score

        metadata = json.loads(metrics_file.read_text(encoding="utf-8"))
        metadata["eval_chrf0"] = float(chrf0_score)
        metadata["routing_warmup_ratio"] = warmup_ratio
        metadata["elapsed_seconds"] = elapsed
        save_json(metrics_file, metadata)

        # Receipt
        receipt = {
            "run": run_name,
            "method": method,
            "alpha": alpha,
            "lambda": jepa_weight,
            "warmup_ratio": warmup_ratio,
            "seed": seed,
            "eval_bleu": metadata.get("eval_bleu"),
            "eval_chrf++": metadata.get("eval_chrf++"),
            "eval_chrf0": metadata.get("eval_chrf0"),
            "validation_prediction_sha256": digest(val_csv),
            "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
        }
        archive_path = backups_dir / run_name / f"{run_name}-best.zip"
        assert archive_path.exists(), 'Missing best archive'
        receipt["archive_sha256"] = digest(archive_path)
        receipt["archive_bytes"] = archive_path.stat().st_size
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ['HF_TOKEN'])
        hf_path = f'amis_routing_warmup_20261004_v2/{run_name}/{archive_path.name}'
        info = api.upload_file(path_or_fileobj=str(archive_path), path_in_repo=hf_path,
                               repo_id='FiveC/amis-rewire-checkpoints', repo_type='model')
        remote = api.get_paths_info('FiveC/amis-rewire-checkpoints', [hf_path], revision=info.oid, repo_type='model')[0]
        assert remote.size == receipt['archive_bytes'] and remote.lfs.sha256 == receipt['archive_sha256']
        receipt.update(hf_repo='FiveC/amis-rewire-checkpoints', hf_path=hf_path, hf_revision=info.oid,
                       best_reload='PASS', trajectory_probes='PASS', schedule_steps=len(schedule))
        assert (folder / 'BEST_RELOAD_PASS.json').exists() and (folder / 'TRAJECTORY_PROBES_PASS.json').exists()
        api.upload_folder(folder_path=str(folder), repo_id='FiveC/amis-rewire-checkpoints', repo_type='model',
                          path_in_repo=f'amis_routing_warmup_20261004_v2/{run_name}',
                          allow_patterns=['*.json', '*.jsonl', '*.csv', '*.npz', '*.txt'])
        save_json(complete_marker, receipt)
        api.upload_file(path_or_fileobj=str(complete_marker), repo_id='FiveC/amis-rewire-checkpoints',
                         path_in_repo=f'amis_routing_warmup_20261004_v2/{run_name}/COMPLETE.json', repo_type='model')

        summaries.append(metadata)
        save_json(out_base / f"warmup_{args.worker}_metrics.json", summaries)

        print(f">>> TRIAL_COMPLETE: {run_name} (elapsed {elapsed:.1f}s)", flush=True)
        print(f"    VALIDATION_BLEU:   {metadata.get('eval_bleu', 0):.4f}", flush=True)
        print(f"    VALIDATION_CHRF++: {metadata.get('eval_chrf++', 0):.4f}", flush=True)
        print(f"    VALIDATION_CHRF0:  {metadata.get('eval_chrf0', 0):.4f}", flush=True)

        # Clean checkpoint folders to save remote disk space, keeping best_model and backups
        for ckpt in folder.glob("checkpoint-*"):
            if ckpt.is_dir():
                shutil.rmtree(ckpt)

    save_json(out_base / "status.json", {
        "status": "WORKER_QUEUE_COMPLETE",
        "worker": args.worker,
        "completed_runs": len(summaries),
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    print(f"=== WORKER_QUEUE_COMPLETE: {args.worker} ===", flush=True)


if __name__ == "__main__":
    main()
