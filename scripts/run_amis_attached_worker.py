"""Validation-only attached-source CLRR confirmation runner (Phase C2)."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url, snapshot_download
from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs_rebuttal/amis_detach_control_20261004'
REPO = 'FiveC/amis-rewire-checkpoints'
PREFIX = 'amis_detach_control_20261004_v1'
REVISION = 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30'

QUEUES = {
    'machine1': [
        ('mbart-amis-full-attached-a010-l010-seed42', 0.10, 0.10, 42),
        ('mbart-amis-full-attached-a010-l010-seed44', 0.10, 0.10, 44),
    ],
    'machine2': [
        ('mbart-amis-full-attached-a010-l010-seed43', 0.10, 0.10, 43),
    ]
}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run_gpu_gate():
    print('STAGE_START gpu_smoke_gate', flush=True)
    smoke_cmd = [sys.executable, str(ROOT/'tests/smoke_attached_routing.py'), '--cuda']
    smoke_res = subprocess.run(smoke_cmd, cwd=ROOT, capture_output=True, text=True)
    if smoke_res.returncode != 0:
        print('GPU_SMOKE_FAILED', smoke_res.stderr, flush=True)
        raise RuntimeError('GPU smoke test failed')
    print('GPU_SMOKE_PASSED', smoke_res.stdout.strip(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['machine1', 'machine2'], required=True)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=os.environ['HF_TOKEN'])
    assert api.whoami()['name'] == 'FiveC'

    # 1. Run GPU Gate before training
    run_gpu_gate()

    # 2. Pin base model
    model = snapshot_download(
        'facebook/mbart-large-50-many-to-many-mmt',
        revision=REVISION,
        token=False,
        allow_patterns=[
            'config.json', 'generation_config.json', 'model.safetensors',
            'sentencepiece.bpe.model', 'tokenizer_config.json', 'special_tokens_map.json'
        ]
    )

    trials = QUEUES[args.worker]
    summaries = []

    for run, alpha, weight, seed in trials:
        folder = OUT/'runs'/run
        if (folder/'COMPLETE.json').exists():
            print('TRIAL_ALREADY_COMPLETE', run, flush=True)
            summaries.append(json.loads((folder/'metrics.json').read_text()))
            continue

        assert not folder.exists(), f'Inspect incomplete trial before resuming: {run}'
        assert shutil.disk_usage(ROOT).free > 20 * 1024**3, 'Insufficient disk space'

        command = [
            sys.executable, '-u', str(ROOT/'scripts/revalidation/train_amis_no_detach.py'),
            '--model', 'mbart-large-50',
            '--model-name', model,
            '--method', 'jepa-clrr-enc',
            '--rewire-stack', 'encoder',
            '--rewire-distance', '2',
            '--rewire-strength', str(alpha),
            '--jepa-weight', str(weight),
            '--data-dir', str(ROOT/'data_processed/amis_mandarin'),
            '--output-dir', str(OUT/'runs'),
            '--backup-dir', str(OUT/'backups'),
            '--run-name', run,
            '--seed', str(seed),
            '--learning-rate', '5e-5',
            '--weight-decay', '0',
            '--warmup-ratio', '0.06',
            '--num-train-epochs', '20',
            '--early-stopping-patience', '4',
            '--per-device-train-batch-size', '32',
            '--gradient-accumulation-steps', '4',
            '--per-device-eval-batch-size', '8',
            '--max-source-length', '256',
            '--max-target-length', '256',
            '--src-lang', 'tl_XX',
            '--tgt-lang', 'zh_CN',
            '--bleu-tokenizer', 'zh',
            '--eval-beams', '1',
            '--num-beams', '4',
            '--generation-precision', 'fp32',
            '--bf16', '--no-fp16',
            '--gradient-checkpointing',
            '--no-auto-resume',
            '--save-total-limit', '2',
            '--dataloader-num-workers', '2',
            '--logging-steps', '10',
            '--validation-only',
            '--hf-backup-repo', REPO,
            '--hf-backup-prefix', PREFIX
        ]

        save(OUT/'status.json', dict(
            status='TRAINING_VALIDATION_ONLY',
            worker=args.worker,
            run=run,
            alpha=alpha,
            lambda_=weight,
            seed=seed,
            completed_runs=len(summaries),
            total_queue_runs=len(trials),
            started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        ))
        save(OUT/'current_command.json', dict(command=command))

        print('TRIAL_START', run, f'seed={seed}', f'alpha={alpha}', f'weight={weight}', flush=True)
        with (OUT/(run + '.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

        if result.returncode:
            save(OUT/'status.json', dict(
                status='FAILED',
                worker=args.worker,
                run=run,
                exit_code=result.returncode,
                completed_runs=len(summaries)
            ))
            raise RuntimeError(f'Trial failed; inspect log: {run}')

        metadata = json.loads((folder/'metrics.json').read_text())
        assert metadata['test_evaluated'] is False and not (folder/'test_predictions.csv').exists()
        assert metadata['parameters'] == metadata['trainable_parameters'] == 610879488
        assert metadata['configuration']['clrr_source_detach'] is False
        assert metadata['configuration']['lsr_target_detach'] is True

        with (folder/'validation_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
            predictions = list(csv.DictReader(stream))
        metadata['eval_chrf0'] = CHRF(word_order=0).corpus_score(
            [r['prediction'] for r in predictions], [[r['target'] for r in predictions]]
        ).score
        save(folder/'metrics.json', metadata)

        archive = OUT/'backups'/run/(run + '-best.zip')
        sha = digest(archive)
        remote_path = PREFIX + '/' + run + '/' + archive.name

        info = get_hf_file_metadata(hf_hub_url(REPO, remote_path), token=os.environ['HF_TOKEN'])
        assert info.size == archive.stat().st_size and info.etag == sha, 'Backup verification failed'

        receipt = dict(
            run=run,
            alpha=alpha,
            lambda_=weight,
            seed=seed,
            repo=REPO,
            path=remote_path,
            revision=info.commit_hash,
            archive_sha256=sha,
            archive_bytes=info.size,
            validation_prediction_sha256=digest(folder/'validation_predictions.csv'),
            test_evaluated=False
        )
        save(folder/'COMPLETE.json', receipt)

        # Upload metrics, receipt, and predictions to HF
        for artifact in (folder/'metrics.json', folder/'COMPLETE.json', folder/'validation_predictions.csv'):
            api.upload_file(
                path_or_fileobj=str(artifact),
                path_in_repo=PREFIX + '/' + run + '/' + artifact.name,
                repo_id=REPO
            )

        summaries.append(metadata)
        save(OUT/f'detach_control_{args.worker}_metrics.json', summaries)
        print('TRIAL_COMPLETE', run,
              'VALIDATION_CHRFPP', metadata['eval_chrf++'],
              'VALIDATION_BLEU', metadata['eval_bleu'],
              'VALIDATION_CHRF0', metadata['eval_chrf0'], flush=True)

        # Remove weights on disk to save disk space
        for path in [*folder.glob('checkpoint-*'), folder/'best_model']:
            assert path.resolve().is_relative_to((OUT/'runs').resolve())
            if path.exists():
                shutil.rmtree(path)
        assert archive.resolve().is_relative_to((OUT/'backups').resolve())
        archive.unlink()

    save(OUT/'status.json', dict(
        status='WORKER_QUEUE_COMPLETE',
        worker=args.worker,
        completed_runs=len(summaries)
    ))
    print('WORKER_QUEUE_COMPLETE', args.worker, flush=True)


if __name__ == '__main__':
    main()
