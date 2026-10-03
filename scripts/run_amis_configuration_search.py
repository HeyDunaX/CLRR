"""Validation-only Amis tuning; upload and verify each run before removing weights."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url, snapshot_download
from sacrebleu.metrics import CHRF
import csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs_rebuttal/amis_configuration_20261003_v1'
REPO = 'FiveC/amis-rewire-checkpoints'
PREFIX = 'amis_configuration_20261003_v1'
REVISION = 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT/'configuration_inputs.json').read_text(encoding='utf-8'))
    for name, expected in protocol['files'].items():
        assert digest(ROOT/name) == expected, name
    api = HfApi(token=os.environ['HF_TOKEN'])
    assert api.whoami()['name'] == 'FiveC'
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text()) == protocol
    else:
        assert not any(path.startswith(PREFIX+'/') for path in api.list_repo_files(REPO)), 'New prefix already used'
        save(OUT/'protocol.json', protocol)
        api.upload_file(path_or_fileobj=str(OUT/'protocol.json'),path_in_repo=PREFIX+'/protocol.json',repo_id=REPO)
    model = snapshot_download('facebook/mbart-large-50-many-to-many-mmt', revision=REVISION,
        token=False, allow_patterns=['config.json','generation_config.json','model.safetensors',
        'sentencepiece.bpe.model','tokenizer_config.json','special_tokens_map.json'])
    trials = [('full_a005_l010','full',.05,.1), ('full_a020_l010','full',.2,.1),
              ('full_a010_l003','full',.1,.03), ('lsr_l003','lsr',0,.03),
              ('full_a010_l030','full',.1,.3), ('lsr_l030','lsr',0,.3),
              ('lsr_l001','lsr',0,.01), ('lsr_l060','lsr',0,.6)]
    summaries = []
    for label, method, alpha, weight in trials:
        run = 'mbart-amis-'+label+'-seed42'
        folder = OUT/'runs'/run
        if (folder/'COMPLETE.json').exists():
            summaries.append(json.loads((folder/'metrics.json').read_text()))
            continue
        assert not folder.exists(), 'Inspect incomplete trial before explicitly resuming: '+run
        assert shutil.disk_usage(ROOT).free > 22*1024**3
        command = [sys.executable,'-u','-m','amis_rewire.train','--model','mbart-large-50',
            '--model-name',model,'--method','jepa-clrr-enc' if method=='full' else 'jepa',
            '--data-dir',str(ROOT/'data_processed/amis_mandarin'), '--output-dir',str(OUT/'runs'),
            '--backup-dir',str(OUT/'backups'),'--run-name',run,'--seed','42',
            '--learning-rate','5e-5','--weight-decay','0','--warmup-ratio','0.06',
            '--num-train-epochs','20','--early-stopping-patience','4',
            '--per-device-train-batch-size','32','--gradient-accumulation-steps','4',
            '--per-device-eval-batch-size','8','--max-source-length','256','--max-target-length','256',
            '--rewire-distance','2','--rewire-strength',str(alpha),'--rewire-stack','encoder',
            '--jepa-weight',str(weight),'--src-lang','tl_XX','--tgt-lang','zh_CN',
            '--bleu-tokenizer','zh','--eval-beams','1','--num-beams','4',
            '--generation-precision','fp32','--bf16','--no-fp16','--gradient-checkpointing',
            '--no-auto-resume','--save-total-limit','2','--dataloader-num-workers','2',
            '--logging-steps','10','--validation-only','--hf-backup-repo',REPO,
            '--hf-backup-prefix',PREFIX]
        save(OUT/'status.json',dict(status='TRAINING_VALIDATION_ONLY',run=run,alpha=alpha,lambda_=weight,
            completed_runs=len(summaries),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        save(OUT/'current_command.json',dict(command=command))
        print('TRIAL_START',run,flush=True)
        with (OUT/(run+'.log')).open('w') as log:
            result = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            save(OUT/'status.json',dict(status='FAILED',run=run,exit_code=result.returncode,completed_runs=len(summaries)))
            raise RuntimeError('Trial failed; inspect log: '+run)
        metadata = json.loads((folder/'metrics.json').read_text())
        assert metadata['test_evaluated'] is False and not (folder/'test_predictions.csv').exists()
        assert metadata['parameters'] == metadata['trainable_parameters'] == 610879488
        with (folder/'validation_predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
            predictions = list(csv.DictReader(stream))
        metadata['eval_chrf0'] = CHRF(word_order=0).corpus_score(
            [r['prediction'] for r in predictions], [[r['target'] for r in predictions]]).score
        save(folder/'metrics.json',metadata)
        archive = OUT/'backups'/run/(run+'-best.zip')
        sha = digest(archive)
        remote = PREFIX+'/'+run+'/'+archive.name
        info = get_hf_file_metadata(hf_hub_url(REPO,remote),token=os.environ['HF_TOKEN'])
        assert info.size == archive.stat().st_size and info.etag == sha, 'Backup verification failed'
        receipt = dict(run=run,alpha=alpha,lambda_=weight,seed=42,repo=REPO,path=remote,
            revision=info.commit_hash,archive_sha256=sha,archive_bytes=info.size,
            validation_prediction_sha256=digest(folder/'validation_predictions.csv'),test_evaluated=False)
        save(folder/'COMPLETE.json',receipt)
        for artifact in (folder/'metrics.json',folder/'COMPLETE.json'):
            api.upload_file(path_or_fileobj=str(artifact),path_in_repo=PREFIX+'/'+run+'/'+artifact.name,repo_id=REPO)
        summaries.append(metadata)
        save(OUT/'screening_metrics.json',summaries)
        print('TRIAL_COMPLETE',run,'VALIDATION_CHRFPP',metadata['eval_chrf++'],
              'VALIDATION_BLEU',metadata['eval_bleu'],'VALIDATION_CHRF0',metadata['eval_chrf0'],flush=True)
        for path in [*folder.glob('checkpoint-*'),folder/'best_model']:
            assert path.resolve().is_relative_to((OUT/'runs').resolve())
            if path.exists():
                shutil.rmtree(path)
        assert archive.resolve().is_relative_to((OUT/'backups').resolve())
        archive.unlink()
    save(OUT/'status.json',dict(status='SCREENING_COMPLETE',completed_runs=8,
        next_stage='Freeze candidates using validation; inspect selection robustness before seed confirmation'))
    print('ALL8_SCREENING_COMPLETE_NO_TEST',flush=True)


if __name__ == '__main__':
    main()
