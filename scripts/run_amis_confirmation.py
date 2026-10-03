"""Execute 12 prespecified, matched mBART Amis trials; never launch Turkish."""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs_rebuttal/amis_confirmation_20261003'
REVISION = 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def semantic(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert set(rows[0]) == {'source', 'target'}
    return hashlib.sha256(json.dumps([[r['source'],r['target']] for r in rows],
                           ensure_ascii=False,separators=(',',':')).encode()).hexdigest(), len(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected = json.loads((ROOT / 'amis_confirmation_inputs.json').read_text())
    data = ROOT / 'data_processed/amis_mandarin'
    data.mkdir(parents=True, exist_ok=True)
    for split, spec in expected['data'].items():
        path = data / (split + '.csv')
        with urllib.request.urlopen(spec['public_url'], timeout=60) as response: path.write_bytes(response.read())
        assert semantic(path) == (spec['semantic_sha256'], spec['rows']), split
    gate = json.loads((ROOT / 'outputs_rebuttal/followup_20261003/precision_training_smoke_gpu.json').read_text())
    assert len(gate) == 4 and all(r['passed'] and r['cuda_bf16_training'] for r in gate)
    trajectory_gate = json.loads((OUT/'trajectory_smoke_gpu.json').read_text())
    assert len(trajectory_gate)==4 and all(r['passed'] for r in trajectory_gate)
    calibration = json.loads((ROOT/'outputs_rebuttal/followup_20261003/batch_calibration.json').read_text())
    assert calibration['selected_microbatch']==32, 'Shared batch32 capacity gate failed; inspect before changing protocol'
    model = snapshot_download(repo_id='facebook/mbart-large-50-many-to-many-mmt', revision=REVISION, token=False,
        allow_patterns=['config.json','generation_config.json','model.safetensors',
                        'sentencepiece.bpe.model','tokenizer_config.json','special_tokens_map.json'])
    commands = []
    for seed in [42,43,44]:
        for name, method, weight in [('baseline','baseline',0),('lsr','jepa',.1),('clrr','clrr-enc',0),('full','jepa-clrr-enc',.1)]:
            run = f'mbart-amis-{name}-seed{seed}'
            cmd = [sys.executable,'-u','scripts/revalidation/measure_training_trajectory.py',
                '--probe-output-dir',str(OUT/'trajectory'/run),'--model','mbart-large-50',
                '--model-name',model,'--method',method,'--jepa-weight',str(weight),
                '--data-dir',str(data),'--output-dir',str(OUT/'runs'),'--backup-dir',str(OUT/'backups'),
                '--run-name',run,'--seed',str(seed),'--learning-rate','5e-5','--weight-decay','0',
                '--warmup-ratio','0.06','--num-train-epochs','20','--early-stopping-patience','4',
                '--per-device-train-batch-size','32','--gradient-accumulation-steps','4',
                '--per-device-eval-batch-size','8','--max-source-length','256','--max-target-length','256',
                '--rewire-distance','2','--rewire-strength','0.1','--rewire-stack','encoder',
                '--src-lang','tl_XX','--tgt-lang','zh_CN','--bleu-tokenizer','zh',
                '--eval-beams','1','--num-beams','4','--generation-precision','fp32',
                '--bf16','--no-fp16','--gradient-checkpointing','--no-auto-resume',
                '--save-total-limit','2','--dataloader-num-workers','2','--logging-steps','10']
            commands.append({'seed':seed,'method':name,'run_name':run,'command':cmd})
    protocol = {'model_revision':REVISION,'inputs':expected,'runs':commands,'seeds':[42,43,44],
        'scope':'fresh matched training; not a continuation of historical runs',
        'selection':'best raw-reference validation chrF++; test not used to choose checkpoints/hyperparameters',
        'precision':'BF16/TF32 training; explicit FP32/TF32-off generation for all branches',
        'probes':'epoch1, fixed epoch5, actual stopped/final epoch; fixed validation data and restored training RNG',
        'primary_comparisons':['Full-Baseline','Full-LSR-only'],
        'secondary_comparisons':['CLRR-only-Baseline','Full-CLRR-only'],
        'statistics':'mean/std and all paired seed deltas; 10k paired sentence bootstrap per seed, Holm24 (4pairs x2metrics x3seeds); no claim sentence bootstrap proves seed stability',
        'decision':'report all12runs and negative outcomes; no test-driven retuning; with only3seeds inference over training seeds remains limited',
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in expected['source_files']}}
    protocol_file = OUT/'protocol.json'
    if protocol_file.exists():
        assert json.loads(protocol_file.read_text()) == protocol, 'Protocol changed; do not resume implicitly'
    else: save(protocol_file,protocol)
    summary = []
    for trial in commands:
        folder = OUT/'runs'/trial['run_name']
        if (folder/'COMPLETE.json').exists():
            summary.append(json.loads((folder/'metrics.json').read_text()));continue
        if folder.exists(): raise RuntimeError('Incomplete run requires inspected explicit resume: '+trial['run_name'])
        assert shutil.disk_usage(ROOT).free > 22*1024**3, 'Insufficient disk before run'
        save(OUT/'status.json',{'status':'TRAINING','running':trial,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'completed_runs':len(summary)})
        print('TRIAL_START',trial['run_name'],flush=True)
        log = OUT/(trial['run_name']+'.log')
        with log.open('w') as stream:
            result = subprocess.run(trial['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            save(OUT/'status.json',{'status':'FAILED','running':trial,'exit_code':result.returncode,'completed_runs':len(summary)})
            raise RuntimeError(f'Trial failed without retry: {trial["run_name"]}')
        metrics = json.loads((folder/'metrics.json').read_text())
        assert metrics['trainable_parameters']==metrics['parameters']==610879488
        assert metrics['configuration']['generation_precision']=='fp32'
        archive = OUT/'backups'/trial['run_name']/(trial['run_name']+'-best.zip')
        assert archive.exists()
        save(folder/'COMPLETE.json',{'run_name':trial['run_name'],'seed':trial['seed'],
            'archive':str(archive),'archive_bytes':archive.stat().st_size,
            'archive_sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()})
        summary.append(metrics)
        save(OUT/'all_metrics.json',summary)
        print('TRIAL_COMPLETE',trial['run_name'],metrics['test_bleu'],metrics['test_chrf++'],flush=True)
        # Preserve best archive; remove redundant resumable checkpoints only inside this completed run.
        for checkpoint in folder.glob('checkpoint-*'):
            assert checkpoint.resolve().is_relative_to((OUT/'runs').resolve())
            shutil.rmtree(checkpoint)
        best = folder/'best_model'
        assert best.resolve().is_relative_to((OUT/'runs').resolve())
        shutil.rmtree(best)
    save(OUT/'status.json',{'status':'TRAINING_COMPLETE','completed_runs':12})
    save(OUT/'COMPLETE.json',{'runs':12,'seeds':[42,43,44]})
    print('ALL12_TRIALS_COMPLETE',flush=True)


if __name__ == '__main__': main()
