"""One deployment and monitor implementation for all configuration files."""
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from .colab import call_pilot, get_hf_token, get_session_name
from .config import ROOT, verify_frozen
from .integrity import sha256


def call(account, args, timeout=180):
    code, stdout, stderr = call_pilot(str(account), args, timeout=timeout)
    if code:
        raise RuntimeError(f'Colab{account}: {stdout}\n{stderr}')
    return stdout


def launch(config, account, stage='gate-train'):
    out, plan = verify_frozen(config)
    selected = [s for s in plan['runs'] if s['account'] == account]
    if not selected:
        raise ValueError('No work assigned to this account')
    cpu = json.loads((out / 'CPU_GATE.json').read_text())
    assert cpu['passed'] and cpu['bundle_sha256'] == sha256(out / 'bundle_manifest.json')
    assert {r['id'] for r in cpu['cases']} == {r['id'] for r in plan['runs']}
    if (out / f'launch_{stage}_account{account}.json').exists():
        raise FileExistsError('This queue already has a launch receipt')
    # Check destinations before any paid GPU work. Reference configurations
    # point to existing checkpoints and must not be launched over those models.
    from huggingface_hub import HfApi
    api = HfApi(token=False)
    assert not api.model_info(plan['repo']).private
    if stage == 'gate-train':
        for spec in selected:
            if api.file_exists(plan['repo'], spec['remote_prefix'] + '/best.zip'):
                raise FileExistsError(f"Existing public checkpoint: {spec['remote_prefix']}")
    else:
        if plan['protocol']['test_barrier'].startswith('No test'):
            raise ValueError('Validation-only configuration')
        lock = json.loads((out / 'selection_lock.json').read_text())
        assert lock['status'] == 'ALL_VALIDATION_VERIFIED'
        assert lock['plan_sha256'] == sha256(out / 'PLAN.json')
    session = get_session_name(str(account))
    current = call(account, ['status', '-s', session])
    if 'gpu-a100-s-' not in current or 'Standard' not in current:
        raise RuntimeError('Prepare an A100 Standard session first; launch does not create/replace sessions')
    name = Path(config).stem
    remote = f'/content/CLRR_experiments/{name}'
    cache = ROOT / '.cache/colab_ops' / name
    cache.mkdir(parents=True, exist_ok=True)
    guard = cache / 'guard.py'
    guard.write_text("import subprocess\nr=subprocess.run(['pgrep','-af','python.*(scripts/revalidation/run_|scripts/experiment.py run)'],capture_output=True,text=True)\nassert not r.stdout.strip(), 'Another research worker is active'\nprint('IDLE_RESEARCH_RUNTIME')\n", encoding='utf-8')
    call(account, ['exec', '-s', session, '-f', str(guard), '--timeout', '60'])
    deployment = cache / 'deployment.zip'
    shutil.copy2(ROOT / '.cache/experiment_bundles' / f'{name}.zip', deployment)
    with zipfile.ZipFile(deployment, 'a', zipfile.ZIP_DEFLATED) as archive:
        archive.write(out / 'CPU_GATE.json', (out / 'CPU_GATE.json').relative_to(ROOT).as_posix())
        if stage == 'test':
            archive.write(out / 'selection_lock.json', (out / 'selection_lock.json').relative_to(ROOT).as_posix())
    call(account, ['upload', str(deployment), f'/content/{name}.zip', '-s', session])
    fd, token_path = tempfile.mkstemp(prefix='clrr-token-')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(get_hf_token())
        call(account, ['upload', token_path, f'/content/.clrr_token_{name}', '-s', session])
    finally:
        Path(token_path).unlink(missing_ok=True)
    shell = cache / 'run.sh'
    shell.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
if [ ! -x /content/miniforge3/bin/conda ]; then
 curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /content/miniforge.sh
 bash /content/miniforge.sh -b -p /content/miniforge3
fi
if [ ! -x /content/miniforge3/envs/clrr/bin/python ]; then
 /content/miniforge3/bin/conda create -y -n clrr python=3.13 pip
fi
mkdir -p {remote}
cd {remote}
unzip -q -o /content/{name}.zip
PY=/content/miniforge3/envs/clrr/bin/python
"$PY" -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install --no-cache-dir -r requirements/colab-experiments.txt
export PYTHONPATH={remote}/src
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
chmod 600 /content/.clrr_token_{name}
export HF_TOKEN=$(cat /content/.clrr_token_{name})
"$PY" -u scripts/experiment.py run --config configs/experiments/{name}.json --account {account} --stage {stage}
''', encoding='utf-8', newline='\n')
    call(account, ['upload', str(shell), f'/content/run_{name}.sh', '-s', session])
    starter = cache / 'start.py'
    starter.write_text(f"import subprocess\nfrom pathlib import Path\nroot=Path({remote!r})\nroot.mkdir(parents=True,exist_ok=True)\npid=root/'queue.pid'\nif pid.exists():\n s=Path('/proc')/pid.read_text().strip()/'stat'\n assert not s.exists() or s.read_text().rsplit(')',1)[1].split()[0] in ('Z','X'), 'Queue active'\nwith (root/'queue.log').open('w') as log:\n p=subprocess.Popen(['bash','/content/run_{name}.sh'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\npid.write_text(str(p.pid))\nprint(p.pid)\n", encoding='utf-8')
    output = call(account, ['exec', '-s', session, '-f', str(starter), '--timeout', '60'])
    receipt = {'account': account, 'session': session, 'remote_root': remote,
        'assigned': [s['id'] for s in selected], 'deployment_sha256': sha256(deployment),
        'status': f'LAUNCHED_{stage.upper()}', 'response': output.strip()}
    (out / f'launch_{stage}_account{account}.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


def monitor(config, account):
    out, plan = verify_frozen(config)
    name = Path(config).stem
    cache = ROOT / '.cache/colab_ops' / name
    cache.mkdir(parents=True, exist_ok=True)
    script = cache / 'monitor.py'
    remote_out = f'/content/CLRR_experiments/{name}/outputs/experiments/{name}'
    script.write_text(f"import json\nfrom pathlib import Path\nroot=Path({remote_out!r})\nfiles=[p for p in root.rglob('*.json') if p.name in ('progress.json','VAL_COMPLETE.json','COMPLETE.json','PASS.json') or p.name.startswith('queue_')]\nprint(json.dumps({{p.relative_to(root).as_posix():json.loads(p.read_text()) for p in files}}))\n", encoding='utf-8')
    response = call(account, ['exec', '-s', get_session_name(str(account)), '-f', str(script), '--timeout', '60'])
    result = next(json.loads(line) for line in response.splitlines() if line.startswith('{'))
    for relative, record in result.items():
        destination = (out / relative).resolve()
        if not destination.is_relative_to(out.resolve()):
            raise ValueError('Unsafe monitor path')
        if destination.name not in ('progress.json', 'VAL_COMPLETE.json', 'COMPLETE.json', 'PASS.json') and not destination.name.startswith('queue_'):
            raise ValueError('Unexpected monitor artifact')
        if destination.name in ('VAL_COMPLETE.json', 'COMPLETE.json', 'PASS.json') and destination.exists():
            assert json.loads(destination.read_text()) == record, 'Immutable receipt changed'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=2), encoding='utf-8')
    target = out / 'status' / f'account{account}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
