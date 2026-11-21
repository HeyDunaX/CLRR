"""Deploy frozen recipes to the author's private RTX project via Tailscale."""
import base64
import json
import re
import subprocess
import zipfile
from pathlib import Path

from .config import ROOT, verify_frozen
from .integrity import sha256

PROJECT = '/workspace/ura/thanhdat/projects/CLRR'
SSH = ['C:/Program Files/Tailscale/tailscale.exe', 'ssh', 'ura@100.90.53.12', PROJECT + '/conda/envs/clrr/bin/python', '-']


def call(code):
    result = subprocess.run(SSH, input=code, capture_output=True, text=True,
                            encoding='utf-8', timeout=90)
    if result.returncode:
        raise RuntimeError(result.stdout + '\n' + result.stderr)
    return json.loads(result.stdout)


def launch_technical(configs):
    """Detached technical-only suite. Never emits training-admission receipts."""
    from methods.difference_preserving.v1 import METHODS
    items, common, commands = [], {}, []
    for config in configs:
        out, plan = verify_frozen(config)
        assert plan.get('technical_only_authorized') is True
        assert plan['protocol']['hardware_profile'] == 'rtx6000-fp32'
        assert all(s['method'] in METHODS for s in plan['runs'])
        assert not plan.get('post_training_test', {}).get('authorized')
        assert plan['scientific_admission'] == {'novelty': 'HOLD', 'semantic_pairs': 'HOLD'}
        name = Path(config).stem
        for file, digest in json.loads((out / 'bundle_manifest.json').read_text()).items():
            if file in common:
                assert common[file] == digest, ('Conflicting frozen source', file)
            common[file] = digest
        bundle = ROOT / '.cache/experiment_bundles' / f'{name}.zip'
        items.append((name, sha256(bundle), base64.b64encode(bundle.read_bytes()).decode()))
        commands.append(f'conda run --no-capture-output -n clrr python -u scripts/experiment.py cpu --config configs/experiments/{name}.json')
        for spec in plan['runs']:
            commands.append(f'conda run --no-capture-output -n clrr python -u scripts/experiment.py run --config configs/experiments/{name}.json --id {spec["id"]} --stage technical-gate')
    assert items and len({n for n, _, _ in items}) == len(items)
    suite = items[0][0] + '_suite'
    local = ROOT / 'outputs/experiments' / suite
    assert not local.exists(), 'Technical suite already launched'
    shell = f'''#!/usr/bin/env bash
set -euo pipefail
source {PROJECT}/environment.sh
cd {PROJECT}
exec 9>{PROJECT}/tmp/training.lock
flock -n 9 || {{ echo PROJECT_BUSY; exit 1; }}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1 HF_XET_CACHE={PROJECT}/cache/xet
export PYTHONPYCACHEPREFIX={PROJECT}/cache/pycache
export TMPDIR={PROJECT}/tmp TMP={PROJECT}/tmp TEMP={PROJECT}/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
''' + '\n'.join(commands) + '\necho TECHNICAL_SUITE_COMPLETE_NO_TRAINING\n'
    state = call(f'''import json,base64,hashlib,zipfile,subprocess,shutil
from pathlib import Path
root=Path({PROJECT!r}).resolve(); out=root/'outputs/experiments'/{suite!r}
assert not out.exists(), 'Technical suite exists'
assert subprocess.run(['flock','-n',str(root/'tmp/training.lock'),'true']).returncode==0, 'Project busy'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],text=True).strip()
assert gpu.startswith('Quadro RTX 6000,') and int(gpu.rsplit(',',1)[1])>=20000, gpu
assert shutil.disk_usage(root).free>40*2**30, 'Insufficient space for transient gate snapshots'
for name,digest,payload in {items!r}:
 assert not (root/'outputs/experiments'/name).exists(), 'Technical workspace exists'
 bundle=root/'cache/bundles'/(name+'.zip');bundle.parent.mkdir(parents=True,exist_ok=True)
 bundle.write_bytes(base64.b64decode(payload));assert hashlib.sha256(bundle.read_bytes()).hexdigest()==digest
 with zipfile.ZipFile(bundle) as z:
  for member in z.namelist():assert (root/member).resolve().is_relative_to(root)
  z.extractall(root)
out.mkdir(parents=True); script=out/'queue.sh';script.write_text({shell!r});script.chmod(0o700)
with (out/'queue.log').open('w') as log:
 p=subprocess.Popen(['bash',str(script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'queue.pid').write_text(str(p.pid))
print(json.dumps({{'status':'TECHNICAL_ONLY_LAUNCHED','pid':p.pid,'suite':{suite!r},'gpu':gpu,'training_authorized':False}}))
''')
    local.mkdir(parents=True)
    state['bundles'] = {n: digest for n, digest, _ in items}
    (local / 'launch.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    return state


def launch_suite(configs):
    """One detached lock-owning queue: all recipe gates before any training."""
    from huggingface_hub import HfApi
    resolved = []
    common_hashes = {}
    api = HfApi(token=False)
    for config in configs:
        out, plan = verify_frozen(config)
        if plan.get('pairing'):
            from methods.difference_preserving.v1.data import require_scientific_admission
            require_scientific_admission(plan)
        assert plan['protocol']['hardware_profile'] == 'rtx6000-fp32'
        assert not plan.get('post_training_test', {}).get('authorized')
        assert all(s['account'] == 1 for s in plan['runs'])
        cpu = json.loads((out/'CPU_GATE.json').read_text())
        assert cpu['passed'] and cpu['bundle_sha256'] == sha256(out/'bundle_manifest.json')
        for name, digest in json.loads((out/'bundle_manifest.json').read_text()).items():
            if name in common_hashes:
                assert common_hashes[name] == digest, ('Incompatible suite bundles', name)
            common_hashes[name] = digest
        assert not api.model_info(plan['repo']).private
        for spec in plan['runs']:
            assert not api.file_exists(plan['repo'], spec['remote_prefix']+'/best.zip'), 'Remote target already exists'
        bundle = ROOT/'.cache/experiment_bundles'/f'{Path(config).stem}.zip'
        resolved.append((Path(config).stem, out, plan, bundle))
    assert len({n for n,_,_,_ in resolved}) == len(resolved) > 0
    suite = resolved[0][0]+'_suite'
    local_out = ROOT/'outputs/experiments'/suite
    assert not local_out.exists(), 'Suite already prepared/launched'
    commands = []
    python = 'conda run --no-capture-output -n clrr python -u scripts/experiment.py'
    for name, _, _, _ in resolved:
        commands.append(f'{python} cpu --config configs/experiments/{name}.json')
    for name, _, plan, _ in resolved:
        for spec in plan['runs']:
            commands.append(f'{python} run --config configs/experiments/{name}.json --id {spec["id"]} --stage gate')
    commands.append('echo SUITE_ALL_GPU_GATES_PASS')
    for name, _, _, _ in resolved:
        commands.append(f'{python} run --config configs/experiments/{name}.json --account 1 --stage train')
    commands.append('echo SUITE_TRAINING_COMPLETE')
    shell = f'''#!/usr/bin/env bash
set -euo pipefail
source {PROJECT}/environment.sh
cd {PROJECT}
exec 9>{PROJECT}/tmp/training.lock
flock -n 9 || {{ echo 'Another CLRR queue is active'; exit 1; }}
export HF_TOKEN=$(cat {PROJECT}/.secrets/hf_token)
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_XET_CACHE={PROJECT}/cache/xet PYTHONPYCACHEPREFIX={PROJECT}/cache/pycache
export TMPDIR={PROJECT}/tmp TMP={PROJECT}/tmp TEMP={PROJECT}/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
''' + '\n'.join(commands) + '\n'
    bundles = [(n, sha256(b), base64.b64encode(b.read_bytes()).decode()) for n,_,_,b in resolved]
    code = f'''import json,base64,hashlib,subprocess,zipfile,shutil
from pathlib import Path
root=Path({PROJECT!r}).resolve();out=root/'outputs/experiments'/{suite!r}
assert not out.exists(), 'Suite already exists remotely'
assert subprocess.run(['flock','-n',str(root/'tmp/training.lock'),'true']).returncode==0, 'Existing project queue active'
assert (root/'.secrets/hf_token').is_file()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],text=True).strip()
assert gpu.startswith('Quadro RTX 6000,')
assert int(gpu.rsplit(',',1)[1]) >= {max(p['protocol']['minimum_free_gpu_mib'] for _,_,p,_ in resolved)}, 'Insufficient free GPU memory for suite'
assert shutil.disk_usage(root).free >= {max(int(p.get('minimum_free_disk_gib', 0)) for _,_,p,_ in resolved)}*2**30, 'Insufficient disk for suite gates/training'
for name,digest,payload in {bundles!r}:
 target=root/'outputs/experiments'/name
 assert not (target/'queue.pid').exists() and not (target/'gates').exists(), 'Recipe already launched'
 bundle=root/'cache/bundles'/(name+'.zip');bundle.parent.mkdir(parents=True,exist_ok=True)
 bundle.write_bytes(base64.b64decode(payload));assert hashlib.sha256(bundle.read_bytes()).hexdigest()==digest
 with zipfile.ZipFile(bundle) as z:
  for n in z.namelist():assert (root/n).resolve().is_relative_to(root)
  z.extractall(root)
out.mkdir(parents=True)
script=out/'queue.sh';script.write_text({shell!r});script.chmod(0o700)
with (out/'queue.log').open('w') as log:
 p=subprocess.Popen(['bash',str(script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'queue.pid').write_text(str(p.pid))
for name,_,_ in {[(n,d,'') for n,d,_ in bundles]!r}:
 target=root/'outputs/experiments'/name
 (target/'queue.pid').write_text(str(p.pid))
 (target/'queue.log').symlink_to(out/'queue.log')
print(json.dumps({{'pid':p.pid,'gpu':gpu,'suite':str(out),'stage':'CPU_THEN_ALL_GPU_GATES_THEN_TRAIN','recipes':{[n for n,_,_,_ in resolved]!r}}}))
'''
    state = call(code)
    local_out.mkdir(parents=True)
    (local_out/'launch_rtx.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    for _, out, _, _ in resolved:
        (out/'launch_rtx.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    return state


def launch(config, account, stage='gate-train'):
    out, plan = verify_frozen(config)
    if plan.get('pairing'):
        from methods.difference_preserving.v1.data import require_scientific_admission
        require_scientific_admission(plan)
    assert account == 1 and stage in ('gate', 'gate-train'), 'RTX supports one validation queue'
    assert plan['protocol']['hardware_profile'] == 'rtx6000-fp32'
    if stage == 'gate':
        assert not plan.get('candidate_pool') and not plan.get('post_training_test', {}).get('authorized'), 'Gate-only launch cannot include pool/test stages'
    assert all(s['account'] == account for s in plan['runs'])
    cpu = json.loads((out / 'CPU_GATE.json').read_text())
    assert cpu['passed'] and cpu['bundle_sha256'] == sha256(out / 'bundle_manifest.json')
    receipt_path = out / ('launch_rtx_gate.json' if stage == 'gate' else 'launch_rtx.json')
    if receipt_path.exists():
        raise FileExistsError('Already launched; monitor or explicitly resume an unfinished ID')
    from huggingface_hub import HfApi
    api = HfApi(token=False)
    assert not api.model_info(plan['repo']).private
    for spec in plan['runs']:
        if api.file_exists(plan['repo'], spec['remote_prefix'] + '/best.zip'):
            raise FileExistsError(spec['remote_prefix'])
    name = Path(config).stem
    bundle = ROOT / '.cache/experiment_bundles' / f'{name}.zip'
    # Reuse the credential already installed inside this authorized project.
    # Credentials never travel in the source bundle or command payload.
    payload = base64.b64encode(bundle.read_bytes()).decode()
    # A later training launch may reuse gate receipts from this exact bundle.
    gates = '\n'.join(f'if [ ! -f outputs/experiments/{name}/gates/{s["id"]}/PASS.json ]; then conda run --no-capture-output -n clrr python -u scripts/experiment.py run --config configs/experiments/{name}.json --id {s["id"]} --stage gate; fi' for s in plan['runs'])
    training = (f'conda run --no-capture-output -n clrr python -u scripts/experiment.py run --config configs/experiments/{name}.json --account 1 --stage train'
                if stage == 'gate-train' else 'echo RTX_GATES_COMPLETE_NO_TRAINING')
    queue_stem = 'gate_queue' if stage == 'gate' else 'queue'
    prepare = ''
    finish = ''
    if plan.get('corrective_order_preparation'):
        assert stage == 'gate-train' and not plan.get('post_training_test', {}).get('authorized')
        prepare = (f'PYTHONPATH={PROJECT}/src conda run --no-capture-output -n clrr python -u '
                   f'-m methods.corrective_order.v1.prepare '
                   f'--config configs/experiments/{name}.json')
    if plan.get('continuation_preparation'):
        prepare = (f'conda run --no-capture-output -n clrr python -u '
                   f'scripts/experiment.py prepare-continuation '
                   f'--config configs/experiments/{name}.json')
    probe = plan.get('pre_training_alignment_probe')
    if probe:
        for key in ('cohort', 'output'):
            assert re.fullmatch(r'outputs/experiments/[a-zA-Z0-9_-]+', probe[key])
        assert stage == 'gate-train', 'Probe is followed by gates and approved controls'
        prepare = (f'conda run --no-capture-output -n clrr python -u scripts/analysis/measure_alignment_residual_checkpoint.py '
                   f'--cohort {probe["cohort"]} --output {probe["output"]} '
                   f'--minimum-free-gpu-mib {plan["protocol"]["minimum_free_gpu_mib"]}')
        finish = 'echo RTX_PIPELINE_COMPLETE'
    if plan.get('pre_training_consistency_probe'):
        assert stage == 'gate-train' and not probe and not plan.get('candidate_pool')
        prepare = (f'conda run --no-capture-output -n clrr python -u '
                   f'scripts/analysis/measure_consistency_checkpoint.py '
                   f'--config configs/experiments/{name}.json')
    if plan.get('prefix_preparation'):
        assert stage == 'gate' and not prepare and not plan.get('candidate_pool')
        # Scientific preparation is not a GPU optimizer gate or long training.
        prepare = (f'conda run --no-capture-output -n clrr python -u '
                   f'scripts/analysis/prepare_prefix_consistency.py '
                   f'--config configs/experiments/{name}.json')
        gates = ''
        training = 'echo PREFIX_PREPARATION_FINISHED_NO_TRAINING'
    if plan.get('two_method_queue'):
        assert stage == 'gate-train' and not prepare
        prepare = f'conda run --no-capture-output -n clrr python -u scripts/analysis/run_two_method_round1.py --config configs/experiments/{name}.json'
        gates = ''
        training = 'echo TWO_METHOD_QUEUE_FINISHED'
    if plan.get('candidate_pool'):
        prefix = f'conda run --no-capture-output -n clrr python -u scripts/experiment.py run --config configs/experiments/{name}.json'
        reference_id = next(s['id'] for s in plan['runs'] if s['method'] == 'sequence-ce-v1')
        risk_id = next(s['id'] for s in plan['runs'] if s['method'] == 'sequence-clrr-risk-v1')
        prepare = f'{prefix} --id {risk_id} --stage preflight\n{prefix} --id {reference_id} --stage pool'
        finish = f'{prefix} --id {reference_id} --stage summarize\n{prefix} --id {reference_id} --stage cleanup-initialization\necho RTX_PIPELINE_COMPLETE'
    post_commands = ''
    post = plan.get('post_training_test', {})
    if post.get('authorized'):
        assert re.fullmatch(r'[a-zA-Z0-9_-]+', post['workspace'])
        assert post.get('design') and post.get('upload_prefix')
        assert re.fullmatch(r'analysis/[a-zA-Z0-9_./-]+', post['upload_prefix'])
        assert '..' not in post['upload_prefix'].split('/')
        test_out = 'outputs/experiments/' + post['workspace']
        python = 'conda run --no-capture-output -n clrr python -u'
        post_commands = f'''
{python} scripts/analysis/evaluate_frozen_rtx.py --stage prepare --cohort outputs/experiments/{name} --output {test_out} --authorize-test --minimum-free-gpu-mib {plan['protocol']['minimum_free_gpu_mib']}
{python} scripts/analysis/evaluate_frozen_rtx.py --stage queue --output {test_out}
{python} scripts/analysis/paired_checkpoint_statistics.py --predictions-dir {test_out} --references data_processed/{plan['dataset']}/test.csv --design configs/experiments/{name}.json --output {test_out}/statistics --upload-prefix {post['upload_prefix']}
echo RTX_PIPELINE_COMPLETE
'''
    shell = f'''#!/usr/bin/env bash
set -euo pipefail
source {PROJECT}/environment.sh
cd {PROJECT}
exec 9>{PROJECT}/tmp/training.lock
flock -n 9 || {{ echo 'Another CLRR training queue is active'; exit 1; }}
export HF_TOKEN=$(cat {PROJECT}/.secrets/hf_token)
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_XET_CACHE={PROJECT}/cache/xet PYTHONPYCACHEPREFIX={PROJECT}/cache/pycache
export TMPDIR={PROJECT}/tmp TMP={PROJECT}/tmp TEMP={PROJECT}/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda run --no-capture-output -n clrr python -u scripts/experiment.py cpu --config configs/experiments/{name}.json
{prepare}
{gates}
{training}
{finish}
{post_commands}
'''
    code = f'''import base64,hashlib,json,subprocess,zipfile,shutil
from pathlib import Path
root=Path({PROJECT!r}).resolve();out=root/'outputs/experiments'/{name!r}
assert not (out/{(queue_stem+'.pid')!r}).exists(), 'Queue already launched'
secret=root/'.secrets/hf_token'
assert secret.is_file() and secret.stat().st_size > 0, 'Install HF token in project .secrets/hf_token first'
lock_check=subprocess.run(['flock','-n',str(root/'tmp/training.lock'),'true'])
assert lock_check.returncode == 0, 'Another CLRR queue is active; do not replace its source'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],universal_newlines=True).strip()
assert gpu.startswith('Quadro RTX 6000,')
assert {bool(plan['protocol'].get('wait_for_gpu', False))!r} or int(gpu.rsplit(',',1)[1]) >= {plan['protocol']['minimum_free_gpu_mib']}
assert shutil.disk_usage(root).free >= {int(plan.get('minimum_free_disk_gib', 0))}*2**30, 'Insufficient project disk space for gates/training'
out.mkdir(parents=True,exist_ok=True)
bundle=root/'cache/bundles'/{(name+'.zip')!r};bundle.parent.mkdir(parents=True,exist_ok=True)
bundle.write_bytes(base64.b64decode({payload!r}))
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=={sha256(bundle)!r}
with zipfile.ZipFile(bundle) as z:
 for n in z.namelist():
  dest=(root/n).resolve();assert root in dest.parents
 z.extractall(root)
script=out/{('run_'+queue_stem+'.sh')!r};script.write_text({shell!r});script.chmod(0o700)
with (out/{(queue_stem+'.log')!r}).open('w') as log:
 p=subprocess.Popen(['bash',str(script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/{(queue_stem+'.pid')!r}).write_text(str(p.pid))
print(json.dumps({{'status':'RTX_GATES_LAUNCHED','pid':p.pid,'gpu':gpu,'project':str(root)}}))
'''
    state = call(code)
    state['stage'] = stage
    state.update(bundle_sha256=sha256(bundle), assigned=[s['id'] for s in plan['runs']])
    receipt_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
    return state


def monitor(config, account):
    out, plan = verify_frozen(config)
    assert account == 1
    name = Path(config).stem
    state = call(f'''import json,subprocess
from pathlib import Path
out=Path({PROJECT!r})/'outputs/experiments'/{name!r}
stem='queue' if (out/'queue.pid').exists() else 'gate_queue'
pid=out/(stem+'.pid');stat=Path('/proc')/pid.read_text().strip()/'stat' if pid.exists() else Path('/nonexistent')
alive=stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
names={{'CPU_GATE.json','PASS.json','progress.json','VAL_COMPLETE.json','run_metadata.json','validation_metrics.json','stop_state.json','collection_receipt.json','UPLOAD_PENDING.json','POOL_PROGRESS.json','POOL_COMPLETE.json','PREFLIGHT.json','PILOT_SUMMARY.json','RESOURCE_ADMISSION.json','POOL_REJECTED.json','ADMISSION.json','TWO_METHOD_GATES_COMPLETE.json','TWO_METHOD_QUEUE.json','PROBE_PROGRESS.json','NUMERICAL_PROGRESS.json','NULL_REPLAY.json','ACTIVATION_AUDIT.json','CALIBRATION.json','FAILED.json','microbatch_progress.json','training_failure.json'}}
files={{p.relative_to(out).as_posix():json.loads(p.read_text()) for p in out.rglob('*.json') if p.name in names or p.name.startswith('queue_')}}
csv={{p.relative_to(out).as_posix():p.read_text() for p in out.glob('runs/*/validation_predictions.csv')}}
print(json.dumps({{'alive':alive,'files':files,'csv':csv,'tail':(out/(stem+'.log')).read_text(errors='replace').splitlines()[-12:] if (out/(stem+'.log')).exists() else [],'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free,utilization.gpu','--format=csv,noheader'],universal_newlines=True)}}))
''')
    for kind in ('files', 'csv'):
        for relative, record in state.pop(kind).items():
            dest = (out / relative).resolve()
            assert dest.is_relative_to(out)
            if dest.name in ('VAL_COMPLETE.json', 'PASS.json') and dest.exists():
                assert json.loads(dest.read_text()) == record, 'Immutable receipt changed'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(record, indent=2) if kind == 'files' else record, encoding='utf-8')
    (out / 'rtx_status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    return state
