"""Launch/collect read-only GPU probes inside the approved RTX workspace."""
import argparse
import base64
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/workspace/ura/thanhdat/projects/CLRR'


def remote(code):
    r = subprocess.run(['C:/Program Files/Tailscale/tailscale.exe', 'ssh', 'ura@100.90.53.12', 'python3', '-'], input=code, text=True, capture_output=True, timeout=90)
    if r.returncode:
        raise RuntimeError(r.stdout + '\n' + r.stderr)
    return json.loads(r.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['launch', 'monitor'], required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--worker', default='scripts/revalidation/probe_token_alignment.py')
    p.add_argument('--plan', default='outputs_rebuttal/method_candidate_diagnostics/PLAN.json')
    args = p.parse_args()
    assert args.job.replace('_', '').replace('-', '').isalnum()
    out = ROOT / 'outputs_rebuttal' / args.job
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'launch':
        worker = (ROOT / args.worker).resolve(); plan = (ROOT / args.plan).resolve()
        assert worker.is_relative_to(ROOT / 'scripts/revalidation') and plan.is_relative_to(ROOT)
        files = {args.worker: worker.read_bytes(), f'outputs/{args.job}/PLAN.json': plan.read_bytes(), 'data_processed/amis_mandarin/validation.csv': (ROOT / 'data_processed/amis_mandarin/validation.csv').read_bytes()}
        files.update({x.relative_to(ROOT).as_posix(): x.read_bytes() for x in (ROOT / 'src/methods').rglob('*.py')})
        shell = f'''#!/usr/bin/env bash
set -euo pipefail
source /workspace/ura/thanhdat/projects/CLRR/environment.sh
export PYTHONPYCACHEPREFIX=/workspace/ura/thanhdat/projects/CLRR/cache/pycache
export XDG_CACHE_HOME=/workspace/ura/thanhdat/projects/CLRR/cache
export CUDA_CACHE_PATH=/workspace/ura/thanhdat/projects/CLRR/cache/cuda
export TRITON_CACHE_DIR=/workspace/ura/thanhdat/projects/CLRR/cache/triton
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
cd {REMOTE}
nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv,noheader
conda run --no-capture-output -n clrr python -u {args.worker} --plan outputs/{args.job}/PLAN.json --output outputs/{args.job}
'''
        files[f'outputs/{args.job}/run.sh'] = shell.encode()
        payload = {k: base64.b64encode(v).decode() for k, v in files.items()}
        code = f'''import base64,json,subprocess
from pathlib import Path
root=Path({REMOTE!r}).resolve();out=root/'outputs'/{args.job!r}
out.mkdir(parents=True,exist_ok=True)
assert not (out/'worker.pid').exists(), 'Job already launched; monitor it'
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
assert int(gpu.strip())>=12288, 'Insufficient free GPU memory'
for relative,data in {payload!r}.items():
 dest=(root/relative).resolve();assert root in dest.parents
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(base64.b64decode(data))
with (out/'worker.log').open('w') as log:
 process=subprocess.Popen(['bash',str(out/'run.sh')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'worker.pid').write_text(str(process.pid))
print(json.dumps({{'status':'LAUNCHED','pid':process.pid,'free_MiB':int(gpu.strip())}}))
'''
    else:
        code = f'''import json,subprocess
from pathlib import Path
out=Path({REMOTE!r})/'outputs'/{args.job!r}
pid=out/'worker.pid';stat=Path('/proc')/pid.read_text().strip()/'stat' if pid.exists() else Path('/nonexistent')
alive=stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
files={{p.relative_to(out).as_posix():json.loads(p.read_text()) for p in out.rglob('*.json') if p.name in ('RESULT.json','COMPLETE.json')}}
print(json.dumps({{'alive':alive,'files':files,'tail':(out/'worker.log').read_text(errors='replace').splitlines()[-12:] if (out/'worker.log').exists() else [],'gpu':subprocess.run(['nvidia-smi','--query-gpu=name,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout}}))
'''
    state = remote(code)
    for name, data in state.get('files', {}).items():
        dest = (out / name).resolve(); assert dest.is_relative_to(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=2), encoding='utf-8')
    (out / 'status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in state.items() if k != 'files'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
