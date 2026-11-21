"""Deploy and monitor a single RTX measurement in the approved private directory."""
import argparse
import base64
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs_rebuttal/rtx_rank_smoke'
OUT.mkdir(parents=True,exist_ok=True)
REMOTE='/workspace/ura/thanhdat/projects/CLRR'


def remote(code):
    r=subprocess.run(['C:/Program Files/Tailscale/tailscale.exe','ssh','ura@100.90.53.12','python3','-'],
        input=code,text=True,capture_output=True,timeout=120)
    if r.returncode:raise RuntimeError(r.stdout+'\n'+r.stderr)
    return r.stdout


def launch():
    plan=json.loads((ROOT/'outputs_rebuttal/method_candidate_diagnostics/PLAN.json').read_text())
    old=json.loads((ROOT/'outputs_rebuttal/method_candidate_diagnostics/BASE/COMPLETE.json').read_text())
    plan['a100_source_covariance_rank']=old['geometry']['source']['effective_rank_covariance_energy']
    files={'PLAN.json':json.dumps(plan,indent=2).encode(),
        'data_processed/amis_mandarin/validation.csv':(ROOT/'data_processed/amis_mandarin/validation.csv').read_bytes(),
        'scripts/revalidation/rtx_rank_smoke.py':(ROOT/'scripts/revalidation/rtx_rank_smoke.py').read_bytes()}
    shell='''#!/usr/bin/env bash
set -euo pipefail
source /workspace/ura/thanhdat/projects/CLRR/environment.sh
export XDG_CACHE_HOME=/workspace/ura/thanhdat/projects/CLRR/cache
export CONDARC=/workspace/ura/thanhdat/projects/CLRR/conda/condarc.yaml
export CONDA_REGISTER_ENVS=false
export PYTHONPYCACHEPREFIX=/workspace/ura/thanhdat/projects/CLRR/cache/pycache
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
cd /workspace/ura/thanhdat/projects/CLRR
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader
test -x /workspace/ura/thanhdat/projects/CLRR/conda/envs/clrr/bin/python
/workspace/ura/thanhdat/conda/runtime/bin/conda run --no-capture-output -n clrr python -u scripts/revalidation/rtx_rank_smoke.py
'''
    files['rtx_rank_smoke.sh']=shell.encode()
    payload={k:base64.b64encode(v).decode() for k,v in files.items()}
    code="""import base64,json,subprocess
from pathlib import Path
root=Path('/workspace/ura/thanhdat').resolve();project=root/'projects/CLRR'
project.mkdir(parents=True,exist_ok=True)
output=project/'outputs/rtx_rank_smoke';output.mkdir(parents=True,exist_ok=True)
assert not (output/'COMPLETE.json').exists(), 'Measurement already complete'
pid=output/'worker.pid'
if pid.exists():
 stat=Path('/proc')/pid.read_text().strip()/'stat'
 assert not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0] in ('Z','X'), 'Worker already active'
files=PAYLOAD
for relative,b64 in files.items():
 dest=(project/relative).resolve();assert root in dest.parents
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(base64.b64decode(b64))
with (output/'worker.log').open('w') as log:
 p=subprocess.Popen(['bash',str(project/'rtx_rank_smoke.sh')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pid.write_text(str(p.pid));print(json.dumps({'status':'LAUNCHED','pid':p.pid,'project':str(project)}))
""".replace('PAYLOAD',repr(payload))
    print(remote(code),flush=True)


def monitor():
    code="""import json,subprocess
from pathlib import Path
out=Path('/workspace/ura/thanhdat/projects/CLRR/outputs/rtx_rank_smoke')
pid=out/'worker.pid';stat=Path('/proc')/pid.read_text().strip()/'stat' if pid.exists() else Path('/nonexistent')
alive=stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
result=json.loads((out/'COMPLETE.json').read_text()) if (out/'COMPLETE.json').exists() else None
log=out/'worker.log'
print(json.dumps({'alive':alive,'result':result,'gpu':subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),'tail':log.read_text(errors='replace').splitlines()[-10:] if log.exists() else []}))
"""
    state=json.loads(remote(code));(OUT/'status.json').write_text(json.dumps(state,indent=2))
    if state['result']:(OUT/'COMPLETE.json').write_text(json.dumps(state['result'],indent=2))
    print(json.dumps(state),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('launch','monitor'),required=True);a=p.parse_args()
    (launch if a.stage=='launch' else monitor)()
