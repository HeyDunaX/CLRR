"""Deploy, monitor or collect a hash-pinned no-training probe in the RTX project."""
import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from experiment_runner.rtx import PROJECT,call


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['launch','monitor','collect'])
    parser.add_argument('--plan',required=True)
    args=parser.parse_args()
    path=Path(args.plan).resolve()
    relative=path.relative_to(ROOT).as_posix()
    plan=json.loads(path.read_text())
    assert plan['optimizer_steps']==0 and plan['split']=='train' and not plan['test_used']
    output=plan['output']
    local=ROOT/output
    if args.action=='launch':
        assert not (local/'LAUNCH.json').exists(),'Already launched; monitor existing probe'
        payloads={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in plan['source_sha256']}
        payloads[relative]=base64.b64encode(path.read_bytes()).decode()
        shell=f'''#!/usr/bin/env bash
set -euo pipefail
source {PROJECT}/environment.sh
cd {PROJECT}
exec 9>{PROJECT}/tmp/training.lock
flock -n 9 || {{ echo PROJECT_BUSY; exit 1; }}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX={PROJECT}/cache/pycache
export TMPDIR={PROJECT}/tmp TMP={PROJECT}/tmp TEMP={PROJECT}/tmp
export PYTHONPATH={PROJECT}/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{PROJECT}/conda/envs/clrr/bin/python -u -m unittest discover -s tests -p {plan['cpu_test']} -v
echo REMOTE_CPU_GATE_PASS
echo '{{"passed":true,"tests":6,"optimizer_steps":0,"source":"unittest exit0; see queue.log"}}' > {PROJECT}/{output}/CPU_GATE.json
exec {PROJECT}/conda/envs/clrr/bin/python -u {plan['worker']} --plan {relative}
'''
        state=call(f'''import json,base64,subprocess,hashlib,shutil
from pathlib import Path
root=Path({PROJECT!r}).resolve();out=root/{output!r}
assert not out.exists(),'Probe output already exists'
assert subprocess.run(['flock','-n',str(root/'tmp/training.lock'),'true']).returncode==0,'Project busy'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],text=True).strip()
assert gpu.startswith('Quadro RTX 6000,') and int(gpu.rsplit(',',1)[1])>={plan['minimum_free_gpu_mib']},gpu
assert shutil.disk_usage(root).free>15*2**30,'Need 15 GiB free project disk'
for name,payload in {payloads!r}.items():
 p=(root/name).resolve();assert p.is_relative_to(root)
 data=base64.b64decode(payload)
 if name in {plan['source_sha256']!r}:assert hashlib.sha256(data).hexdigest()=={plan['source_sha256']!r}[name]
 if p.exists():assert p.read_bytes()==data,('Refuse overwrite existing different source',name)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
out.mkdir(parents=True,exist_ok=True)
script=out/'queue.sh';script.write_text({shell!r});script.chmod(0o700)
with (out/'queue.log').open('w') as log:
 proc=subprocess.Popen(['bash',str(script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'queue.pid').write_text(str(proc.pid))
print(json.dumps(dict(pid=proc.pid,gpu=gpu,optimizer_steps=0,output={output!r})))
''')
        (local/'LAUNCH.json').write_text(json.dumps(state,indent=2))
        print(json.dumps(state,indent=2));return
    state=call(f'''import json,os,hashlib,base64
from pathlib import Path
out=Path({PROJECT!r})/{output!r};pid=int((out/'queue.pid').read_text())
try:os.kill(pid,0);alive=True
except ProcessLookupError:alive=False
state={{p.name:json.loads(p.read_text()) for p in out.glob('*.json') if p.name in ['COMPLETE.json','ERROR.json','PROGRESS.json','BENCHMARK.json']}}
files={{}}
if {args.action=='collect'!r}:
 assert (out/'COMPLETE.json').exists() or (out/'ERROR.json').exists(),'Probe not terminal'
 for p in out.glob('*.json'):
  assert p.stat().st_size<4*2**20
  data=p.read_bytes();files[p.name]=dict(sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode())
print(json.dumps(dict(pid=pid,alive=alive,state=state,files=files,log=(out/'queue.log').read_text()[-3500:])))
''')
    if args.action=='collect':
        receipts={}
        for name,item in state.pop('files').items():
            data=base64.b64decode(item['data'])
            assert hashlib.sha256(data).hexdigest()==item['sha256']
            if name=='PLAN.json':assert data==path.read_bytes()
            (local/name).write_bytes(data)
            receipts[name]=item['sha256']
        (local/'COLLECTION.json').write_text(json.dumps(dict(verified=True,files=receipts),indent=2))
    print(json.dumps(state,indent=2))


if __name__=='__main__':main()
