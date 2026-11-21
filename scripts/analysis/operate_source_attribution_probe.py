"""Launch/inspect/collect authorized frozen diagnostics in the private RTX project."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from experiment_runner.rtx import call, PROJECT


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['launch','monitor','collect'])
    parser.add_argument('--output', default='outputs/experiments/amis_source_attribution_frozen_rtx')
    args=parser.parse_args()
    relative=args.output
    out=ROOT/relative
    assert out.resolve().is_relative_to((ROOT/'outputs/experiments').resolve())
    if args.action=='launch':
        plan=json.loads((out/'PLAN.json').read_text())
        assert plan['optimizer_steps']==0 and plan['split']=='validation'
        workers={'source_attribution_v1':'scripts/analysis/measure_source_attribution.py',
                 'decision_branches_v1':'scripts/analysis/measure_decision_branches.py'}
        worker=workers[plan['metric_family']]
        files={f:base64.b64encode((ROOT/f).read_bytes()).decode() for f in plan['source_sha256']}
        files[relative+'/PLAN.json']=base64.b64encode((out/'PLAN.json').read_bytes()).decode()
        shell=f'''#!/usr/bin/env bash
set -euo pipefail
source {PROJECT}/environment.sh
cd {PROJECT}
exec 9>{PROJECT}/tmp/training.lock
flock -n 9 || {{ echo PROJECT_BUSY; exit 1; }}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec {PROJECT}/conda/envs/clrr/bin/python -u {worker} --plan {relative}/PLAN.json --output {relative}
'''
        state=call(f'''import json,base64,subprocess,hashlib,shutil
from pathlib import Path
root=Path({PROJECT!r}).resolve();out=root/{relative!r}
assert not out.exists(),'Probe already launched'
assert subprocess.run(['flock','-n',str(root/'tmp/training.lock'),'true']).returncode==0,'CLRR busy'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],text=True).strip()
assert gpu.startswith('Quadro RTX 6000,') and int(gpu.rsplit(',',1)[1])>=18000,gpu
assert shutil.disk_usage(root).free>40*2**30
for name,payload in {files!r}.items():
 target=(root/name).resolve();assert target.is_relative_to(root)
 target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(payload))
plan=json.loads((out/'PLAN.json').read_text())
for name,digest in plan['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
script=out/'queue.sh';script.write_text({shell!r});script.chmod(0o700)
with (out/'queue.log').open('w') as log:
 p=subprocess.Popen(['bash',str(script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'queue.pid').write_text(str(p.pid))
print(json.dumps({{'pid':p.pid,'gpu':gpu,'optimizer_steps':0,'stage':'CPU_GATE_THEN_NATIVE_GPU_REPLAY_THEN_FROZEN_PROBE'}}))
''')
        (out/'launch_rtx.json').write_text(json.dumps(state,indent=2))
        print(json.dumps(state));return
    state=call(f'''import json,os,base64,hashlib
from pathlib import Path
out=Path({PROJECT!r})/{relative!r}
pid=int((out/'queue.pid').read_text())
try:os.kill(pid,0);alive=True
except ProcessLookupError:alive=False
files={{}}
if {args.action=='collect'!r}:
 assert (out/'COMPLETE.json').exists(), 'Collect only completed immutable probe outputs'
 for p in out.rglob('*'):
  if p.is_file() and 'restore' not in p.relative_to(out).parts and p.suffix in ('.json','.csv','.md'):
   assert p.stat().st_size<3*2**20
   data=p.read_bytes();files[str(p.relative_to(out))]={{'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}}
print(json.dumps({{'alive':alive,'pid':pid,'complete':(out/'COMPLETE.json').exists(),'log':(out/'queue.log').read_text()[-5000:],'files':files}}))
''')
    if args.action=='collect':
        receipts={}
        for name,item in state.pop('files').items():
            path=(out/name).resolve();assert path.is_relative_to(out.resolve())
            chunks=[]
            for offset in range(0,item['size'],3000):
                part=call(f'''import json,base64
from pathlib import Path
out=(Path({PROJECT!r})/{relative!r}).resolve();p=(out/{name!r}).resolve()
assert p.is_relative_to(out) and p.is_file()
with p.open('rb') as f:
 f.seek({offset});data=f.read(3000)
print(json.dumps({{'data':base64.b64encode(data).decode()}}))
''')
                chunks.append(base64.b64decode(part['data']))
            data=b''.join(chunks)
            assert hashlib.sha256(data).hexdigest()==item['sha256']
            if path.name=='PLAN.json':assert data==path.read_bytes()
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            receipts[name]=item['sha256']
        (out/'COLLECTION.json').write_text(json.dumps(dict(verified=True,files=receipts),indent=2))
    if 'Traceback' not in state['log']:
        state['log']='\n'.join(state['log'].splitlines()[-3:])
    print(json.dumps(state))


if __name__=='__main__':main()
