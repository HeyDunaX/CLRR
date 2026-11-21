"""Deploy/monitor/collect a frozen, TRAIN-only geometry diagnostic on RTX."""
import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from experiment_runner.rtx import PROJECT,call


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['launch','monitor','collect'])
    parser.add_argument('--output',default='outputs/experiments/amis_cmc_update_geometry_rtx')
    args = parser.parse_args()
    relative = args.output
    local = (ROOT/relative).resolve()
    assert local.is_relative_to((ROOT/'outputs/experiments').resolve())
    if args.action == 'launch':
        plan = json.loads((local/'PLAN.json').read_text())
        assert plan['split']=='train' and plan['optimizer_steps']==0
        worker = 'scripts/analysis/measure_cmc_update_geometry.py'
        sources = plan['source_sha256']
        payloads = {name:base64.b64encode((ROOT/name).read_bytes()).decode() for name in (worker,'scripts/analysis/analyze_cmc_gradient_estimator.py')}
        payloads[relative+'/PLAN.json'] = base64.b64encode((local/'PLAN.json').read_bytes()).decode()
        shell = f'''#!/usr/bin/env bash
set -euo pipefail
source {PROJECT}/environment.sh
cd {PROJECT}
exec 9>{PROJECT}/tmp/training.lock
flock -n 9 || {{ echo PROJECT_BUSY; exit 1; }}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX={PROJECT}/cache/pycache
export TMPDIR={PROJECT}/tmp TMP={PROJECT}/tmp TEMP={PROJECT}/tmp
exec {PROJECT}/conda/envs/clrr/bin/python -u {worker} --plan {relative}/PLAN.json
'''
        state = call(f'''import json,base64,subprocess,hashlib
from pathlib import Path
root=Path({PROJECT!r}).resolve();out=root/{relative!r}
assert not out.exists(), 'Diagnostic already exists'
assert subprocess.run(['flock','-n',str(root/'tmp/training.lock'),'true']).returncode==0,'Project busy'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],text=True).strip()
assert gpu.startswith('Quadro RTX 6000,') and int(gpu.rsplit(',',1)[1])>=20000,gpu
for name,digest in {sources!r}.items():
 if name not in {list(payloads)!r}:assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,('Remote source differs',name)
for name,payload in {payloads!r}.items():
 path=(root/name).resolve();assert path.is_relative_to(root)
 data=base64.b64decode(payload)
 if name in {sources!r}:assert hashlib.sha256(data).hexdigest()=={sources!r}[name]
 if path.exists() and name.endswith('.py'):assert path.read_bytes()==data,'Refuse overwrite changed worker'
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
script=out/'queue.sh';script.write_text({shell!r});script.chmod(0o700)
with (out/'queue.log').open('w') as log:
 process=subprocess.Popen(['bash',str(script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'queue.pid').write_text(str(process.pid))
print(json.dumps(dict(pid=process.pid,gpu=gpu,stage='CPU_THEN_GPU_GATE_THEN_FROZEN_GEOMETRY',optimizer_steps=0)))
''')
        (local/'LAUNCH.json').write_text(json.dumps(state,indent=2))
        print(json.dumps(state));return
    state = call(f'''import json,os,base64,hashlib
from pathlib import Path
out=Path({PROJECT!r})/{relative!r}
pid=int((out/'queue.pid').read_text())
try:os.kill(pid,0);alive=True
except ProcessLookupError:alive=False
names=['CPU_GATE.json','GPU_GATE.json','PROGRESS.json','ERROR.json','COMPLETE.json']
state={{name:json.loads((out/name).read_text()) for name in names if (out/name).exists()}}
files={{}}
if {args.action=='collect'!r}:
 assert (out/'COMPLETE.json').exists() or (out/'ERROR.json').exists(),'Only collect terminal diagnostics'
 for p in out.glob('*.json'):
  assert p.stat().st_size<4*2**20
  data=p.read_bytes();files[p.name]=dict(size=len(data),sha256=hashlib.sha256(data).hexdigest())
print(json.dumps(dict(alive=alive,pid=pid,files=files,state=state,log=(out/'queue.log').read_text()[-2500:])))
''')
    if args.action == 'collect':
        receipts = {}
        for name,item in state.pop('files').items():
            data = b''
            for offset in range(0,item['size'],5000):
                part = call(f'''import json,base64
from pathlib import Path
p=Path({PROJECT!r})/{relative!r}/{name!r}
with p.open('rb') as f:
 f.seek({offset});data=f.read(5000)
print(json.dumps(dict(data=base64.b64encode(data).decode())))
''')
                data += base64.b64decode(part['data'])
            assert hashlib.sha256(data).hexdigest()==item['sha256']
            if name=='PLAN.json':assert data==(local/name).read_bytes()
            (local/name).write_bytes(data)
            receipts[name]=item['sha256']
        (local/'COLLECTION.json').write_text(json.dumps(dict(verified=True,files=receipts),indent=2))
    if 'Traceback' not in state['log']:
        state['log']='\n'.join(state['log'].splitlines()[-2:])
    print(json.dumps(state,indent=2))


if __name__=='__main__':
    main()
