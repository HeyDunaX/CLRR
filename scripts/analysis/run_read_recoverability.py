"""Dispatch/inspect/collect the frozen W42 read diagnostic, small artifacts only."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from experiment_runner.rtx import call, PROJECT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('action', choices=['launch','status','collect'])
args = parser.parse_args()
output = 'outputs/experiments/w_read_recoverability_rtx'
script = 'scripts/analysis/probe_read_recoverability.py'
local = ROOT/output
if args.action == 'launch':
    payload = base64.b64encode((ROOT/script).read_bytes()).decode()
    digest = hashlib.sha256((ROOT/script).read_bytes()).hexdigest()
    config_digest = hashlib.sha256((ROOT/'configs/experiments/amis_drm_validation_fix_rtx.json').read_bytes()).hexdigest()
    shell = f'''#!/usr/bin/env bash
set -euo pipefail
cd {PROJECT}
source environment.sh
exec 9>{PROJECT}/tmp/training.lock
flock -n 9 || {{ echo PROJECT_BUSY; exit 1; }}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
export HF_XET_CACHE={PROJECT}/cache/xet PYTHONPYCACHEPREFIX={PROJECT}/cache/pycache
export TMPDIR={PROJECT}/tmp TMP={PROJECT}/tmp TEMP={PROJECT}/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{PROJECT}/conda/envs/clrr/bin/python -u {script} --output {output} --rows 32
'''
    result = call(f'''import json,base64,hashlib,subprocess,shutil
from pathlib import Path
root=Path({PROJECT!r}).resolve();out=root/{output!r}
assert out.resolve().is_relative_to(root/'outputs/experiments')
assert not out.exists(), 'Output exists; inspect, do not launch duplicate'
assert subprocess.run(['flock','-n',str(root/'tmp/training.lock'),'true'],capture_output=True).returncode==0,'Project busy'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
assert gpu.startswith('Quadro RTX 6000,') and int(gpu.split(',')[1])>=20000,gpu
assert shutil.disk_usage(root).free>20*2**30
assert hashlib.sha256((root/'configs/experiments/amis_drm_validation_fix_rtx.json').read_bytes()).hexdigest()=={config_digest!r}
target=root/{script!r};data=base64.b64decode({payload!r})
assert hashlib.sha256(data).hexdigest()=={digest!r}
assert not target.exists() or target.read_bytes()==data,'Existing source differs'
target.write_bytes(data)
out.mkdir(parents=True)
(out/'SOURCE.py').write_bytes(data)
queue=out/'queue.sh';queue.write_text({shell!r});queue.chmod(0o700)
with (out/'queue.log').open('w') as log:
 p=subprocess.Popen(['bash',str(queue)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=p.pid,output={output!r},gpu_before=gpu,script_sha256={digest!r},config_sha256={config_digest!r},model_training=False,test_used=False)
(out/'queue.pid').write_text(str(p.pid));(out/'launch.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
''')
    local.mkdir(parents=True,exist_ok=True)
    (local/'launch.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
else:
    result = call(f'''import json,hashlib,os
from pathlib import Path
p=Path({PROJECT!r})/{output!r}
pid=int((p/'queue.pid').read_text())
try:os.kill(pid,0);alive=True
except ProcessLookupError:alive=False
r=dict(alive=alive,log_tail=(p/'queue.log').read_text()[-2500:])
for name in ['status.json','COMPLETE.json','CLEANUP.json']:
 if (p/name).exists():r[name]=json.loads((p/name).read_text())
if {args.action!r}=='collect':
 assert (p/'COMPLETE.json').exists() and (p/'CLEANUP.json').exists(),'Wait for completion and cleanup'
 r['files']={{}}
 for name in ['PLAN.json','GATES.json','measurements.json','COMPLETE.json','CLEANUP.json','launch.json']:
  data=(p/name).read_bytes();assert len(data)<4*1024*1024
  r['files'][name]=dict(text=data.decode(),sha256=hashlib.sha256(data).hexdigest())
print(json.dumps(r))
''')
    if 'files' in result:
        local.mkdir(parents=True,exist_ok=True)
        hashes = {}
        for name, record in result.pop('files').items():
            data = record['text'].encode()
            assert hashlib.sha256(data).hexdigest()==record['sha256']
            (local/name).write_bytes(data)
            hashes[name] = record['sha256']
        (local/'download_receipt.json').write_text(json.dumps(hashes,indent=2))
        result['collected'] = hashes
print(json.dumps(result,indent=2))
