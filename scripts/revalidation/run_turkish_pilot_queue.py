"""Run one approved queue after gates on the same pinned bundle."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
PREFIX='turkish_old_method_pilot_es10_20261007'
OUT=ROOT/'outputs_rebuttal'/PREFIX


def write(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--account',choices=['1','2','3'],required=True)
    a=p.parse_args();plan=json.loads((OUT/'PLAN.json').read_text())
    assert plan['status']=='APPROVED'
    assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0)
    assert torch.cuda.is_bf16_supported() and 35 < torch.cuda.get_device_properties(0).total_memory/2**30 < 45
    manifest=json.loads((ROOT/'fresh_bundle_files.json').read_text())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==d for n,d in manifest.items())
    OUT.mkdir(parents=True,exist_ok=True)
    state={'account':a.account,'status':'GATES','completed':[],'started_utc_epoch':time.time()}
    status=OUT/f'queue_account{a.account}.json';write(status,state)
    gate=OUT/'gpu_gate';gate.mkdir(exist_ok=True)
    try:
        for script, extra in [('tests/smoke_rerun_optimization.py',[]),
                              ('tests/smoke_precision_training.py',['--audit-rerun'])]:
            subprocess.run([sys.executable,'-u',script,'--cuda','--turkish-pilot',
                            '--result-dir',str(gate),*extra],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'-u','scripts/revalidation/smoke_fullsize_turkish.py'],cwd=ROOT,check=True)
        write(gate/'GATE_COMPLETE.json',{'passed':True,'gpu':torch.cuda.get_device_name(0),
            'torch':torch.__version__,'account':a.account,'completed_utc_epoch':time.time(),
            'bundle_manifest_sha256':hashlib.sha256((ROOT/'fresh_bundle_files.json').read_bytes()).hexdigest()})
        for run in plan['queues'][a.account]:
            state.update(status='TRAINING',current=run,run_started_utc_epoch=time.time());write(status,state)
            subprocess.run([sys.executable,'-u','scripts/revalidation/run_turkish_old_method_pilot.py',
                            '--family',run['family'],'--seed',str(run['seed']),'--stage','train'],cwd=ROOT,check=True)
            d=OUT/run['family']/f"seed{run['seed']}"
            assert (d/'VAL_COMPLETE.json').exists()
            receipt=json.loads((d/'collection_receipt.json').read_text());assert receipt['verified']
            state['completed'].append(run);write(status,state)
        state.update(status='VALIDATION_QUEUE_COMPLETE',completed_utc_epoch=time.time());write(status,state)
    except Exception as e:
        state.update(status='FAILED',error_type=type(e).__name__,error=str(e));write(status,state)
        raise


if __name__=='__main__':main()
