import subprocess
from pathlib import Path
root=Path('/content/CLRR_controls')
pidfile=Path('/content/strong_mbart_supplement.pid')
assert root.exists() and Path('/content/.standardized_hf_token').exists()
if pidfile.exists():
    p=Path('/proc')/pidfile.read_text().strip()/'stat'
    assert not p.exists() or p.read_text().rsplit(')',1)[1].split()[0] in ('Z','X'), 'Supplement already scheduled'
subprocess.run(['unzip','-q','-o','/content/strong_mbart_supplement_bundle.zip','-d',str(root)],check=True)
controller=root/'outputs_rebuttal/amis_standardized_strong_mbart/controller.py'
controller.write_text("""import json, os, subprocess, time
from pathlib import Path
root=Path('/content/CLRR_controls')
out=root/'outputs_rebuttal/amis_standardized_strong_mbart'
base=root/'outputs_rebuttal/amis_standardized_controls'
status=out/'schedule_status.json'
def record(state):
    status.write_text(json.dumps({'status':state,'account':3,'runs':['A05','A06'],'updated_unix':time.time()}))
record('WAITING_FOR_C01_VERIFIED_BACKUP')
while True:
    q=base/'queue_account3.json'
    if q.exists():
        state=json.loads(q.read_text())
        if state['status']=='FAILED':
            record('BLOCKED_ORIGINAL_QUEUE_FAILED'); raise RuntimeError(state)
        if state['status']=='VALIDATION_QUEUE_COMPLETE':
            receipt=json.loads((base/'runs/C01/VAL_COMPLETE.json').read_text())
            assert receipt['status']=='VALIDATION_COMPLETE_VERIFIED'
            break
    time.sleep(30)
# The queue marks completion just before exit; wait for the original process group.
pid=int(Path('/content/standardized_controls.pid').read_text())
while Path('/proc',str(pid),'stat').exists():
    stat=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[0]
    if stat in ('Z','X'): break
    time.sleep(3)
env=os.environ.copy()
env['HF_TOKEN']=Path('/content/.standardized_hf_token').read_text().strip()
env['TOKENIZERS_PARALLELISM']='false'
env['PYTHONUNBUFFERED']='1'
env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
record('GATES_THEN_TRAIN_RUNNING')
result=subprocess.run(['/content/miniforge3/envs/clrr/bin/python','-u','scripts/revalidation/run_standardized_strong_mbart.py','--stage','gate-train'],cwd=root,env=env)
record('VALIDATION_QUEUE_COMPLETE' if result.returncode==0 else 'FAILED')
raise SystemExit(result.returncode)
""")
with open('/content/strong_mbart_supplement.log','w') as log:
    p=subprocess.Popen(['/content/miniforge3/envs/clrr/bin/python','-u',str(controller)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(p.pid))
print('SCHEDULED_COLAB3_AFTER_C01',p.pid)
