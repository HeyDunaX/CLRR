import subprocess
from pathlib import Path
root=Path('/content/CLRR_controls')
pidfile=Path('/content/strong_mbart_b_account1.pid')
if pidfile.exists():
    stat=Path('/proc',pidfile.read_text().strip(),'stat')
    assert not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0] in ('Z','X'), 'B already scheduled'
subprocess.run(['unzip','-q','-o','/content/strong_mbart_supplement_bundle.zip','-d',str(root)],check=True)
controller=root/'outputs_rebuttal/amis_standardized_strong_mbart/controller_b_account1.py'
controller.write_text("""import json,os,subprocess,time
from pathlib import Path
root=Path('/content/CLRR_controls')
out=root/'outputs_rebuttal/amis_standardized_strong_mbart'
base=root/'outputs_rebuttal/amis_standardized_controls'
status=out/'queue_account1.json'
def record(state):
    status.write_text(json.dumps({'account':1,'status':state,'current':'A06','updated_unix':time.time()}))
record('WAITING_FOR_NLLB_QUEUE')
while True:
    queue=json.loads((base/'queue_account1.json').read_text())
    if queue['status']=='FAILED':
        record('BLOCKED_ORIGINAL_QUEUE_FAILED'); raise RuntimeError(queue)
    if queue['status']=='VALIDATION_QUEUE_COMPLETE':
        for ident in ['B01','B02','B03','B04','B05']:
            assert json.loads((base/'runs'/ident/'VAL_COMPLETE.json').read_text())['status']=='VALIDATION_COMPLETE_VERIFIED'
        break
    time.sleep(30)
pid=int(Path('/content/standardized_controls.pid').read_text())
while Path('/proc',str(pid),'stat').exists():
    if Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[0] in ('Z','X'): break
    time.sleep(3)
env=os.environ.copy()
env['HF_TOKEN']=Path('/content/.standardized_hf_token').read_text().strip()
env['TOKENIZERS_PARALLELISM']='false'
env['PYTHONUNBUFFERED']='1'
env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
for stage in ['gate','train']:
    record(stage.upper())
    result=subprocess.run(['/content/miniforge3/envs/clrr/bin/python','-u','scripts/revalidation/run_standardized_strong_mbart.py','--id','A06','--stage',stage],cwd=root,env=env)
    if result.returncode:
        record('FAILED'); raise SystemExit(result.returncode)
record('VALIDATION_QUEUE_COMPLETE')
""")
with open('/content/strong_mbart_b_account1.log','w') as log:
    proc=subprocess.Popen(['/content/miniforge3/envs/clrr/bin/python','-u',str(controller)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(proc.pid))
print('A06_SCHEDULED_ACCOUNT1_AFTER_NLLB',proc.pid)
