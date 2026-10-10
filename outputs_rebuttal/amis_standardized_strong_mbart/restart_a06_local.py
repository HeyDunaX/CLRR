from pathlib import Path
import hashlib,json,os,sys,tempfile,zipfile
root=Path(r'D:\Code\CLRR')
sys.path.insert(0,str(root/'scratch'))
from launch_fresh_confirmation_remote import ensure_session,get_session_name,call_pilot,get_hf_token
out=root/'outputs_rebuttal/amis_standardized_strong_mbart'
bundles=[root/'scratch/standardized_controls_bundle.zip',root/'scratch/strong_mbart_supplement_bundle.zip']
manifest=json.loads((out/'bundle_manifest.json').read_text())
combined={}
for b in bundles:
    with zipfile.ZipFile(b) as z:
        combined.update({n:z.read(n) for n in z.namelist()})
for name,digest in manifest.items():
    assert hashlib.sha256(combined[name]).hexdigest()==digest, name
cpu=json.loads((out/'CPU_GATE.json').read_text())
assert cpu['passed'] and cpu['bundle_sha256']==hashlib.sha256((out/'bundle_manifest.json').read_bytes()).hexdigest()
def call(args):
    code,stdout,stderr=call_pilot('1',args,timeout=300)
    if code:raise RuntimeError(f'{args[0]} failed: {stdout}\n{stderr}')
    print(stdout,flush=True)
    return stdout
ensure_session('1')
session=get_session_name('1')
check=out/'check_restart_a06_remote.py'
check.write_text("import subprocess\nr=subprocess.run(['pgrep','-af','python.*scripts/revalidation/run_'],capture_output=True,text=True)\nassert not r.stdout.strip(), 'A worker is already active'\nprint('NO_ACTIVE_RESEARCH_WORKER')\n",encoding='utf-8')
call(['exec','-s',session,'-f',str(check),'--timeout','60'])
for b in bundles:call(['upload',str(b),'/content/'+b.name,'-s',session])
fd,tmp=tempfile.mkstemp(prefix='clrr-hf-token-')
try:
    with os.fdopen(fd,'w') as f:f.write(get_hf_token())
    call(['upload',tmp,'/content/.standardized_hf_token','-s',session])
finally:Path(tmp).unlink(missing_ok=True)
controller='''import json,subprocess,sys,time
from pathlib import Path
root=Path('/content/CLRR_controls')
out=root/'outputs_rebuttal/amis_standardized_strong_mbart'
path=out/'queue_account1.json'
def record(state):
    path.write_text(json.dumps({'account':1,'status':state,'current':'A06','updated_unix':time.time(),'restart_from_scratch':True}))
for stage in ['gate','train']:
    record(stage.upper())
    result=subprocess.run([sys.executable,'-u','scripts/revalidation/run_standardized_strong_mbart.py','--id','A06','--stage',stage],cwd=root)
    if result.returncode:
        record('FAILED');raise SystemExit(result.returncode)
record('VALIDATION_QUEUE_COMPLETE')
'''
remote_controller=out/'controller_b_restart.py'
remote_controller.write_text(controller,encoding='utf-8')
call(['upload',str(remote_controller),'/content/controller_b_restart.py','-s',session])
shell=(root/'outputs_rebuttal/amis_standardized_controls/run_account1.sh').read_text(encoding='utf-8')
old='"$PY" -u scripts/revalidation/run_standardized_controls.py --account 1 --stage gate-train'
assert old in shell
shell=shell.replace(old,'unzip -q -o /content/strong_mbart_supplement_bundle.zip\n"$PY" -u /content/controller_b_restart.py')
script=out/'restart_a06.sh'
script.write_text(shell,encoding='utf-8',newline='\n')
call(['upload',str(script),'/content/restart_a06.sh','-s',session])
starter=out/'start_restart_a06_remote.py'
starter.write_text("import subprocess\nfrom pathlib import Path\np=Path('/content/strong_mbart_b_account1.pid')\nif p.exists():\n s=Path('/proc')/p.read_text().strip()/'stat'\n assert not s.exists() or s.read_text().rsplit(')',1)[1].split()[0] in ('Z','X'), 'A06 already active'\nwith open('/content/strong_mbart_b_account1.log','w') as log:\n proc=subprocess.Popen(['bash','/content/restart_a06.sh'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\np.write_text(str(proc.pid))\nprint('A06_RESTART_SETUP_LAUNCHED',proc.pid)\n",encoding='utf-8')
result=call(['exec','-s',session,'-f',str(starter),'--timeout','60'])
receipt={'id':'A06','account':1,'status':'RESTART_SETUP_LAUNCHED','restart_from_scratch':True,'bundles':{b.name:hashlib.sha256(b.read_bytes()).hexdigest() for b in bundles},'method_and_hyperparameters_unchanged':True,'result':result}
(out/'restart_a06_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print('A06_RESTART_LAUNCHED',flush=True)
