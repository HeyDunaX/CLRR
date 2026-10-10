import json,subprocess,sys,time
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
