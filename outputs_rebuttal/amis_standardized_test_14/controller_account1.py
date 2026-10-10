import json,subprocess,sys,time
from pathlib import Path
root=Path('/content/CLRR_controls')
out=root/'outputs_rebuttal/amis_standardized_test_14'
out.mkdir(parents=True,exist_ok=True)
state={'account':1,'assigned':['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'D01'],'completed':[],'status':'STARTING'}
def record():
 state['updated_unix']=time.time()
 (out/'queue_account1.json').write_text(json.dumps(state))
record()
for ident in state['assigned']:
 state.update(status='TEST',current=ident);record()
 result=subprocess.run([sys.executable,'-u','scripts/revalidation/test_standardized_14.py','--id',ident],cwd=root)
 if result.returncode:
  state.update(status='FAILED',returncode=result.returncode);record();raise SystemExit(result.returncode)
 state['completed'].append(ident);record()
state.update(status='TEST_QUEUE_COMPLETE');record()
