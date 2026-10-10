import subprocess
from pathlib import Path
p=Path('/content/strong_mbart_b_account1.pid')
if p.exists():
 s=Path('/proc')/p.read_text().strip()/'stat'
 assert not s.exists() or s.read_text().rsplit(')',1)[1].split()[0] in ('Z','X'), 'A06 already active'
with open('/content/strong_mbart_b_account1.log','w') as log:
 proc=subprocess.Popen(['bash','/content/restart_a06.sh'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
p.write_text(str(proc.pid))
print('A06_RESTART_SETUP_LAUNCHED',proc.pid)
