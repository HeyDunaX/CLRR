import subprocess
from pathlib import Path
with open('/content/diagnostics.log','a') as log:
 p=subprocess.Popen(['bash','/content/diagnostics.sh'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
Path('/content/diagnostics.pid').write_text(str(p.pid))
print('RESTARTED',p.pid)
