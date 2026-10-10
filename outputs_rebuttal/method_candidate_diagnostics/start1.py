import subprocess
from pathlib import Path
Path('/content/diagnostic_queue_complete').unlink(missing_ok=True)
with open('/content/diagnostics.log','w') as log:
 p=subprocess.Popen(['bash','/content/diagnostics.sh'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
Path('/content/diagnostics.pid').write_text(str(p.pid))
print('DIAGNOSTICS_LAUNCHED',p.pid)
