import json,subprocess
from pathlib import Path
p=Path('/content/diagnostics.pid');stat=Path('/proc')/p.read_text().strip()/'stat' if p.exists() else Path('/nonexistent')
alive=stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
root=Path('/content/CLRR_diagnostics/outputs_rebuttal/method_candidate_diagnostics')
results={p.parent.name:json.loads(p.read_text()) for p in root.glob('*/COMPLETE.json')}
log=Path('/content/diagnostics.log')
print(json.dumps({'alive':alive,'queue_complete':Path('/content/diagnostic_queue_complete').exists(),'results':results,'tail':log.read_text(errors='replace').splitlines()[-12:] if log.exists() else []}))
