import json
from pathlib import Path

root = Path('/content/CLRR_controls/outputs_rebuttal')
for name in ['amis_standardized_strong_mbart/schedule_status.json',
             'amis_standardized_strong_mbart/queue_account3.json',
             'amis_standardized_strong_mbart/runs/A05/progress.json',
             'amis_standardized_strong_mbart/runs/A06/progress.json',
             'amis_standardized_controls/queue_account3.json',
             'amis_standardized_controls/runs/C01/progress.json']:
    path = root / name
    print(name, path.read_text() if path.exists() else 'PENDING')
pid = Path('/content/strong_mbart_supplement.pid').read_text().strip()
print('CONTROLLER_ALIVE', Path('/proc', pid).exists())
for gate in (root / 'amis_standardized_strong_mbart').rglob('GPU_GATE.json'):
    print('GPU_GATE', gate.parent.name, gate.read_text())
log = Path('/content/strong_mbart_supplement.log')
if log.exists():
    print('LOG_TAIL', '\n'.join(log.read_text(errors='replace').splitlines()[-10:]))
