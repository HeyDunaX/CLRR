import json
from pathlib import Path

root = Path('/content/CLRR_controls/outputs_rebuttal/amis_standardized_strong_mbart')
for name in ['migration_guard.json', 'queue_account3.json', 'runs/A05/VAL_COMPLETE.json']:
    path = root / name
    print(name, path.exists(), path.read_text()[:1200] if path.exists() else '')
for name in ['/content/strong_mbart_migration_guard.pid', '/content/strong_mbart_migration_guard.log']:
    path = Path(name)
    print(name, path.exists(), path.read_text()[-2500:] if path.exists() else '')
