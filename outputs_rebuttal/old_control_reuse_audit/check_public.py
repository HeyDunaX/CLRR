import json
from pathlib import Path
import requests
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
REPO = 'FiveC/amis-rewire-checkpoints'
api = HfApi()
info = api.model_info(REPO)
assert not info.private
response = requests.get(f'https://huggingface.co/{REPO}/resolve/{info.sha}/registry/checkpoint_index.json', timeout=60)
response.raise_for_status()
index = response.json()['checkpoints']
proofs = []
controls = json.loads((OUT / 'number_replay.json').read_text())['seed42_controls']
for control in controls:
    folder = ROOT / control['metadata_folder']
    v = json.loads((folder / 'local_verification.json').read_text())
    digest = v['hf_archive']['archive_sha256']
    matches = [r for r in index if r['sha256'] == digest]
    assert matches, control['family']
    for row in matches:
        entry = api.get_paths_info(REPO, [row['path']], revision=info.sha)[0]
        assert entry.lfs.sha256 == digest and entry.size == v['hf_archive']['archive_size']
    command_path = folder / 'command.json'
    command = json.loads(command_path.read_text()) if command_path.exists() else []
    def arg(name):
        return command[command.index(name) + 1] if name in command else None
    proofs.append(dict(control, archive_sha256=digest, public_paths=[r['path'] for r in matches],
        patience=arg('--early-stopping-patience'), learning_rate=arg('--learning-rate'),
        microbatch=arg('--per-device-train-batch-size'), accumulation=arg('--gradient-accumulation-steps'),
        public_revision=info.sha))
(OUT / 'public_seed42_controls.json').write_text(json.dumps(proofs, indent=2), encoding='utf-8')
print(json.dumps([{k: r[k] for k in ['family', 'completed_epochs', 'selected_epoch', 'patience', 'public_paths']} for r in proofs], indent=2))
print('ALL_5_CORRECTED_SEED42_ARCHIVES_PUBLIC_SHA_SIZE_PASS')
