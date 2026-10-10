"""Retrieve verified public model artifacts; diagnostics remain local."""
import json
from pathlib import Path

from .config import ROOT
from .integrity import sha256

INDEX = ROOT / 'outputs_rebuttal/repository_cleanup/model_output_index.json'


def fetch(local_path):
    from huggingface_hub import hf_hub_download
    index = json.loads(INDEX.read_text(encoding='utf-8'))
    key = Path(local_path).as_posix()
    records = {r['local']: r for r in index['verified']}
    if key not in records:
        raise KeyError('No verified public copy for that original path')
    record = records[key]
    if record['size'] > 8 * 1024**2:
        raise ValueError('Use the stored HF link for large model artifacts; local fetch is limited to small files')
    path = Path(hf_hub_download(record.get('repo', index['repo']), record['remote'],
        revision=record.get('revision', index['revision']), token=False,
        local_dir=ROOT / '.cache/public_model_outputs'))
    if sha256(path) != record['sha256']:
        raise ValueError('Public artifact checksum mismatch')
    return path
