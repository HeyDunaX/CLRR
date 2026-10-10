"""Checks required before deleting weights or evaluating the locked cohort."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def weight_file(folder):
    for name in ('pytorch_model.bin', 'model.safetensors'):
        path = Path(folder) / name
        if path.exists():
            return path
    raise FileNotFoundError(f'No supported weights in {folder}')

def verify_remote_files(api, repo, files):
    """Pin one revision and compare every remote artifact with its local bytes."""
    revision = api.model_info(repo).sha
    entries = {e.path: e for e in api.get_paths_info(repo, list(files), revision=revision)}
    verified = {}
    for remote, local in files.items():
        local = Path(local)
        entry = entries[remote]
        assert entry.size == local.stat().st_size, f'Size mismatch: {remote}'
        digest = sha256(local)
        if entry.lfs:
            assert entry.lfs.sha256 == digest, f'LFS hash mismatch: {remote}'
        else:
            data = local.read_bytes()
            assert entry.blob_id == hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(), remote
        verified[remote] = {'size': entry.size, 'sha256': digest}
    return {'repo': repo, 'revision': revision, 'files': verified, 'verified': True}

def restore_archive(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert all((destination / n).resolve().is_relative_to(destination.resolve()) for n in z.namelist())
        z.extractall(destination)

def strict_wrapped_model(folder, method, **kwargs):
    from transformers import AutoModelForSeq2SeqLM
    from unittest.mock import patch
    from amis_rewire.modeling import load_model
    base, info = AutoModelForSeq2SeqLM.from_pretrained(str(folder), output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')), info
    with patch('amis_rewire.modeling.AutoModelForSeq2SeqLM.from_pretrained', return_value=base):
        return load_model(str(folder), method=method, **kwargs)

def remove_weights(out, folder, archive):
    """Caller must have a verified collection receipt before cleanup."""
    out, folder, archive = Path(out), Path(folder), Path(archive)
    receipt = json.loads((out / 'collection_receipt.json').read_text())
    assert receipt['verified'] and receipt['files']
    targets = [folder/'best_model', archive, *folder.glob('checkpoint-*')]
    for target in targets:
        assert target.resolve().is_relative_to(out.resolve()), target
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
