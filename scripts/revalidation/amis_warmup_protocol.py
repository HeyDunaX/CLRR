"""Fail closed on changed inputs/runtime for the corrected warmup experiment."""
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

from amis_rewire.scientific_records import digest, save

ROOT = Path(__file__).resolve().parents[2]
REVISION = 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30'
VERSIONS = {'torch': '2.6.0+cu124', 'transformers': '4.57.6', 'huggingface-hub': '0.36.0',
            'accelerate': '1.15.0', 'sacrebleu': '2.6.0', 'datasets': '3.6.0',
            'numpy': '2.5.3', 'pandas': '3.0.6', 'tokenizers': '0.22.2',
            'sentencepiece': '0.2.2', 'safetensors': '0.8.0', 'protobuf': '7.36.2'}


def preflight(out):
    import torch
    from huggingface_hub import HfApi
    manifest = json.loads((ROOT / 'warmup_inputs_manifest_v2.json').read_text())
    for name, expected in manifest['files'].items():
        assert digest(ROOT / name) == expected, f'Input hash mismatch: {name}'
    control_config = json.loads((ROOT / 'warmup_control_training_configuration_v2.json').read_text())
    assert control_config['status'] == 'PASS' and len(control_config['controls']) == 9
    assert all(control_config['model_ast_matches_existing_confirmation_bundle'].values())
    versions = {k: importlib.metadata.version(k) for k in VERSIONS}
    assert versions == VERSIONS, versions
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    gpu = torch.cuda.get_device_properties(0)
    assert 'A100' in gpu.name and 39 * 1024 ** 3 < gpu.total_memory < 41 * 1024 ** 3
    ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    # Standard A100 runtimes have ~83 GiB host RAM; verify SKU in the CLI launch receipt.
    token = (Path('/content/.hf_warmup_token').read_text().strip()
             if Path('/content/.hf_warmup_token').exists() else os.environ.get('HF_TOKEN'))
    assert token, 'HF backup authentication must be ready before training'
    os.environ['HF_TOKEN'] = token
    api = HfApi(token=token)
    assert api.whoami()['name'] == 'FiveC', 'Unexpected HF account'
    # Existing repository only, use a fresh suite prefix.
    api.repo_info('FiveC/amis-rewire-checkpoints', repo_type='model')
    receipt = {'versions': versions, 'python': __import__('platform').python_version(),
               'gpu': gpu.name, 'gpu_bytes': gpu.total_memory, 'host_ram_bytes': ram,
               'manifest_sha256': digest(ROOT / 'warmup_inputs_manifest_v2.json'),
               'input_checks': 'PASS', 'backbone_revision': REVISION,
               'hf_repo': 'FiveC/amis-rewire-checkpoints', 'hf_prefix': 'amis_routing_warmup_20261004_v2',
               'test_access': 'No test tokenization/generation; hash verification only'}
    save(out / 'preflight.json', receipt)
    (out / 'environment_freeze.txt').write_text(subprocess.check_output(
        [__import__('sys').executable, '-m', 'pip', 'freeze'], text=True))
    return manifest
