"""Admission checks for the single initialization-calibrated pilot."""
import hashlib
import json


def validate_recipe(config, run, root):
    expected = dict(backbone='mbart-large-50',microbatch=4,accumulation=32,
        virtual_rank=32,basis_seed=11,virtual_eta=1.,order_chunk=16,seed=42)
    if any(run.get(k) != v for k,v in expected.items()):
        raise ValueError('Virtual-order pilot differs from the reviewed seed42 design')
    p = config['protocol']
    if (p['hardware_profile'] != 'rtx6000-fp32' or p['attention_implementation'] != 'sdpa'
            or p['selection_metric'] != 'eval_bleu' or p['patience'] != 10
            or not p['gradient_checkpointing_use_reentrant']):
        raise ValueError('Keep the locked native RTX baseline protocol')
    if config.get('post_training_test',{}).get('authorized'):
        raise ValueError('Pilot is validation-only')
    record = config['virtual_order_calibration']
    path = (root / record['path']).resolve()
    if not path.is_relative_to(root / 'outputs/experiments'):
        raise ValueError('Calibration must be an experiment receipt')
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Initialization calibration changed')
    c = json.loads(path.read_text())
    if (not c['passed'] or c['optimizer_steps'] != 0 or c['validation_used'] or c['test_used']
            or c['model_id'] != run['model_id'] or c['model_revision'] != run['model_revision']
            or c['train_sha256'] != config['data']['train']['sha256']
            or c['coefficient'] != run['order_coefficient']):
        raise ValueError('Calibration/init/data/scale mismatch')
