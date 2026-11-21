"""Shared train-only probes, frozen calibration, and admission checks."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from transformers import DataCollatorForSeq2Seq


def validate_recipe(config, run):
    p = config['protocol']
    expected = dict(attention_implementation='sdpa', sdpa_kernel='math',
        gradient_checkpointing_use_reentrant=False,
        view_checkpointing='whole-functional-call-non-reentrant',
        saved_tensors=run.get('order_offload', 'cpu')+'-offload', epochs=20, patience=10, selection_metric='eval_bleu')
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Corrective-order execution differs from reviewed recipe')
    if (not config.get('corrective_order_preparation') or run['seed'] != 42
            or run['microbatch'] != 4 or run['accumulation'] != 32
            or run['learning_rate'] != 5e-5 or run['backbone'] != 'mbart-large-50'
            or config.get('post_training_test', {}).get('authorized')):
        raise ValueError('This bounded pilot locks seed42, mBART B4/GA32, no test')
    for key in ('virtual_eta', 'order_multiplier'):
        if not math.isfinite(run[key]) or run[key] <= 0:
            raise ValueError('Positive virtual step and multiplier required')
    preparation = config['corrective_order_preparation']
    if preparation.get('mode') == 'finite_relu':
        if (preparation['eta_grid'] != [run['learning_rate']]
                or run['virtual_eta'] != run['learning_rate']
                or not preparation.get('require_null_replay')
                or preparation['gradient_ratio'] != .1 or preparation['train_batches'] != 3):
            raise ValueError('Finite-map admission locks one a-priori eta and original gradient calibration')
    review = config['design_review']
    root = Path(__file__).resolve().parents[4]
    file = (root / review['path']).resolve()
    if not file.is_relative_to(root/'docs/research') or hashlib.sha256(file.read_bytes()).hexdigest() != review['sha256']:
        raise ValueError('Corrective-order design changed')


def add_probes(dataset, seed):
    # Fixed random ordering with a cyclic successor is a derangement. All six
    # arms receive the SAME probe assignments; no validation/test is sampled.
    order = np.random.default_rng(seed + 731).permutation(len(dataset))
    peers = np.empty(len(order), dtype=np.int64)
    peers[order] = np.roll(order, -1)
    if len(order) < 2 or np.any(peers == np.arange(len(order))):
        raise ValueError('At least two distinct training rows required')
    for key in ('input_ids', 'attention_mask', 'labels'):
        values = dataset[key]
        dataset = dataset.add_column('probe_'+key, [values[int(i)] for i in peers])
    return dataset


class ProbeCollator:
    def __init__(self, tokenizer, model):
        self.native = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    def __call__(self, examples):
        output = self.native([{k: v for k, v in e.items() if not k.startswith('probe_')} for e in examples])
        if 'probe_input_ids' in examples[0]:
            probe = self.native([{k.removeprefix('probe_'): v for k, v in e.items()
                                  if k.startswith('probe_')} for e in examples])
            output.update({'probe_'+k: v for k, v in probe.items()})
        return output


def apply_calibration(model, output, spec):
    path = Path(output)/'order_preparation/CALIBRATION.json'
    if path.exists():
        receipt = json.loads(path.read_text())
        if not receipt['passed']:
            raise ValueError('Calibration did not pass')
        model.eta = receipt['eta']
        model.coefficient = receipt['curvature_coefficient' if model.mode == 'curvature' else 'order_coefficient']


def require_admission(output, plan):
    from experiment_runner.integrity import sha256
    output = Path(output)
    bundle = sha256(output/'bundle_manifest.json')
    cal = json.loads((output/'order_preparation/CALIBRATION.json').read_text())
    assert cal['passed'] and cal['bundle_sha256'] == bundle
    assert cal['optimizer_steps'] == 0 and not cal['test_used']
    for spec in plan['runs']:
        gate = json.loads((output/'gates'/spec['id']/'PASS.json').read_text())
        assert gate['passed'] and gate['bundle_sha256'] == bundle
