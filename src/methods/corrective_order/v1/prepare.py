"""Train-only numerical calibration before ALL GPU gates and long training."""
import argparse
import gc
import hashlib
import time
import traceback

import numpy as np
import torch
from transformers import set_seed


def fingerprint(model):
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_norm(loss, model):
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
    norm = sum(g.detach().double().square().sum() for g in gradients).sqrt()
    result = float(norm)
    del gradients
    if not np.isfinite(result) or result <= 0:
        raise ValueError('Invalid calibration gradient norm')
    return result


def eligible_steps(records):
    """Require the declared asymptotic scaling; never weaken it on failure."""
    eligible = []
    for a, b in zip(records, records[1:]):
        if not all(np.isfinite(r[k]) and r[k] > 0 for r in (a, b)
                   for k in ('auxiliary', 'parameter_order_distance', 'eta')):
            continue
        ratio = b['auxiliary']/a['auxiliary']
        distance_ratio = ((b['parameter_order_distance']/b['eta']**2) /
                          (a['parameter_order_distance']/a['eta']**2))
        if .5 <= ratio <= 2. and .5 <= distance_ratio <= 2.:
            eligible.append(a['eta'])
    return eligible


def main(config):
    from experiment_runner.config import verify_frozen
    from experiment_runner import engine as e
    out, plan = verify_frozen(config)
    e.configure(out)
    e.wait_for_gpu()
    e.validate_environment()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = out/'order_preparation'
    if (target/'CALIBRATION.json').exists():
        raise FileExistsError('Calibration already frozen')
    mem = dict((line.split(':')[0], int(line.split()[1]))
               for line in open('/proc/meminfo') if line.startswith(('MemAvailable:', 'MemTotal:')))
    if mem['MemAvailable'] < 12*2**20:
        raise RuntimeError('Need 12 GiB available host RAM before admission; 6 GiB reserve enforced during offload')
    spec = next(s for s in plan['runs'] if s['id'] == 'R2_ORDER_S42')
    set_seed(spec['seed'])
    model, tokenizer, _ = e.load_model(spec)
    model.to('cuda').train()
    model.gradient_checkpointing_enable({'use_reentrant': False})
    before = fingerprint(model)
    _, ds = e.dataset('train', tokenizer, spec)
    collate = e.collator(model, tokenizer, spec)
    order = np.random.default_rng(20261118).permutation(len(ds))
    batches = []
    for start in (0, 4, 8):
        tensors = {k: v.to('cuda') for k, v in collate([ds[int(i)] for i in order[start:start+4]]).items()}
        batch = {k: v for k, v in tensors.items() if not k.startswith('probe_')}
        probe = {k.removeprefix('probe_'): v for k, v in tensors.items() if k.startswith('probe_')}
        batches.append((batch, probe))
    records = []
    started = time.time()
    preparation = plan['corrective_order_preparation']
    mode = preparation.get('mode', 'smooth_scaling')
    if mode not in ('smooth_scaling', 'finite_relu'):
        raise ValueError('Unknown numerical admission policy')
    etas = preparation['eta_grid']
    if mode == 'finite_relu':
        # A fixed finite-map penalty does not require a smooth Lie-bracket limit.
        # This is an explicit protocol amendment, never a relaxed ratio cutoff.
        from .audit import ActivationAudit
        assert len(etas) == 1 and etas[0] == spec['learning_rate']
        assert preparation['require_null_replay']
        print('ORDER_ACTIVATION_AUDIT_START', flush=True)
        audit = ActivationAudit(model.base_model, batches[0][0])
        evidence = audit.run(model, batches[0][0], etas[0])
        e.write(target/'ACTIVATION_AUDIT.json', evidence)
        if evidence['crossings'] <= 0:
            raise ValueError('No observed ReLU crossings; do not claim boundary explanation')
        del audit
        gc.collect(); torch.cuda.empty_cache()
        print('ORDER_ACTIVATION_AUDIT_COMPLETE', evidence['crossings'], evidence['positions'], flush=True)
    # Identical masks define the identical finite composition in either order.
    # This catches RNG/recomputation drift before interpreting any eta scaling.
    if plan['corrective_order_preparation'].get('require_null_replay', False):
        print('ORDER_NULL_REPLAY_START', flush=True)
        model.eta = plan['corrective_order_preparation']['eta_grid'][0]
        with model.offload_context():
            loss, ce, penalty = model.objective(*batches[0], seeds=(101, 101))
        null = dict(model.last_diagnostics)
        passed = null['parameter_order_distance'] == 0 and null['raw_probe_skl'] == 0
        e.write(target/'NULL_REPLAY.json', dict(passed=passed, diagnostics=null,
            optimizer_steps=0, test_used=False))
        if not passed:
            raise ValueError('Identical-mask composition is not identical; stop before eta calibration')
        del loss, ce, penalty
        gc.collect(); torch.cuda.empty_cache()
        print('ORDER_NULL_REPLAY_PASS', flush=True)
    for eta in plan['corrective_order_preparation']['eta_grid']:
        print('ORDER_ETA_PROBE_START', eta, flush=True)
        model.eta = eta
        with model.offload_context():
            loss, ce, penalty = model.objective(*batches[0], seeds=(101, 211))
        record = dict(model.last_diagnostics)
        record['eta'] = eta
        records.append(record)
        e.write(target/'NUMERICAL_PROGRESS.json', dict(status='CALIBRATING', records=records,
            optimizer_steps=0, test_used=False, seconds=time.time()-started))
        print('ORDER_ETA_PROBE', record, flush=True)
        del loss, ce, penalty
        gc.collect(); torch.cuda.empty_cache()
    # Explicit numerical tolerance, not evidence of harmfulness or novelty.
    if mode == 'finite_relu':
        eligible = [r['eta'] for r in records if all(np.isfinite(r[k]) and r[k] > 0
            for k in ('auxiliary', 'raw_probe_skl', 'parameter_order_distance', 'ce'))]
    else:
        eligible = eligible_steps(records)
    if not eligible:
        raise ValueError('No resolvable small-step scaling region; stop, do not train')
    eta = eligible[0]
    norms = {'order': [], 'curvature': []}
    for mode in norms:
        model.mode, model.eta = mode, eta
        for i, (batch, probe) in enumerate(batches):
            print('ORDER_GRADIENT_CALIBRATION_START', mode, i, flush=True)
            with model.offload_context():
                loss, ce, penalty = model.objective(batch, probe, seeds=(101+i, 211+i))
            ce_norm = gradient_norm(ce, model)
            aux_norm = gradient_norm(penalty, model)
            norms[mode].append(dict(ce_norm=ce_norm, auxiliary_norm=aux_norm,
                ce=float(ce.detach()), raw_auxiliary=float(penalty.detach())))
            print('ORDER_GRADIENT_CALIBRATION', mode, i, norms[mode][-1], flush=True)
            del loss, ce, penalty
            gc.collect(); torch.cuda.empty_cache()
    coefficients = {key: float(np.median([.1*r['ce_norm']/r['auxiliary_norm'] for r in rows]))
                    for key, rows in norms.items()}
    assert fingerprint(model) == before, 'Frozen calibration mutated model state'
    assert all(p.grad is None for p in model.parameters()), 'Calibration accumulated .grad'
    receipt = dict(passed=True, eta=eta, order_coefficient=coefficients['order'],
        admission_policy=mode, smooth_bracket_limit_verified=False,
        curvature_coefficient=coefficients['curvature'],
        policy='median initial auxiliary/CE gradient-norm ratio 0.10; 3 TRAIN batches',
        numerical_records=records, gradient_records=norms, optimizer_steps=0,
        null_replay_required=plan['corrective_order_preparation'].get('require_null_replay', False),
        parameters_unchanged=True, test_used=False, validation_used=False,
        probe_sampling='fixed TRAIN derangement shared by all arms',
        mechanism_causally_verified=False,
        bundle_sha256=e.sha256(out/'bundle_manifest.json'),
        max_memory_gib=torch.cuda.max_memory_allocated()/2**30, seconds=time.time()-started)
    e.write(target/'CALIBRATION.json', receipt)
    print('ORDER_CALIBRATION_PASS', receipt, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    try:
        main(args.config)
    except Exception as error:
        from experiment_runner import engine as e
        e.write(e.OUT/'order_preparation/FAILED.json', dict(passed=False,
            error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(),
            training_admitted=False, recipe_changed=False, updated_unix=time.time()))
        raise
