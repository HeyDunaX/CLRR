"""Configuration-driven workflow. No GPU action is performed by inspect/build."""
import argparse
import importlib.util
import json
import subprocess
import sys
import time

from .config import ROOT, build, create_recipe, read_config, verify_frozen, workspace


def cpu_gate(config):
    out, plan = verify_frozen(config)
    from methods.registry import ALIASES, CANDIDATES
    candidate_only = all(ALIASES.get(s['method'], s['method']) in CANDIDATES for s in plan['runs'])
    name = 'smoke_method_candidates' if candidate_only else 'smoke_standardized_controls'
    from methods.alignment_residual import METHODS as ALIGNMENT_RESIDUAL_METHODS
    if (any(s['method'] in ALIGNMENT_RESIDUAL_METHODS for s in plan['runs'])
            and all(s['method'] in {*ALIGNMENT_RESIDUAL_METHODS, 'baseline'} for s in plan['runs'])):
        name = 'smoke_alignment_residual'
    from methods.decoder_residual_bridge.v1 import METHODS as DRB_METHODS
    if all(s['method'] in DRB_METHODS for s in plan['runs']):
        name = 'smoke_decoder_residual_bridge'
    from methods.difference_preserving.v1 import METHODS as DIFFERENCE_METHODS
    from methods.difference_preserving.v2.model import METHODS as RELATIONAL_METHODS
    if all(s['method'] in DIFFERENCE_METHODS for s in plan['runs']):
        name = 'smoke_difference_preserving'
    if all(s['method'] in RELATIONAL_METHODS for s in plan['runs']):
        name = 'smoke_relational_v2'
    from methods.transition_lsr import METHODS as TRANSITION_METHODS
    if all(s['method'] in TRANSITION_METHODS for s in plan['runs']):
        name = 'smoke_transition_lsr'
    from methods.learned_strength import METHODS as STRENGTH_METHODS
    if all(s['method'] in STRENGTH_METHODS for s in plan['runs']):
        name = 'smoke_learned_strength'
    from methods.paired_route import METHODS as PAIRED_ROUTE_METHODS
    if all(s['method'] in PAIRED_ROUTE_METHODS for s in plan['runs']):
        name = 'smoke_paired_route'
    from methods.decoupled_consistency import METHODS as CONSISTENCY_METHODS
    if all(s['method'] in CONSISTENCY_METHODS for s in plan['runs']):
        name = 'smoke_decoupled_consistency'
    from methods.prefix_consistency import METHODS as PREFIX_METHODS
    if all(s['method'] in PREFIX_METHODS for s in plan['runs']):
        name = 'smoke_prefix_consistency'
    from methods.continuation_consistency import METHODS as CMC_METHODS
    if all(s['method'] in CMC_METHODS for s in plan['runs']):
        name = 'smoke_continuation_consistency'
    from methods.continuation_consistency.v2 import METHODS as ESTIMATOR_METHODS
    if all(s['method'] in ESTIMATOR_METHODS for s in plan['runs']):
        name = 'smoke_continuation_estimator'
    from methods.registry import TWO_METHODS
    if all(s['method'] in TWO_METHODS for s in plan['runs']):
        name = 'smoke_two_method_round1'
    from methods.corrective_order import METHODS as ORDER_METHODS
    if all(s['method'] in ORDER_METHODS for s in plan['runs']):
        name = 'smoke_corrective_order'
    from methods.virtual_corrective_order.v1.model import METHODS as VIRTUAL_ORDER_METHODS
    if all(s['method'] in VIRTUAL_ORDER_METHODS for s in plan['runs']):
        name = 'smoke_virtual_order_training'
    from methods.gold_ordered_consistency import METHODS as GOC_METHODS
    if (any(s['method'] in GOC_METHODS for s in plan['runs'])
            and all(s['method'] in {*GOC_METHODS, 'baseline', 'weak-full-consistency-v1'} for s in plan['runs'])):
        name = 'smoke_gold_ordered_consistency'
    sys.path.insert(0, str(ROOT / 'scripts/revalidation'))
    sys.path.insert(0, str(ROOT / 'tests'))
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests' / f'{name}.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    from . import engine
    engine.configure(out)
    gate.main()
    receipt = json.loads((out / 'CPU_GATE.json').read_text())
    assert receipt['passed'] and {s['id'] for s in receipt['cases']} == {s['id'] for s in plan['runs']}
    return receipt


def run(config, stage, ident=None, account=None, resume=None):
    out, plan = verify_frozen(config)
    if (ident is None) == (account is None):
        raise ValueError('Choose exactly one --id or --account')
    selected = [s for s in plan['runs'] if s['id'] == ident or (account is not None and s['account'] == account)]
    if not selected:
        raise ValueError('No run assigned to this ID/account')
    if account is not None:
        if resume:
            raise ValueError('Resume one explicitly selected run')
        state = {'account': account, 'status': 'STARTING', 'completed': []}
        from .engine import write
        phase = 'test' if stage == 'test' else 'validation'
        status = out / f'queue_{phase}_account{account}.json'
        if status.exists():
            raise FileExistsError('Queue state exists; select an unfinished --id explicitly')
        stages = ['gate', 'train'] if stage == 'gate-train' else [stage]
        try:
            for spec in selected:
                for current in stages:
                    state.update(status=current.upper(), current=spec['id'], updated_unix=time.time())
                    write(status, state)
                    subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/experiment.py'),
                        'run', '--config', str(config), '--id', spec['id'], '--stage', current],
                        check=True, cwd=ROOT)
                state['completed'].append(spec['id'])
            state.update(status='QUEUE_COMPLETE', updated_unix=time.time())
        except Exception as error:
            state.update(status='FAILED', error=str(error), updated_unix=time.time())
            raise
        finally:
            write(status, state)
        return state
    if stage == 'gate-train':
        raise ValueError('Use --account for a gate/train queue, or run the two stages explicitly')
    from . import engine
    engine.configure(out)
    if stage in ('gate', 'technical-gate', 'train', 'pool', 'preflight'):
        engine.wait_for_gpu()
    from .sequence import prepare_pool, preflight, summarize, cleanup_initialization
    dispatch = {'gate': engine.gpu_gate,
                'technical-gate': lambda s: engine.gpu_gate(s, technical_only=True),
                'train': lambda s: engine.train_run(s, resume),
                'test': engine.test_run, 'pool': prepare_pool, 'preflight': preflight,
                'summarize': summarize, 'cleanup-initialization': cleanup_initialization}
    dispatch[stage](selected[0])
    return {'id': ident, 'stage': stage, 'finished': True}


def lock(config):
    out, plan = verify_frozen(config)
    if plan['protocol']['test_barrier'].startswith('No test'):
        raise ValueError('This configuration is validation-only')
    from huggingface_hub import HfApi
    from .integrity import sha256
    api = HfApi(token=False)
    info = api.model_info(plan['repo'])
    assert not info.private
    runs = {}
    for spec in plan['runs']:
        complete = json.loads((out / 'runs' / spec['id'] / 'VAL_COMPLETE.json').read_text())
        assert complete['status'] == 'VALIDATION_COMPLETE_VERIFIED'
        assert complete['spec'] == spec and complete['backup']['verified']
        archive = spec['remote_prefix'] + '/best.zip'
        entries = api.get_paths_info(plan['repo'], [archive], revision=info.sha)
        assert len(entries) == 1 and entries[0].lfs
        assert entries[0].lfs.sha256 == complete['archive_sha256']
        runs[spec['id']] = {'revision': info.sha, 'archive_path': archive,
            'archive_sha256': complete['archive_sha256']}
    target = out / 'selection_lock.json'
    if target.exists():
        raise FileExistsError('Selection already locked')
    receipt = {'status': 'ALL_VALIDATION_VERIFIED', 'runs': runs,
        'plan_sha256': sha256(out / 'PLAN.json'), 'test_used_for_selection': False}
    target.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


def status(config):
    out = workspace(config)
    return {p.relative_to(out).as_posix(): json.loads(p.read_text())
        for p in sorted(out.rglob('*.json'))
        if p.name in ('progress.json', 'VAL_COMPLETE.json', 'COMPLETE.json') or p.name.startswith('queue_')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    technical = sub.add_parser('technical-suite', help='RTX technical fixtures only; no training admission')
    technical.add_argument('--configs', nargs='+', required=True)
    suite = sub.add_parser('launch-suite', help='RTX multi-dataset queue; all gates precede training')
    suite.add_argument('--configs', nargs='+', required=True)
    p = sub.add_parser('create', help='Create one reusable recipe from explicit CLI arguments')
    for flag in ('template', 'from-run', 'name', 'method'):
        p.add_argument('--' + flag, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--account', type=int, choices=(1, 2, 3), required=True)
    for flag in ('dataset', 'backbone', 'model-id', 'model-revision', 'src-lang',
                 'tgt-lang', 'bleu-tokenizer', 'storage-method', 'configuration', 'source-prefix'):
        p.add_argument('--' + flag)
    p.add_argument('--stack', choices=('encoder', 'decoder', 'both'))
    for flag in ('batch-size', 'accumulation', 'max-source-length', 'max-target-length',
                 'validation-beams', 'workers', 'patience', 'middle-layer-idx', 'distance', 'adapter-rank', 'module-seed', 'donor-layer', 'decoder-layer'):
        p.add_argument('--' + flag, type=int)
    for flag in ('lr', 'epochs', 'alignment-weight', 'temperature', 'strength', 'jepa-weight', 'interaction-weight'):
        p.add_argument('--' + flag, type=float)
    p.add_argument('--allow-test', action='store_true', help='Declare a later locked test phase; does not evaluate test')
    for name in ('inspect', 'build', 'cpu', 'run', 'lock', 'status', 'launch', 'monitor', 'prepare-continuation'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        if name == 'run':
            p.add_argument('--stage', required=True, choices=('gate', 'technical-gate', 'train', 'test', 'gate-train', 'pool', 'preflight', 'summarize', 'cleanup-initialization'))
            group = p.add_mutually_exclusive_group(required=True)
            group.add_argument('--id')
            group.add_argument('--account', type=int, choices=(1, 2, 3))
            p.add_argument('--resume')
        if name in ('launch', 'monitor'):
            p.add_argument('--account', type=int, choices=(1, 2, 3), required=True)
        if name == 'launch':
            p.add_argument('--stage', choices=('gate', 'gate-train', 'test'), default='gate-train')
    p = sub.add_parser('fetch')
    p.add_argument('--local-path', required=True, help='Original repo-relative path recorded by the cleanup index')
    args = parser.parse_args()
    if args.command == 'technical-suite':
        from .rtx import launch_technical
        result = launch_technical(args.configs)
    elif args.command == 'launch-suite':
        from .rtx import launch_suite
        result = launch_suite(args.configs)
    elif args.command == 'create':
        result = create_recipe(args)
    elif args.command == 'fetch':
        from .storage import fetch
        result = {'cache_path': str(fetch(args.local_path))}
    else:
        path, config = read_config(args.config)
        if args.command == 'inspect':
            result = {'dataset': config['dataset'], 'protocol': config['protocol'],
                'runs': config['runs'], 'gpu_started': False}
        elif args.command == 'build': result = build(path)
        elif args.command == 'cpu': result = cpu_gate(path)
        elif args.command == 'prepare-continuation':
            from methods.continuation_consistency.v1.probe import main
            result = main(path)
        elif args.command == 'run': result = run(path, args.stage, args.id, args.account, args.resume)
        elif args.command == 'lock': result = lock(path)
        elif args.command == 'status': result = status(path)
        else:
            if config['protocol'].get('hardware_profile') == 'rtx6000-fp32':
                from .rtx import launch, monitor
            else:
                from .remote import launch, monitor
            result = launch(path, args.account, args.stage) if args.command == 'launch' else monitor(path, args.account)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
