"""Validate and freeze recipes without starting a GPU or changing old receipts."""
import copy
import json
import re
import subprocess
import zipfile
from pathlib import Path

from .integrity import sha256

ROOT = Path(__file__).resolve().parents[2]
REPO = 'FiveC/amis-rewire-checkpoints'


def read_config(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'configs/experiments'):
        raise ValueError('Put recipes under configs/experiments/')
    config = json.loads(path.read_text(encoding='utf-8'))
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', path.stem):
        raise ValueError('Unsafe experiment name')
    if config['repo'] != REPO:
        raise ValueError('Use the canonical public checkpoint repository')
    if not re.fullmatch(r'[a-z0-9_-]+', config['dataset']):
        raise ValueError('Unsafe dataset name')
    required = {'epochs', 'patience', 'weight_decay', 'warmup_ratio', 'optim',
        'scheduler', 'adam_beta1', 'adam_beta2', 'adam_epsilon', 'max_grad_norm',
        'bf16', 'tf32_training', 'gradient_checkpointing',
        'gradient_checkpointing_use_reentrant', 'eval_batch', 'pad_multiple',
        'generation_precision', 'test_beams', 'bleu_tokenizer', 'selection',
        'test_barrier', 'save_limit', 'dataloader_num_workers'}
    missing = required - config['protocol'].keys()
    if missing:
        raise ValueError(f'Missing explicit protocol fields: {sorted(missing)}')
    if config['protocol']['generation_precision'] != 'fp32':
        raise ValueError('This engine requires the verified FP32 generation protocol')
    if config['protocol']['epochs'] <= 0 or config['protocol']['patience'] < 1:
        raise ValueError('Epoch cap and patience must be positive')
    from methods.registry import ALIASES, CANDIDATES, ORIGINAL_ALIASES
    supported = {'baseline', 'middle-alignment', 'bitfit', 'narrow-lora',
                 'strong-lora-a', 'strong-lora-b', *ORIGINAL_ALIASES,
                 *ORIGINAL_ALIASES.values(), *ALIASES, *CANDIDATES}
    if not config['runs']:
        raise ValueError('No runs declared')
    ids, targets = set(), set()
    for run in config['runs']:
        if run['method'] not in supported:
            raise ValueError(f"Unimplemented method: {run['method']}")
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', run['id']) or run['id'] in ids:
            raise ValueError('Run IDs must be unique safe names')
        ids.add(run['id'])
        prefix = run['remote_prefix']
        parts = prefix.split('/')
        if len(parts) < 5 or parts[0] != 'checkpoints' or parts[1] != config['dataset']:
            raise ValueError('Remote paths must start checkpoints/<dataset>/')
        if parts[2] != run['backbone'] or parts[4] != f"seed-{run['seed']}":
            raise ValueError('Remote path does not match backbone/seed')
        if any(not re.fullmatch(r'[a-zA-Z0-9_.-]+', p) or p in ('.', '..') for p in parts):
            raise ValueError('Unsafe public checkpoint path')
        if prefix in targets:
            raise ValueError('Two runs would upload to the same destination')
        targets.add(prefix)
        if run.get('data_seed', run['seed']) != run['seed']:
            raise ValueError('This engine uses matched seed/data_seed')
        if run['microbatch'] <= 0 or run['accumulation'] <= 0:
            raise ValueError('Batch and accumulation must be positive')
        if run['learning_rate'] <= 0 or run['validation_beams'] < 1:
            raise ValueError('Learning rate and beam count must be positive')
        if run['method'] == 'middle-alignment':
            if run.get('middle_layer_idx') is None or run.get('temperature', 0) <= 0 or run.get('alignment_weight') is None:
                raise ValueError('Middle requires layer index, temperature and alignment weight')
        if run['method'] in ('clrr-precache', 'full-precache', 'lsr-centered', 'full-centered',
                             'precache-clrr-v1', 'precache-full-v1', 'centered-lsr-v1', 'centered-full-v1'):
            if run['backbone'] != 'mbart-large-50' or run.get('stack', 'encoder') != 'encoder':
                raise ValueError('Frozen candidate gates currently support mBART encoder only')
            for field, expected in {'distance': 2, 'strength': .10, 'jepa_weight': .10}.items():
                if run.get(field, expected) != expected:
                    raise ValueError(f'Frozen candidate does not support overriding {field}')
    if config['protocol']['pad_multiple'] != 8:
        raise ValueError('The verified shared collator uses pad_multiple=8')
    return path, config


def workspace(path):
    path, _ = read_config(path)
    return ROOT / 'outputs/experiments' / path.stem


def create_recipe(args):
    """Explicit CLI overrides on a selected template run; never mutate template."""
    _, original = read_config(args.template)
    matches = [s for s in original['runs'] if s['id'] == args.from_run]
    if len(matches) != 1:
        raise ValueError('Choose an existing template run with --from-run')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.name):
        raise ValueError('Unsafe recipe name')
    target = ROOT / 'configs/experiments' / f'{args.name}.json'
    if target.exists():
        raise FileExistsError('Recipe name already exists')
    config = copy.deepcopy(original)
    spec = copy.deepcopy(matches[0])
    from methods.registry import ALIASES, ORIGINAL_ALIASES
    method = ALIASES.get(args.method, ORIGINAL_ALIASES.get(args.method, args.method))
    clrr_methods = {'original-clrr', 'original-full', 'precache-clrr-v1', 'precache-full-v1', 'centered-full-v1'}
    lsr_methods = {'original-lsr', 'original-full', 'precache-full-v1', 'centered-lsr-v1', 'centered-full-v1'}
    if any(getattr(args, flag) is not None for flag in ('distance', 'strength', 'stack')) and method not in clrr_methods:
        raise ValueError('Rewiring arguments require a CLRR method')
    if args.jepa_weight is not None and method not in lsr_methods:
        raise ValueError('LSR weight requires an LSR method')
    if any(getattr(args, flag) is not None for flag in ('middle_layer_idx', 'alignment_weight', 'temperature')) and method != 'middle-alignment':
        raise ValueError('Middle arguments require middle-alignment')
    config.pop('reference_cohort', None)
    config['purpose'] = 'User-defined recipe; execution requires explicit build/gate/run commands'
    config['status'] = 'PREPARED_NOT_LAUNCHED'
    spec.update(id=args.name, method=args.method, seed=args.seed, data_seed=args.seed,
                account=args.account)
    spec.pop('definition', None)
    for flag, field in {'lr': 'learning_rate', 'batch_size': 'microbatch',
                       'accumulation': 'accumulation', 'max_source_length': 'max_source_length',
                       'max_target_length': 'max_target_length', 'validation_beams': 'validation_beams',
                       'src_lang': 'src_lang', 'tgt_lang': 'tgt_lang',
                       'model_id': 'model_id', 'model_revision': 'model_revision',
                       'middle_layer_idx': 'middle_layer_idx', 'alignment_weight': 'alignment_weight',
                       'temperature': 'temperature', 'distance': 'distance',
                       'strength': 'strength', 'jepa_weight': 'jepa_weight',
                       'source_prefix': 'source_prefix', 'stack': 'stack'}.items():
        value = getattr(args, flag)
        if value is not None:
            spec[field] = None if flag in ('src_lang', 'tgt_lang') and value.lower() == 'none' else value
    if args.backbone and args.backbone != spec['backbone']:
        if any(value is None for value in (args.model_id, args.model_revision, args.src_lang,
                                          args.tgt_lang, args.max_source_length, args.max_target_length)):
            raise ValueError('Changing backbone requires model ID/revision, languages and lengths explicitly')
        spec['backbone'] = args.backbone
    if args.dataset and args.dataset != config['dataset']:
        if args.src_lang is None or args.tgt_lang is None or args.bleu_tokenizer is None:
            raise ValueError('Changing dataset requires explicit language codes and BLEU tokenizer')
        if not re.fullmatch(r'[a-z0-9_-]+', args.dataset):
            raise ValueError('Unsafe dataset name')
        import csv
        config['dataset'] = args.dataset
        config['data'] = {}
        for split in ('train', 'validation', 'test'):
            file = ROOT / 'data_processed' / args.dataset / f'{split}.csv'
            with file.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != ['source', 'target']: raise ValueError('Expected source,target CSV')
                rows = sum(1 for _ in reader)
            config['data'][split] = {'rows': rows, 'sha256': sha256(file)}
    for flag, field in {'epochs': 'epochs', 'patience': 'patience',
                       'workers': 'dataloader_num_workers', 'bleu_tokenizer': 'bleu_tokenizer'}.items():
        value = getattr(args, flag)
        if value is not None: config['protocol'][field] = value
    config['protocol']['test_barrier'] = ('All declared validation selections verified and locked before test'
                                        if args.allow_test else 'No test stage in this pilot')
    config['protocol']['selection'] = f"original-reference validation chrF++; beam {spec['validation_beams']}, FP32"
    storage_method = args.storage_method or args.method
    spec['remote_prefix'] = f"checkpoints/{config['dataset']}/{spec['backbone']}/{storage_method}/seed-{args.seed}"
    if args.configuration: spec['remote_prefix'] += '/' + args.configuration
    config['runs'] = [spec]
    # Validate before committing the new recipe.
    target.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    try: read_config(target)
    except Exception:
        target.unlink()
        raise
    return {'config': str(target.relative_to(ROOT)), 'resolved_run': spec,
            'protocol': config['protocol'], 'gpu_started': False}


def build(path):
    path, config = read_config(path)
    out = workspace(path)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Frozen workspace already exists; use a new configuration name')
    dataset = ROOT / 'data_processed' / config['dataset']
    for split, record in config['data'].items():
        if sha256(dataset / f'{split}.csv') != record['sha256']:
            raise ValueError(f'Dataset changed: {split}')
    frozen = copy.deepcopy(config)
    frozen['code_git_head'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    frozen['configuration_sha256'] = sha256(path)
    frozen['status'] = 'FROZEN_NOT_LAUNCHED'
    sources = list((ROOT / 'src').rglob('*.py'))
    sources += list((ROOT / 'tests').glob('*.py'))
    sources += [ROOT / 'scripts/experiment.py', ROOT / 'scripts/colab.py',
                ROOT / 'requirements/colab-experiments.txt', path]
    sources += [dataset / f'{s}.csv' for s in config['data']]
    if (dataset / 'manifest.json').exists():
        sources.append(dataset / 'manifest.json')
    # Credentials and scratch are deliberately excluded from source bundles.
    for file in sources:
        if re.search(rb'hf_[A-Za-z0-9]{20,}', file.read_bytes()):
            raise ValueError(f'Credential-like literal in bundle input: {file.relative_to(ROOT)}')
    out.mkdir(parents=True)
    plan_path = out / 'PLAN.json'
    plan_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding='utf-8')
    sources.append(plan_path)
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}
    manifest = out / 'bundle_manifest.json'
    manifest.write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    bundle = ROOT / '.cache/experiment_bundles' / f'{path.stem}.zip'
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(ROOT / name, name)
        archive.write(manifest, manifest.relative_to(ROOT).as_posix())
    return {'workspace': str(out), 'bundle': str(bundle), 'sha256': sha256(bundle),
            'runs': len(config['runs']), 'gpu_started': False}


def verify_frozen(path):
    path, _ = read_config(path)
    out = workspace(path)
    frozen = json.loads((out / 'PLAN.json').read_text(encoding='utf-8'))
    if frozen['configuration_sha256'] != sha256(path):
        raise ValueError('Configuration changed after freezing')
    for name, digest in json.loads((out / 'bundle_manifest.json').read_text()).items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    return out, frozen
