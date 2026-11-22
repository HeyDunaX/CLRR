"""Validate and freeze recipes without starting a GPU or changing old receipts."""
import copy
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

from .integrity import sha256

ROOT = Path(__file__).resolve().parents[2]
REPO = 'FiveC/amis-rewire-checkpoints'


def _sha256_matches(path: Path, expected_sha256: str) -> bool:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == expected_sha256:
        return True
    lf_to_crlf = hashlib.sha256(data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()
    crlf_to_lf = hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()
    return expected_sha256 in (lf_to_crlf, crlf_to_lf)


def read_config(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'configs/experiments'):
        raise ValueError('Put recipes under configs/experiments/')
    config = json.loads(path.read_text(encoding='utf-8'))
    if config.get('protocol_lock'):
        lock_path = (ROOT / config['protocol_lock']['path']).resolve()
        if not lock_path.is_relative_to(ROOT / 'configs/protocols'):
            raise ValueError('Protocol lock must live under configs/protocols/')
        if not _sha256_matches(lock_path, config['protocol_lock']['sha256']):
            raise ValueError('Locked baseline protocol changed')
        locked = json.loads(lock_path.read_text(encoding='utf-8'))
        for field in ('data', 'runtime'):
            if config[field] != locked[field]:
                raise ValueError(f'Baseline lock mismatch: {field}')
        for key, value in locked['protocol'].items():
            if config['protocol'].get(key) != value:
                raise ValueError(f'Baseline lock mismatch: protocol.{key}')
        for run in config['runs']:
            for key, value in locked['run'].items():
                if run.get(key) != value:
                    raise ValueError(f'Baseline lock mismatch: {run["id"]}.{key}')
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
    metric = config['protocol'].get('selection_metric', 'eval_chrf++')
    if metric not in ('eval_bleu', 'eval_chrf++'):
        raise ValueError('Unsupported validation checkpoint metric')
    if metric == 'eval_bleu' and 'validation BLEU' not in config['protocol']['selection']:
        raise ValueError('Selection description must declare validation BLEU')
    if config['protocol'].get('retain_secondary_chrf', False) and metric != 'eval_bleu':
        raise ValueError('Dual selection requires BLEU primary and chrF++ secondary')
    from methods.registry import TWO_METHODS, ALIASES, CANDIDATES, ORIGINAL_ALIASES, TOKEN_METHODS, CONTRASTIVE_METHODS, INTERACTION_METHODS, SEQUENCE_METHODS, DEPTH_METHODS
    from methods.decoder_residual_bridge.v1 import METHODS as DRB_METHODS
    from methods.difference_preserving.v1 import METHODS as DIFFERENCE_METHODS
    from methods.difference_preserving.v2.model import METHODS as RELATIONAL_METHODS
    from methods.transition_lsr import METHODS as TRANSITION_METHODS
    from methods.learned_strength import METHODS as STRENGTH_METHODS
    from methods.alignment_residual import METHODS as ALIGNMENT_RESIDUAL_METHODS
    supported = {'baseline', 'middle-alignment', 'bitfit', 'narrow-lora',
                 'strong-lora-a', 'strong-lora-b', *ORIGINAL_ALIASES,
                 *ORIGINAL_ALIASES.values(), *ALIASES, *CANDIDATES, *TOKEN_METHODS, *CONTRASTIVE_METHODS, *INTERACTION_METHODS, *SEQUENCE_METHODS, *DEPTH_METHODS, *ALIGNMENT_RESIDUAL_METHODS, *DRB_METHODS, *DIFFERENCE_METHODS, *RELATIONAL_METHODS}
    supported.update(TRANSITION_METHODS)
    supported.update(STRENGTH_METHODS)
    from methods.paired_route import METHODS as PAIRED_ROUTE_METHODS
    supported.update(PAIRED_ROUTE_METHODS)
    from methods.decoupled_consistency import METHODS as CONSISTENCY_METHODS
    supported.update(CONSISTENCY_METHODS)
    from methods.prefix_consistency import METHODS as PREFIX_METHODS
    supported.update(PREFIX_METHODS)
    from methods.continuation_consistency import METHODS as CMC_METHODS
    supported.update(CMC_METHODS)
    from methods.corrective_order import METHODS as ORDER_METHODS
    supported.update(ORDER_METHODS)
    from methods.virtual_corrective_order.v1.model import METHODS as VIRTUAL_ORDER_METHODS
    supported.update(VIRTUAL_ORDER_METHODS)
    supported.update(TWO_METHODS)
    from methods.gold_ordered_consistency import METHODS as GOC_METHODS
    supported.update(GOC_METHODS)
    from methods.cross_ratio_dual import METHODS as CR_DUAL_METHODS
    supported.update(CR_DUAL_METHODS)
    if config['protocol'].get('hardware_profile', 'a100-standard') == 'rtx6000-fp32':
        p = config['protocol']
        if p['bf16'] or p.get('fp16', False) or p['tf32_training']:
            raise ValueError('RTX FP32 pilot must disable BF16/FP16/TF32')
        if not p['test_barrier'].startswith('No test'):
            raise ValueError('RTX development pilot is validation-only')
    elif config['protocol'].get('hardware_profile', 'a100-standard') != 'a100-standard':
        raise ValueError('Unknown hardware profile')
    if not config['runs']:
        raise ValueError('No runs declared')
    ids, targets = set(), set()
    for run in config['runs']:
        if run['method'] not in supported:
            raise ValueError(f"Unimplemented method: {run['method']}")
        if run['method'] in VIRTUAL_ORDER_METHODS:
            from methods.virtual_corrective_order.v1.validation import validate_recipe
            validate_recipe(config, run, ROOT)
        if run['method'] in GOC_METHODS:
            from methods.gold_ordered_consistency.v1.validation import validate_recipe
            validate_recipe(config, run, ROOT)
        if run['method'] in CR_DUAL_METHODS:
            from methods.cross_ratio_dual.v1.validation import validate_recipe
            validate_recipe(config, run, ROOT)
        if run['method'] in ORDER_METHODS:
            from methods.corrective_order.v1.data import validate_recipe
            validate_recipe(config, run)
        if run['method'] in TWO_METHODS:
            expected = dict(backbone='mbart-large-50', consistency_beta=1.25, strength=0., jepa_weight=0.)
            if any(run.get(k) != v for k,v in expected.items()):
                raise ValueError('Two-method round differs from reviewed definition')
            if run['method'] == 'decision-residual-memory-v1' and run.get('memory_eta') != .1:
                raise ValueError('DRM eta must equal reviewed .1')
            review = config['design_review']
            review_path = (ROOT/review['path']).resolve()
            if not review_path.is_relative_to(ROOT/'docs/research') or sha256(review_path) != review['sha256']:
                raise ValueError('Two-method design changed')
            if not config.get('protocol_lock') or metric != 'eval_bleu' or not config.get('two_method_queue'):
                raise ValueError('Two-method pilot requires locked protocol and conditional queue')
            if config.get('post_training_test',{}).get('authorized'):
                raise ValueError('Two-method pilot is validation-only')
        if run['method'] in CMC_METHODS:
            expected = dict(backbone='mbart-large-50', continuation_top_k=4,
                probe_chunk=2, consistency_beta=1.25, strength=0., jepa_weight=0.)
            if any(run.get(k) != v for k, v in expected.items()):
                raise ValueError('CMC differs from reviewed sparse channel design')
            review = config.get('design_review', {})
            review_path = (ROOT / review.get('path', '')).resolve()
            if (not review_path.is_relative_to(ROOT / 'docs/research') or not review_path.is_file()
                    or sha256(review_path) != review.get('sha256')):
                raise ValueError('CMC requires pinned scientific design')
            if not config.get('protocol_lock') or metric != 'eval_bleu':
                raise ValueError('CMC requires locked baseline protocol and BLEU')
            if config.get('post_training_test', {}).get('authorized'):
                raise ValueError('CMC pilot is validation-only')
            if run['method'] in ('continuation-control-variate-v2','continuation-sampled-direct-v2'):
                admission=config.get('estimator_admission',{})
                receipt=(ROOT/admission.get('path','')).resolve()
                if (not receipt.is_relative_to(ROOT/'outputs/experiments') or not receipt.is_file()
                        or sha256(receipt)!=admission.get('sha256')):
                    raise ValueError('Estimator pilot requires pinned completed gradient audit')
                evidence=json.loads(receipt.read_text())
                if (not evidence['passed'] or evidence['optimizer_steps']!=0 or evidence['test_used']
                        or not evidence['parameters_unchanged'] or len(evidence['results'])!=8
                        or evidence['variance_ratio_pooled']>.8 or evidence['improved_conditions']<6):
                    raise ValueError('Estimator gradient audit did not pass the locked gate')
        if run['method'] in PREFIX_METHODS:
            import math
            for key in ('path_weight', 'consistency_beta'):
                if not math.isfinite(run[key]) or run[key] < 0:
                    raise ValueError('Invalid prefix consistency weight')
            if run['method'] == 'calibrated-consistency-v1' and run['path_weight'] != 0.:
                raise ValueError('Wcal must have zero path weight')
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
        if run['method'] in CONSISTENCY_METHODS:
            component, beta = CONSISTENCY_METHODS[run['method']]
            expected = dict(backbone='mbart-large-50', distance=2, strength=0.,
                consistency_component=component, consistency_weight=beta, jepa_weight=0.)
            if any(run.get(k) != v for k, v in expected.items()):
                raise ValueError('Consistency differs from reviewed G/R/W design')
            review = config.get('design_review', {})
            review_path = (ROOT / review.get('path', '')).resolve()
            if (review.get('decision') != 'ENGINEERING_PILOT_AUTHORIZED_NOT_NOVELTY_CLAIM'
                    or not review_path.is_relative_to(ROOT / 'docs/research')
                    or not review_path.is_file() or sha256(review_path) != review.get('sha256')):
                raise ValueError('Consistency requires pinned design/prior review')
            if not config.get('protocol_lock') or metric != 'eval_bleu':
                raise ValueError('Consistency requires locked baseline protocol and BLEU')
            if config.get('post_training_test', {}).get('authorized'):
                raise ValueError('Consistency pilot is validation-only')
        if run['method'] in PAIRED_ROUTE_METHODS:
            rewired, beta = PAIRED_ROUTE_METHODS[run['method']]
            expected = dict(backbone='mbart-large-50', distance=2,
                strength=.1 if rewired else 0., consistency_weight=beta, jepa_weight=0.)
            if any(run.get(k) != v for k, v in expected.items()):
                raise ValueError('Paired route differs from reviewed four-arm design')
            review = config.get('design_review', {})
            review_path = (ROOT / review.get('path', '')).resolve()
            if (review.get('decision') != 'ENGINEERING_PILOT_AUTHORIZED_NOT_NOVELTY_CLAIM'
                    or not review_path.is_relative_to(ROOT / 'docs/research')
                    or not review_path.is_file() or sha256(review_path) != review.get('sha256')):
                raise ValueError('Paired route requires pinned engineering/prior review')
            if not config.get('protocol_lock') or metric != 'eval_bleu':
                raise ValueError('Paired route requires locked baseline protocol and BLEU')
            if config.get('post_training_test', {}).get('authorized'):
                raise ValueError('Paired route pilot is validation-only')
        if run['method'] in STRENGTH_METHODS:
            expected = dict(backbone='mbart-large-50', distance=2, strength=0.,
                jepa_weight=.1 if STRENGTH_METHODS[run['method']] else 0.)
            if any(run.get(k) != v for k, v in expected.items()):
                raise ValueError('Learned strength differs from approved two-arm design')
            review = config.get('design_review', {})
            review_path = (ROOT / review.get('path', '')).resolve()
            if (review.get('decision') != 'ENGINEERING_PILOT_AUTHORIZED_NOT_NOVELTY_CLAIM'
                    or not review_path.is_relative_to(ROOT / 'docs/research')
                    or not review_path.is_file() or sha256(review_path) != review.get('sha256')):
                raise ValueError('Learned strength requires pinned engineering/prior review')
            if not config.get('protocol_lock') or metric != 'eval_bleu':
                raise ValueError('Learned strength requires pinned baseline protocol/BLEU')
            if config.get('post_training_test', {}).get('authorized'):
                raise ValueError('Learned strength pilot is validation-only')
        if run['method'] in TRANSITION_METHODS:
            expected = dict(backbone='mbart-large-50', alignment_weight=.1,
                            distance=2, strength=.1, jepa_weight=0.)
            if any(run.get(k) != v for k, v in expected.items()):
                raise ValueError('Transition LSR differs from its reviewed design')
            review = config.get('design_review', {})
            review_path = (ROOT / review.get('path', '')).resolve()
            if (review.get('decision') != 'DESIGN_REVIEW_PASS_FOR_BOUNDED_PILOT'
                    or not review_path.is_relative_to(ROOT / 'docs/research')
                    or not review_path.is_file() or sha256(review_path) != review.get('sha256')):
                raise ValueError('Transition LSR requires its pinned pre-training novelty/design review')
            if not config.get('protocol_lock') or metric != 'eval_bleu':
                raise ValueError('Transition LSR requires pinned baseline protocol and BLEU')
            if config.get('post_training_test', {}).get('authorized'):
                raise ValueError('Transition LSR pilot is validation-only')
        if run['method'] in RELATIONAL_METHODS:
            expected = dict(backbone='mbart-large-50', alignment_weight=.1,
                            distance=2, strength=.1, jepa_weight=0.)
            if any(run.get(k) != v for k, v in expected.items()):
                raise ValueError('Relational v2 recipe differs from its locked design')
            if not config.get('protocol_lock') or metric != 'eval_bleu':
                raise ValueError('Relational v2 requires pinned protocol and BLEU selector')
            if config.get('scientific_admission') != dict(purpose='EXPLORATORY_MECHANISM_SCREEN',
                    novelty='UNESTABLISHED', semantic_minimal_pairs_claimed=False):
                raise ValueError('Relational v2 must declare exploratory scientific scope')
            if config.get('exploratory_screen_authorized') is not True or 'pairing' in config:
                raise ValueError('Authorize the all-training screen explicitly; no semantic pair manifest')
            if config.get('post_training_test', {}).get('authorized'):
                raise ValueError('Relational screen is validation-only')
        if run['method'] in DIFFERENCE_METHODS:
            if (run['backbone'] != 'mbart-large-50' or run.get('alignment_weight') != .1
                    or run.get('distance') != 2 or run.get('strength') != .1
                    or run.get('jepa_weight') != 0.):
                raise ValueError('Difference pilot locks mBART, weight=.1, CLRR d2/a.1, no old LSR')
            if config['protocol'].get('attention_implementation') != 'sdpa' or metric != 'eval_bleu':
                raise ValueError('Difference pilot requires SDPA and validation BLEU')
            pairing = config.get('pairing', {})
            relative = Path(pairing.get('manifest', ''))
            if (relative.is_absolute() or '..' in relative.parts
                    or not relative.as_posix().startswith('outputs/experiments/')
                    or not re.fullmatch('[a-f0-9]{64}', pairing.get('sha256', ''))):
                raise ValueError('Declare a pinned workspace-local reviewed train-pair manifest')
            if config.get('post_training_test', {}).get('authorized'):
                raise ValueError('Difference pilot is validation-only')
        if run['method'] in ALIGNMENT_RESIDUAL_METHODS:
            expected = {'adapter_rank': 64, 'module_seed': run['seed'] + 10000,
                'distance': 2, 'strength': .1,
                'jepa_weight': 0. if 'capacity' in ALIGNMENT_RESIDUAL_METHODS[run['method']] else .1}
            if run['backbone'] != 'mbart-large-50' or any(run.get(k) != v for k, v in expected.items()):
                raise ValueError('Alignment residual v1 must match its frozen four-arm design')
            if config['protocol'].get('attention_implementation') != 'sdpa':
                raise ValueError('Alignment residual development uses the locked SDPA recipe')
            if config.get('post_training_test', {}).get('authorized'):
                raise ValueError('Alignment residual pilot is validation-only')
        if run['method'] in DRB_METHODS:
            expected = {'adapter_rank':64, 'module_seed':run['seed']+10000,
                        'donor_layer':6, 'decoder_layer':12, 'jepa_weight':0.}
            if run['backbone'] != 'mbart-large-50' or any(run.get(k) != v for k,v in expected.items()):
                raise ValueError('DRB pilot differs from locked design')
            if config['protocol'].get('attention_implementation') != 'sdpa' or metric != 'eval_bleu':
                raise ValueError('DRB locks SDPA and validation BLEU selection')
            if config.get('post_training_test', {}).get('authorized'):
                raise ValueError('DRB development is validation-only')
        if run['method'] in DEPTH_METHODS:
            if run['backbone'] != 'mbart-large-50' or run.get('donor_layer') != 6 or run.get('decoder_layer') != 12:
                raise ValueError('Depth-v2 pilot locks mBART donor6/decoder12')
            if config['protocol'].get('attention_implementation') not in ('eager', 'sdpa') or run.get('jepa_weight',0) != 0:
                raise ValueError('Depth-v2 requires an explicit eager/sdpa backend and CE only')
        if run['method'] in SEQUENCE_METHODS:
            import math
            if run['backbone'] != 'mbart-large-50' or run.get('jepa_weight', 0) != 0:
                raise ValueError('Sequence v1 is mBART only and has no LSR')
            if config['protocol'].get('view_checkpointing') != 'whole-view-non-reentrant':
                raise ValueError('Sequence v1 requires whole-view-non-reentrant checkpointing')
            for field in ('risk_weight', 'risk_alpha'):
                if not math.isfinite(run.get(field, -1)):
                    raise ValueError('Declare finite risk weight and alpha')
            is_risk = run['method'] in ('sequence-risk-v1', 'sequence-clrr-risk-v1')
            if run['risk_alpha'] <= 0 or (run['risk_weight'] <= 0 if is_risk else run['risk_weight'] != 0):
                raise ValueError('Risk/CE method ID must agree with its weight')
            if not isinstance(run.get('score_chunk'), int) or run['score_chunk'] < 1:
                raise ValueError('Declare score_chunk >=1')
            initial = config.get('initialization', {})
            if initial.get('repo') != REPO or not re.fullmatch(r'[a-f0-9]{40}', initial.get('revision', '')):
                raise ValueError('Pin the common initialization repository/revision')
            if not re.fullmatch(r'[a-f0-9]{64}', initial.get('archive_sha256', '')):
                raise ValueError('Pin the common initialization SHA256')
            if not initial.get('archive_path', '').startswith(f'checkpoints/{config["dataset"]}/{run["backbone"]}/baseline/'):
                raise ValueError('Continuation must initialize from the declared dataset/backbone baseline')
            pool = config.get('candidate_pool', {})
            if pool.get('split') != 'train' or pool.get('samples', 0) < 1 or not pool.get('include_gold'):
                raise ValueError('Declare a train-only pool with gold and sampled candidates')
            if not math.isfinite(pool.get('temperature', 0)) or pool.get('temperature', 0) <= 0:
                raise ValueError('Declare positive finite proposal temperature')
            if not isinstance(pool.get('source_batch'), int) or pool['source_batch'] < 1:
                raise ValueError('Declare positive source_batch for reproducible sampling')
            for field in ('max_no_eos_fraction', 'max_length_cap_fraction'):
                if field in pool and (not math.isfinite(pool[field]) or not 0 < pool[field] <= 1):
                    raise ValueError(f'Invalid pool threshold: {field}')
        if run['method'] in TOKEN_METHODS:
            if run['backbone'] != 'mbart-large-50':
                raise ValueError('Token v1 is verified for mBART only')
            for field, expected in {'distance': 2, 'strength': .10, 'jepa_weight': .10}.items():
                if run.get(field, expected) != expected:
                    raise ValueError(f'Token v1 does not override {field}')
        if run['method'] in INTERACTION_METHODS:
            import math
            if run['backbone'] != 'mbart-large-50' or run.get('stack', 'encoder') != 'encoder':
                raise ValueError('Interaction v1 is verified for mBART encoder only')
            if run.get('jepa_weight', 0) != 0:
                raise ValueError('Interaction v1 has no LSR term; set jepa_weight=0')
            if run.get('distance', 2) != 2 or run.get('strength', .1) != .1:
                raise ValueError('Interaction v1 recipe locks distance=2 and strength=.1')
            if run.get('distance', 2) != 2 or run.get('strength', .1) != .1:
                raise ValueError('Interaction v1 recipe locks distance=2 and strength=.1')
            if not math.isfinite(run.get('interaction_weight', -1)) or run.get('interaction_weight', -1) < 0:
                raise ValueError('Declare finite interaction_weight >=0, including paired CE control')
            if not math.isfinite(run.get('temperature', 0)) or run.get('temperature', 0) <= 0:
                raise ValueError('Declare positive finite interaction temperature')
            if run['microbatch'] < 2:
                raise ValueError('Interaction negatives require microbatch >=2')
            if config['protocol'].get('view_checkpointing') != 'whole-view-non-reentrant':
                raise ValueError('Declare whole-view-non-reentrant view_checkpointing explicitly')
        if run['method'] in CONTRASTIVE_METHODS:
            if run['backbone'] != 'mbart-large-50' or run.get('stack', 'encoder') != 'encoder':
                raise ValueError('Effect v1 supports mBART encoder only')
            if run.get('alignment_weight', 0) <= 0 or run.get('temperature', 0) <= 0 or run['microbatch'] < 2:
                raise ValueError('Contrastive alignment needs weight, temperature and microbatch >=2')
            for field, expected in {'distance': 2, 'strength': .10,
                    'jepa_weight': 0.0 if run['method'] in ('effect-clrr-v1', 'nce-only-v1', 'clrr-nce-v1') else .10}.items():
                if run.get(field) != expected:
                    raise ValueError(f'Effect v1 requires {field}={expected}')
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
    from methods.registry import ALIASES, ORIGINAL_ALIASES, CONTRASTIVE_METHODS, INTERACTION_METHODS
    method = ALIASES.get(args.method, ORIGINAL_ALIASES.get(args.method, args.method))
    from methods.alignment_residual import METHODS as ALIGNMENT_RESIDUAL_METHODS
    from methods.decoder_residual_bridge.v1 import METHODS as DRB_METHODS
    clrr_methods = {'original-clrr', 'original-full', 'precache-clrr-v1', 'precache-full-v1', 'centered-full-v1'}
    from methods.transition_lsr import METHODS as TRANSITION_METHODS
    clrr_methods.update(TRANSITION_METHODS)
    lsr_methods = {'original-lsr', 'original-full', 'precache-full-v1', 'centered-lsr-v1', 'centered-full-v1'}
    clrr_methods |= CONTRASTIVE_METHODS - {'ordinary-lsr-v1', 'nce-only-v1'}
    clrr_methods |= INTERACTION_METHODS
    clrr_methods |= ALIGNMENT_RESIDUAL_METHODS.keys()
    lsr_methods |= CONTRASTIVE_METHODS
    lsr_methods |= ALIGNMENT_RESIDUAL_METHODS.keys()
    if any(getattr(args, flag, None) is not None for flag in ('adapter_rank', 'module_seed')) and method not in {*ALIGNMENT_RESIDUAL_METHODS, *DRB_METHODS}:
        raise ValueError('Adapter arguments require an alignment residual or DRB method')
    if method in DRB_METHODS:
        spec.update(adapter_rank=64, module_seed=args.seed + 10000, donor_layer=6, decoder_layer=12, jepa_weight=0.)
    if any(getattr(args, k, None) is not None for k in ('donor_layer', 'decoder_layer')) and method not in DRB_METHODS:
        raise ValueError('Bridge layer arguments require a DRB method')
    if method in ALIGNMENT_RESIDUAL_METHODS:
        spec.update(adapter_rank=64, module_seed=args.seed + 10000, distance=2, strength=.1,
                    jepa_weight=0. if 'capacity' in ALIGNMENT_RESIDUAL_METHODS[method] else .1)
    if any(getattr(args, flag) is not None for flag in ('distance', 'strength', 'stack')) and method not in clrr_methods:
        raise ValueError('Rewiring arguments require a CLRR method')
    if args.jepa_weight is not None and method not in lsr_methods:
        raise ValueError('LSR weight requires an LSR method')
    if args.middle_layer_idx is not None and method != 'middle-alignment':
        raise ValueError('Middle layer index requires middle-alignment')
    if args.alignment_weight is not None and method not in {'middle-alignment', *CONTRASTIVE_METHODS, *TRANSITION_METHODS}:
        raise ValueError('Alignment weight/temperature require a contrastive method')
    if args.temperature is not None and method not in {'middle-alignment', *CONTRASTIVE_METHODS, *INTERACTION_METHODS}:
        raise ValueError('Temperature requires a contrastive/interaction method')
    if getattr(args, 'interaction_weight', None) is not None and method not in INTERACTION_METHODS:
        raise ValueError('Interaction weight requires an interaction method')
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
                       'temperature': 'temperature', 'interaction_weight': 'interaction_weight', 'distance': 'distance',
                       'strength': 'strength', 'jepa_weight': 'jepa_weight',
                       'source_prefix': 'source_prefix', 'stack': 'stack',
                       'adapter_rank': 'adapter_rank', 'module_seed': 'module_seed',
                       'donor_layer': 'donor_layer', 'decoder_layer': 'decoder_layer'}.items():
        value = getattr(args, flag, None)
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
    if method in ALIGNMENT_RESIDUAL_METHODS and config.get('protocol_lock'):
        config['protocol']['selection'] = original['protocol']['selection']
    if method in DRB_METHODS:
        config['protocol'].update(selection_metric='eval_bleu', selection=f"original-reference validation BLEU; beam {spec['validation_beams']}, FP32")
    if method in TRANSITION_METHODS:
        config['protocol']['selection'] = original['protocol']['selection']
    if method in INTERACTION_METHODS:
        config['protocol']['view_checkpointing'] = 'whole-view-non-reentrant'
        spec['jepa_weight'] = 0.
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
        ['git', '-c', f'safe.directory={ROOT.as_posix()}', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    frozen['configuration_sha256'] = sha256(path)
    frozen['status'] = 'FROZEN_NOT_LAUNCHED'
    sources = list((ROOT / 'src').rglob('*.py'))
    sources += list((ROOT / 'tests').glob('*.py'))
    sources += [ROOT / 'scripts/experiment.py', ROOT / 'scripts/colab.py',
                ROOT / 'requirements/colab-experiments.txt', path]
    if config.get('protocol_lock'):
        sources.append(ROOT / config['protocol_lock']['path'])
    if config.get('design_review'):
        sources.append(ROOT / config['design_review']['path'])
    if config.get('pairing'):
        pairs = ROOT / config['pairing']['manifest']
        if sha256(pairs) != config['pairing']['sha256']:
            raise ValueError('Train-pair manifest changed')
        sources.append(pairs)
        sources.append(ROOT / 'scripts/analysis/audit_difference_pairs.py')
    if config.get('post_training_test', {}).get('authorized'):
        sources += [ROOT / 'scripts/analysis/evaluate_frozen_rtx.py',
                    ROOT / 'scripts/analysis/paired_checkpoint_statistics.py']
    if config.get('pre_training_alignment_probe'):
        sources.append(ROOT / 'scripts/analysis/measure_alignment_residual_checkpoint.py')
    from methods.decoupled_consistency import METHODS as CONSISTENCY_METHODS
    # The consistency correctness gate imports the probe statistics helper too.
    if config.get('pre_training_consistency_probe') or any(r['method'] in CONSISTENCY_METHODS for r in config['runs']):
        sources.append(ROOT / 'scripts/analysis/measure_consistency_checkpoint.py')
    from methods.prefix_consistency import METHODS as PREFIX_METHODS
    if config.get('prefix_preparation') or any(r['method'] in PREFIX_METHODS for r in config['runs']):
        sources.append(ROOT / 'scripts/analysis/prepare_prefix_consistency.py')
    if config.get('prefix_admission'):
        for record in config['prefix_admission'].values():
            receipt = ROOT / record['path']
            if sha256(receipt) != record['sha256']:
                raise ValueError('Prefix scientific receipt changed')
            sources.append(receipt)
    if config.get('two_method_queue'):
        sources.append(ROOT/'scripts/analysis/run_two_method_round1.py')
    if config.get('virtual_order_calibration'):
        record = config['virtual_order_calibration']
        sources.append(ROOT / record['path'])
    if config.get('goc_zero_step_admission'):
        record = config['goc_zero_step_admission']
        sources.append(ROOT / record['path'])
    if config.get('estimator_admission'):
        receipt=ROOT/config['estimator_admission']['path']
        if sha256(receipt)!=config['estimator_admission']['sha256']:
            raise ValueError('Estimator scientific receipt changed')
        sources.append(receipt)
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
