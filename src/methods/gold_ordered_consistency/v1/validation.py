"""Admission and recipe validation for Gold-Ordered Consistency (GOC) pilots."""
import hashlib
import json
from pathlib import Path

from .model import METHODS


ALLOWED_BETAS = (1.25, 2.5)


def _verify_file_hash(path: Path, expected_sha256: str) -> None:
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual == expected_sha256:
        return
    # Allow Git CRLF/LF checkout normalization only if exact content matches.
    lf_to_crlf = hashlib.sha256(data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()
    crlf_to_lf = hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()
    if expected_sha256 not in (lf_to_crlf, crlf_to_lf):
        raise ValueError(f'Pinned file SHA256 mismatch: {path}')


def validate_recipe(config, run, root):
    root = Path(root).resolve()
    if run['method'] not in METHODS:
        raise ValueError(f"Unknown GOC method: {run['method']}")
    expected = dict(backbone='mbart-large-50', distance=2, strength=0., jepa_weight=0.)
    if any(run.get(k) != v for k, v in expected.items()):
        raise ValueError('GOC pilot locks native mBART-50 with zero CLRR/LSR weights')
    if run.get('consistency_beta') not in ALLOWED_BETAS:
        raise ValueError(f'GOC pilot locks consistency_beta in {ALLOWED_BETAS}')
    p = config['protocol']
    if (p.get('attention_implementation') != 'sdpa'
            or p.get('selection_metric') != 'eval_bleu'
            or p.get('patience') != 10
            or not p.get('gradient_checkpointing_use_reentrant')):
        raise ValueError('GOC pilot requires locked SDPA, ES10, reentrant checkpointing, and eval_bleu')
    if not config.get('protocol_lock'):
        raise ValueError('GOC pilot requires a locked baseline protocol')
    if config.get('post_training_test', {}).get('authorized'):
        raise ValueError('GOC pilot is validation-only')
    review = config.get('design_review', {})
    review_path = (root / review.get('path', '')).resolve()
    if (not review_path.is_relative_to(root / 'docs/research')
            or not review_path.is_file()):
        raise ValueError('GOC pilot requires pinned design review under docs/research/')
    _verify_file_hash(review_path, review.get('sha256', ''))
    admission = config.get('goc_zero_step_admission', {})
    receipt_path = (root / admission.get('path', '')).resolve()
    if (not receipt_path.is_relative_to(root / 'docs/tasks/reports')
            or not receipt_path.is_file()):
        raise ValueError('GOC pilot requires pinned completed zero-step diagnostic receipt')
    _verify_file_hash(receipt_path, admission.get('sha256', ''))
    evidence = json.loads(receipt_path.read_text(encoding='utf-8'))
    conditions = evidence.get('decision', {}).get('conditions', {})
    if (evidence.get('status') != 'COMPLETE'
            or evidence.get('optimizer_steps') != 0
            or evidence.get('generation_calls') != 0
            or evidence.get('test_split_opened') is not False
            or not evidence.get('trained', {}).get('unchanged')
            or not evidence.get('initial', {}).get('unchanged')
            or evidence.get('decision', {}).get('failed_conditions') != []
            or len(conditions) != 6
            or not all(conditions.values())):
        raise ValueError('GOC zero-step diagnostic receipt did not pass all six locked quantitative screens')


def require_admission(plan, root):
    root = Path(root).resolve()
    for run in plan['runs']:
        if run['method'] in METHODS:
            validate_recipe(plan, run, root)
