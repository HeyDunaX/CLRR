"""Recipe validation for 2x2 Cross-Tail Lexical Fork Preference pilots."""
import hashlib
import math
from pathlib import Path

from .model import METHODS


def _verify_file_hash(path: Path, expected_sha256: str) -> None:
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual == expected_sha256:
        return
    lf_to_crlf = hashlib.sha256(data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()
    crlf_to_lf = hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()
    if expected_sha256 not in (lf_to_crlf, crlf_to_lf):
        raise ValueError(f'Pinned file SHA256 mismatch: {path}')


def validate_recipe(config, run, root):
    root = Path(root).resolve()
    method = run['method']
    if method not in METHODS:
        raise ValueError(f'Unknown lexical_fork method: {method}')
    if run.get('backbone') not in ('mbart-large-50', 'nllb-200-distilled-600m'):
        raise ValueError('lexical_fork v1 requires mBART-50 or NLLB-200')
    if not run.get('src_lang') or not run.get('tgt_lang'):
        raise ValueError('lexical_fork requires explicit src_lang and tgt_lang codes')
    if run.get('strength', 0.0) != 0.0 or run.get('jepa_weight', 0.0) != 0.0:
        raise ValueError('lexical_fork pilot locks native backbone with zero CLRR/LSR weights')

    lam2 = run.get('lambda_epoch2', 0.05)
    lam3 = run.get('lambda_epoch3_plus', 0.10)
    min_pos = run.get('min_pos_2x2', 32)
    num_strata = run.get('num_margin_strata', 5)
    for name, val in (('lambda_epoch2', lam2), ('lambda_epoch3_plus', lam3)):
        if val is None or not math.isfinite(val) or val < 0.0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if lam2 != 0.05 or lam3 != 0.10:
        raise ValueError('Locked exploratory pilot fixes lambda_epoch2=0.05 and lambda_epoch3_plus=0.10')
    if int(min_pos) != 32 or int(num_strata) != 5:
        raise ValueError('Locked exploratory pilot fixes min_pos_2x2=32 and num_margin_strata=5')
    if int(run.get('microbatch', 0)) < 1:
        raise ValueError(f'{method} requires microbatch >= 1')

    p = config['protocol']
    if (p.get('attention_implementation') != 'sdpa'
            or p.get('selection_metric') != 'eval_bleu'
            or p.get('patience') != 10
            or not p.get('gradient_checkpointing_use_reentrant')):
        raise ValueError('lexical_fork pilot requires locked SDPA, ES10, reentrant checkpointing, and eval_bleu')
    if not config.get('protocol_lock'):
        raise ValueError('lexical_fork pilot requires a locked baseline protocol')
    if config.get('post_training_test', {}).get('authorized'):
        raise ValueError('lexical_fork pilot is validation-only')

    review = config.get('design_review', {})
    review_path = (root / review.get('path', '')).resolve()
    if not review_path.is_relative_to(root / 'docs/research') or not review_path.is_file():
        raise ValueError('lexical_fork pilot requires pinned design review under docs/research/')
    _verify_file_hash(review_path, review.get('sha256', ''))
