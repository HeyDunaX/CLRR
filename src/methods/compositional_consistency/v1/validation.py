"""Recipe validation for Compositional Context-Invariance Consistency (CoCC) pilots."""
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
        raise ValueError(f'Unknown CoCC method: {method}')
    if run.get('backbone') not in ('mbart-large-50', 'nllb-200-distilled-600m'):
        raise ValueError('CoCC v1 requires a multilingual seq2seq backbone (mBART-50 or NLLB-200)')
    if not run.get('src_lang') or not run.get('tgt_lang'):
        raise ValueError('CoCC requires explicit src_lang and tgt_lang codes')
    if run.get('strength', 0.0) != 0.0 or run.get('jepa_weight', 0.0) != 0.0:
        raise ValueError('CoCC pilot locks native backbone with zero CLRR/LSR weights')
    if run.get('layout', 's1') != 's1':
        raise ValueError('CoCC Stage-1 pilot locks native Layout S1 concatenation')
    if not run.get('mask_y_j1', True):
        raise ValueError('CoCC Stage-1 pilot locks boundary mask_y_j1=True')

    beta = run.get('consistency_beta')
    if beta is None or not math.isfinite(beta) or beta < 0.0:
        raise ValueError('consistency_beta must be finite and nonnegative')
    if method == 'concat-ce-v1' and beta != 0.0:
        raise ValueError('concat-ce-v1 requires consistency_beta == 0.0')
    if method != 'concat-ce-v1' and beta <= 0.0:
        raise ValueError(f'{method} requires consistency_beta > 0.0')
    if int(run.get('microbatch', 0)) < 2:
        raise ValueError(f'{method} requires microbatch >= 2')

    p = config['protocol']
    if (p.get('attention_implementation') != 'sdpa'
            or p.get('selection_metric') != 'eval_bleu'
            or p.get('patience') != 10
            or not p.get('gradient_checkpointing_use_reentrant')):
        raise ValueError('CoCC pilot requires locked SDPA, ES10, reentrant checkpointing, and eval_bleu')
    if not config.get('protocol_lock'):
        raise ValueError('CoCC pilot requires a locked baseline protocol')
    if config.get('post_training_test', {}).get('authorized'):
        raise ValueError('CoCC pilot is validation-only')

    review = config.get('design_review', {})
    review_path = (root / review.get('path', '')).resolve()
    if not review_path.is_relative_to(root / 'docs/research') or not review_path.is_file():
        raise ValueError('CoCC pilot requires pinned design review under docs/research/')
    _verify_file_hash(review_path, review.get('sha256', ''))
