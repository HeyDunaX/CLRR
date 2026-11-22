"""Recipe validation for Marginal-Free Cross-Ratio Dual Consistency (CR-Dual) pilots."""
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
        raise ValueError(f'Unknown CR-Dual method: {method}')
    if run.get('backbone') not in ('mbart-large-50', 'nllb-200-distilled-600m'):
        raise ValueError('CR-Dual v1 requires a many-to-many pretrained backbone (mBART-50 or NLLB-200)')
    if not run.get('src_lang') or not run.get('tgt_lang'):
        raise ValueError('CR-Dual requires explicit src_lang and tgt_lang codes')
    if run.get('strength', 0.0) != 0.0 or run.get('jepa_weight', 0.0) != 0.0:
        raise ValueError('CR-Dual pilot locks native backbone with zero CLRR/LSR weights')

    alpha_rev = run.get('alpha_rev', 0.5)
    cr_lambda = run.get('cr_lambda', 0.25)
    cr_delta = run.get('cr_delta', 1.0)
    cr_warmup = run.get('cr_warmup_fraction', 0.2)
    contrast_margin = run.get('contrast_margin', 2.0)

    if not math.isfinite(alpha_rev) or alpha_rev < 0.0:
        raise ValueError('alpha_rev must be finite and nonnegative')
    if not math.isfinite(cr_lambda) or cr_lambda < 0.0:
        raise ValueError('cr_lambda must be finite and nonnegative')
    if not math.isfinite(cr_delta) or cr_delta <= 0.0:
        raise ValueError('cr_delta must be finite and positive')
    if not math.isfinite(cr_warmup) or not (0.0 <= cr_warmup <= 1.0):
        raise ValueError('cr_warmup_fraction must be in [0, 1]')
    if not math.isfinite(contrast_margin) or contrast_margin <= 0.0:
        raise ValueError('contrast_margin must be finite and positive')

    if method == 'bidirectional-ce-v1':
        if alpha_rev <= 0.0 or cr_lambda != 0.0:
            raise ValueError('bidirectional-ce-v1 requires alpha_rev > 0 and cr_lambda == 0')
    elif method in ('cross-ratio-dual-v1', 'cross-ratio-shuffled-v1'):
        if alpha_rev <= 0.0 or cr_lambda <= 0.0:
            raise ValueError(f'{method} requires alpha_rev > 0 and cr_lambda > 0')
        min_bs = 4 if method == 'cross-ratio-shuffled-v1' else 2
        if int(run.get('microbatch', 0)) < min_bs:
            raise ValueError(f'{method} requires microbatch >= {min_bs}')
    elif method == 'unidirectional-contrast-v1':
        if alpha_rev != 0.0 or cr_lambda <= 0.0:
            raise ValueError('unidirectional-contrast-v1 requires alpha_rev == 0 and cr_lambda > 0')
        if int(run.get('microbatch', 0)) < 2:
            raise ValueError('unidirectional-contrast-v1 requires microbatch >= 2')

    p = config['protocol']
    if (p.get('attention_implementation') != 'sdpa'
            or p.get('selection_metric') != 'eval_bleu'
            or p.get('patience') != 10
            or not p.get('gradient_checkpointing_use_reentrant')):
        raise ValueError('CR-Dual pilot requires locked SDPA, ES10, reentrant checkpointing, and eval_bleu')
    if not config.get('protocol_lock'):
        raise ValueError('CR-Dual pilot requires a locked baseline protocol')
    if config.get('post_training_test', {}).get('authorized'):
        raise ValueError('CR-Dual pilot is validation-only')

    review = config.get('design_review', {})
    review_path = (root / review.get('path', '')).resolve()
    if (not review_path.is_relative_to(root / 'docs/research')
            or not review_path.is_file()):
        raise ValueError('CR-Dual pilot requires pinned design review under docs/research/')
    _verify_file_hash(review_path, review.get('sha256', ''))
