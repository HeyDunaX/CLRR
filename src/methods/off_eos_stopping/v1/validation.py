"""Recipe validation for Off-EOS Trajectory Huber Margin pilots."""
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
        raise ValueError(f'Unknown off-eos stopping method: {method}')
    if run.get('backbone') not in ('mbart-large-50', 'nllb-200-distilled-600m'):
        raise ValueError('off_eos_stopping v1 requires a multilingual seq2seq backbone (mBART-50 or NLLB-200)')
    if not run.get('src_lang') or not run.get('tgt_lang'):
        raise ValueError('off_eos_stopping requires explicit src_lang and tgt_lang codes')
    if run.get('strength', 0.0) != 0.0 or run.get('jepa_weight', 0.0) != 0.0:
        raise ValueError('off_eos_stopping pilot locks native backbone with zero CLRR/LSR weights')

    stop_weight = run.get('stop_weight', 0.10)
    stop_margin = run.get('stop_margin', 0.1)
    stop_tau = run.get('stop_tau', 0.25)
    chrf_band = run.get('chrf_band', 0.5)
    stop_warmup_fraction = run.get('stop_warmup_fraction', 0.10)
    extra_past_eos = run.get('extra_past_eos', 12)
    max_rollout_cap = run.get('max_rollout_cap', 96)
    max_len_ratio = run.get('max_len_ratio', 1.5)

    for name, val in (
        ('stop_weight', stop_weight),
        ('stop_margin', stop_margin),
        ('stop_tau', stop_tau),
    ):
        if val is None or not math.isfinite(val) or val <= 0.0:
            raise ValueError(f'{name} must be finite and strictly positive')
    if chrf_band is None or not math.isfinite(chrf_band) or chrf_band < 0.0:
        raise ValueError('chrf_band must be finite and nonnegative')
    if stop_warmup_fraction is None or not math.isfinite(stop_warmup_fraction) or not (0.0 <= stop_warmup_fraction <= 0.5):
        raise ValueError('stop_warmup_fraction must be in [0.0, 0.5]')
    if int(extra_past_eos) < 1 or int(max_rollout_cap) < 8:
        raise ValueError('extra_past_eos must be >= 1 and max_rollout_cap >= 8')
    if max_len_ratio is None or not math.isfinite(max_len_ratio) or max_len_ratio < 1.0:
        raise ValueError('max_len_ratio must be finite and >= 1.0')
    if int(run.get('microbatch', 0)) < 1:
        raise ValueError(f'{method} requires microbatch >= 1')

    p = config['protocol']
    if (p.get('attention_implementation') != 'sdpa'
            or p.get('selection_metric') != 'eval_bleu'
            or p.get('patience') != 10
            or not p.get('gradient_checkpointing_use_reentrant')):
        raise ValueError('off_eos_stopping pilot requires locked SDPA, ES10, reentrant checkpointing, and eval_bleu')
    if not config.get('protocol_lock'):
        raise ValueError('off_eos_stopping pilot requires a locked baseline protocol')
    if config.get('post_training_test', {}).get('authorized'):
        raise ValueError('off_eos_stopping pilot is validation-only')

    review = config.get('design_review', {})
    review_path = (root / review.get('path', '')).resolve()
    if not review_path.is_relative_to(root / 'docs/research') or not review_path.is_file():
        raise ValueError('off_eos_stopping pilot requires pinned design review under docs/research/')
    _verify_file_hash(review_path, review.get('sha256', ''))
