"""Off-EOS stopping v1: ChrF utility-band stopping and matched length control."""
from .diagnostic import run_step0_diagnostic
from .model import (
    METHODS,
    OffEosStoppingMT,
    detect_repetition_horizon,
    extract_eos_margins,
    huber_one_sided,
    off_eos_stop_loss,
    select_chrf_stop_set,
)
from .validation import validate_recipe

__all__ = [
    'METHODS',
    'OffEosStoppingMT',
    'detect_repetition_horizon',
    'extract_eos_margins',
    'huber_one_sided',
    'off_eos_stop_loss',
    'run_step0_diagnostic',
    'select_chrf_stop_set',
    'validate_recipe',
]
