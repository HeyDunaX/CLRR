"""CoCC v1: token-aligned isolated-vs-concatenated distribution invariance."""
from .diagnostic import run_step0_diagnostic
from .model import (
    CompositionalConsistencyMT,
    METHODS,
    build_composite_batch_s1,
    chunked_aligned_kl_loss,
    chunked_ce_loss,
)
from .validation import validate_recipe

__all__ = [
    'CompositionalConsistencyMT',
    'METHODS',
    'build_composite_batch_s1',
    'chunked_aligned_kl_loss',
    'chunked_ce_loss',
    'run_step0_diagnostic',
    'validate_recipe',
]
