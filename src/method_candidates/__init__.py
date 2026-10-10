"""Compatibility imports; implementation lives in src/methods/."""
if __package__.startswith("src."):
    from src.methods.registry import wrap_candidate
else:
    from methods.registry import wrap_candidate
from .precache import PrecacheResidualRewire
from .centered_lsr import CenteredLSRSeq2SeqLM
