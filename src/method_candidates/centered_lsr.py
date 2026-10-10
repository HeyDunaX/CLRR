"""Compatibility imports; implementation lives in src/methods/."""
if __package__.startswith("src."):
    from src.methods.centered_lsr.v1.lsr import CenteredLSRSeq2SeqLM, centered_cosine_loss
else:
    from methods.centered_lsr.v1.lsr import CenteredLSRSeq2SeqLM, centered_cosine_loss
