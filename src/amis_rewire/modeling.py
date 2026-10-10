"""Compatibility imports; implementation lives in src/methods/."""
if __package__.startswith("src."):
    from src.methods.original import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM, load_model
else:
    from methods.original import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM, load_model
from transformers import AutoModelForSeq2SeqLM
if __package__.startswith("src."):
    from src.methods.common.helpers import _hidden_and_repack, _find_stack_layers, _masked_mean_pool
else:
    from methods.common.helpers import _hidden_and_repack, _find_stack_layers, _masked_mean_pool
