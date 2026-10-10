from __future__ import annotations
from torch import nn
from transformers import AutoModelForSeq2SeqLM
from .clrr import CrossLayerResidualRewire
from .lsr import JEPAGuidedSeq2SeqLM

def load_model(
    model_name: str,
    method: str,
    distance: int = 2,
    strength: float = 0.1,
    stack: str = "encoder",
    jepa_weight: float = 0.1,
    revision: str | None = None,
    middle_layer_idx: int = 6,
    alignment_temperature: float = 0.1,
) -> nn.Module:
    """Load model with specified architecture and rewiring configuration."""
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)

    method_clean = method.lower().strip()
    if method_clean in ("baseline", "none"):
        return base_model

    if method_clean == "middle-align":
        from comparative_baselines.middle_align_acl2025.model import MiddleAlignMT5
        return MiddleAlignMT5(base_model, middle_layer_idx=middle_layer_idx,
                              temperature=alignment_temperature, align_weight=jepa_weight,
                              pad_token_id=base_model.config.pad_token_id)

    if method_clean in ("clrr", "clrr-enc", "rewire"):
        return CrossLayerResidualRewire(base_model, distance=distance, strength=strength, stack=stack)

    if method_clean in ("jepa",):
        return JEPAGuidedSeq2SeqLM(base_model, jepa_weight=jepa_weight)

    if method_clean in ("jepa-clrr", "jepa-clrr-enc"):
        rewired = CrossLayerResidualRewire(base_model, distance=distance, strength=strength, stack=stack)
        return JEPAGuidedSeq2SeqLM(rewired, jepa_weight=jepa_weight)

    raise ValueError(f"Unknown method: {method}. Valid options: baseline, clrr-enc, jepa, jepa-clrr-enc")
