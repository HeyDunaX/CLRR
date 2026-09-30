"""NLLB-200 modeling module supporting Baseline, PEFT, Comparative, and CLRR variants."""

from __future__ import annotations

from typing import Any
import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.amis_rewire.modeling import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM
from src.comparative_baselines.layerskip_acl2024.model import LayerSkipMT5
from src.comparative_baselines.middle_align_acl2025.model import MiddleAlignMT5
from src.peft_baselines.bitfit_adapter import apply_bitfit_to_model
from src.peft_baselines.lora_adapter import apply_lora_to_model


DEFAULT_NLLB_MODEL = "facebook/nllb-200-distilled-600M"
DEFAULT_TGT_LANG = "zho_Hant"


def get_nllb_tokenizer(
    model_name: str = DEFAULT_NLLB_MODEL,
    src_lang: str = DEFAULT_TGT_LANG,
    tgt_lang: str = DEFAULT_TGT_LANG,
) -> Any:
    """Load and configure the NLLB SentencePiece tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        use_fast=True,
    )
    return tokenizer


def load_nllb_model(
    method: str,
    model_name: str = DEFAULT_NLLB_MODEL,
    distance: int = 2,
    strength: float = 0.1,
    jepa_weight: float = 0.1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    layerskip_p_max: float = 0.2,
    middle_layer_idx: int = 6,
    middle_align_weight: float = 0.1,
) -> tuple[nn.Module, dict[str, Any]]:
    """Loads and wraps NLLB-200 according to the specified experimental method.

    Supported methods:
      - 'baseline': Standard full parameter fine-tuning
      - 'bitfit': BitFit bias-only parameter-efficient fine-tuning (ACL 2022)
      - 'lora': LoRA low-rank adaptation on attention q_proj/v_proj (ICLR 2022)
      - 'layerskip': LayerSkip stochastic layer dropout on encoder (ACL 2024)
      - 'middle_align': Middle-Layer representation alignment at Layer 6 (ACL 2025)
      - 'clrr_enc': CLRR on encoder stack (d=2, alpha=0.1) + LSR (lambda=0.1)
      - 'clrr_dec': CLRR on decoder stack (d=2, alpha=0.1) + LSR (lambda=0.1)
    """
    print(f"[NLLB-200] Loading base model: {model_name}...", flush=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    metadata: dict[str, Any] = {
        "model": model_name,
        "method": method,
    }

    if method == "baseline":
        model = base_model
        metadata["description"] = "Standard Full Fine-Tuning"

    elif method == "bitfit":
        model, stats = apply_bitfit_to_model(base_model)
        metadata.update(stats)
        metadata["description"] = "BitFit Bias-Only Tuning (ACL 2022)"

    elif method == "lora":
        model, stats = apply_lora_to_model(
            base_model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )
        metadata.update(stats)
        metadata["description"] = f"LoRA (r={lora_r}, alpha={lora_alpha}, ICLR 2022)"

    elif method == "layerskip":
        model = LayerSkipMT5(base_model, p_max=layerskip_p_max)
        metadata["p_max"] = layerskip_p_max
        metadata["description"] = f"LayerSkip Stochastic Depth (p_max={layerskip_p_max}, ACL 2024)"

    elif method == "middle_align":
        pad_id = getattr(base_model.config, "pad_token_id", 0) or 0
        model = MiddleAlignMT5(
            base_model,
            middle_layer_idx=middle_layer_idx,
            align_weight=middle_align_weight,
            pad_token_id=pad_id,
        )
        metadata["middle_layer_idx"] = middle_layer_idx
        metadata["align_weight"] = middle_align_weight
        metadata["description"] = f"Middle-Layer Alignment (Layer {middle_layer_idx}, ACL 2025)"

    elif method == "clrr_enc":
        rewired = CrossLayerResidualRewire(
            base_model,
            distance=distance,
            strength=strength,
            stack="encoder",
        )
        model = JEPAGuidedSeq2SeqLM(rewired, jepa_weight=jepa_weight)
        metadata["rewire_stack"] = "encoder"
        metadata["distance"] = distance
        metadata["strength"] = strength
        metadata["jepa_weight"] = jepa_weight
        metadata["description"] = "CLRR-Enc + LSR (Proposed Main)"

    elif method == "clrr_dec":
        rewired = CrossLayerResidualRewire(
            base_model,
            distance=distance,
            strength=strength,
            stack="decoder",
        )
        model = JEPAGuidedSeq2SeqLM(rewired, jepa_weight=jepa_weight)
        metadata["rewire_stack"] = "decoder"
        metadata["distance"] = distance
        metadata["strength"] = strength
        metadata["jepa_weight"] = jepa_weight
        metadata["description"] = "CLRR-Dec + LSR (Proposed Decoder)"

    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported methods: "
            f"['baseline', 'bitfit', 'lora', 'layerskip', 'middle_align', 'clrr_enc', 'clrr_dec']"
        )

    return model, metadata
