"""LoRA Adapter Implementation for Seq2Seq Models.

Strictly follows official LoRA formulation:
- Reference: Hu et al., ICLR 2022 ("LoRA: Low-Rank Adaptation of Large Language Models")
- Mathematical form: W = W_0 + (alpha / r) * B * A, with A ~ N(0, sigma^2), B = 0.
- Applied to attention projection matrices: ["q_proj", "v_proj"].
"""

from __future__ import annotations

from typing import Any
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    modules_to_save: list[str] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Applies LoRA to a Seq2Seq model using official Hugging Face PEFT library.

    Args:
        model: Base Seq2Seq model (e.g. MBartForConditionalGeneration).
        r: LoRA decomposition rank (default: 8).
        lora_alpha: LoRA scaling factor (default: 16).
        lora_dropout: Dropout probability for LoRA layers (default: 0.05).
        target_modules: List of module names to apply LoRA to. Defaults to ["q_proj", "v_proj"].
        modules_to_save: Modules to unfreeze and train alongside LoRA adapters.

    Returns:
        peft_model: The wrapped PeftModel.
        stats: Dictionary containing parameter statistics.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
    )

    peft_model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in peft_model.parameters())
    percent = (100.0 * trainable_params / all_params) if all_params > 0 else 0.0

    print(
        f"[LoRA] Initialized successfully with r={r}, alpha={lora_alpha}, target_modules={target_modules}. "
        f"Trainable params: {trainable_params:,} / {all_params:,} ({percent:.4f}%)",
        flush=True,
    )

    stats = {
        "method": "LoRA",
        "rank": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": percent,
    }

    return peft_model, stats
