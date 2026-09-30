"""BitFit Adapter Implementation for Seq2Seq Models.

Strictly follows official BitFit formulation and code from authors:
- Reference: Ben-Zaken, Ravfogel, and Goldberg (ACL 2022)
  "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models"
- Official code: https://github.com/benzakenelad/BitFit (glue_evaluator.py lines 612-629)
- Freezes all weight matrices and optimizes only bias vectors.
"""

from __future__ import annotations

from typing import Any
import torch.nn as nn


def apply_bitfit_to_model(
    model: nn.Module,
    bias_components: list[str] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Applies BitFit to a Seq2Seq model by freezing weight matrices and unfreezing bias terms.

    Args:
        model: Base Seq2Seq model (e.g. MBartForConditionalGeneration).
        bias_components: Specific bias components to unfreeze. Defaults to ["bias"] (all bias terms).

    Returns:
        model: Model with updated parameter gradient flags.
        stats: Dictionary containing parameter statistics.
    """
    if bias_components is None:
        bias_components = ["bias"]

    # 1. Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # 2. Unfreeze bias components as specified in official author implementation
    activated_tensors = []
    for name, param in model.named_parameters():
        for component in bias_components:
            if component in name:
                param.requires_grad = True
                activated_tensors.append((name, list(param.shape)))
                break

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    percent = (100.0 * trainable_params / all_params) if all_params > 0 else 0.0

    print(
        f"[BitFit] Initialized successfully. "
        f"Trainable bias params: {trainable_params:,} / {all_params:,} ({percent:.4f}%), "
        f"activated tensors: {len(activated_tensors)}",
        flush=True,
    )

    stats = {
        "method": "BitFit",
        "bias_components": bias_components,
        "num_bias_tensors": len(activated_tensors),
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": percent,
        "sample_bias_tensors": [name for name, _ in activated_tensors[:5]],
    }

    return model, stats
