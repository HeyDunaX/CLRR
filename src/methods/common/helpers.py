from __future__ import annotations
from typing import Any
import torch
from torch import nn

def _hidden_and_repack(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


def _find_stack_layers(model: nn.Module, stack: str) -> list[nn.Module]:
    """Find encoder/decoder blocks across T5, mT5, ByT5, and mBART layouts."""
    if hasattr(model, stack):
        owner = getattr(model, stack)
        for attribute in ("block", "layers"):
            if hasattr(owner, attribute):
                return list(getattr(owner, attribute))
    if hasattr(model, "model"):
        owner = getattr(model.model, stack, None)
        if owner is not None:
            for attribute in ("block", "layers"):
                if hasattr(owner, attribute):
                    return list(getattr(owner, attribute))
    raise ValueError(f"Could not locate {stack} layers for {model.__class__.__name__}.")


def _masked_mean_pool(hidden_states: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool hidden states over sequence dimension with an attention mask."""
    if mask is None:
        return hidden_states.mean(dim=1)
    mask_float = mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    return (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-6)
