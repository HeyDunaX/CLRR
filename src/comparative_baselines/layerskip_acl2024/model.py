"""LayerSkip (Elhoushi et al., ACL 2024 Long Paper).

Implements Stochastic Layer Dropout on Transformer Encoder Blocks according to:
- Eq. (1): x_{l+1} = x_l + M(p_l) * f_l(x_l)
- Eq. (2): p_l = S(t) * D(l) * p_max  (S(t) = 1.0 for fine-tuning)
- Eq. (3): D(l) = exp(l * ln(2) / (L - 1)) - 1
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, MT5ForConditionalGeneration


def _hidden_and_repack(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


def _find_encoder_layers(model: nn.Module) -> list[nn.Module]:
    """Locate encoder blocks for T5 / mT5 models."""
    if hasattr(model, "encoder") and hasattr(model.encoder, "block"):
        return list(model.encoder.block)
    if hasattr(model, "model") and hasattr(model.model, "encoder") and hasattr(model.model.encoder, "block"):
        return list(model.model.encoder.block)
    raise ValueError(f"Could not locate encoder blocks for {model.__class__.__name__}")


class LayerSkipMT5(nn.Module):
    """Wraps an mT5 model with LayerSkip Stochastic Layer Dropout (ACL 2024).

    Applies layer-wise stochastic dropping to encoder blocks during training.
    During evaluation and inference, all layers are active (dropout probability = 0).
    Has zero extra parameters (Delta theta = 0).
    """

    def __init__(
        self,
        base_model: nn.Module,
        p_max: float = 0.2,
    ):
        super().__init__()
        self.base_model = base_model
        self.p_max = p_max
        self._layers = _find_encoder_layers(self.base_model)
        self.num_layers = len(self._layers)
        self.drop_probs: list[float] = self._calculate_dropout_rates()
        self._handles: list[Any] = []
        self._install_hooks()

    def _calculate_dropout_rates(self) -> list[float]:
        """Calculates per-layer dropout rate p_l using Eq. (2) and Eq. (3)."""
        L = self.num_layers
        rates = []
        for l in range(L):
            if L <= 1:
                d_l = 0.0
            else:
                # Eq. (3): D(l) = exp(l * ln(2) / (L - 1)) - 1
                d_l = math.exp(l * math.log(2.0) / (L - 1)) - 1.0
            # Eq. (2): p_l = D(l) * p_max  (S(t) = 1.0 for fine-tuning)
            p_l = float(d_l * self.p_max)
            rates.append(p_l)
        return rates

    def _install_hooks(self) -> None:
        """Installs forward hooks implementing Eq. (1)."""
        for l, (layer, p_l) in enumerate(zip(self._layers, self.drop_probs)):
            self._handles.append(layer.register_forward_hook(self._make_layer_hook(l, p_l)))

    def _make_layer_hook(self, layer_idx: int, p_l: float):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            # Only apply layer dropout during training and when p_l > 0
            if (not self.training) or p_l <= 0.0:
                return output

            input_hidden = inputs[0]
            block_hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(block_hidden, torch.Tensor) or input_hidden.shape != block_hidden.shape:
                return output

            batch_size = block_hidden.size(0)
            keep_prob = 1.0 - p_l
            # Sample-level Bernoulli dropout: shape (batch_size, 1, 1)
            keep_mask = (torch.rand((batch_size, 1, 1), device=block_hidden.device) < keep_prob)
            
            # Eq. (1): x_{l+1} = keep_mask * f_l(x_l) + (1 - keep_mask) * x_l
            dropped_hidden = torch.where(keep_mask, block_hidden, input_hidden)
            return _hidden_and_repack(output, dropped_hidden)

        return hook

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        return self.base_model.save_pretrained(save_directory, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def load_layerskip_model(
    model_name_or_path: str = "google/mt5-small",
    p_max: float = 0.2,
) -> LayerSkipMT5:
    """Loads mT5 and wraps with LayerSkip."""
    base = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    return LayerSkipMT5(base, p_max=p_max)
