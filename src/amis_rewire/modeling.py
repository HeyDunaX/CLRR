"""Parameter-neutral cross-layer residual rewiring and JEPA-guided Seq2Seq modeling."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM


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


class CrossLayerResidualRewire(nn.Module):
    """Wrap a base model and rewire hidden states across layers.

    The module owns zero nn.Parameter. When applied to the context encoder,
    at layer i it adds a fixed, deterministic residual from layer i-distance
    (using stop-gradient detach). This preserves low-level morphosyntactic
    features against over-smoothing in deep layers for polysynthetic languages.
    """

    def __init__(
        self,
        base_model: nn.Module,
        distance: int = 2,
        strength: float = 0.1,
        stack: str = "encoder",
    ):
        super().__init__()
        if distance < 1:
            raise ValueError("distance must be >= 1")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        if stack not in ("encoder", "decoder", "both"):
            raise ValueError(f"stack must be 'encoder', 'decoder', or 'both', got: {stack}")
        self.base_model = base_model
        self.distance = distance
        self.strength = strength
        self.stack = stack
        self._cache: dict[str, dict[int, torch.Tensor]] = {}
        self._handles: list[Any] = []
        self._install_hooks()

    def _install_hooks(self) -> None:
        stacks = ("encoder", "decoder") if self.stack == "both" else (self.stack,)
        for stack_name in stacks:
            layers = _find_stack_layers(self.base_model, stack_name)
            self._cache[stack_name] = {}
            for index, layer in enumerate(layers):
                self._handles.append(
                    layer.register_forward_hook(self._make_hook(stack_name, index))
                )

    def _make_hook(self, stack_name: str, index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden, torch.Tensor):
                return output
            source = self._cache[stack_name].get(index - self.distance)
            if source is not None and source.shape == hidden.shape:
                hidden = hidden + self.strength * source.to(dtype=hidden.dtype)
            self._cache[stack_name][index] = hidden.detach()
            return _hidden_and_repack(output, hidden)

        return hook

    @contextmanager
    def _fresh_cache(self):
        self._cache = {k: {} for k in ("encoder", "decoder")}
        try:
            yield
        finally:
            self._cache = {k: {} for k in ("encoder", "decoder")}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        with self._fresh_cache():
            return self.base_model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        with self._fresh_cache():
            return self.base_model.generate(*args, **kwargs)

    def num_parameters(self, only_trainable: bool = False) -> int:
        parameters: Iterable[nn.Parameter] = self.parameters()
        if only_trainable:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

    def extra_parameter_count(self) -> int:
        return 0

    @property
    def config(self) -> Any:
        return self.base_model.config

    def get_encoder(self) -> nn.Module:
        if hasattr(self.base_model, "get_encoder"):
            return self.base_model.get_encoder()
        if hasattr(self.base_model, "encoder"):
            return self.base_model.encoder
        raise AttributeError("Underlying model has no encoder attribute.")

    def get_decoder(self) -> nn.Module:
        if hasattr(self.base_model, "get_decoder"):
            return self.base_model.get_decoder()
        if hasattr(self.base_model, "decoder"):
            return self.base_model.decoder
        raise AttributeError("Underlying model has no decoder attribute.")

    def save_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.save_pretrained(*args, **kwargs)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return self.base_model.prepare_decoder_input_ids_from_labels(labels=labels)

    def gradient_checkpointing_enable(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self) -> Any:
        return self.base_model.gradient_checkpointing_disable()

    def get_input_embeddings(self) -> Any:
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self) -> Any:
        return self.base_model.get_output_embeddings()

    def enable_input_require_grads(self) -> Any:
        return self.base_model.enable_input_require_grads()

    def disable_input_require_grads(self) -> Any:
        return self.base_model.disable_input_require_grads()


class JEPAGuidedSeq2SeqLM(nn.Module):
    """JEPA-guided Seq2Seq model for low-resource translation.

    Combines autoregressive translation (Cross-Entropy) with representation-space
    predictive alignment (JEPA loss):
      L = L_NMT + lambda_jepa * L_JEPA

    Where L_JEPA aligns the Context Encoder source representation (Amis) with the
    target representation (Chinese) with stop-gradient, preventing semantic collapse.
    Adds zero extra parameters.
    """

    def __init__(self, base_model: nn.Module, jepa_weight: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.jepa_weight = jepa_weight

    @property
    def config(self) -> Any:
        return self.base_model.config

    def extra_parameter_count(self) -> int:
        return 0

    def num_parameters(self, only_trainable: bool = False) -> int:
        parameters: Iterable[nn.Parameter] = self.parameters()
        if only_trainable:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

    def get_encoder(self) -> nn.Module:
        if hasattr(self.base_model, "get_encoder"):
            return self.base_model.get_encoder()
        if hasattr(self.base_model, "encoder"):
            return self.base_model.encoder
        raise AttributeError("Underlying model has no encoder attribute.")

    def get_decoder(self) -> nn.Module:
        if hasattr(self.base_model, "get_decoder"):
            return self.base_model.get_decoder()
        if hasattr(self.base_model, "decoder"):
            return self.base_model.decoder
        raise AttributeError("Underlying model has no decoder attribute.")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        # Request hidden states to compute JEPA latent alignment
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )

        if labels is not None and self.jepa_weight > 0.0:
            encoder_outputs = getattr(outputs, "encoder_last_hidden_state", None)
            if encoder_outputs is None and hasattr(outputs, "encoder_hidden_states") and outputs.encoder_hidden_states:
                encoder_outputs = outputs.encoder_hidden_states[-1]

            if encoder_outputs is not None:
                # Target tokens: replace ignore index -100 with pad token id
                pad_token_id = getattr(self.config, "pad_token_id", 0) or 0
                target_ids = labels.clone()
                target_ids[target_ids == -100] = pad_token_id
                target_mask = (target_ids != pad_token_id).long()

                # Target encoder representation under stop-gradient (JEPA anchor)
                encoder = self.get_encoder()
                fresh_cache_ctx = (
                    self.base_model._fresh_cache()
                    if hasattr(self.base_model, "_fresh_cache")
                    else nullcontext()
                )
                with fresh_cache_ctx:
                    with torch.no_grad():
                        target_out = encoder(input_ids=target_ids, attention_mask=target_mask)
                        target_hidden = (
                            target_out.last_hidden_state
                            if hasattr(target_out, "last_hidden_state")
                            else target_out[0]
                        ).detach()

                # Pool and compute cosine distance in latent space
                src_rep = F.normalize(_masked_mean_pool(encoder_outputs, attention_mask), p=2, dim=-1)
                tgt_rep = F.normalize(_masked_mean_pool(target_hidden, target_mask), p=2, dim=-1)
                jepa_loss = (1.0 - (src_rep * tgt_rep).sum(dim=-1)).mean()

                if hasattr(outputs, "loss") and outputs.loss is not None:
                    outputs.loss = outputs.loss + self.jepa_weight * jepa_loss.to(outputs.loss.dtype)

        return outputs

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.save_pretrained(*args, **kwargs)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return self.base_model.prepare_decoder_input_ids_from_labels(labels=labels)

    def gradient_checkpointing_enable(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self) -> Any:
        return self.base_model.gradient_checkpointing_disable()

    def get_input_embeddings(self) -> Any:
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self) -> Any:
        return self.base_model.get_output_embeddings()

    def enable_input_require_grads(self) -> Any:
        return self.base_model.enable_input_require_grads()

    def disable_input_require_grads(self) -> Any:
        return self.base_model.disable_input_require_grads()


def load_model(
    model_name: str,
    method: str,
    distance: int = 2,
    strength: float = 0.1,
    stack: str = "encoder",
    jepa_weight: float = 0.1,
) -> nn.Module:
    """Load model with specified architecture and rewiring configuration."""
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    method_clean = method.lower().strip()
    if method_clean in ("baseline", "none"):
        return base_model

    if method_clean in ("clrr", "clrr-enc", "rewire"):
        return CrossLayerResidualRewire(base_model, distance=distance, strength=strength, stack=stack)

    if method_clean in ("jepa",):
        return JEPAGuidedSeq2SeqLM(base_model, jepa_weight=jepa_weight)

    if method_clean in ("jepa-clrr", "jepa-clrr-enc"):
        rewired = CrossLayerResidualRewire(base_model, distance=distance, strength=strength, stack=stack)
        return JEPAGuidedSeq2SeqLM(rewired, jepa_weight=jepa_weight)

    raise ValueError(f"Unknown method: {method}. Valid options: baseline, clrr-enc, jepa, jepa-clrr-enc")