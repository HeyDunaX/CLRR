from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Iterable
import torch
from torch import nn
from ..common.helpers import _hidden_and_repack, _find_stack_layers

class CrossLayerResidualRewire(nn.Module):
    """Wrap a base model and rewire hidden states across layers.

    The module owns zero nn.Parameter. When applied to the context encoder,
    at layer i it adds a fixed, deterministic residual from layer i-distance
    (using stop-gradient detach). Morphological preservation and reduced
    over-smoothing are hypotheses to evaluate, not guarantees of this operation.
    """

    # Forward returns a mean loss and discards num_items_in_batch. Tell Trainer
    # to divide each microbatch loss by the gradient accumulation count.
    accepts_loss_kwargs = False

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

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("num_items_in_batch", None)
        with self._fresh_cache():
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                **kwargs,
            )

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        with self._fresh_cache():
            return self.base_model.generate(*args, **kwargs)

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False, **kwargs: Any) -> int:
        if hasattr(self.base_model, "num_parameters"):
            return self.base_model.num_parameters(only_trainable=only_trainable, exclude_embeddings=exclude_embeddings, **kwargs)
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
        if hasattr(self.base_model, "prepare_decoder_input_ids_from_labels"):
            return self.base_model.prepare_decoder_input_ids_from_labels(labels=labels)
        decoder_start_token_id = getattr(self.config, "decoder_start_token_id", None)
        pad_token_id = getattr(self.config, "pad_token_id", None)
        if decoder_start_token_id is None and hasattr(self.base_model, "config"):
            decoder_start_token_id = getattr(self.base_model.config, "decoder_start_token_id", None)
        if pad_token_id is None and hasattr(self.base_model, "config"):
            pad_token_id = getattr(self.base_model.config, "pad_token_id", None)

        shifted_input_ids = labels.new_zeros(labels.shape)
        shifted_input_ids[:, 1:] = labels[:, :-1].clone()
        if decoder_start_token_id is not None:
            shifted_input_ids[:, 0] = decoder_start_token_id
        if pad_token_id is not None:
            shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

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

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Any, strict: bool = True) -> Any:
        return self.base_model.load_state_dict(state_dict, strict=strict)
