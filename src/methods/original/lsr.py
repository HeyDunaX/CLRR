from __future__ import annotations
from contextlib import nullcontext
from typing import Any, Iterable
import torch
from torch import nn
import torch.nn.functional as F
from ..common.helpers import _masked_mean_pool

class JEPAGuidedSeq2SeqLM(nn.Module):
    """JEPA-guided Seq2Seq model for low-resource translation.

    Combines autoregressive translation (Cross-Entropy) with representation-space
    predictive alignment (JEPA loss):
      L = L_NMT + lambda_jepa * L_JEPA

    Where L_JEPA aligns the Context Encoder source representation (Amis) with the
    target representation (Chinese) with stop-gradient. This is shared-encoder
    cosine regularization, without a predictor or EMA teacher; it adds zero
    extra parameters and does not guarantee that representations avoid collapse.
    """

    # Both CE and LSR are microbatch means, not sums normalized by Trainer's
    # accumulated token count.
    accepts_loss_kwargs = False

    def __init__(self, base_model: nn.Module, jepa_weight: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.jepa_weight = jepa_weight

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    @property
    def config(self) -> Any:
        return self.base_model.config

    def extra_parameter_count(self) -> int:
        return 0

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False, **kwargs: Any) -> int:
        if hasattr(self.base_model, "num_parameters"):
            return self.base_model.num_parameters(only_trainable=only_trainable, exclude_embeddings=exclude_embeddings, **kwargs)
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
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("num_items_in_batch", None)
        # Request hidden states to compute JEPA latent alignment
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
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
