"""Middle-Layer Representation Alignment (Liu & Niehues, ACL 2025 Long Paper).

Implements Cross-Lingual Semantic Representation Alignment at the middle layer:
- Eq. (1): L_align = - 1/|B| sum_{(s,t)} log( exp(sim(h_s^i, h_t^i) / tau) / sum_v exp(sim(h_s^i, h_v^i) / tau) )
- Target middle layer: Encoder layer index i = 4 (for 8-layer mT5-small)
- Temperature: tau = 0.1 (Appendix D.1)
- Loss weight: lambda_align = 0.1
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import Seq2SeqLMOutput


def _masked_mean_pool(hidden_states: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean-pools hidden states over sequence dimension, ignoring masked tokens."""
    if mask is None:
        return hidden_states.mean(dim=1)
    mask_float = mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    sum_hidden = (hidden_states * mask_float).sum(dim=1)
    sum_mask = mask_float.sum(dim=1).clamp(min=1e-6)
    return sum_hidden / sum_mask


class MiddleAlignMT5(nn.Module):
    """Wraps an mT5 model with Middle-Layer Representation Alignment (ACL 2025).

    Extracts representations at middle layer (layer 4) for both source (Amis)
    and target (Chinese) sequences and computes a Cosine Contrastive Alignment Loss.
    Zero extra parameters (Delta theta = 0).
    """

    def __init__(
        self,
        base_model: nn.Module,
        middle_layer_idx: int = 4,
        temperature: float = 0.1,
        align_weight: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.base_model = base_model
        self.middle_layer_idx = middle_layer_idx
        self.temperature = temperature
        self.align_weight = align_weight
        self.pad_token_id = pad_token_id

    def _extract_middle_rep(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Passes tokens through the encoder and extracts mean-pooled layer-4 representation."""
        encoder = getattr(self.base_model, "encoder", None)
        if encoder is None and hasattr(self.base_model, "model"):
            encoder = getattr(self.base_model.model, "encoder", None)
        if encoder is None:
            raise ValueError(f"Could not locate encoder in {self.base_model.__class__.__name__}")

        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # hidden_states: tuple of (initial_embeds, layer_1, ..., layer_L)
        # layer_idx 4 is hidden_states[4] or hidden_states[middle_layer_idx]
        idx = min(self.middle_layer_idx, len(encoder_outputs.hidden_states) - 1)
        layer_hidden = encoder_outputs.hidden_states[idx]
        return _masked_mean_pool(layer_hidden, attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Seq2SeqLMOutput:
        kwargs.pop("num_items_in_batch", None)
        # Standard Seq2Seq forward pass
        outputs = self.base_model(

            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )

        ce_loss = outputs.loss
        if (not self.training) or (ce_loss is None) or (labels is None) or (input_ids is None):
            return outputs

        batch_size = input_ids.size(0)
        if batch_size <= 1:
            return outputs

        # Extract source representation h_s^4
        idx = min(self.middle_layer_idx, len(outputs.encoder_hidden_states) - 1)
        h_s = _masked_mean_pool(outputs.encoder_hidden_states[idx], attention_mask)

        # Build target tokens from labels (replacing -100 with pad_token_id)
        target_ids = torch.where(labels != -100, labels, self.pad_token_id)
        target_mask = (labels != -100).long()

        # Extract target representation h_t^4 through the shared encoder
        h_t = self._extract_middle_rep(input_ids=target_ids, attention_mask=target_mask)

        # L2-normalize
        h_s_norm = F.normalize(h_s, p=2, dim=-1)
        h_t_norm = F.normalize(h_t, p=2, dim=-1)

        # Cosine similarity matrix scaled by temperature tau
        logits = torch.matmul(h_s_norm, h_t_norm.T) / self.temperature
        targets = torch.arange(batch_size, device=logits.device)

        # Symmetric contrastive alignment loss
        loss_s2t = F.cross_entropy(logits, targets)
        loss_t2s = F.cross_entropy(logits.T, targets)
        align_loss = 0.5 * (loss_s2t + loss_t2s)

        # Total combined loss
        total_loss = ce_loss + self.align_weight * align_loss

        return Seq2SeqLMOutput(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        return self.base_model.save_pretrained(save_directory, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def load_middle_align_model(
    model_name_or_path: str = "google/mt5-small",
    middle_layer_idx: int = 4,
    temperature: float = 0.1,
    align_weight: float = 0.1,
) -> MiddleAlignMT5:
    """Loads mT5 and wraps with Middle-Layer Alignment."""
    base = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    pad_id = getattr(base.config, "pad_token_id", 0) or 0
    return MiddleAlignMT5(
        base,
        middle_layer_idx=middle_layer_idx,
        temperature=temperature,
        align_weight=align_weight,
        pad_token_id=pad_id,
    )
