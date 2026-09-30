"""PEFT Baselines for CLRR Comparative Evaluation.

Implements official parameter-efficient fine-tuning protocols:
- LoRA: Low-Rank Adaptation (Hu et al., ICLR 2022)
- BitFit: Bias-Term Fine-Tuning (Ben-Zaken et al., ACL 2022)
"""

from .lora_adapter import apply_lora_to_model
from .bitfit_adapter import apply_bitfit_to_model

__all__ = ["apply_lora_to_model", "apply_bitfit_to_model"]
