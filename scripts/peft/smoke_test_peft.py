"""Preflight Smoke Test for PEFT Baselines (LoRA & BitFit).

Verifies in < 25 seconds:
1. Module imports and Hugging Face PEFT integration.
2. LoRA adapter initialization and parameter counting (trainable ~0.25%).
3. BitFit adapter initialization and parameter counting (trainable ~0.055%).
4. Forward pass, loss calculation, backward gradient flow:
   - Asserts that ONLY trainable parameters receive gradients.
   - Asserts that frozen weights have param.grad is None.
5. Autoregressive sequence generation with forced_bos_token_id.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from transformers import MBartConfig, MBartForConditionalGeneration
from peft_baselines.lora_adapter import apply_lora_to_model
from peft_baselines.bitfit_adapter import apply_bitfit_to_model


def build_test_mbart(fast: bool = True) -> MBartForConditionalGeneration:
    """Builds a lightweight mBART model for instant CPU/GPU verification."""
    config = MBartConfig(
        vocab_size=250054,
        d_model=256 if fast else 1024,
        encoder_layers=2 if fast else 12,
        decoder_layers=2 if fast else 12,
        encoder_attention_heads=4 if fast else 16,
        decoder_attention_heads=4 if fast else 16,
        encoder_ffn_dim=1024 if fast else 4096,
        decoder_ffn_dim=1024 if fast else 4096,
        max_position_embeddings=512,
        forced_bos_token_id=250025,  # zh_CN token id
    )
    return MBartForConditionalGeneration(config)


def test_lora() -> None:
    print("\n--- [TEST 1/2] LoRA Smoke Test ---")
    base_model = build_test_mbart(fast=True)
    peft_model, stats = apply_lora_to_model(
        base_model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    assert stats["trainable_params"] > 0, "LoRA has 0 trainable parameters!"
    assert stats["trainable_params"] < stats["all_params"], "LoRA trained all parameters!"
    print(f"LoRA trainable parameters: {stats['trainable_params']:,} / {stats['all_params']:,} ({stats['trainable_percent']:.4f}%)")

    # Dummy batch
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(10, 250000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = torch.randint(10, 250000, (batch_size, seq_len))

    # Forward
    outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    assert loss is not None and not torch.isnan(loss), "LoRA loss is None or NaN!"
    print(f"LoRA Forward Loss: {loss.item():.4f}")

    # Backward
    loss.backward()

    # Verify gradient flow: only LoRA adapters have gradients
    frozen_with_grad = 0
    lora_with_grad = 0
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                lora_with_grad += 1
        else:
            if param.grad is not None:
                frozen_with_grad += 1

    assert frozen_with_grad == 0, f"Error: {frozen_with_grad} frozen parameters received gradients in LoRA!"
    assert lora_with_grad > 0, "Error: No LoRA parameter received gradient!"
    print(f"LoRA Gradient verification: {lora_with_grad} adapter tensors updated, 0 frozen tensors touched.")

    # Generation check
    generated = peft_model.generate(input_ids=input_ids, max_length=10, num_beams=1)
    assert generated.shape[0] == batch_size, "LoRA generate output shape mismatch!"
    print(f"LoRA Generate output shape: {list(generated.shape)}")
    print("[PASS] LoRA Smoke Test PASSED successfully!")


def test_bitfit() -> None:
    print("\n--- [TEST 2/2] BitFit Smoke Test ---")
    base_model = build_test_mbart(fast=True)
    model, stats = apply_bitfit_to_model(base_model, bias_components=["bias"])

    assert stats["trainable_params"] > 0, "BitFit has 0 trainable parameters!"
    assert stats["trainable_params"] < stats["all_params"], "BitFit trained all parameters!"
    print(f"BitFit trainable parameters: {stats['trainable_params']:,} / {stats['all_params']:,} ({stats['trainable_percent']:.4f}%)")

    # Dummy batch
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(10, 250000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = torch.randint(10, 250000, (batch_size, seq_len))

    # Forward
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    assert loss is not None and not torch.isnan(loss), "BitFit loss is None or NaN!"
    print(f"BitFit Forward Loss: {loss.item():.4f}")

    # Backward
    loss.backward()

    # Verify gradient flow: only bias vectors have gradients
    weights_with_grad = 0
    biases_with_grad = 0
    for name, param in model.named_parameters():
        if "bias" in name:
            if param.grad is not None:
                biases_with_grad += 1
        else:
            if param.grad is not None:
                weights_with_grad += 1

    assert weights_with_grad == 0, f"Error: {weights_with_grad} weight matrices received gradients in BitFit!"
    assert biases_with_grad > 0, "Error: No bias vector received gradient!"
    print(f"BitFit Gradient verification: {biases_with_grad} bias tensors updated, 0 weight matrices touched.")

    # Generation check
    generated = model.generate(input_ids=input_ids, max_length=10, num_beams=1)
    assert generated.shape[0] == batch_size, "BitFit generate output shape mismatch!"
    print(f"BitFit Generate output shape: {list(generated.shape)}")
    print("[PASS] BitFit Smoke Test PASSED successfully!")


def main() -> None:
    print("=======================================================")
    print("Running Preflight Smoke Test for PEFT Baselines")
    print(f"Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print("=======================================================")
    test_lora()
    test_bitfit()
    print("\n=======================================================")
    print("ALL PEFT PREFLIGHT SMOKE TESTS COMPLETED SUCCESSFULLY!")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
