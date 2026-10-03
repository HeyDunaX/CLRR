"""Preflight Smoke Test for NLLB-200 Suite.

Verifies model initialization, parameter counts, forward pass, backward gradient flow,
and generation for all 7 experimental methods on NLLB-200 in < 15 seconds on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root and src to sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

import torch
from transformers import M2M100Config, M2M100ForConditionalGeneration

from src.amis_rewire.modeling import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM
from src.comparative_baselines.layerskip_acl2024.model import LayerSkipMT5
from src.comparative_baselines.middle_align_acl2025.model import MiddleAlignMT5
from src.peft_baselines.bitfit_adapter import apply_bitfit_to_model
from src.peft_baselines.lora_adapter import apply_lora_to_model
from src.nllb_suite.modeling_nllb import DEFAULT_NLLB_MODEL, DEFAULT_TGT_LANG, get_nllb_tokenizer


METHODS_TO_TEST = [
    "baseline",
    "bitfit",
    "lora",
    "layerskip",
    "middle_align",
    "clrr_enc",
    "clrr_dec",
]


def build_test_nllb(fast: bool = True) -> M2M100ForConditionalGeneration:
    """Builds a lightweight NLLB (M2M100) model for instant CPU/GPU verification."""
    config = M2M100Config(
        vocab_size=256204,
        d_model=128 if fast else 1024,
        encoder_layers=2 if fast else 12,
        decoder_layers=2 if fast else 12,
        encoder_attention_heads=2 if fast else 16,
        decoder_attention_heads=2 if fast else 16,
        encoder_ffn_dim=512 if fast else 4096,
        decoder_ffn_dim=512 if fast else 4096,
        max_position_embeddings=512,
        forced_bos_token_id=256201,  # zho_Hant token id
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    return M2M100ForConditionalGeneration(config)


def apply_method_to_base(base_model: torch.nn.Module, method: str) -> torch.nn.Module:
    if method == "baseline":
        return base_model
    elif method == "bitfit":
        model, _ = apply_bitfit_to_model(base_model)
        return model
    elif method == "lora":
        model, _ = apply_lora_to_model(base_model, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
        return model
    elif method == "layerskip":
        return LayerSkipMT5(base_model, p_max=0.2)
    elif method == "middle_align":
        return MiddleAlignMT5(base_model, middle_layer_idx=1, align_weight=0.1, pad_token_id=1)
    elif method == "clrr_enc":
        rewired = CrossLayerResidualRewire(base_model, distance=1, strength=0.1, stack="encoder")
        return JEPAGuidedSeq2SeqLM(rewired, jepa_weight=0.1)
    elif method == "clrr_dec":
        rewired = CrossLayerResidualRewire(base_model, distance=1, strength=0.1, stack="decoder")
        return JEPAGuidedSeq2SeqLM(rewired, jepa_weight=0.1)
    else:
        raise ValueError(f"Unknown method {method}")


def run_smoke_test() -> bool:
    print("=" * 65)
    print("Preflight Fast Smoke Test: NLLB-200 Experimental Suite")
    print("=" * 65)

    print("\n[1/2] Testing NLLB Tokenizer & Language Code...")
    tokenizer = get_nllb_tokenizer(DEFAULT_NLLB_MODEL, src_lang="zho_Hant", tgt_lang="zho_Hant")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_TGT_LANG)
    assert forced_bos_token_id == 256201, f"Expected 256201, got {forced_bos_token_id}"
    print(f"  Tokenizer OK! forced_bos_token_id for '{DEFAULT_TGT_LANG}' = {forced_bos_token_id}")

    # Dummy batch
    batch_size = 2
    seq_len = 6
    input_ids = torch.randint(10, 500, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = torch.randint(10, 500, (batch_size, seq_len), dtype=torch.long)

    print(f"\n[2/2] Testing 7 Methods on NLLB Architecture (fast config)...")

    all_passed = True
    for method in METHODS_TO_TEST:
        print(f"\n--- Testing Method: {method.upper()} ---")
        try:
            base_model = build_test_nllb(fast=True)
            model = apply_method_to_base(base_model, method)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            pct = 100.0 * trainable / total if total > 0 else 0.0
            print(f"  Params: trainable={trainable:,} / total={total:,} ({pct:.4f}%)")

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            assert loss is not None and not torch.isnan(loss), f"Loss is NaN or None for {method}!"
            print(f"  Forward pass OK: loss = {loss.item():.4f}")

            # Backward pass
            loss.backward()
            print("  Backward pass OK: gradients successfully computed")

            # Generation test
            gen_model = model.base_model if hasattr(model, "base_model") and hasattr(model.base_model, "generate") else model
            gen_out = gen_model.generate(
                input_ids=input_ids[:1],
                attention_mask=attention_mask[:1],
                max_length=10,
                num_beams=2,
                forced_bos_token_id=forced_bos_token_id,
            )
            assert gen_out is not None and gen_out.shape[0] == 1, "Generation output shape mismatch!"
            print(f"  Generation OK: shape = {list(gen_out.shape)}")
            print(f"  [PASS] {method}")

            del base_model, model, outputs, loss, gen_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [FAIL] {method}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 65)
    if all_passed:
        print("ALL 7 NLLB-200 METHODS PASSED PREFLIGHT SMOKE TEST!")
    else:
        print("SOME METHODS FAILED PREFLIGHT SMOKE TEST!")
    print("=" * 65)
    return all_passed


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
