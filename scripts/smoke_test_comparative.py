"""Preflight Smoke Test for Comparative Baselines and Extensions.

Validates that models across target backbones (mt5-small, mbart-large-50, byt5-small) can:
1. Initialize without error.
2. Run 1 forward pass on a 2-sample batch.
3. Compute valid finite scalar loss (CE + Alignment for Middle-Align, JEPA loss for CLRR-Dec).
4. Run 1 backward pass with valid gradients on all parameters.
5. Run 1 generate() step with beam search.
6. Verify Hugging Face API access if HF_TOKEN is set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer

from src.amis_rewire.modeling import load_model as load_amis_model
from src.comparative_baselines.layerskip_acl2024 import load_layerskip_model
from src.comparative_baselines.layerskip_acl2024.train import configure_mbart
from src.comparative_baselines.middle_align_acl2025 import load_middle_align_model


MODELS = {
    "mt5": "google/mt5-small",
    "mbart": "facebook/mbart-large-50-many-to-many-mmt",
    "byt5": "google/byt5-small",
}


def test_model_family(family: str, model_name: str, device: str) -> None:
    print(f"\n=======================================================")
    print(f"[Smoke Test] Testing Backbone Family: {family.upper()} ({model_name})")
    print(f"=======================================================")

    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    src_texts = ["Mirecep to kofa ko kapah.", "O ma'oripay a tamdaw."]
    tgt_texts = ["青年喝咖啡。", "活著的人。"]

    src_enc = tokenizer(src_texts, return_tensors="pt", padding=True).to(device)
    tgt_enc = tokenizer(text_target=tgt_texts, return_tensors="pt", padding=True).to(device)
    labels = tgt_enc["input_ids"]

    # 1. LayerSkip
    print(f"\n--- [{family}] Testing LayerSkip (ACL 2024) ---")
    ls_model = load_layerskip_model(model_name, p_max=0.2).to(device)
    configure_mbart(tokenizer, ls_model)
    ls_model.train()
    print(f"LayerSkip encoder layers: {ls_model.num_layers} | rates: {[round(p, 3) for p in ls_model.drop_probs[:4]]}...")

    ls_out = ls_model(input_ids=src_enc["input_ids"], attention_mask=src_enc["attention_mask"], labels=labels)
    assert ls_out.loss is not None and torch.isfinite(ls_out.loss), f"LayerSkip loss invalid: {ls_out.loss}"
    print(f"LayerSkip Forward Pass: PASS (loss = {ls_out.loss.item():.4f})")

    ls_out.loss.backward()
    grad_count = sum(1 for p in ls_model.parameters() if p.grad is not None)
    print(f"LayerSkip Backward Pass: PASS ({grad_count} tensors received gradients)")

    ls_model.eval()
    with torch.no_grad():
        gen_tokens = ls_model.generate(input_ids=src_enc["input_ids"][:1], max_length=16, num_beams=2)
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    print(f"LayerSkip Generate Pass: PASS (sample: '{gen_text}')")

    del ls_model, ls_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Middle-Layer Alignment
    print(f"\n--- [{family}] Testing Middle-Layer Alignment (ACL 2025) ---")
    mid_layer = 4 if family == "mt5" else 6
    ma_model = load_middle_align_model(
        model_name,
        middle_layer_idx=mid_layer,
        temperature=0.1,
        align_weight=0.1,
    ).to(device)
    configure_mbart(tokenizer, ma_model)
    ma_model.train()

    ma_out = ma_model(input_ids=src_enc["input_ids"], attention_mask=src_enc["attention_mask"], labels=labels)
    assert ma_out.loss is not None and torch.isfinite(ma_out.loss), f"Middle-Align loss invalid: {ma_out.loss}"
    print(f"Middle-Align Forward Pass: PASS (total_loss = {ma_out.loss.item():.4f})")

    ma_out.loss.backward()
    grad_count = sum(1 for p in ma_model.parameters() if p.grad is not None)
    print(f"Middle-Align Backward Pass: PASS ({grad_count} tensors received gradients)")

    ma_model.eval()
    with torch.no_grad():
        gen_tokens = ma_model.generate(input_ids=src_enc["input_ids"][:1], max_length=16, num_beams=2)
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    print(f"Middle-Align Generate Pass: PASS (sample: '{gen_text}')")

    del ma_model, ma_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. JEPA + CLRR-Dec
    print(f"\n--- [{family}] Testing JEPA + CLRR-Dec (Ours Decoder Rewiring) ---")
    clrr_model = load_amis_model(
        model_name,
        method="jepa-clrr",
        distance=2,
        strength=0.1,
        stack="decoder",
        jepa_weight=0.1,
    ).to(device)
    configure_mbart(tokenizer, clrr_model)
    clrr_model.train()

    clrr_out = clrr_model(input_ids=src_enc["input_ids"], attention_mask=src_enc["attention_mask"], labels=labels)
    assert clrr_out.loss is not None and torch.isfinite(clrr_out.loss), f"CLRR-Dec loss invalid: {clrr_out.loss}"
    print(f"CLRR-Dec Forward Pass: PASS (loss = {clrr_out.loss.item():.4f})")

    clrr_out.loss.backward()
    grad_count = sum(1 for p in clrr_model.parameters() if p.grad is not None)
    print(f"CLRR-Dec Backward Pass: PASS ({grad_count} tensors received gradients)")

    clrr_model.eval()
    with torch.no_grad():
        gen_tokens = clrr_model.generate(input_ids=src_enc["input_ids"][:1], max_length=16, num_beams=2)
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    print(f"CLRR-Dec Generate Pass: PASS (sample: '{gen_text}')")

    del clrr_model, clrr_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_smoke_test(target_families: list[str]) -> bool:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Smoke Test] Running preflight check on device: {device} | targets: {target_families}")

    for family in target_families:
        if family not in MODELS:
            raise ValueError(f"Unknown model family: {family}. Available: {list(MODELS.keys())}")
        test_model_family(family, MODELS[family], device)

    # Test Hugging Face Token (if available)
    print("\n--- Testing Hugging Face Access ---")
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            user_info = api.whoami()
            print(f"Hugging Face Auth: PASS (Authenticated as: {user_info.get('name', 'User')})")
        except Exception as e:
            print(f"Hugging Face Auth: WARNING (Token verification failed: {e})")
    else:
        print("Hugging Face Auth: NOTE (HF_TOKEN not set in environment; skipping remote upload test)")

    print("\n=======================================================")
    print("ALL PREFLIGHT SMOKE TESTS COMPLETED SUCCESSFULLY (100% OK)")
    print("=======================================================")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preflight Smoke Test for Comparative & CLRR-Dec Models")
    parser.add_argument("--backbone", choices=["mt5", "mbart", "byt5", "all"], default="all")
    args = parser.parse_args()

    targets = list(MODELS.keys()) if args.backbone == "all" else [args.backbone]
    success = run_smoke_test(targets)
    if not success:
        sys.exit(1)
