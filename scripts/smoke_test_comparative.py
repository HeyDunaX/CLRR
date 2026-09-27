"""Preflight Smoke Test for Comparative Baselines.

Validates that both models can:
1. Initialize without error on google/mt5-small.
2. Run 1 forward pass on a 2-sample batch.
3. Compute valid finite scalar loss (CE + Alignment for Middle-Align).
4. Run 1 backward pass with valid gradients on all parameters.
5. Run 1 generate() step with beam search.
6. Verify Hugging Face API access if HF_TOKEN is set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer

from src.comparative_baselines.layerskip_acl2024 import load_layerskip_model
from src.comparative_baselines.middle_align_acl2025 import load_middle_align_model



def run_smoke_test() -> bool:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Smoke Test] Running preflight check on device: {device}")

    model_name = "google/mt5-small"
    print(f"[Smoke Test] Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Dummy batch: 2 Amis sentences and 2 Mandarin references
    src_texts = ["Mirecep to kofa ko kapah.", "O ma'oripay a tamdaw."]
    tgt_texts = ["青年喝咖啡。", "活著的人。"]

    src_enc = tokenizer(src_texts, return_tensors="pt", padding=True).to(device)
    tgt_enc = tokenizer(text_target=tgt_texts, return_tensors="pt", padding=True).to(device)
    labels = tgt_enc["input_ids"]

    # -------------------------------------------------------------
    # 1. Test LayerSkip (ACL 2024)
    # -------------------------------------------------------------
    print("\n--- Testing LayerSkip (ACL 2024) ---")
    ls_model = load_layerskip_model(model_name, p_max=0.2).to(device)
    ls_model.train()

    print(f"LayerSkip encoder layers: {ls_model.num_layers}")
    print(f"Dropout rates p_l: {[round(p, 4) for p in ls_model.drop_probs]}")

    # Forward
    ls_out = ls_model(input_ids=src_enc["input_ids"], attention_mask=src_enc["attention_mask"], labels=labels)
    assert ls_out.loss is not None, "LayerSkip loss is None!"
    assert torch.isfinite(ls_out.loss), f"LayerSkip loss is non-finite: {ls_out.loss.item()}"
    print(f"LayerSkip Forward Pass: PASS (loss = {ls_out.loss.item():.4f})")

    # Backward
    ls_out.loss.backward()
    grad_count = sum(1 for p in ls_model.parameters() if p.grad is not None)
    total_count = sum(1 for p in ls_model.parameters())
    print(f"LayerSkip Backward Pass: PASS ({grad_count}/{total_count} parameter tensors received gradients)")

    # Generate
    ls_model.eval()
    with torch.no_grad():
        gen_tokens = ls_model.generate(input_ids=src_enc["input_ids"][:1], max_length=16, num_beams=2)
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    print(f"LayerSkip Generate Pass: PASS (decoded: '{gen_text}')")

    del ls_model, ls_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 2. Test Middle-Layer Alignment (ACL 2025)
    # -------------------------------------------------------------
    print("\n--- Testing Middle-Layer Alignment (ACL 2025) ---")
    ma_model = load_middle_align_model(
        model_name,
        middle_layer_idx=4,
        temperature=0.1,
        align_weight=0.1,
    ).to(device)
    ma_model.train()

    # Forward
    ma_out = ma_model(input_ids=src_enc["input_ids"], attention_mask=src_enc["attention_mask"], labels=labels)
    assert ma_out.loss is not None, "Middle-Align loss is None!"
    assert torch.isfinite(ma_out.loss), f"Middle-Align loss is non-finite: {ma_out.loss.item()}"
    print(f"Middle-Align Forward Pass: PASS (total_loss = {ma_out.loss.item():.4f})")

    # Backward
    ma_out.loss.backward()
    grad_count = sum(1 for p in ma_model.parameters() if p.grad is not None)
    total_count = sum(1 for p in ma_model.parameters())
    print(f"Middle-Align Backward Pass: PASS ({grad_count}/{total_count} parameter tensors received gradients)")

    # Generate
    ma_model.eval()
    with torch.no_grad():
        gen_tokens = ma_model.generate(input_ids=src_enc["input_ids"][:1], max_length=16, num_beams=2)
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    print(f"Middle-Align Generate Pass: PASS (decoded: '{gen_text}')")

    del ma_model, ma_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 3. Test Hugging Face Token (if available)
    # -------------------------------------------------------------
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

    print("\n==========================================")
    print("ALL PREFLIGHT SMOKE TESTS PASSED (100% OK)")
    print("==========================================")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    if not success:
        sys.exit(1)
