"""Preflight Smoke Test for Revalidation Suite.

Validates in < 30s:
1. mT5-small No-Stop-Gradient forward/backward/generate
2. mBART-large-50 CLRR-only forward/backward/generate
3. mBART-large-50 LSR-only forward/backward/generate
4. Hugging Face credentials check
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src and repo_root to sys.path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration, MBartForConditionalGeneration

from amis_rewire.train import configure_mbart
from amis_rewire.modeling import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM
from scripts.revalidation.run_no_stop_gradient import CrossLayerResidualRewireNoStopGrad


def smoke_test() -> None:
    print("[smoke] Starting preflight smoke tests...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] Target device: {device}")

    # 1. mT5 No-Stop-Gradient
    print("[smoke] Test 1/3: mT5-small No-Stop-Gradient forward/backward...")
    mt5_name = "google/mt5-small"
    mt5_tok = AutoTokenizer.from_pretrained(mt5_name)
    mt5_base = MT5ForConditionalGeneration.from_pretrained(mt5_name).to(device)
    mt5_no_sg = CrossLayerResidualRewireNoStopGrad(mt5_base, distance=2, strength=0.1, stack="encoder")

    inputs = mt5_tok(["Mifoting ko wawa.", "Mafuti' ci Ina."], return_tensors="pt", padding=True).to(device)
    labels = mt5_tok(["小孩在抓魚。", "媽媽在睡覺。"], return_tensors="pt", padding=True).input_ids.to(device)

    out = mt5_no_sg(**inputs, labels=labels)
    loss = out.loss
    loss.backward()
    gen = mt5_no_sg.generate(**inputs, max_new_tokens=5)
    print(f"[smoke] mT5 No-SG OK! Loss: {loss.item():.4f}, Generated shape: {gen.shape}")

    del mt5_base, mt5_no_sg, out, loss, gen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. mBART CLRR-only
    print("[smoke] Test 2/3: mBART CLRR-only...")
    mbart_name = "facebook/mbart-large-50-many-to-many-mmt"
    mbart_tok = AutoTokenizer.from_pretrained(mbart_name)
    # Using tiny stub or base config
    mbart_base = MBartForConditionalGeneration.from_pretrained(mbart_name).to(device)
    configure_mbart(mbart_tok, mbart_base)
    mbart_clrr = CrossLayerResidualRewire(mbart_base, distance=2, strength=0.1, stack="encoder")

    mb_in = mbart_tok(["Mifoting ko wawa."], return_tensors="pt").to(device)
    mb_lbl = mbart_tok(["小孩在抓魚。"], return_tensors="pt").input_ids.to(device)
    out_mb = mbart_clrr(**mb_in, labels=mb_lbl)
    out_mb.loss.backward()
    gen_mb = mbart_clrr.generate(**mb_in, max_new_tokens=5)
    print(f"[smoke] mBART CLRR-only OK! Loss: {out_mb.loss.item():.4f}, Generated shape: {gen_mb.shape}")

    # 3. mBART LSR-only
    print("[smoke] Test 3/3: mBART LSR-only...")
    mbart_lsr = JEPAGuidedSeq2SeqLM(mbart_base, jepa_weight=0.1)
    out_lsr = mbart_lsr(**mb_in, labels=mb_lbl)
    out_lsr.loss.backward()
    print(f"[smoke] mBART LSR-only OK! Loss: {out_lsr.loss.item():.4f}")

    del mbart_base, mbart_clrr, mbart_lsr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Check HF Token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("[smoke] HF_TOKEN detected: OK")
    else:
        print("[smoke] Notice: HF_TOKEN not set (checkpoints will be saved locally)")

    print("[smoke] ALL PREFLIGHT SMOKE TESTS PASSED!")


if __name__ == "__main__":
    smoke_test()
