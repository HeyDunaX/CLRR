"""Helper module to load any checkpoint (Baseline, CLRR, LSR) across NLLB and mBART."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def load_eval_model_and_tokenizer(model_path: str | Path, model_family: str = "auto") -> tuple[Any, Any]:
    model_path = Path(model_path)
    str_path = str(model_path).lower()

    leaf_name = model_path.name.lower()
    if "nllb" in leaf_name or "nllb" in str_path or model_family == "nllb":
        family = "nllb"
        base_name = "facebook/nllb-200-distilled-600M"
    else:
        family = "mbart"
        base_name = "facebook/mbart-large-50-many-to-many-mmt"

    if "baseline" in leaf_name:
        is_clrr = False
    else:
        is_clrr = any(tag in leaf_name for tag in ("clrr", "lsr", "jepa"))

    # Tokenizer
    tok_dir = model_path if (model_path / "tokenizer_config.json").is_file() else base_name
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    if family == "nllb":
        tokenizer.src_lang = "zho_Hant"
        tokenizer.tgt_lang = "zho_Hant"
    elif family == "mbart":
        tokenizer.src_lang = "tl_XX" if "tl_XX" in tokenizer.lang_code_to_id else "en_XX"
        tokenizer.tgt_lang = "zh_CN"

    # Weights
    bin_file = model_path / "pytorch_model.bin"
    safe_file = model_path / "model.safetensors"

    print(f"[LoadHelper] Loading {model_path} (leaf={leaf_name}, family={family}, is_clrr={is_clrr})...", flush=True)

    if is_clrr:
        if family == "nllb":
            from src.nllb_suite.modeling_nllb import load_nllb_model
            model, _ = load_nllb_model("clrr_enc", model_name=base_name)
        else:
            from src.amis_rewire.modeling import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM
            from transformers import MBartForConditionalGeneration
            raw = MBartForConditionalGeneration.from_pretrained(base_name)
            model = JEPAGuidedSeq2SeqLM(CrossLayerResidualRewire(raw, distance=2, strength=0.1, stack="encoder"), jepa_weight=0.1)

        if bin_file.is_file():
            state_dict = torch.load(bin_file, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"[LoadHelper] Loaded state_dict from {bin_file}", flush=True)
        elif safe_file.is_file():
            from safetensors.torch import load_file
            state_dict = load_file(str(safe_file))
            model.load_state_dict(state_dict, strict=False)
            print(f"[LoadHelper] Loaded state_dict from {safe_file}", flush=True)
    else:
        if (model_path / "config.json").is_file():
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            except Exception:
                model = AutoModelForSeq2SeqLM.from_pretrained(base_name)
                if bin_file.is_file():
                    model.load_state_dict(torch.load(bin_file, map_location="cpu"), strict=False)
                elif safe_file.is_file():
                    from safetensors.torch import load_file
                    model.load_state_dict(load_file(str(safe_file)), strict=False)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(base_name)
            if bin_file.is_file():
                model.load_state_dict(torch.load(bin_file, map_location="cpu"), strict=False)
            elif safe_file.is_file():
                from safetensors.torch import load_file
                model.load_state_dict(load_file(str(safe_file)), strict=False)

    return model, tokenizer
