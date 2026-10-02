import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path("d:/Code/CLRR/data_processed/opus100_turkish_english_20k")
assert (DATA_DIR / "train.csv").exists(), "Missing train.csv"

train_df = pd.read_csv(DATA_DIR / "train.csv", nrows=10)
print(f"Loaded 10 samples from {DATA_DIR}")

# Test 1: mBART-50 tokenizer & config
print("\n--- Testing mBART-50 with Turkish (tr_TR -> en_XX) ---")
tok_mbart = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tok_mbart.src_lang = "tr_TR"
tok_mbart.tgt_lang = "en_XX"

inputs = tok_mbart(list(train_df["source"][:2]), text_target=list(train_df["target"][:2]), return_tensors="pt", padding=True)
print(f"mBART input_ids shape: {inputs.input_ids.shape}, labels shape: {inputs.labels.shape}")
assert inputs.input_ids.shape[0] == 2
print("mBART Turkish tokenization PASSED!")

# Test 2: NLLB-200 tokenizer & config
print("\n--- Testing NLLB-200 with Turkish (tur_Latn -> eng_Latn) ---")
tok_nllb = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="tur_Latn", tgt_lang="eng_Latn")
inputs_nllb = tok_nllb(list(train_df["source"][:2]), text_target=list(train_df["target"][:2]), return_tensors="pt", padding=True)
print(f"NLLB input_ids shape: {inputs_nllb.input_ids.shape}, labels shape: {inputs_nllb.labels.shape}")
assert inputs_nllb.input_ids.shape[0] == 2
print("NLLB Turkish tokenization PASSED!")

print("\n=== ALL TURKISH SMOKE TESTS PASSED CLEANLY! ===")
