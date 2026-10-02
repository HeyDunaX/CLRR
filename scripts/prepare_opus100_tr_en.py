import os
import sys
import pandas as pd
from datasets import load_dataset

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = "d:/Code/CLRR/data_processed/opus100_turkish_english_20k"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Helsinki-NLP/opus-100 en-tr dataset from local cache...")
ds = load_dataset("Helsinki-NLP/opus-100", "en-tr")

print("Dataset loaded successfully!")
print(f"Original Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")

# 1. Validation split: 2,000 pairs
val_records = []
for item in ds["validation"]:
    tr_text = item["translation"]["tr"].strip()
    en_text = item["translation"]["en"].strip()
    if tr_text and en_text:
        val_records.append({"source": tr_text, "target": en_text})
df_val = pd.DataFrame(val_records)
df_val.to_csv(os.path.join(OUTPUT_DIR, "validation.csv"), index=False, encoding="utf-8")
print(f"Validation saved: {len(df_val)} pairs.")

# 2. Test split: 2,000 pairs
test_records = []
for item in ds["test"]:
    tr_text = item["translation"]["tr"].strip()
    en_text = item["translation"]["en"].strip()
    if tr_text and en_text:
        test_records.append({"source": tr_text, "target": en_text})
df_test = pd.DataFrame(test_records)
df_test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, encoding="utf-8")
print(f"Test saved: {len(df_test)} pairs.")

# 3. Train split: Fixed sample of 20,000 pairs with seed 42
# Shuffle train with seed 42 and take 20,000 valid non-empty pairs
print("Sampling 20,000 pairs from train split (seed 42)...")
shuffled_train = ds["train"].shuffle(seed=42)

val_sources = set(df_val["source"])
test_sources = set(df_test["source"])

train_records = []
for item in shuffled_train:
    tr_text = item["translation"]["tr"].strip()
    en_text = item["translation"]["en"].strip()
    if not tr_text or not en_text:
        continue
    # Ensure no contamination with val/test
    if tr_text in val_sources or tr_text in test_sources:
        continue
    train_records.append({"source": tr_text, "target": en_text})
    if len(train_records) == 20000:
        break

df_train = pd.DataFrame(train_records)
df_train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, encoding="utf-8")
print(f"Train saved: {len(df_train)} pairs.")

# Statistics
print("\n=== Dataset Statistics ===")
for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    src_lens = df["source"].apply(lambda s: len(str(s).split()))
    tgt_lens = df["target"].apply(lambda s: len(str(s).split()))
    print(f"{name}: {len(df)} pairs | Source word count: mean={src_lens.mean():.1f}, min={src_lens.min()}, max={src_lens.max()} | Target word count: mean={tgt_lens.mean():.1f}, min={tgt_lens.min()}, max={tgt_lens.max()}")

print("\nSample 1-3 from Test:")
for i in range(3):
    print(f"Test #{i+1}:")
    print(f"  TR: {df_test.iloc[i]['source']}")
    print(f"  EN: {df_test.iloc[i]['target']}")
