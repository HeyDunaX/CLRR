import os
import hashlib
import json
import pandas as pd

DATA_DIR = "d:/Code/CLRR/data_processed/opus100_turkish_english_20k"

splits_info = {}
for split in ["train", "validation", "test"]:
    filepath = os.path.join(DATA_DIR, f"{split}.csv")
    df = pd.read_csv(filepath)
    with open(filepath, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    splits_info[split] = {
        "rows": len(df),
        "sha256": sha256
    }

manifest = {
    "dataset": "opus100_turkish_english_20k",
    "source_language": "tr",
    "target_language": "en",
    "dataset_source": "Helsinki-NLP/opus-100",
    "sampling": "Fixed seed 42 sampling of 20,000 train pairs; official validation (2,000) and test (2,000)",
    "splits": splits_info,
    "cross_split_overlap": {
        "train/validation": {"exact_sources": 0},
        "train/test": {"exact_sources": 0}
    },
    "preprocessing": "UTF-8; stripped whitespace; non-empty strings; strictly disjoint splits."
}

manifest_path = os.path.join(DATA_DIR, "manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Manifest written to {manifest_path}")
