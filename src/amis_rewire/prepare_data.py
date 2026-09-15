"""Validate and normalize an Amis-Chinese CSV into train/validation/test CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--source-col", default="amis")
    parser.add_argument("--target-col", default="chinese")
    parser.add_argument("--split-col", default="split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv, encoding="utf-8")
    required = {args.source_col, args.target_col, args.split_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Found: {list(frame.columns)}")
    frame = frame[[args.source_col, args.target_col, args.split_col]].copy()
    frame.columns = ["source", "target", "split"]
    frame = frame.dropna().astype({"source": str, "target": str, "split": str})
    frame["source"] = frame["source"].str.strip()
    frame["target"] = frame["target"].str.strip()
    frame = frame[(frame.source != "") & (frame.target != "")]
    frame["split"] = frame["split"].str.lower().replace({"valid": "validation", "dev": "validation"})
    allowed = {"train", "validation", "test"}
    if not set(frame["split"]).issubset(allowed):
        raise ValueError(f"split must contain only {sorted(allowed)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in sorted(allowed):
        split_frame = frame[frame["split"] == split][["source", "target"]]
        if split_frame.empty:
            raise ValueError(f"No examples found for split={split}")
        split_frame.to_csv(output_dir / f"{split}.csv", index=False, encoding="utf-8")
        print(f"[data] {split}: {len(split_frame)} examples -> {output_dir / (split + '.csv')}")


if __name__ == "__main__":
    main()