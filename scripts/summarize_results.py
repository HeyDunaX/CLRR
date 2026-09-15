"""Aggregate multi-seed metrics.json files into mean and standard deviation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--save", default="results_summary.csv")
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.outputs_dir).glob("*/metrics.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {args.outputs_dir}")
    frame = pd.DataFrame(rows)
    metric_columns = ["test_loss", "test_bleu", "test_chrf++"]
    summary = frame.groupby(["model", "method"])[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        for column in summary.columns.to_flat_index()
    ]
    summary.to_csv(args.save, index=False)
    print(summary.to_string(index=False))
    print(f"Saved {args.save}")


if __name__ == "__main__":
    main()