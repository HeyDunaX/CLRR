"""Unpack parallel.zip into data/processed/{train,validation,test}.csv.

Reads the aligned Amis and Chinese (Mandarin) text files from the ZIP archive
and writes CSV files with columns ``source,target`` for the training pipeline.
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path


SPLIT_MAP = {
    "train": ("parallel-data/ami.train", "parallel-data/cmn.train"),
    "validation": ("parallel-data/ami.dev", "parallel-data/cmn.dev"),
    "test": ("parallel-data/ami.test", "parallel-data/cmn.test"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-path",
        default="parallel.zip",
        help="Path to the parallel.zip archive (default: parallel.zip)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for output CSV files (default: data/processed)",
    )
    return parser.parse_args()


def read_lines(zf: zipfile.ZipFile, name: str) -> list[str]:
    """Read a text file from the ZIP and return stripped non-empty lines."""
    with zf.open(name) as f:
        return [line.strip() for line in f.read().decode("utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path)
    output_dir = Path(args.output_dir)

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset archive not found: {zip_path}.\n"
            "Please download the dataset from Google Drive:\n"
            "  https://drive.google.com/drive/folders/1W-glGBpCz9R16Oy-P96jdK7YGSVuvdg2\n"
            "and place parallel.zip in the workspace root or pass --zip-path."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        for split, (ami_name, cmn_name) in SPLIT_MAP.items():
            ami_lines = read_lines(zf, ami_name)
            cmn_lines = read_lines(zf, cmn_name)

            if len(ami_lines) != len(cmn_lines):
                raise ValueError(
                    f"Line count mismatch for {split}: "
                    f"{ami_name} has {len(ami_lines)} lines, "
                    f"{cmn_name} has {len(cmn_lines)} lines"
                )

            out_path = output_dir / f"{split}.csv"
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["source", "target"])
                for ami, cmn in zip(ami_lines, cmn_lines):
                    writer.writerow([ami, cmn])

            print(f"[data] {split}: {len(ami_lines)} examples -> {out_path}")

    print(f"[data] Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
