"""Download pinned AmericasNLP parallel files and write source,target CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

REPOSITORY = "https://github.com/AmericasNLP/americasnlp2021"
REVISION = "d3f519c6b38299d8149311e65a369047350e6849"
FILES = {
    "train": ("data/ashaninka-spanish/train.cni", "data/ashaninka-spanish/train.es"),
    "validation": ("data/ashaninka-spanish/dev.cni", "data/ashaninka-spanish/dev.es"),
    "test": ("test_data/test.cni", "test_data/test.es"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data_processed/ashaninka_spanish"))
    args = parser.parse_args()
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "ashaninka_spanish", "source_language": "cni", "target_language": "es",
                "repository": REPOSITORY, "revision": REVISION, "raw_files": {}, "splits": {}}
    pairs_by_split = {}
    for split, paths in FILES.items():
        columns = []
        for path in paths:
            url = f"https://raw.githubusercontent.com/AmericasNLP/americasnlp2021/{REVISION}/{path}"
            with urlopen(Request(url, headers={"User-Agent": "CLRR-dataset-preparation"}), timeout=60) as response:
                payload = response.read()
            filename = Path(path).name
            (raw_dir / filename).write_bytes(payload)
            manifest["raw_files"][filename] = {"url": url, "sha256": hashlib.sha256(payload).hexdigest()}
            columns.append(payload.decode("utf-8-sig").splitlines())
        sources, targets = columns
        if len(sources) != len(targets):
            raise ValueError(f"Unaligned files for {split}: {len(sources)} != {len(targets)}")
        pairs, dropped = [], []
        for line_number, (source, target) in enumerate(zip(sources, targets), 1):
            source, target = source.strip(), target.strip()
            if not source or not target:
                dropped.append(line_number)
                continue
            pairs.append((source, target))
        if not pairs:
            raise ValueError(f"Empty split: {split}")
        output = args.output_dir / f"{split}.csv"
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["source", "target"])
            writer.writerows(pairs)
        pairs_by_split[split] = pairs
        manifest["splits"][split] = {
            "raw_rows": len(sources), "rows": len(pairs), "empty_pair_line_numbers": dropped,
            "duplicate_pair_rows": len(pairs) - len(set(pairs)),
            "duplicate_source_rows": len(pairs) - len({s for s, _ in pairs}),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
        print(f"{split}: {len(pairs)} pairs; dropped empty aligned pairs: {dropped}")
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        a, b = pairs_by_split[left], pairs_by_split[right]
        overlaps[f"{left}/{right}"] = {
            "exact_pairs": len(set(a) & set(b)),
            "exact_sources": len({s for s, _ in a} & {s for s, _ in b}),
        }
    manifest["cross_split_overlap"] = overlaps
    manifest["preprocessing"] = "UTF-8; strip outer whitespace; drop empty aligned pairs jointly; preserve official split and order. No synthetic data."
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("CROSS_SPLIT_OVERLAP", json.dumps(overlaps))


if __name__ == "__main__":
    main()
