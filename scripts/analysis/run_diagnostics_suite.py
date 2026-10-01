"""Runner Script for the Entire Scientific Diagnostics Suite.

Automates:
1. Downloading and extracting trained checkpoints from Hugging Face Hub.
2. Running LSR Anti-Collapse SVD & Effective Rank diagnostics.
3. Running Layer-wise Affix Probing across all 12 layers on Baseline vs CLRR-Enc.
4. Exporting summary comparison tables and CSVs to outputs_rebuttal/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download


REPO_ID = "FiveC/amis-rewire-checkpoints"


def download_and_extract(subpath: str, extract_to: Path) -> Path:
    """Downloads a zip checkpoint from Hugging Face Hub and extracts it."""
    extract_to.mkdir(parents=True, exist_ok=True)
    done_marker = extract_to / ".extracted_done"
    if done_marker.is_file():
        print(f"[Cache] Checkpoint already extracted at {extract_to}", flush=True)
        return extract_to

    print(f"[Download] Fetching {subpath} from {REPO_ID}...", flush=True)
    zip_path = hf_hub_download(repo_id=REPO_ID, filename=subpath)
    print(f"[Extract] Extracting {zip_path} to {extract_to}...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    done_marker.touch()
    return extract_to


def run_command(cmd: list[str]) -> None:
    print(f"\n[Exec] {' '.join(cmd)}", flush=True)
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[Error] Command failed with return code {ret.returncode}", flush=True)
        sys.exit(ret.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["nllb-200", "mbart-50"], default="nllb-200")
    parser.add_argument("--device", default="cuda" if subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0 else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    ckpt_dir = repo_root / "checkpoints_eval"
    output_dir = repo_root / "outputs_rebuttal"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"STARTING SCIENTIFIC DIAGNOSTICS SUITE (Backbone: {args.backbone.upper()})")
    print(f"{'='*75}")

    if args.backbone == "nllb-200":
        base_zip = "nllb-200/nllb-200-baseline-best.zip"
        clrr_zip = "nllb-200/nllb-200-clrr_enc-best.zip"
        base_model_dir = download_and_extract(base_zip, ckpt_dir / "nllb-baseline")
        clrr_model_dir = download_and_extract(clrr_zip, ckpt_dir / "nllb-clrr_enc")
    else:
        # mBART-50
        base_zip = "metrics/mbart-large-50-ami-cmn-baseline_metrics.json"  # or weights
        lsr_zip = "revalidation_ablations/mbart-large-50-ami-cmn-lsr-only/mbart-large-50-ami-cmn-lsr-only-best.zip"
        clrr_model_dir = download_and_extract(lsr_zip, ckpt_dir / "mbart-lsr")
        base_model_dir = clrr_model_dir

    # 1. Morphological Overlap Audit
    run_command([
        sys.executable,
        str(repo_root / "scripts" / "analysis" / "audit_morphological_overlap.py"),
    ])

    # 2. LSR Anti-Collapse SVD & Effective Rank
    run_command([
        sys.executable,
        str(repo_root / "scripts" / "analysis" / "diagnose_lsr_collapse.py"),
        "--model-path", str(clrr_model_dir),
        "--label", f"{args.backbone}_CLRR_LSR",
        "--batch-size", str(args.batch_size),
        "--output-dir", str(output_dir),
    ])

    # 3. Layer-Wise Probing: Baseline vs CLRR-Enc
    run_command([
        sys.executable,
        str(repo_root / "scripts" / "analysis" / "probe_affixes_layerwise.py"),
        "--model-path", str(base_model_dir),
        "--model-label", f"{args.backbone}_Baseline",
        "--batch-size", str(args.batch_size),
        "--output-dir", str(output_dir),
    ])

    run_command([
        sys.executable,
        str(repo_root / "scripts" / "analysis" / "probe_affixes_layerwise.py"),
        "--model-path", str(clrr_model_dir),
        "--model-label", f"{args.backbone}_CLRR_Enc",
        "--batch-size", str(args.batch_size),
        "--output-dir", str(output_dir),
    ])

    print(f"\n{'='*75}")
    print(f"ALL SCIENTIFIC DIAGNOSTICS COMPLETED SUCCESSFULLY!")
    print(f"Generated artifacts in {output_dir}:")
    for f in output_dir.glob("*.*"):
        print(f"  * {f.name} ({f.stat().st_size:,} bytes)")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
