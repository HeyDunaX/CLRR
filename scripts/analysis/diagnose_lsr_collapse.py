"""LSR Representation Anti-Collapse Diagnostic Script.

Computes empirical representation metrics on the test set (575 samples):
1. Singular Value Decomposition (SVD) spectrum of pooled encoder states.
2. Effective Rank (Roy & Vetterli, 2007: exp of normalized singular value entropy).
3. Mean Pairwise Cosine Similarity (checks for dimensional collapse toward a single ray).
4. Variance / Covariance trace across batches.

Answers reviewer critiques regarding mathematical vs empirical guarantee against collapse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def compute_effective_rank(singular_values: np.ndarray) -> float:
    """Computes Effective Rank = exp(Entropy(p_i)) where p_i = s_i / sum(s)."""
    s_sum = np.sum(singular_values)
    if s_sum == 0:
        return 0.0
    p = singular_values / s_sum
    # Filter out zeros for entropy calculation
    p_nonzero = p[p > 1e-12]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    return float(np.exp(entropy))


def compute_pairwise_cosine(representations: np.ndarray) -> tuple[float, float]:
    """Computes mean and std of pairwise cosine similarities among normalized vectors."""
    norms = np.linalg.norm(representations, axis=1, keepdims=True)
    normed = representations / np.maximum(norms, 1e-12)
    cos_sim_matrix = np.dot(normed, normed.T)
    # Extract upper triangle excluding diagonal
    n = len(normed)
    triu_indices = np.triu_indices(n, k=1)
    pairwise_cos = cos_sim_matrix[triu_indices]
    return float(np.mean(pairwise_cos)), float(np.std(pairwise_cos))


def extract_encoder_representations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_length: int = 256,
) -> np.ndarray:
    """Extracts mean-pooled encoder representations for a list of texts."""
    model.eval()
    model.to(device)
    dataset = TextDataset(texts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    pooled_reps = []

    with torch.no_grad():
        for batch in loader:
            inputs = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            encoder = model.get_encoder()
            outputs = encoder(**inputs)
            hidden_states = outputs.last_hidden_state  # [B, L, H]

            # Masked mean-pooling
            mask = inputs["attention_mask"].unsqueeze(-1)  # [B, L, 1]
            sum_hidden = (hidden_states * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask  # [B, H]

            pooled_reps.append(pooled.cpu().numpy())

    return np.vstack(pooled_reps)


def run_diagnostics(
    representations: np.ndarray,
    label: str,
) -> dict[str, Any]:
    """Performs SVD and rank diagnostics on an extracted representation matrix."""
    n_samples, hidden_dim = representations.shape
    print(f"[{label}] Representation shape: {n_samples} samples x {hidden_dim} dimensions")

    # Center representations
    centered = representations - np.mean(representations, axis=0, keepdims=True)

    # SVD
    _, s, _ = np.linalg.svd(centered, full_matrices=False)

    effective_rank = compute_effective_rank(s)
    mean_cos, std_cos = compute_pairwise_cosine(representations)
    condition_number = float(s[0] / max(s[-1], 1e-12))
    variance_explained_top1 = float((s[0] ** 2) / np.sum(s ** 2) * 100)
    variance_explained_top10 = float(np.sum(s[:10] ** 2) / np.sum(s ** 2) * 100)

    print(f"[{label}] Effective Rank:                 {effective_rank:.2f} / {min(n_samples, hidden_dim)}")
    print(f"[{label}] Mean Pairwise Cosine:          {mean_cos:.4f} +/- {std_cos:.4f}")
    print(f"[{label}] Top-1 Singular Value Variance:  {variance_explained_top1:.2f}%")
    print(f"[{label}] Top-10 Singular Value Variance: {variance_explained_top10:.2f}%")
    print(f"[{label}] Condition Number (s_0/s_min):   {condition_number:.2e}")

    # Check for collapse
    # A collapsed representation has Effective Rank near 1.0 and Top-1 variance near 100%
    is_collapsed = (effective_rank < 3.0) or (mean_cos > 0.98 and variance_explained_top1 > 90.0)
    print(f"[{label}] Dimensional Collapse Detected:   {'YES (COLLAPSED)' if is_collapsed else 'NO (DIVERSE MANIFOLD)'}")

    return {
        "label": label,
        "n_samples": n_samples,
        "hidden_dim": hidden_dim,
        "effective_rank": effective_rank,
        "effective_rank_ratio": float(effective_rank / min(n_samples, hidden_dim)),
        "mean_pairwise_cosine": mean_cos,
        "std_pairwise_cosine": std_cos,
        "top1_variance_percent": variance_explained_top1,
        "top10_variance_percent": variance_explained_top10,
        "condition_number": condition_number,
        "is_collapsed": is_collapsed,
        "singular_values": s[:50].tolist(),  # First 50 singular values
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint directory or HF hub ID.")
    parser.add_argument("--data-csv", default="data/processed/test.csv", help="Path to test CSV.")
    parser.add_argument("--output-dir", default="outputs_rebuttal", help="Output directory for reports.")
    parser.add_argument("--label", default="LSR_Model", help="Label for this model evaluation.")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_test = pd.read_csv(args.data_csv)
    source_texts = [str(s).strip() for s in df_test["source"]]
    target_texts = [str(t).strip() for t in df_test["target"]]

    print(f"[Init] Loading model and tokenizer from {args.model_path}...")
    try:
        from scripts.analysis.load_helper import load_eval_model_and_tokenizer
    except ImportError:
        from load_helper import load_eval_model_and_tokenizer
    model, tokenizer = load_eval_model_and_tokenizer(args.model_path)

    print(f"[Extract] Extracting source encoder representations (Amis)...")
    source_reps = extract_encoder_representations(model, tokenizer, source_texts, batch_size=args.batch_size)
    source_diag = run_diagnostics(source_reps, f"{args.label}_Source_Amis")

    print(f"\n[Extract] Extracting target encoder representations (Mandarin)...")
    target_reps = extract_encoder_representations(model, tokenizer, target_texts, batch_size=args.batch_size)
    target_diag = run_diagnostics(target_reps, f"{args.label}_Target_Mandarin")

    report = {
        "model_path": args.model_path,
        "source_diagnostics": source_diag,
        "target_diagnostics": target_diag,
    }

    report_path = out_dir / f"lsr_anti_collapse_{args.label.lower()}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save singular value CSV
    max_sv = max(len(source_diag["singular_values"]), len(target_diag["singular_values"]))
    df_sv = pd.DataFrame({
        "Rank_i": list(range(1, max_sv + 1)),
        "Source_Amis_Singular_Value": source_diag["singular_values"] + [None]*(max_sv - len(source_diag["singular_values"])),
        "Target_Mandarin_Singular_Value": target_diag["singular_values"] + [None]*(max_sv - len(target_diag["singular_values"])),
    })
    sv_path = out_dir / f"lsr_singular_values_{args.label.lower()}.csv"
    df_sv.to_csv(sv_path, index=False)

    print(f"\n[Done] Exported diagnostic report to {report_path}")
    print(f"[Done] Exported singular values to {sv_path}\n")


if __name__ == "__main__":
    main()
