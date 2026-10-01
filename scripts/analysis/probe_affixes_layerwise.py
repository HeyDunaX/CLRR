"""Layer-Wise Linear Probing for Amis Morphological Affixes.

Trains linear probes (Logistic Regression) on intermediate hidden states
across encoder depth (Layers 1 to N) to predict the presence of verbal affixes:
- ma- (Patient/Stative Voice)
- mi- (Actor Voice)
- pa- (Causative Prefix)

Compares Vanilla Baseline vs CLRR-Enc to test the hypothesis:
"CLRR prevents over-smoothing and preserves localized morphosyntactic affixes
at deep encoder layers where vanilla representations degrade."
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class SimpleTextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def extract_layerwise_hidden_states(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_length: int = 256,
) -> list[np.ndarray]:
    """Extracts mean-pooled hidden states for every layer from 0 to N.

    Returns:
        A list of numpy arrays, where index l contains [N_samples, H] pooled features.
    """
    model.eval()
    model.to(device)
    dataset = SimpleTextDataset(texts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    encoder = model.get_encoder()
    all_layer_pools: list[list[np.ndarray]] = []

    with torch.no_grad():
        for batch in loader:
            inputs = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = encoder(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (N_layers + 1) tensors [B, L, H]

            if not all_layer_pools:
                all_layer_pools = [[] for _ in range(len(hidden_states))]

            mask = inputs["attention_mask"].unsqueeze(-1)  # [B, L, 1]
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)

            for l_idx, h in enumerate(hidden_states):
                pooled = (h * mask).sum(dim=1) / sum_mask  # [B, H]
                all_layer_pools[l_idx].append(pooled.cpu().numpy())

    return [np.vstack(layer_batches) for layer_batches in all_layer_pools]


def evaluate_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[float, float]:
    """Evaluates a linear probe using Stratified K-Fold cross validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    f1s = []

    for train_idx, test_idx in skf.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(max_iter=200, random_state=seed, solver="lbfgs", C=1.0)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, zero_division=0))

    return float(np.mean(accs)), float(np.mean(f1s))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint directory or HF ID.")
    parser.add_argument("--model-label", required=True, help="Label for this model (e.g. Vanilla_Baseline or CLRR_Enc).")
    parser.add_argument("--data-csv", default="data/processed/test.csv")
    parser.add_argument("--output-dir", default="outputs_rebuttal")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_test = pd.read_csv(args.data_csv)
    sources = [str(s).strip() for s in df_test["source"]]

    # Affix labels
    has_ma = np.array([1 if re.search(r"\b(ma|Ma)[a-z']+", s) else 0 for s in sources])
    has_mi = np.array([1 if re.search(r"\b(mi|Mi)[a-z']+", s) else 0 for s in sources])
    has_pa = np.array([1 if re.search(r"\b(pa|Pa)[a-z']+", s) else 0 for s in sources])

    affix_tasks = {
        "ma_stative": has_ma,
        "mi_actor": has_mi,
        "pa_causative": has_pa,
    }

    print(f"[Init] Loading {args.model_label} from {args.model_path}...", flush=True)
    try:
        from scripts.analysis.load_helper import load_eval_model_and_tokenizer
    except ImportError:
        from load_helper import load_eval_model_and_tokenizer
    model, tokenizer = load_eval_model_and_tokenizer(args.model_path)

    print(f"[Extract] Extracting layer-wise hidden states for {len(sources)} test sentences...", flush=True)
    layer_states = extract_layerwise_hidden_states(model, tokenizer, sources, batch_size=args.batch_size)
    num_layers = len(layer_states) - 1
    print(f"[Extract] Extracted {len(layer_states)} layers (Layer 0 = embedding, Layers 1..{num_layers} = blocks).", flush=True)

    records = []
    print(f"\n{'='*75}", flush=True)
    print(f"LAYER-WISE AFFIX PROBING: {args.model_label.upper()}", flush=True)
    print(f"{'='*75}", flush=True)

    for l_idx, feats in enumerate(layer_states):
        layer_name = f"Layer_{l_idx}" if l_idx > 0 else "Embedding"
        layer_f1s = []
        for task_name, targets in affix_tasks.items():
            acc, f1 = evaluate_probe(feats, targets)
            records.append({
                "Model": args.model_label,
                "Layer_Index": l_idx,
                "Layer_Name": layer_name,
                "Affix_Task": task_name,
                "Positive_Count": int(np.sum(targets)),
                "Total_Count": len(targets),
                "Accuracy": acc,
                "F1_Score": f1,
            })
            layer_f1s.append(f"{task_name[:2]}:{f1:.3f}")
        print(f"  * {layer_name:<11} | F1: {' | '.join(layer_f1s)}", flush=True)

    df_results = pd.DataFrame(records)
    out_csv = out_dir / f"layerwise_probing_{args.model_label.lower()}.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n[Done] Exported layer-wise probing results to {out_csv}\n", flush=True)


if __name__ == "__main__":
    main()
