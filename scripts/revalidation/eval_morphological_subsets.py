"""Evaluation of translation accuracy across Amis morphological voice subsets.

Splits test set (575 pairs) into distinct morphosyntactic categories:
- mi- (Actor Voice)
- ma- (Patient / Stative Voice)
- pa- (Causative prefix)
- Simple / Root-only sentences
Computes BLEU and chrF++ per subset to quantify morphosyntactic preservation.
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import sacrebleu


def eval_morphology() -> pd.DataFrame:
    repo_root = Path(__file__).resolve().parents[2]
    pred_dir = repo_root / "results" / "analysis" / "predictions"
    out_dir = repo_root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_csv = repo_root / "data" / "processed" / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing {test_csv}")

    df_test = pd.read_csv(test_csv)
    sources = [str(s).strip() for s in df_test["source"]]
    targets = [str(t).strip() for t in df_test["target"]]

    # Define subsets
    subsets = {
        "Full Test Set": list(range(len(sources))),
        "mi- (Actor Voice)": [i for i, s in enumerate(sources) if re.search(r"\b(mi|Mi)[a-z']+", s)],
        "ma- (Patient/Stative)": [i for i, s in enumerate(sources) if re.search(r"\b(ma|Ma)[a-z']+", s)],
        "pa- (Causative)": [i for i, s in enumerate(sources) if re.search(r"\b(pa|Pa)[a-z']+", s)],
        "Root / Simple (No mi/ma/pa)": [
            i for i, s in enumerate(sources)
            if not re.search(r"\b(mi|Mi|ma|Ma|pa|Pa)[a-z']+", s)
        ],
    }

    # Load predictions
    models = {
        "mBART Baseline": pred_dir / "mbart-large-50-ami-cmn-baseline.csv",
        "mBART CLRR-Enc": pred_dir / "mbart-large-50-ami-cmn-jepa-clrr-enc.csv",
        "mT5 Baseline": pred_dir / "mt5-small-ami-cmn-baseline.csv",
        "mT5 CLRR-Enc": pred_dir / "mt5-small-ami-cmn-clrr-enc.csv",
    }

    model_preds = {}
    for mname, mpath in models.items():
        if mpath.exists():
            df_p = pd.read_csv(mpath)
            model_preds[mname] = [str(p).strip() for p in df_p["prediction"]]

    results = []

    for sname, indices in subsets.items():
        sub_refs = [targets[i] for i in indices]
        n_sentences = len(indices)

        row = {
            "Morphological Subset": sname,
            "N Sentences": n_sentences,
        }

        for mname, preds in model_preds.items():
            sub_preds = [preds[i] for i in indices]
            bleu = sacrebleu.corpus_bleu(sub_preds, [sub_refs], tokenize="zh").score
            chrf = sacrebleu.corpus_chrf(sub_preds, [sub_refs], word_order=2).score
            row[f"{mname} BLEU"] = round(bleu, 2)
            row[f"{mname} chrF++"] = round(chrf, 2)

        # Marginal gains for mBART and mT5
        if "mBART Baseline chrF++" in row and "mBART CLRR-Enc chrF++" in row:
            row["mBART Delta chrF++"] = round(row["mBART CLRR-Enc chrF++"] - row["mBART Baseline chrF++"], 2)

        results.append(row)

    res_df = pd.DataFrame(results)
    out_csv = out_dir / "morphological_breakdown.csv"
    res_df.to_csv(out_csv, index=False)

    print("Morphological Breakdown Evaluation:")
    print(res_df.to_string(index=False))
    print(f"\nSaved results to: {out_csv}")
    return res_df


if __name__ == "__main__":
    eval_morphology()
