"""Quantitative analysis of ByT5 generation dynamics and degradation.

Compares ByT5 Baseline, ByT5 LayerSkip, ByT5 Middle-Align, ByT5 CLRR-Enc, and ByT5 CLRR-Dec
to quantify degenerate repetition loops, length distortion, and character-level errors.
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import sacrebleu
from huggingface_hub import hf_hub_download


def detect_repeated_ngrams(text: str, n: int = 4) -> int:
    """Counts number of repeated n-character grams in text."""
    if len(text) < n * 2:
        return 0
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
    seen = set()
    repeats = 0
    for ng in ngrams:
        if ng in seen:
            repeats += 1
        else:
            seen.add(ng)
    return repeats


def analyze_byt5() -> pd.DataFrame:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "FiveC/amis-rewire-checkpoints"
    model_files = {
        "ByT5 Baseline": "analysis/predictions/byt5-small-ami-cmn-baseline.csv",
        "ByT5 LayerSkip (ACL 2024)": "comparative_baselines/byt5-small-layerskip-acl2024/test_predictions.csv",
        "ByT5 Middle-Align (ACL 2025)": "comparative_baselines/byt5-small-middle-align-acl2025/test_predictions.csv",
        "ByT5 CLRR-Enc (Ours)": "analysis/predictions/byt5-small-ami-cmn-jepa-clrr-enc.csv",
        "ByT5 CLRR-Dec (Ours)": "comparative_baselines/byt5-small-ami-cmn-jepa-clrr-dec/test_predictions.csv",
    }

    results = []

    for name, remote_path in model_files.items():
        local_path = hf_hub_download(repo_id=repo_id, filename=remote_path, repo_type="model")
        df = pd.read_csv(local_path)
        refs = [str(t).strip() for t in df["target"]]
        preds = [str(p).strip() for p in df["prediction"]]

        # Metrics
        bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="zh").score
        chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score

        ref_lens = [len(r) for r in refs]
        pred_lens = [len(p) for p in preds]
        avg_ref_len = sum(ref_lens) / len(ref_lens)
        avg_pred_len = sum(pred_lens) / len(pred_lens)
        len_ratio = avg_pred_len / avg_ref_len

        ufffd_count = sum(p.count("\ufffd") for p in preds)
        
        # Repetition metrics
        repeat_counts = [detect_repeated_ngrams(p, n=4) for p in preds]
        severe_loop_sentences = sum(1 for c in repeat_counts if c >= 3)
        loop_percentage = (severe_loop_sentences / len(preds)) * 100.0
        avg_repeat_4grams = sum(repeat_counts) / len(preds)

        results.append({
            "Method": name,
            "BLEU (zh)": round(bleu, 2),
            "chrF++ (w=2)": round(chrf, 2),
            "Avg Char Length": round(avg_pred_len, 2),
            "Length Ratio": round(len_ratio, 3),
            "Severe Loop Sentences (%)": round(loop_percentage, 1),
            "Avg Repeated 4-grams": round(avg_repeat_4grams, 2),
            "Invalid UTF-8 (\\ufffd)": ufffd_count,
        })

    res_df = pd.DataFrame(results)
    out_csv = out_dir / "byt5_utf8_corruption_stats.csv"
    res_df.to_csv(out_csv, index=False)

    print("ByT5 Analysis Results:")
    print(res_df.to_string(index=False))
    print(f"\nSaved results to: {out_csv}")
    return res_df


if __name__ == "__main__":
    analyze_byt5()
