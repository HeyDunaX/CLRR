"""Audit script for mBART baseline evaluation anomaly.

Investigates why mBART baseline achieves BLEU 19.61 but chrF++ 14.05,
verifying character vs word n-gram matching behavior on unsegmented Chinese.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import sacrebleu
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh


def run_audit() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    pred_dir = repo_root / "results" / "analysis" / "predictions"
    out_dir = repo_root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = pred_dir / "mbart-large-50-ami-cmn-baseline.csv"
    enc_path = pred_dir / "mbart-large-50-ami-cmn-jepa-clrr-enc.csv"

    if not base_path.exists() or not enc_path.exists():
        raise FileNotFoundError(f"Missing prediction files in {pred_dir}")

    df_base = pd.read_csv(base_path)
    df_enc = pd.read_csv(enc_path)

    refs = [str(t).strip() for t in df_base["target"]]
    preds_base = [str(p).strip() for p in df_base["prediction"]]
    preds_enc = [str(p).strip() for p in df_enc["prediction"]]

    tok_zh = TokenizerZh()
    seg_refs = [tok_zh(r) for r in refs]
    seg_preds_base = [tok_zh(p) for p in preds_base]
    seg_preds_enc = [tok_zh(p) for p in preds_enc]

    # Metrics
    bleu_base = sacrebleu.corpus_bleu(preds_base, [refs], tokenize="zh").score
    bleu_enc = sacrebleu.corpus_bleu(preds_enc, [refs], tokenize="zh").score

    chrf_w2_raw_base = sacrebleu.corpus_chrf(preds_base, [refs], word_order=2).score
    chrf_w2_raw_enc = sacrebleu.corpus_chrf(preds_enc, [refs], word_order=2).score

    chrf_w0_raw_base = sacrebleu.corpus_chrf(preds_base, [refs], word_order=0).score
    chrf_w0_raw_enc = sacrebleu.corpus_chrf(preds_enc, [refs], word_order=0).score

    chrf_w2_seg_base = sacrebleu.corpus_chrf(seg_preds_base, [seg_refs], word_order=2).score
    chrf_w2_seg_enc = sacrebleu.corpus_chrf(seg_preds_enc, [seg_refs], word_order=2).score

    report = {
        "num_test_sentences": len(refs),
        "mbart_baseline": {
            "bleu_zh": round(bleu_base, 4),
            "chrf_plus_plus_raw_w2": round(chrf_w2_raw_base, 4),
            "chrf_raw_w0": round(chrf_w0_raw_base, 4),
            "chrf_plus_plus_segmented_w2": round(chrf_w2_seg_base, 4),
        },
        "mbart_clrr_enc": {
            "bleu_zh": round(bleu_enc, 4),
            "chrf_plus_plus_raw_w2": round(chrf_w2_raw_enc, 4),
            "chrf_raw_w0": round(chrf_w0_raw_enc, 4),
            "chrf_plus_plus_segmented_w2": round(chrf_w2_seg_enc, 4),
        },
        "explanation": (
            "In sacrebleu chrF++ (word_order=2), word n-grams are extracted via whitespace splitting. "
            "Because Chinese sentences have no inter-word whitespace, unsegmented sentences yield zero "
            "word bigram matches, heavily deflating raw chrF++ to ~14.05. When segmented into Chinese words, "
            "chrF++ is 22.72 for baseline and 23.59 for CLRR-Enc, aligning with BLEU."
        ),
    }

    out_file = out_dir / "mbart_audit_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Audit completed successfully. Report saved to:", out_file)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run_audit()
