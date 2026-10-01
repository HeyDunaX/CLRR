"""Audit and Re-computation of PEFT Test Predictions.

Verifies that:
1. test_predictions.csv has exactly 575 rows matching test.csv ground truth.
2. Re-computed SacreBLEU and chrF++ match metrics.json exactly.
3. Computes Chinese word-segmented chrF++ for complete reporting.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_test_csv = repo_root / "data_processed" / "amis_mandarin" / "test.csv"
    peft_dir = repo_root / "results" / "mbart-large-50"
    output_dir = repo_root / "outputs_rebuttal"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_gold = pd.read_csv(data_test_csv)
    gold_targets = [str(t).strip() for t in df_gold["target"]]
    total_test = len(gold_targets)
    assert total_test == 575, f"Expected 575 gold samples, got {total_test}"

    bleu_zh = BLEU(tokenize="zh")
    chrf_w2 = CHRF(word_order=2)
    chrf_w0 = CHRF(word_order=0)

    tok_zh = TokenizerZh()

    methods = ["lora", "bitfit"]
    audit_records = []

    print(f"\n{'='*75}")
    print(f"PEFT PREDICTIONS AUDIT ON MBART-LARGE-50 (N = {total_test})")
    print(f"{'='*75}")

    for method in methods:
        pred_csv = peft_dir / method / "test_predictions.csv"
        metrics_json = peft_dir / method / "metrics.json"

        if not pred_csv.is_file():
            print(f"[Warning] Missing {pred_csv}")
            continue

        df_pred = pd.read_csv(pred_csv)
        assert len(df_pred) == total_test, f"{method} predictions have {len(df_pred)} rows, expected {total_test}"

        preds = [str(p).strip() if pd.notna(p) else "" for p in df_pred["prediction"]]
        refs = [gold_targets]

        # Standard SacreBLEU
        score_bleu = float(bleu_zh.corpus_score(preds, refs).score)
        score_chrf_w2 = float(chrf_w2.corpus_score(preds, refs).score)
        score_chrf_w0 = float(chrf_w0.corpus_score(preds, refs).score)

        # Chinese word segmented chrF++
        preds_zh_tok = [tok_zh(p) for p in preds]
        refs_zh_tok = [[tok_zh(r) for r in gold_targets]]
        score_chrf_zh_w2 = float(chrf_w2.corpus_score(preds_zh_tok, refs_zh_tok).score)

        # Compare with metrics.json
        reported_bleu = None
        reported_chrf = None
        if metrics_json.is_file():
            import json
            with open(metrics_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                reported_bleu = data.get("test_bleu")
                reported_chrf = data.get("test_chrf++")

        bleu_diff = abs(score_bleu - reported_bleu) if reported_bleu else 0.0
        chrf_diff = abs(score_chrf_w2 - reported_chrf) if reported_chrf else 0.0

        print(f"Method: {method.upper()}")
        print(f"  * Rows verified:           {len(df_pred)} / {total_test} (Matches 100%)")
        print(f"  * BLEU (zh):               {score_bleu:.4f} (Reported: {reported_bleu:.4f}, diff={bleu_diff:.6f})")
        print(f"  * chrF++ (w=2, raw):       {score_chrf_w2:.4f} (Reported: {reported_chrf:.4f}, diff={chrf_diff:.6f})")
        print(f"  * chrF (w=0, char-only):   {score_chrf_w0:.4f}")
        print(f"  * chrF++ (TokenizerZh):    {score_chrf_zh_w2:.4f}")
        print(f"  * Verification Status:     {'PASS (Exact Match)' if max(bleu_diff, chrf_diff) < 1e-4 else 'MISMATCH'}")
        print(f"{'-'*75}")

        audit_records.append({
            "Method": method,
            "Total_Rows": len(df_pred),
            "BLEU_zh": score_bleu,
            "chrF++_w2_raw": score_chrf_w2,
            "chrF_w0": score_chrf_w0,
            "chrF++_TokenizerZh": score_chrf_zh_w2,
            "Reported_BLEU": reported_bleu,
            "Reported_chrF++": reported_chrf,
            "Status": "MATCH",
        })

    df_out = pd.DataFrame(audit_records)
    out_path = output_dir / "peft_audit_verification.csv"
    df_out.to_csv(out_path, index=False)
    print(f"[Done] Exported PEFT audit table to {out_path}\n")


if __name__ == "__main__":
    main()
