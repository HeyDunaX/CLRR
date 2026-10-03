"""Verify isolated audit files and manuscript drafts without changing old reports."""
from __future__ import annotations
import csv
import hashlib
import json
import re
from pathlib import Path
from amis_rewire.metrics import score_prediction_csv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs_rebuttal/metric_audit_20261002/full"

def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def main():
    rows = read_csv(OUT / "all_verified_translation_scores.csv")
    audit = json.loads((OUT / "translation_audit.json").read_text(encoding="utf-8"))
    assert len(rows) == len(audit["results"]) == 36
    hash_checks = []
    for row, record in zip(rows, audit["results"]):
        assert (row["dataset"], row["backbone"], row["method"]) == (
            record["dataset"], record["backbone"], record["method"])
        pred = ROOT / row["prediction_path"]
        ref = ROOT / "data_processed" / row["dataset"] / "test.csv"
        assert hashlib.sha256(pred.read_bytes()).hexdigest() == row["prediction_sha256"]
        assert hashlib.sha256(ref.read_bytes()).hexdigest() == row["reference_sha256"]
        got = score_prediction_csv(pred, ref, row["bleu_tokenizer"])
        for metric, col in [("bleu", "bleu"), ("chrf++", "chrfpp")]:
            assert abs(got[metric] - float(row[col])) < 1e-10
            assert abs(float(row[col]) - record[col]) < 1e-10
        hash_checks.append({
            "backbone": row["backbone"], "method": row["method"],
            "dataset": row["dataset"], "predictions_unchanged": True,
            "references_unchanged": True, "score_recomputed": True,
        })
    stats = read_csv(OUT / "paired_bootstrap_standardized.csv")
    assert len(stats) == 42
    adjusted = 0.0
    for i, row in enumerate(sorted(stats, key=lambda r: float(r["p_value"]))):
        adjusted = max(adjusted, min(1.0, (42-i)*float(row["p_value"])))
        assert abs(adjusted-float(row["p_value_holm"])) < 1e-12
        assert float(row["delta_ci_low"]) <= float(row["delta_ci_high"])

    isolated = OUT / "isolated_reports"
    text = (isolated / "docs/clrr_main.tex").read_text(encoding="utf-8")
    assert (isolated / "docs/clrr_main.tex").read_bytes() == (
        isolated / "docs/CLRR-paper/latex/clrr_main.tex").read_bytes()
    clean = re.sub(r"(?m)(?<!\\)%.*", "", text)
    labels = re.findall(r"\\label\{([^}]+)\}", clean)
    refs = re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", clean)
    assert not set(refs)-set(labels)
    assert len(labels) == len(set(labels))
    stack = []
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", clean):
        if match[1] == "begin":
            stack.append(match[2])
        else:
            assert stack and stack.pop() == match[2]
    assert not stack
    balance = 0
    for match in re.finditer(r"(?<!\\)[{}]", clean):
        balance += 1 if match[0] == "{" else -1
        assert balance >= 0
    assert balance == 0

    start = text.index(r"\label{tab:main_comparative_matrix}")
    main_table = text[start:text.index(r"\end{table*}", start)]
    for row in rows:
        if row["dataset"] == "amis_mandarin" and row["backbone"] in ["mBART", "NLLB"] and row["method"] in [
            "Baseline", "CLRR+LSR", "CLRR-Dec+LSR", "Middle-Layer Alignment", "BitFit",
            "Narrow LoRA", "Strong LoRA A", "Strong LoRA B",
        ]:
            for col in ["bleu", "chrfpp"]:
                assert f'{float(row[col]):.4f}' in main_table

    start = text.index(r"\label{tab:bootstrap_details}")
    bootstrap_tex = text[start:text.index(r"\end{table*}", start)]
    checked_rows = 0
    for line in bootstrap_tex.splitlines():
        if "&" not in line or not line.rstrip().endswith(r"\\") or line.startswith(r"\textbf"):
            continue
        parts = [p.strip() for p in line.replace(r"\\", "").split("&")]
        if len(parts) != 7 or not parts[0].startswith(("mBART ", "NLLB ", "mT5 ")):
            continue
        bb, baseline = parts[0].split(" ", 1)
        challenger, metric = parts[1:3]
        matches = [r for r in stats if r["dataset"] == "amis_mandarin" and r["backbone"] == bb
                   and r["baseline"] == baseline and r["challenger"] == challenger and r["metric"] == metric]
        assert len(matches) == 1, parts
        row = matches[0]
        expected = [f'{float(row["delta"]):.4f}',
                    f'[{float(row["delta_ci_low"]):.4f}, {float(row["delta_ci_high"]):.4f}]',
                    f'{float(row["p_value"]):.4f}', f'{float(row["p_value_holm"]):.4f}']
        assert parts[3:] == expected, (parts, expected)
        checked_rows += 1
    assert checked_rows == 12

    receipt = {
        "verified_outputs": 36, "paired_tests": 42, "tex_scope": "ISOLATED_AUDIT_DRAFTS",
        "unchanged_prediction_reference_hashes": hash_checks,
        "canonical_csv_matches_audit": True, "all_scores_recomputed": True,
        "holm_recomputed": True, "tex_copies_identical": True,
        "tex_environment_brace_and_references_checked": True,
        "tex_primary_and_bootstrap_values_checked": True, "bootstrap_tex_rows_checked": checked_rows,
        "pdf_compilation": "NOT_RUN_NO_LOCAL_COMPILER",
        "training_or_weight_loading": False,
    }
    (OUT / "final_full_verification.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k != "unchanged_prediction_reference_hashes"}))

if __name__ == "__main__":
    main()
