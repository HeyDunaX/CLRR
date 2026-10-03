"""Leave-one-row-out influence diagnostics; retain the complete primary test set."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "outputs_rebuttal/metric_audit_20261002/full"
OUT = ROOT / "outputs_rebuttal/easy_diagnostics_20261002"


def read(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    verified = {(r["dataset"],r["backbone"],r["method"]):r for r in read(AUDIT/"all_verified_translation_scores.csv")}
    comparisons = [("mBART","Baseline","CLRR+LSR"),("mBART","LSR-only","CLRR+LSR"),
                   ("NLLB","Baseline","CLRR+LSR"),("mT5","Baseline","CLRR+LSR")]
    raw = read(ROOT/"data_processed/amis_mandarin/test.csv")
    refs = [r["target"].strip() for r in raw]
    summaries, details, api_checks = [],[],[]
    for backbone,base,challenger in comparisons:
        pair = []
        for method in (base,challenger):
            row = verified[("amis_mandarin",backbone,method)]
            assert digest(ROOT/row["prediction_path"]) == row["prediction_sha256"]
            assert digest(ROOT/"data_processed/amis_mandarin/test.csv") == row["reference_sha256"]
            predictions = read(ROOT/row["prediction_path"])
            assert len(predictions)==len(raw)
            assert [r["source"] for r in predictions]==[r["source"] for r in raw]
            pair.append([r["prediction"].strip() for r in predictions])
        for word_order in (0,2):
            metric = CHRF(word_order=word_order)
            stats = [np.asarray(metric._extract_corpus_statistics(p,[refs]),dtype="float64") for p in pair]
            totals = [s.sum(axis=0) for s in stats]
            scores = [metric._compute_score_from_stats(t).score for t in totals]
            delta = scores[1]-scores[0]
            leave_out = np.asarray([
                metric._compute_score_from_stats(totals[1]-stats[1][i]).score -
                metric._compute_score_from_stats(totals[0]-stats[0][i]).score for i in range(len(refs))])
            reduction = delta-leave_out
            minimum_index = int(np.argmin(leave_out))
            maximum_index = int(np.argmax(leave_out))
            summary = dict(backbone=backbone,baseline=base,challenger=challenger,word_order=word_order,
                full_test_delta=delta,leave_one_out_min_delta=float(leave_out.min()),
                leave_one_out_max_delta=float(leave_out.max()),
                removed_index_at_min_delta=minimum_index,removed_index_at_max_delta=maximum_index,
                nonpositive_deltas_count=int((leave_out<=0).sum()),rows=len(refs))
            summaries.append(summary)
            for i,value in enumerate(leave_out):
                details.append(dict(backbone=backbone,baseline=base,challenger=challenger,word_order=word_order,
                                    removed_index=i,leave_one_out_delta=float(value),delta_reduction=float(reduction[i])))
            # Verify the sufficient-statistic shortcut at the most influential row.
            i = minimum_index
            short_refs = refs[:i]+refs[i+1:]
            direct = [metric.corpus_score(p[:i]+p[i+1:],[short_refs]).score for p in pair]
            error = abs((direct[1]-direct[0])-leave_out[i])
            assert error < 1e-10
            api_checks.append(dict(backbone=backbone,baseline=base,word_order=word_order,
                                   removed_index=i,direct_corpus_max_error=error))
            print(f"[influence] {summary}",flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,table in [("influence_summary.csv",summaries),("influence_all_rows.csv",details)]:
        with (OUT/name).open("w",encoding="utf-8",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    (OUT/"influence_protocol.json").write_text(json.dumps({
        "scope":"Post-hoc sentence influence, not a new primary test or significance test",
        "rows":len(refs),"rule":"Every original test row removed once; primary scores retain all rows",
        "metrics":"chrF word_order=0 and chrF++ word_order=2, raw text, standard defaults",
        "warning":"Leave-one-out robustness does not establish training-seed robustness or morphology",
        "direct_api_checks":api_checks,"script_sha256":digest(__file__)},indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
