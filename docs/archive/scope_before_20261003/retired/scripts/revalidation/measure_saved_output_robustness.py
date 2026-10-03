"""Character-only chrF diagnostics from verified predictions; no training."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF
from sacrebleu.significance import PairedTest, _compute_p_value

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "outputs_rebuttal/metric_audit_20261002/full"
OUT = ROOT / "outputs_rebuttal/easy_diagnostics_20261002"
SAMPLES, SEED = 10000, 42


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(values):
    order = np.argsort(values)
    result = np.empty(len(values))
    running = 0.0
    for rank,index in enumerate(order):
        running = max(running,min(1.0,(len(values)-rank)*values[index]))
        result[index] = running
    return result


def main():
    rows = read_csv(AUDIT/"all_verified_translation_scores.csv")
    comparisons = [
        ("amis_mandarin","mBART","Baseline","CLRR+LSR"),
        ("amis_mandarin","mBART","LSR-only","CLRR+LSR"),
        ("amis_mandarin","mBART","Baseline","CLRR-only"),
        ("amis_mandarin","mBART","Baseline","LSR-only"),
        ("amis_mandarin","NLLB","Baseline","CLRR+LSR"),
        ("amis_mandarin","mT5","Baseline","CLRR+LSR"),
        ("amis_mandarin","ByT5","Baseline","CLRR+LSR"),
        ("ashaninka_spanish","mBART","Baseline (patience 4)","CLRR+LSR (patience 10, selected epoch 14)"),
    ]
    metric = CHRF(char_order=6,word_order=0,beta=2,lowercase=False,whitespace=False,eps_smoothing=False)
    systems,refs_by_dataset,provenance = {},{},[]
    needed = {(d,b,m) for d,b,a,c in comparisons for m in [a,c]}
    for row in rows:
        key = (row["dataset"],row["backbone"],row["method"])
        if key not in needed:
            continue
        reference_path = ROOT / "data_processed" / row["dataset"] / "test.csv"
        prediction_path = ROOT / row["prediction_path"]
        assert digest(reference_path) == row["reference_sha256"]
        assert digest(prediction_path) == row["prediction_sha256"]
        raw, prediction_rows = read_csv(reference_path), read_csv(prediction_path)
        assert len(raw) == len(prediction_rows)
        assert [r["source"] for r in raw] == [r["source"] for r in prediction_rows]
        assert [r["target"] for r in raw] == [r["target"] for r in prediction_rows]
        if "index" in prediction_rows[0]:
            assert [int(r["index"]) for r in prediction_rows] == list(range(len(raw)))
        refs = [r["target"].strip() for r in raw]
        preds = [r["prediction"].strip() for r in prediction_rows]
        stats = np.asarray(metric._extract_corpus_statistics(preds,[refs]),dtype="float32")
        score = metric.corpus_score(preds,[refs]).score
        sentence_scores = np.asarray([metric.sentence_score(p,[r]).score for p,r in zip(preds,refs)])
        systems[key] = dict(preds=preds,refs=refs,stats=stats,score=score,sentence_scores=sentence_scores)
        refs_by_dataset[row["dataset"]] = refs
        provenance.append({k:row[k] for k in ["dataset","backbone","method","prediction_path","prediction_sha256","reference_sha256"]})
    assert set(systems) == needed
    boot = {}
    for dataset,refs in refs_by_dataset.items():
        indices = np.random.default_rng(SEED).choice(len(refs),size=(SAMPLES,len(refs)),replace=True)
        for key,system in systems.items():
            if key[0] != dataset:
                continue
            samples = np.empty(SAMPLES)
            for start in range(0,SAMPLES,128):
                sums = system["stats"][indices[start:start+128]].sum(axis=1)
                samples[start:start+len(sums)] = [metric._compute_score_from_stats(s).score for s in sums]
            boot[key] = samples
            print(f"[chrF] {key}: {system['score']:.6f}",flush=True)
    tests = []
    for dataset,backbone,base,challenger in comparisons:
        a,b = systems[(dataset,backbone,base)],systems[(dataset,backbone,challenger)]
        delta = b["score"]-a["score"]
        sampled = boot[(dataset,backbone,challenger)]-boot[(dataset,backbone,base)]
        lo,hi = np.percentile(sampled,[2.5,97.5])
        absolute = np.abs(sampled)
        sentence_delta = b["sentence_scores"]-a["sentence_scores"]
        row = dict(dataset=dataset,backbone=backbone,baseline=base,challenger=challenger,
                   baseline_chrf=a["score"],challenger_chrf=b["score"],delta=delta,
                   delta_ci_low=float(lo),delta_ci_high=float(hi),
                   p_value=float(_compute_p_value(absolute-absolute.mean(),abs(delta))),
                   sentence_wins=int((sentence_delta>1e-10).sum()),sentence_losses=int((sentence_delta<-1e-10).sum()),
                   sentence_ties=int((np.abs(sentence_delta)<=1e-10).sum()))
        tests.append(row)
    prior = read_csv(AUDIT/"paired_bootstrap_standardized.csv")
    joint = holm([float(r["p_value"]) for r in prior]+[r["p_value"] for r in tests])
    for row,adjusted,joint_adjusted in zip(tests,holm([r["p_value"] for r in tests]),joint[len(prior):]):
        row["p_value_holm_8_diagnostics"] = float(adjusted)
        row["p_value_holm_50_combined_posthoc"] = float(joint_adjusted)
    os.environ["SACREBLEU_SEED"] = str(SEED)
    a,b = systems[comparisons[0][:2]+(comparisons[0][2],)],systems[comparisons[0][:2]+(comparisons[0][3],)]
    signatures,native = PairedTest(named_systems=[("Baseline",a["preds"]),("CLRR+LSR",b["preds"])],
        metrics={"chrF":CHRF(word_order=0)},references=[a["refs"]],test_type="bs",n_samples=SAMPLES,n_jobs=1)()
    native_values = next(v for k,v in native.items() if k != "System")
    assert abs(native_values[1].p_value-tests[0]["p_value"]) < 1e-12
    summary = []
    for (dataset,backbone,method),s in systems.items():
        ref_chars = sum(len("".join(r.split())) for r in s["refs"])
        summary.append(dict(dataset=dataset,backbone=backbone,method=method,rows=len(s["preds"]),chrf=s["score"],
            exact_matches=sum(p==r for p,r in zip(s["preds"],s["refs"])),
            empty_predictions=sum(not p for p in s["preds"]),
            character_length_ratio=sum(len("".join(p.split())) for p in s["preds"])/ref_chars))
    OUT.mkdir(parents=True,exist_ok=True)
    for name,table in [("character_chrf_bootstrap.csv",tests),("output_summary.csv",summary)]:
        with (OUT/name).open("w",encoding="utf-8",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    protocol = {"sacrebleu":sacrebleu.__version__,"signature":str(metric.get_signature()),
        "samples":SAMPLES,"seed":SEED,"unit":"paired original test rows",
        "references":"original CSV targets; strip outer whitespace only",
        "ci":"95% percentile CI of signed corpus chrF difference",
        "p":"SacreBLEU centered absolute-difference bootstrap",
        "families":"8 new diagnostic tests; also 50 = previous 42 plus new 8, all post-hoc",
        "status":"exploratory diagnostic; does not replace original BLEU/chrF++ results",
        "limitations":"fixed checkpoints, one training seed; no inference about training-seed variance or morphology",
        "native_api_check":{"p_value":native_values[1].p_value,"matches":True,
                            "signatures":{k:str(v) for k,v in signatures.items()}},
        "input_provenance":provenance,"script_sha256":digest(__file__)}
    (OUT/"robustness_protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    for row in tests:
        print(f"[result] {row}",flush=True)


if __name__ == "__main__":
    main()
