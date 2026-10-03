"""Export audited table copies to an isolated folder; preserve old reports."""
from __future__ import annotations
import csv
import hashlib
import json
import shutil
from pathlib import Path
from sacrebleu.metrics import BLEU, CHRF

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "outputs_rebuttal/metric_audit_20261002/full"

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))

def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def backup(path):
    target = AUDIT / "files_before" / path.relative_to(ROOT)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

def main():
    scores = read_csv(AUDIT / "all_verified_translation_scores.csv")
    lookup = {(r["backbone"], r["method"]): r for r in scores if r["dataset"] == "amis_mandarin"}
    canonical = AUDIT / "isolated_reports/results/metrics_standardized_scores.csv"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(AUDIT / "all_verified_translation_scores.csv", canonical)
    changed = [canonical.relative_to(ROOT).as_posix()]
    methods = {
        "Standard Fine-Tuning": "Baseline", "CLRR-Enc (Ours)": "CLRR-only",
        "JEPA + CLRR-Enc (Ours)": "CLRR+LSR", "CLRR-Enc + LSR (Ours)": "CLRR+LSR",
        "JEPA + CLRR-Dec (Ours)": "CLRR-Dec+LSR", "LORA": "Narrow LoRA", "BITFIT": "BitFit",
    }
    def backbone(name):
        if "mbart" in name: return "mBART"
        if "byt5" in name: return "ByT5"
        if "mt5" in name: return "mT5"
        if "nllb" in name: return "NLLB"
        raise ValueError(name)

    paths = [
        "results/all_results.csv", "results/comparative_scores.csv",
        "results/full_comparative_matrix_acl.csv", "results/peft_scores.csv",
        "outputs_comparative/comparative_scores.csv", "outputs_comparative/full_comparative_matrix_acl.csv",
    ]
    for rel in paths:
        path = ROOT / rel
        if not path.exists(): continue
        backup(path)
        rows = read_csv(path)
        fields = list(rows[0])
        if rel.endswith("all_results.csv") and "extra_params" not in fields:
            fields.append("extra_params")
        for r in rows:
            model = r.get("backbone") or r.get("model") or r.get("Model")
            method = r.get("method") or r.get("Method")
            bb, mm = backbone(model), methods.get(method, method)
            v = lookup[bb, mm]
            r["bleu" if "bleu" in r else "BLEU (zh)"] = v["bleu"]
            r["chrf++" if "chrf++" in r else "chrF++ (w=2)"] = v["chrfpp"]
            if "status" in r: r["status"] = "VERIFIED_RAW_REFERENCES_20261002"
            if "trainable_params" in r:
                # Old zeros meant added params, not trainable params.
                if bb == "mBART":
                    r["trainable_params"] = {"BitFit":335872,"Narrow LoRA":1179648}.get(mm,610879488)
                else:
                    r["trainable_params"] = ""  # No independently audited count for these historical rows.
                r["extra_params"] = 1179648 if mm == "Narrow LoRA" else 0
            if "Trainable Params" in r and bb == "mBART":
                r["Trainable Params"] = {"BitFit":"335,872 (0.0550%)",
                  "Narrow LoRA":"1,179,648 (0.1927% active)"}.get(mm,"610,879,488 (100%; 0 added)")
        destination = AUDIT / "isolated_reports" / rel
        write_csv(destination, rows, fields)
        changed.append(destination.relative_to(ROOT).as_posix())

    for rel in ["results/analysis/mbart_ablation_results.csv","outputs_revalidation/mbart_ablation_results.csv"]:
        path=ROOT/rel
        if not path.exists():continue
        backup(path); rows=read_csv(path)
        for r in rows:
            mm="LSR-only" if "lsr-only" in r["Run"] else "CLRR-only"
            r["BLEU"]=lookup["mBART",mm]["bleu"]; r["chrF++"]=lookup["mBART",mm]["chrfpp"]
        destination = AUDIT / "isolated_reports" / rel
        write_csv(destination,rows,list(rows[0]));changed.append(destination.relative_to(ROOT).as_posix())

    # Exact existing source-regex groups. mT5 here is CLRR-only, not CLRR+LSR.
    import re
    refs=read_csv(ROOT/"data_processed/amis_mandarin/test.csv")
    masks={x:[i for i,r in enumerate(refs) if re.search(rf"\b({x}|{x.capitalize()})[a-z']+\b",r["source"])] for x in ["mi","ma","pa"]}
    union=set().union(*[set(v) for v in masks.values()])
    masks={"Full":list(range(len(refs))),**masks,"Root":[i for i in range(len(refs)) if i not in union]}
    predictions={}
    for key in [("mBART","Baseline"),("mBART","CLRR+LSR"),("mT5","Baseline"),("mT5","CLRR-only")]:
        predictions[key]=read_csv(ROOT/lookup[key]["prediction_path"])
        assert len(predictions[key])==len(refs)
    bleu,chrf=BLEU(tokenize="zh"),CHRF(word_order=2)
    morphology=[]
    for name,indices in masks.items():
        r={"Morphological Subset":name,"N Sentences":len(indices)}
        for key, label in [(("mBART","Baseline"),"mBART Baseline"),(("mBART","CLRR+LSR"),"mBART CLRR-Enc"),
                            (("mT5","Baseline"),"mT5 Baseline"),(("mT5","CLRR-only"),"mT5 CLRR-Enc")]:
            hyp=[predictions[key][i]["prediction"].strip() for i in indices]
            target=[[refs[i]["target"].strip() for i in indices]]
            r[label+" BLEU"]=bleu.corpus_score(hyp,target).score
            r[label+" chrF++"]=chrf.corpus_score(hyp,target).score
        r["mBART Delta chrF++"]=r["mBART CLRR-Enc chrF++"]-r["mBART Baseline chrF++"]
        morphology.append(r)
    for rel in ["results/analysis/morphological_breakdown.csv","outputs_revalidation/morphological_breakdown.csv"]:
        path=ROOT/rel
        if path.exists():backup(path)
        destination = AUDIT / "isolated_reports" / rel
        write_csv(destination,morphology,list(morphology[0]));changed.append(destination.relative_to(ROOT).as_posix())

    # Diagnostic only: preserve primary scores with all 575 rows.
    hyp={k:[r["prediction"].strip() for r in predictions["mBART",k]] for k in ["Baseline","CLRR+LSR"]}
    ref=[r["target"].strip() for r in refs]
    result={"removed_index":170,"purpose":"DIAGNOSTIC_ONLY_NOT_PRIMARY_TEST_SELECTION","primary_rows":575,"diagnostic_rows":574}
    for k,v in hyp.items():
        result[k]=chrf.corpus_score([t for i,t in enumerate(v) if i!=170],[[t for i,t in enumerate(ref) if i!=170]]).score
    result["delta"]=result["CLRR+LSR"]-result["Baseline"]
    (AUDIT/"leave_one_out_index170.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (AUDIT/"derived_csv_corrections.json").write_text(json.dumps({"files":changed,"source":"all_verified_translation_scores.csv"},indent=2),encoding="utf-8")
    print(json.dumps({"corrected_tables":len(changed),"verified_scores":len(scores),"leave_one_out":result},ensure_ascii=False))

if __name__=="__main__":
    main()
