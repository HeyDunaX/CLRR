"""Verify cached translation outputs and reported statistics without model loading.

Run with the existing clrr Python environment. Raw reports remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest, _compute_p_value

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs_rebuttal/metric_audit_20261002/full"
REVISION = "63b0dbc4fab41a669ccb36ed2f22894868634b8a"
SAMPLES = 10000
SEED = 42


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values)-rank)*values[index]))
        adjusted[index] = previous
    return adjusted


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path("outputs_rebuttal/metric_audit_20261002/standardized_metrics.json").read_text(encoding="utf-8"))
    rows = report["results"].copy()
    extras = [
        ("ByT5", "Baseline", "analysis/predictions/byt5-small-ami-cmn-baseline.csv"),
        ("ByT5", "CLRR+LSR", "analysis/predictions/byt5-small-ami-cmn-jepa-clrr-enc.csv"),
        ("ByT5", "CLRR-Dec+LSR", "comparative_baselines/byt5-small-ami-cmn-jepa-clrr-dec/test_predictions.csv"),
        ("ByT5", "LayerSkip", "comparative_baselines/byt5-small-layerskip-acl2024/test_predictions.csv"),
        ("ByT5", "Middle-Layer Alignment", "comparative_baselines/byt5-small-middle-align-acl2025/test_predictions.csv"),
        ("mT5", "LayerSkip", "comparative_baselines/layerskip_acl2024/test_predictions.csv"),
        ("mBART", "LayerSkip", "comparative_baselines/mbart-large-50-layerskip-acl2024/test_predictions.csv"),
    ]
    downloads = []
    for backbone, method, remote in extras:
        dest = OUT / "historical_predictions" / (backbone + "-" + method.replace("/", "-").replace(" ", "-") + ".csv")
        dest.parent.mkdir(exist_ok=True)
        url = f"https://huggingface.co/FiveC/amis-rewire-checkpoints/resolve/{REVISION}/{remote}"
        if not dest.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(2_000_001)
            if len(data) > 2_000_000:
                raise ValueError("Refusing large prediction response")
            dest.write_bytes(data)
        downloads.append({"remote":remote,"revision":REVISION,"local":dest.relative_to(ROOT).as_posix(),"sha256":digest(dest)})
        rows.append({"dataset":"amis_mandarin","backbone":backbone,"method":method,"prediction_path":dest.relative_to(ROOT).as_posix()})
    systems = {}
    datasets = {}
    for name in report["datasets"]:
        frames = {split:pd.read_csv(f"data_processed/{name}/{split}.csv",keep_default_na=False) for split in ["train","validation","test"]}
        stats = {}
        for split, frame in frames.items():
            assert list(frame.columns) == ["source","target"]
            stats[split] = {
                "rows":len(frame), "source_whitespace_tokens":sum(len(s.split()) for s in frame.source),
                "target_han_characters":sum(sum("\u3400" <= c <= "\u4dbf" or "\u4e00" <= c <= "\u9fff" or "\uf900" <= c <= "\ufaff" for c in t) for t in frame.target),
                "source_empty":int((frame.source.str.strip()=="").sum()),
                "target_empty":int((frame.target.str.strip()=="").sum()),
                "duplicate_pairs":int(frame.duplicated(["source","target"]).sum()),
                "sha256":digest(f"data_processed/{name}/{split}.csv")}
        for a,b in [("train","validation"),("train","test"),("validation","test")]:
            aa=set(zip(frames[a].source.str.strip(),frames[a].target.str.strip()))
            bb=set(zip(frames[b].source.str.strip(),frames[b].target.str.strip()))
            stats[a+"_"+b+"_exact_pair_overlap"]=len(aa & bb)
        datasets[name]=stats
        systems[name]={}
    for row in rows:
        frame=pd.read_csv(row["prediction_path"],keep_default_na=False)
        raw=pd.read_csv(f"data_processed/{row['dataset']}/test.csv",keep_default_na=False)
        assert len(frame)==len(raw),row["prediction_path"]
        assert frame.source.tolist()==raw.source.tolist(),row["prediction_path"]
        assert frame.target.tolist()==raw.target.tolist(),row["prediction_path"]
        if "index" in frame:
            assert frame["index"].tolist()==list(range(len(raw)))
        if "prediction_sha256" in row:
            assert digest(row["prediction_path"])==row["prediction_sha256"]
        predictions=frame.prediction.str.strip().tolist()
        references=raw.target.str.strip().tolist()
        tokenizer="zh" if row["dataset"]=="amis_mandarin" else "13a"
        b=BLEU(tokenize=tokenizer)
        c=CHRF(word_order=2)
        scores={"bleu":b.corpus_score(predictions,[references]).score,"chrfpp":c.corpus_score(predictions,[references]).score}
        for key,value in scores.items():
            if key in row:
                assert abs(value-row[key])<1e-10,(row["method"],key)
            row[key]=value
        row.update(rows=len(raw),bleu_tokenizer=tokenizer,bleu_signature=str(b.get_signature()),chrfpp_signature=str(c.get_signature()),
                   prediction_sha256=digest(row["prediction_path"]),reference_sha256=digest(f"data_processed/{row['dataset']}/test.csv"),
                   status="VERIFIED_FROM_ALIGNED_SAVED_PREDICTIONS")
        systems[row["dataset"]][(row["backbone"],row["method"])]=(predictions,references,row)
    pd.DataFrame(rows).to_csv(OUT/"all_verified_translation_scores.csv",index=False)
    (OUT/"translation_audit.json").write_text(json.dumps({"revision":REVISION,"sacrebleu":sacrebleu.__version__,"results":rows,"datasets":datasets,"downloads":downloads},indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"[verified] {len(rows)} translation outputs; dataset statistics saved",flush=True)

    comparisons=[]
    def compare(dataset,backbone,base,challenger):
        comparisons.append((dataset,backbone,base,challenger))
    for base in ["Baseline","LSR-only","CLRR-only","Middle-Layer Alignment","CLRR-Dec+LSR","BitFit","Narrow LoRA","Strong LoRA A","Strong LoRA B"]:
        compare("amis_mandarin","mBART",base,"CLRR+LSR")
    for base in ["Baseline","Middle-Layer Alignment","Narrow LoRA","Strong LoRA A","Strong LoRA B"]:
        compare("amis_mandarin","NLLB",base,"CLRR+LSR")
    for base in ["Baseline","Middle-Layer Alignment"]:
        compare("amis_mandarin","mT5",base,"CLRR-Dec+LSR")
    compare("amis_mandarin","mT5","Baseline","CLRR+LSR")
    compare("amis_mandarin","mT5","CLRR+LSR","CLRR-Both+LSR")
    for challenger in ["CLRR+LSR (patience 4)","CLRR+LSR (patience 10, selected epoch 14)","CLRR+LSR (patience 10, fixed epoch 20)"]:
        compare("ashaninka_spanish","mBART","Baseline (patience 4)",challenger)
    (OUT/"bootstrap_protocol.json").write_text(json.dumps({
        "samples":SAMPLES,"seed":SEED,"unit":"sentence; paired sampling of original test rows",
        "metrics":"BLEU zh/13a and raw chrF++; original CSV references",
        "ci":"percentile 2.5 and 97.5 of signed corpus-score delta",
        "p":"SacreBLEU centered absolute-difference bootstrap p, not one-sided win probability",
        "holm_family":"all 42 metric comparisons in this audit; post-hoc, one training seed",
        "comparisons":comparisons},indent=2),encoding="utf-8")
    boot={}
    for dataset in systems:
        n=len(next(iter(systems[dataset].values()))[0])
        idxs=np.random.default_rng(SEED).choice(n,size=(SAMPLES,n),replace=True)
        required={(backbone,name) for d,backbone,a,b in comparisons if d==dataset for name in [a,b]}
        for backbone,name in sorted(required):
            predictions,references,row=systems[dataset][(backbone,name)]
            for metric_name,metric in [("BLEU",BLEU(tokenize=row["bleu_tokenizer"])),("chrF++",CHRF(word_order=2))]:
                stats=np.asarray(metric._extract_corpus_statistics(predictions,[references]),dtype="float32")
                scores=np.empty(SAMPLES)
                for start in range(0,SAMPLES,128):
                    sums=stats[idxs[start:start+128]].sum(axis=1)
                    scores[start:start+len(sums)]=[metric._compute_score_from_stats(s).score for s in sums]
                boot[(dataset,backbone,name,metric_name)]=scores
            print(f"[bootstrap] cached {dataset} / {backbone} / {name}",flush=True)
    tests=[]
    for dataset,backbone,base,challenger in comparisons:
        for metric_name,key in [("BLEU","bleu"),("chrF++","chrfpp")]:
            delta=systems[dataset][(backbone,challenger)][2][key]-systems[dataset][(backbone,base)][2][key]
            sampled=boot[(dataset,backbone,challenger,metric_name)]-boot[(dataset,backbone,base,metric_name)]
            absdiff=np.abs(sampled)
            p=_compute_p_value(absdiff-absdiff.mean(),abs(delta))
            lo,hi=np.percentile(sampled,[2.5,97.5])
            tests.append({"dataset":dataset,"backbone":backbone,"baseline":base,"challenger":challenger,"metric":metric_name,
                          "delta":delta,"delta_ci_low":float(lo),"delta_ci_high":float(hi),"p_value":float(p)})
    for row,adjusted in zip(tests,holm([x["p_value"] for x in tests])):
        row["p_value_holm"]=float(adjusted)
    pd.DataFrame(tests).to_csv(OUT/"paired_bootstrap_standardized.csv",index=False)

    # Independent equivalence check against the public SacreBLEU PairedTest API.
    os.environ["SACREBLEU_SEED"]=str(SEED)
    checks=[]
    for dataset,backbone,base,challenger in [comparisons[0],comparisons[1]]:
        a,refs,_=systems[dataset][(backbone,base)]
        b,_,_=systems[dataset][(backbone,challenger)]
        signatures,official=PairedTest(named_systems=[(base,a),(challenger,b)],metrics={"BLEU":BLEU(tokenize="zh"),"chrF++":CHRF(word_order=2)},
                                      references=[refs],test_type="bs",n_samples=SAMPLES,n_jobs=1)()
        for metric_name,values in official.items():
            if metric_name=="System":
                continue
            label="BLEU" if metric_name=="BLEU" else "chrF++"
            expected=next(x for x in tests if x["dataset"]==dataset and x["backbone"]==backbone and x["baseline"]==base and x["challenger"]==challenger and x["metric"]==label)
            assert abs(values[1].p_value-expected["p_value"])<1e-12
            checks.append({"baseline":base,"challenger":challenger,"metric":label,"p_value":values[1].p_value,"matches_cached_implementation":True,"signature":str(signatures[metric_name])})
    (OUT/"bootstrap_api_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(f"[done] {len(tests)} tests; {len(checks)} native SacreBLEU equivalence checks",flush=True)


if __name__ == "__main__":
    main()
