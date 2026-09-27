"""Inspect main-run checkpoints and prepare the follow-up paper analyses."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer

from amis_rewire.metrics import generation_metrics, safe_decode_inputs
from amis_rewire.modeling import _find_stack_layers, load_model
from amis_rewire.train import build_compute_metrics, configure_mbart, tokenize_dataset


REPO = "FiveC/amis-rewire-checkpoints"
PREFIX = "checkpoints"
MAIN_RUNS = {
    "mt5-small-ami-cmn-baseline": ("baseline", 2.81, 4.49),
    "mt5-small-ami-cmn-clrr-enc": ("clrr-enc", 4.50, 5.90),
    "mt5-small-ami-cmn-jepa-clrr-enc": ("jepa-clrr-enc", 5.17, 5.76),
    "mbart-large-50-ami-cmn-baseline": ("baseline", 20.09, 15.72),
    "mbart-large-50-ami-cmn-jepa-clrr-enc": ("jepa-clrr-enc", 20.81, 16.56),
}
NEW_RUNS = (
    "byt5-small-ami-cmn-baseline",
    "byt5-small-ami-cmn-jepa-clrr-enc",
    "mt5-small-ami-cmn-jepa-clrr-dec",
    "mt5-small-ami-cmn-jepa-clrr-both",
)
EXPECTED_ARGS = {
    "num_train_epochs": 20.0,
    "per_device_train_batch_size": 128,
    "per_device_eval_batch_size": 128,
    "gradient_accumulation_steps": 1,
    "seed": 42,
    "data_seed": 42,
    "warmup_ratio": 0.06,
    "gradient_checkpointing": False,
    "bf16": True,
}


def token() -> str:
    value = os.environ.get("HF_TOKEN")
    if not value:
        raise RuntimeError("Set the HF_TOKEN Colab Secret before accessing the private checkpoint repo.")
    return value


def remote_file(run_name: str, filename: str) -> str:
    return f"{PREFIX}/{run_name}/{filename}"


def fetch_file(repo: str, filename: str) -> Path:
    return Path(
        hf_hub_download(repo_id=repo, filename=filename, repo_type="model", token=token())
    )


def extract_best(archive_path: Path, destination: Path) -> None:
    has_weights = (destination / "pytorch_model.bin").exists() or (
        destination / "model.safetensors"
    ).exists()
    if all((destination / name).exists() for name in ("training_args.bin", "config.json", "tokenizer_config.json")) and has_weights:
        return
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            target = (destination / item.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in {archive_path}: {item.filename}")
        archive.extractall(destination)
    if not (destination / "training_args.bin").exists():
        raise FileNotFoundError(f"training_args.bin is missing from {archive_path}")
    if not (destination / "config.json").exists():
        raise FileNotFoundError(f"config.json is missing from {archive_path}")
    if not (destination / "tokenizer_config.json").exists() or not (
        (destination / "pytorch_model.bin").exists() or (destination / "model.safetensors").exists()
    ):
        raise FileNotFoundError(f"Tokenizer or model weights are missing from {archive_path}")


def read_training_args(best_dir: Path, run_name: str) -> dict[str, object]:
    args = torch.load(best_dir / "training_args.bin", map_location="cpu", weights_only=False)
    expected = dict(EXPECTED_ARGS)
    expected["learning_rate"] = 5e-5 if run_name.startswith("mbart") else 3e-4
    actual = {key: getattr(args, key, None) for key in expected}
    mismatches = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] is None or actual[key] != value
    }
    if mismatches:
        raise ValueError(f"Training arguments differ from run_all_models.sh for {run_name}: {mismatches}")
    return actual


def test_frame(data_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / "test.csv", encoding="utf-8").fillna("")
    if list(frame.columns) != ["source", "target"] or len(frame) != 575:
        raise ValueError("test.csv must contain exactly 575 aligned source,target rows")
    return frame


def preflight(repo: str, output_root: Path, data_dir: Path) -> dict[str, dict[str, object]]:
    test_frame(data_dir)
    files = set(HfApi(token=token()).list_repo_files(repo, repo_type="model"))
    required = {remote_file(run, f"{run}-best.zip") for run in MAIN_RUNS}
    missing = sorted(required - files)
    if missing:
        raise FileNotFoundError(f"Required best-model archives are missing: {missing}")
    checked: dict[str, dict[str, object]] = {}
    for run in MAIN_RUNS:
        archive = fetch_file(repo, remote_file(run, f"{run}-best.zip"))
        best_dir = output_root / "reference_models" / run / "best_model"
        extract_best(archive, best_dir)
        checked[run] = {"best_dir": str(best_dir), "training_args": read_training_args(best_dir, run)}
        print(f"[preflight] {run}: archive and training arguments OK", flush=True)
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "preflight.json").write_text(
        json.dumps(
            {
                "repo": repo,
                "prefix": PREFIX,
                "runs": checked,
                "script_only_settings": {
                    "early_stopping_patience": 4,
                    "max_source_length": 256,
                    "max_target_length": 256,
                    "validation_beams": 1,
                    "test_beams": 4,
                    "note": "Some CLI settings are not recoverable from training_args.bin; these values come from run_all_models.sh.",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[preflight] patience, sequence limits, and validation beams use the confirmed script values where archives lack them", flush=True)
    return checked


def model_and_tokenizer(best_dir: Path, method: str):
    tokenizer = AutoTokenizer.from_pretrained(best_dir, use_fast=True)
    model = load_model(str(best_dir), method, distance=2, strength=0.1, stack="encoder")
    configure_mbart(tokenizer, model.base_model if hasattr(model, "base_model") else model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, device


def generate_rows(model, tokenizer, device, frame: pd.DataFrame, batch_size: int) -> list[str]:
    predictions: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            sources = frame["source"].iloc[start : start + batch_size].tolist()
            encoded = tokenizer(sources, padding=True, truncation=True, max_length=256, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(**encoded, max_length=256, num_beams=4)
            predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [prediction.strip() for prediction in predictions]


def predict_like_main_run(model, tokenizer, frame: pd.DataFrame, best_dir: Path, output_dir: Path) -> list[str]:
    """Use the same collator and Seq2SeqTrainer generation path as train.py."""
    args = torch.load(best_dir / "training_args.bin", map_location="cpu", weights_only=False)
    args.output_dir = str(output_dir)
    args.generation_num_beams = 4
    args.generation_max_length = 256
    args.report_to = []
    raw_dataset = Dataset.from_pandas(frame, preserve_index=False)
    dataset = tokenize_dataset(
        raw_dataset, tokenizer,
        SimpleNamespace(max_source_length=256, max_target_length=256, source_prefix=""),
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8),
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
    )
    result = trainer.predict(dataset, metric_key_prefix="test")
    generated = result.predictions[0] if isinstance(result.predictions, tuple) else result.predictions
    predictions = tokenizer.batch_decode(safe_decode_inputs(generated), skip_special_tokens=True)
    predictions = [prediction.strip() for prediction in predictions]
    scores = generation_metrics(predictions, frame["target"].tolist())
    if abs(scores["bleu"] - result.metrics["test_bleu"]) > 1e-5 or abs(scores["chrf++"] - result.metrics["test_chrf++"]) > 1e-5:
        raise ValueError("Decoded predictions do not reproduce Seq2SeqTrainer test metrics")
    return predictions


def sentence_cosines(hidden: torch.Tensor, mask: torch.Tensor) -> list[float | None]:
    scores: list[float | None] = []
    for states, valid in zip(hidden, mask):
        vectors = F.normalize(states[valid].float(), p=2, dim=-1)
        n = len(vectors)
        if n < 2:
            scores.append(None)
            continue
        total = vectors.sum(dim=0)
        diagonal = (vectors * vectors).sum()
        score = ((total * total).sum() - diagonal) / (n * (n - 1))
        scores.append(float(score.item()))
    return scores


def measure_cosines(model, tokenizer, device, frame: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    layers = _find_stack_layers(model, "encoder")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for index, layer in enumerate(layers):
        def capture(_module, _inputs, output, layer_index=index):
            captured[layer_index] = (output[0] if isinstance(output, (tuple, list)) else output).detach()
        handles.append(layer.register_forward_hook(capture))
    records = []
    try:
        with torch.inference_mode():
            for start in range(0, len(frame), batch_size):
                sources = frame["source"].iloc[start : start + batch_size].tolist()
                encoded = tokenizer(sources, padding=True, truncation=True, max_length=256, return_tensors="pt")
                valid = encoded["attention_mask"].bool()
                for special_id in tokenizer.all_special_ids:
                    valid &= encoded["input_ids"] != special_id
                encoded = {key: value.to(device) for key, value in encoded.items()}
                fresh_cache = model._fresh_cache() if hasattr(model, "_fresh_cache") else nullcontext()
                captured.clear()
                with fresh_cache:
                    model.get_encoder()(**encoded, return_dict=True)
                if len(captured) != len(layers):
                    raise RuntimeError("Did not capture every encoder layer")
                for layer_index in range(len(layers)):
                    values = sentence_cosines(captured[layer_index], valid.to(device))
                    records.extend(
                        {"index": start + offset, "layer": layer_index + 1, "cosine": value}
                        for offset, value in enumerate(values)
                    )
    finally:
        for handle in handles:
            handle.remove()
    return pd.DataFrame(records)


def cosine_figure(cosines: pd.DataFrame, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = cosines.groupby(["run", "layer"], as_index=False)["cosine"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    for run, group in summary.groupby("run"):
        label = "mT5 CLRR-Enc" if run.endswith("clrr-enc") else "mT5 Baseline"
        ax.plot(group["layer"], group["cosine"], marker="o", label=label)
    ax.set(xlabel="Encoder layer", ylabel="Mean within-sentence token cosine")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def analyze(repo: str, output_root: Path, data_dir: Path, batch_size: int) -> None:
    analysis_dir = output_root / "analysis"
    expected = [
        "old_model_scores.csv", "cosine_by_sentence.csv", "cosine_by_layer.csv", "encoder_cosine.png",
        *(f"predictions/{run}.csv" for run in MAIN_RUNS),
    ]
    remote = set(HfApi(token=token()).list_repo_files(repo, repo_type="model"))
    if all((analysis_dir / name).exists() for name in expected):
        if not all(f"analysis/{name}" in remote for name in expected):
            HfApi(token=token()).upload_folder(
                folder_path=str(analysis_dir), path_in_repo="analysis", repo_id=repo, repo_type="model"
            )
        print("[skip] old-model analysis is already available locally", flush=True)
        return
    if all(f"analysis/{name}" in remote for name in expected):
        for name in expected:
            destination = analysis_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fetch_file(repo, f"analysis/{name}"), destination)
        print("[restore] old-model analysis downloaded from Hugging Face", flush=True)
        return
    checked = preflight(repo, output_root, data_dir)
    frame = test_frame(data_dir)
    prediction_dir = analysis_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    all_cosines = []
    score_rows = []
    for run, (method, expected_bleu, expected_chrf) in MAIN_RUNS.items():
        model, tokenizer, device = model_and_tokenizer(Path(checked[run]["best_dir"]), method)
        if run in ("mt5-small-ami-cmn-baseline", "mt5-small-ami-cmn-clrr-enc"):
            layer_scores = measure_cosines(model, tokenizer, device, frame, batch_size)
            layer_scores.insert(0, "run", run)
            all_cosines.append(layer_scores)
        predictions = predict_like_main_run(
            model, tokenizer, frame, Path(checked[run]["best_dir"]), output_root / "trainer_tmp" / run
        )
        if len(predictions) != 575:
            raise ValueError(f"{run} produced {len(predictions)} predictions instead of 575")
        scores = generation_metrics(predictions, frame["target"].tolist())
        if abs(scores["bleu"] - expected_bleu) > 0.02 or abs(scores["chrf++"] - expected_chrf) > 0.02:
            raise ValueError(
                f"{run} score mismatch: regenerated {scores}, reported BLEU={expected_bleu}, chrF++={expected_chrf}"
            )
        pd.DataFrame(
            {"index": range(575), "source": frame["source"], "target": frame["target"], "prediction": predictions}
        ).to_csv(prediction_dir / f"{run}.csv", index=False, encoding="utf-8")
        score_rows.append({"run": run, **scores})
        print(f"[analyze] {run}: {scores}", flush=True)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(score_rows).to_csv(analysis_dir / "old_model_scores.csv", index=False)
    cosines = pd.concat(all_cosines, ignore_index=True)
    cosines.to_csv(analysis_dir / "cosine_by_sentence.csv", index=False)
    cosines.groupby(["run", "layer"], as_index=False)["cosine"].mean().to_csv(
        analysis_dir / "cosine_by_layer.csv", index=False
    )
    cosine_figure(cosines, analysis_dir / "encoder_cosine.png")
    HfApi(token=token()).upload_folder(
        folder_path=str(analysis_dir), path_in_repo="analysis", repo_id=repo, repo_type="model"
    )


def read_predictions(path: Path, frame: pd.DataFrame) -> list[str]:
    saved = pd.read_csv(path, encoding="utf-8").fillna("")
    if list(saved.columns) != ["index", "source", "target", "prediction"]:
        raise ValueError(f"Unexpected prediction columns in {path}")
    if len(saved) != 575 or saved["index"].tolist() != list(range(575)):
        raise ValueError(f"Prediction rows are not aligned in {path}")
    if saved["source"].tolist() != frame["source"].tolist() or saved["target"].tolist() != frame["target"].tolist():
        raise ValueError(f"Prediction text differs from test.csv in {path}")
    return saved["prediction"].tolist()


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [0.0] * len(p_values)
    maximum = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=lambda i: p_values[i])):
        maximum = max(maximum, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = maximum
    return adjusted


def report(repo: str, output_root: Path, data_dir: Path) -> None:
    frame = test_frame(data_dir)
    analysis_dir = output_root / "analysis"
    prediction_dir = analysis_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    runs = (*MAIN_RUNS, *NEW_RUNS)
    predictions = {}
    scores = []
    for run in runs:
        path = prediction_dir / f"{run}.csv"
        if not path.exists() and run in NEW_RUNS:
            local = output_root / run / "test_predictions.csv"
            if local.exists():
                shutil.copyfile(local, path)
            else:
                shutil.copyfile(fetch_file(repo, remote_file(run, "test_predictions.csv")), path)
        if not path.exists() and run in MAIN_RUNS:
            raise FileNotFoundError(f"Run analyze first: {path}")
        predictions[run] = read_predictions(path, frame)
        run_scores = generation_metrics(predictions[run], frame["target"].tolist())
        if run in NEW_RUNS:
            metrics_path = output_root / run / "metrics.json"
            if not metrics_path.exists():
                metrics_path = fetch_file(repo, remote_file(run, "metrics.json"))
            recorded = json.loads(metrics_path.read_text(encoding="utf-8"))
            if abs(run_scores["bleu"] - recorded["test_bleu"]) > 1e-4 or abs(
                run_scores["chrf++"] - recorded["test_chrf++"]
            ) > 1e-4:
                raise ValueError(f"Prediction scores disagree with metrics.json for {run}")
        scores.append({"run": run, **run_scores})
    pd.DataFrame(scores).to_csv(analysis_dir / "all_scores.csv", index=False)

    comparisons = [
        ("primary", "mt5-small-ami-cmn-baseline", "mt5-small-ami-cmn-jepa-clrr-enc"),
        ("primary", "mbart-large-50-ami-cmn-baseline", "mbart-large-50-ami-cmn-jepa-clrr-enc"),
        ("primary", "byt5-small-ami-cmn-baseline", "byt5-small-ami-cmn-jepa-clrr-enc"),
        ("exploratory", "mt5-small-ami-cmn-jepa-clrr-enc", "mt5-small-ami-cmn-jepa-clrr-dec"),
        ("exploratory", "mt5-small-ami-cmn-jepa-clrr-enc", "mt5-small-ami-cmn-jepa-clrr-both"),
    ]
    os.environ["SACREBLEU_SEED"] = "42"
    rows = []
    for kind, base, challenger in comparisons:
        signatures, results = PairedTest(
            named_systems=[(base, predictions[base]), (challenger, predictions[challenger])],
            metrics={"BLEU": BLEU(tokenize="zh"), "chrF++": CHRF(word_order=2)},
            references=[frame["target"].tolist()],
            test_type="bs",
            n_samples=10000,
            n_jobs=1,
        )()
        for metric, values in results.items():
            if metric == "System":
                continue
            rows.append(
                {
                    "type": kind,
                    "baseline": base,
                    "challenger": challenger,
                    "metric": metric,
                    "baseline_score": values[0].score,
                    "challenger_score": values[1].score,
                    "delta": values[1].score - values[0].score,
                    "p_value": values[1].p_value,
                    "signature": str(signatures[metric]),
                }
            )
    primary_indexes = [i for i, row in enumerate(rows) if row["type"] == "primary"]
    adjusted = holm_adjust([rows[i]["p_value"] for i in primary_indexes])
    for i, value in zip(primary_indexes, adjusted):
        rows[i]["p_value_holm"] = value
    pd.DataFrame(rows).to_csv(analysis_dir / "paired_bootstrap.csv", index=False)

    baseline = predictions["mt5-small-ami-cmn-baseline"]
    proposed = predictions["mt5-small-ami-cmn-jepa-clrr-enc"]
    metric = CHRF(word_order=2)
    case_rows = []
    for index, (source, target, first, second) in enumerate(
        zip(frame["source"], frame["target"], baseline, proposed)
    ):
        first_score = metric.sentence_score(first, [target]).score
        second_score = metric.sentence_score(second, [target]).score
        case_rows.append(
            {"index": index, "source": source, "target": target, "baseline": first,
             "jepa_clrr_enc": second, "sentence_chrf_delta": second_score - first_score}
        )
    ranked = sorted(case_rows, key=lambda row: row["sentence_chrf_delta"])
    pd.DataFrame(ranked[:20] + ranked[-20:][::-1]).to_csv(
        analysis_dir / "case_candidates.csv", index=False, encoding="utf-8"
    )
    (analysis_dir / "analysis_notes.txt").write_text(
        "Paired bootstrap resamples 575 test sentences, seed 42, 10,000 samples. "
        "Holm correction applies to the six primary p-values (three backbones x two metrics). "
        "Decoder/both comparisons are exploratory. All models use one training seed, "
        "so these tests do not measure variation across training seeds. "
        "Case candidates require human linguistic review before claims about Amis morphology.\n",
        encoding="utf-8",
    )
    HfApi(token=token()).upload_folder(
        folder_path=str(analysis_dir), path_in_repo="analysis", repo_id=repo, repo_type="model"
    )


def smoke(repo: str, output_root: Path, data_dir: Path) -> None:
    checked = preflight(repo, output_root, data_dir)
    run = "mt5-small-ami-cmn-baseline"
    model, tokenizer, device = model_and_tokenizer(Path(checked[run]["best_dir"]), "baseline")
    frame = test_frame(data_dir).iloc[:2]
    predictions = generate_rows(model, tokenizer, device, frame, batch_size=2)
    if len(predictions) != 2:
        raise RuntimeError("Smoke test did not produce two translations")
    print("[smoke] two translations generated successfully", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "smoke", "analyze", "report"))
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_extra"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight(args.repo, args.output_dir, args.data_dir)
    elif args.stage == "smoke":
        smoke(args.repo, args.output_dir, args.data_dir)
    elif args.stage == "analyze":
        analyze(args.repo, args.output_dir, args.data_dir, args.batch_size)
    else:
        report(args.repo, args.output_dir, args.data_dir)


if __name__ == "__main__":
    main()
