"""Generation metrics used by every experiment."""

from __future__ import annotations

from typing import Any

import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF


# Standard tokenizers for SacreBLEU (Mandarin: zh, Spanish/Latin: 13a)
BLEU_METRIC_ZH = BLEU(tokenize="zh")
BLEU_METRIC_13A = BLEU(tokenize="13a")
CHRFPP_METRIC = CHRF(word_order=2)


def generation_metrics(
    predictions: list[str],
    references: list[str],
    tokenize: str = "zh",
) -> dict[str, float]:
    if not predictions or len(predictions) != len(references):
        raise ValueError("Predictions and original references must have the same nonzero length")
    predictions = [prediction.strip() for prediction in predictions]
    references = [[reference.strip() for reference in references]]
    bleu_metric = BLEU_METRIC_ZH if tokenize == "zh" else (BLEU_METRIC_13A if tokenize == "13a" else BLEU(tokenize=tokenize))
    return {
        "bleu": float(bleu_metric.corpus_score(predictions, references).score),
        "chrf++": float(CHRFPP_METRIC.corpus_score(predictions, references).score),
    }


def safe_decode_inputs(predictions: np.ndarray, pad_token_id: int = 0) -> np.ndarray:
    predictions = np.asarray(predictions)
    return np.where(predictions != -100, predictions, pad_token_id)

def build_compute_metrics(tokenizer: Any, references: list[str], tokenize: str = "zh"):
    """Bind the original references for one ordered evaluation split."""
    raw_references = tuple(references)

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        predictions, _labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        pad_id = getattr(tokenizer, "pad_token_id", 0)
        decoded = tokenizer.batch_decode(
            safe_decode_inputs(predictions, 0 if pad_id is None else pad_id),
            skip_special_tokens=True,
        )
        return generation_metrics(decoded, list(raw_references), tokenize=tokenize)

    return compute_metrics


def metric_protocol(tokenize: str = "zh") -> dict[str, Any]:
    return {
        "reference": "original_csv",
        "library": "sacrebleu",
        "version": sacrebleu.__version__,
        "aggregation": "corpus",
        "bleu_tokenizer": tokenize,
        "bleu_lowercase": False,
        "bleu_smooth_method": "exp",
        "bleu_effective_order": False,
        "chrf_char_order": 6,
        "chrf_word_order": 2,
        "chrf_beta": 2,
        "chrf_lowercase": False,
        "chrf_whitespace": False,
        "chrf_eps_smoothing": False,
        "chrf_external_tokenizer": None,
    }

def score_prediction_csv(prediction_path, reference_path, tokenize: str = "zh") -> dict[str, float]:
    """Score saved ordered hypotheses against original CSV targets."""
    import csv

    with open(reference_path, encoding="utf-8-sig", newline="") as handle:
        references = list(csv.DictReader(handle))
    with open(prediction_path, encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if len(predictions) != len(references) or not references:
        raise ValueError("Prediction/reference row count mismatch")
    for index, (prediction, reference) in enumerate(zip(predictions, references)):
        if "source" in prediction and prediction["source"].strip() != reference["source"].strip():
            raise ValueError(f"Prediction source order mismatch at row {index}")
        if "index" in prediction and int(prediction["index"]) != index:
            raise ValueError(f"Prediction index mismatch at row {index}")
        if "target" in prediction and prediction["target"].strip() != reference["target"].strip():
            raise ValueError(f"Prediction target mismatch at row {index}")
    return generation_metrics(
        [row["prediction"] for row in predictions],
        [row["target"] for row in references],
        tokenize=tokenize,
    )


def verified_reference_metrics(backbone: str, method: str, reference_path) -> dict[str, float]:
    """Use the canonical audited table only for the identical reference file."""
    import csv
    import hashlib
    from pathlib import Path

    canonical = Path(__file__).resolve().parents[2] / "outputs_rebuttal/metric_audit_20261002/full/all_verified_translation_scores.csv"
    reference_hash = hashlib.sha256(Path(reference_path).read_bytes()).hexdigest()
    with canonical.open(encoding="utf-8-sig", newline="") as handle:
        matches = [
            row for row in csv.DictReader(handle)
            if row["backbone"] == backbone and row["method"] == method
            and row["reference_sha256"] == reference_hash
        ]
    if len(matches) != 1:
        raise ValueError(f"No unique audited reference score for {backbone}/{method} on this dataset")
    row = matches[0]
    return {"bleu": float(row["bleu"]), "chrf++": float(row["chrfpp"])}
