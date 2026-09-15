"""Generation metrics used by every experiment."""

from __future__ import annotations

import numpy as np
from sacrebleu.metrics import BLEU, CHRF


BLEU_METRIC = BLEU()
CHRFPP_METRIC = CHRF(word_order=2)


def generation_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    predictions = [prediction.strip() for prediction in predictions]
    references = [[reference.strip() for reference in references]]
    return {
        "bleu": float(BLEU_METRIC.corpus_score(predictions, references).score),
        "chrf++": float(CHRFPP_METRIC.corpus_score(predictions, references).score),
    }


def safe_decode_inputs(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions)
    return np.where(predictions != -100, predictions, 0)