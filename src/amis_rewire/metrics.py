"""Generation metrics used by every experiment."""

from __future__ import annotations

import numpy as np
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
    predictions = [prediction.strip() for prediction in predictions]
    references = [[reference.strip() for reference in references]]
    bleu_metric = BLEU_METRIC_ZH if tokenize == "zh" else (BLEU_METRIC_13A if tokenize == "13a" else BLEU(tokenize=tokenize))
    return {
        "bleu": float(bleu_metric.corpus_score(predictions, references).score),
        "chrf++": float(CHRFPP_METRIC.corpus_score(predictions, references).score),
    }


def safe_decode_inputs(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions)
    return np.where(predictions != -100, predictions, 0)