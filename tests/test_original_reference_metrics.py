"""Regression checks for the original-reference metric protocol."""
import unittest
import csv
import tempfile
from pathlib import Path

import numpy as np
from amis_rewire.metrics import (
    build_compute_metrics, generation_metrics, score_prediction_csv,
    verified_reference_metrics,
)


class FakeTokenizer:
    def batch_decode(self, ids, skip_special_tokens=True):
        # A labels decode would alter the reference, and must never occur.
        values=np.asarray(ids).reshape(-1).tolist()
        if values != [1, 2]:
            raise AssertionError("Attempted to decode labels instead of predictions")
        return [" 那個人來了。 ", ""]


class OriginalReferenceMetricsTest(unittest.TestCase):
    def test_original_references_and_empty_prediction_are_preserved(self):
        refs=["那個人來了。", "來吧。"]
        callback=build_compute_metrics(FakeTokenizer(), refs, tokenize="zh")
        refs[0]="mutated after binding"
        result=callback((np.array([[1], [2]]), np.array([[90], [-100]])))
        self.assertEqual(result, generation_metrics(["那個人來了。",""],["那個人來了。","來吧。"],tokenize="zh"))

    def test_split_count_mismatch_is_rejected(self):
        callback=build_compute_metrics(FakeTokenizer(), ["one"], tokenize="13a")
        with self.assertRaises(ValueError):
            callback((np.array([[1],[2]]), np.array([[90],[-100]])))

    def test_saved_predictions_reject_wrong_source_order(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / "reference.csv"
            predictions = Path(folder) / "predictions.csv"
            for path, fields, rows in [
                (reference, ["source", "target"], [["first", "甲"], ["second", "乙"]]),
                (predictions, ["source", "prediction"], [["second", "乙"], ["first", "甲"]]),
            ]:
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(fields)
                    writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "source order mismatch"):
                score_prediction_csv(predictions, reference)

    def test_reference_lookup_rejects_other_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / "test.csv"
            reference.write_text("source,target\nother,其他\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No unique audited reference score"):
                verified_reference_metrics("mBART", "Baseline", reference)


if __name__=="__main__":
    unittest.main()
