"""Small checks for the follow-up run boundaries and paired data."""

from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from followup_analysis import (
    EXPECTED_ARGS,
    extract_best,
    holm_adjust,
    read_predictions,
    read_training_args,
    sentence_cosines,
)
from run_followup_models import restore_latest, training_command


class FollowupTests(unittest.TestCase):
    def test_rejects_checkpoint_with_wrong_training_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            best_dir = Path(temporary)
            args = SimpleNamespace(**EXPECTED_ARGS, learning_rate=3e-4)
            args.per_device_train_batch_size = 64
            torch.save(args, best_dir / "training_args.bin")
            with self.assertRaisesRegex(ValueError, "per_device_train_batch_size"):
                read_training_args(best_dir, "mt5-small-ami-cmn-baseline")

    def test_accepts_checkpoint_with_main_run_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            best_dir = Path(temporary)
            torch.save(SimpleNamespace(**EXPECTED_ARGS, learning_rate=3e-4), best_dir / "training_args.bin")
            values = read_training_args(best_dir, "mt5-small-ami-cmn-baseline")
            self.assertEqual(values["num_train_epochs"], 20.0)

    def test_best_archive_without_weights_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "incomplete.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name in ("training_args.bin", "config.json", "tokenizer_config.json"):
                    archive.writestr(name, b"placeholder")
            with self.assertRaisesRegex(FileNotFoundError, "model weights"):
                extract_best(archive_path, root / "best_model")

    def test_wrapped_archive_restores_missing_base_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "wrapped.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name in ("training_args.bin", "tokenizer_config.json", "pytorch_model.bin"):
                    archive.writestr(name, b"placeholder")
            best_dir = root / "best_model"
            with patch("followup_analysis.AutoConfig.from_pretrained") as from_pretrained:
                from_pretrained.return_value.save_pretrained.side_effect = lambda path: (
                    Path(path) / "config.json"
                ).write_text("{}", encoding="utf-8")
                extract_best(archive_path, best_dir, "mt5-small-ami-cmn-clrr-enc")
            from_pretrained.assert_called_once_with("google/mt5-small")
            self.assertTrue((best_dir / "config.json").exists())

    def test_baseline_archive_without_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "baseline.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name in ("training_args.bin", "tokenizer_config.json", "pytorch_model.bin"):
                    archive.writestr(name, b"placeholder")
            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                extract_best(archive_path, root / "best_model", "mt5-small-ami-cmn-baseline")

    def test_prediction_rows_must_match_original_test_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            frame = pd.DataFrame({"source": [f"ami-{i}" for i in range(575)], "target": [f"zh-{i}" for i in range(575)]})
            saved = frame.copy()
            saved.insert(0, "index", range(575))
            saved["prediction"] = "generated"
            saved.loc[1, "source"] = "wrong sentence"
            saved.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "differs from test.csv"):
                read_predictions(path, frame)

    def test_cosine_uses_only_unmasked_tokens(self) -> None:
        hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
        mask = torch.tensor([[True, True, False]])
        self.assertAlmostEqual(sentence_cosines(hidden, mask)[0], 0.0, places=6)

    def test_followup_command_keeps_protocol_and_remote_folder(self) -> None:
        command = training_command(
            "mt5-small-ami-cmn-jepa-clrr-dec", "mt5-small", "jepa-clrr", "decoder",
            "FiveC/amis-rewire-checkpoints", Path("outputs_extra"), Path("backups_extra"), Path("data/processed"),
        )
        self.assertEqual(command[command.index("--rewire-stack") + 1], "decoder")
        self.assertEqual(command[command.index("--num-train-epochs") + 1], "20")
        self.assertEqual(command[command.index("--hf-backup-prefix") + 1], "checkpoints")

    def test_resume_restores_latest_and_selected_best_checkpoint(self) -> None:
        run = "byt5-small-ami-cmn-baseline"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archives = {}
            for step in (5, 10):
                remote = f"checkpoints/{run}/checkpoint-{step}.zip"
                archive_path = root / f"checkpoint-{step}.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr(
                        "trainer_state.json",
                        '{"best_model_checkpoint": "outputs_extra/' + run + '/checkpoint-5"}',
                    )
                    archive.writestr("pytorch_model.bin", b"model")
                    archive.writestr("optimizer.pt", b"optimizer")
                archives[remote] = archive_path
            with patch("run_followup_models.fetch_file", side_effect=lambda _repo, name: archives[name]):
                restore_latest("private/repo", run, root / "outputs", set(archives))
            for step in (5, 10):
                self.assertTrue((root / "outputs" / run / f"checkpoint-{step}" / "trainer_state.json").exists())

    def test_holm_adjustment_preserves_order_and_monotonicity(self) -> None:
        self.assertEqual(holm_adjust([0.04, 0.001, 0.02]), [0.04, 0.003, 0.04])


if __name__ == "__main__":
    unittest.main()
