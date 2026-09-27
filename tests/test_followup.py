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
    MAIN_RUNS,
    NEW_RUNS,
    extract_best,
    holm_adjust,
    measure_cosines,
    predict_like_main_run,
    read_predictions,
    read_training_args,
    report,
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

    def test_local_analysis_smoke_generates_predictions_and_cosines(self) -> None:
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Seq2SeqTrainingArguments, T5Config, T5ForConditionalGeneration
        from amis_rewire.modeling import CrossLayerResidualRewire

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            best_dir = root / "best_model"
            best_dir.mkdir()
            vocab = {"[PAD]": 0, "[UNK]": 1, "</s>": 2, "a": 3, "b": 4, "x": 5, "y": 6}
            vocab.update({f"extra{i}": i for i in range(7, 20)})
            backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="</s>")
            model = T5ForConditionalGeneration(T5Config(
                vocab_size=20, d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1,
                num_heads=2, pad_token_id=0, eos_token_id=2, decoder_start_token_id=0,
            ))
            model.generate = lambda **kwargs: torch.full(
                (kwargs["input_ids"].shape[0], 2), 2, dtype=torch.long, device=kwargs["input_ids"].device
            )
            args = Seq2SeqTrainingArguments(
                output_dir=str(root / "training"), eval_strategy="epoch", save_strategy="epoch",
                load_best_model_at_end=True, predict_with_generate=True,
                per_device_eval_batch_size=2, use_cpu=True, report_to=[],
            )
            torch.save(args, best_dir / "training_args.bin")
            frame = pd.DataFrame({"source": ["a b", "b a"], "target": ["x  y", "y  x"]})
            predictions = predict_like_main_run(
                model, tokenizer, frame, best_dir, root / "prediction"
            )
            self.assertEqual(len(predictions), 2)
            self.assertEqual(len(measure_cosines(model, tokenizer, torch.device("cpu"), frame, 2)), 2)
            wrapped = CrossLayerResidualRewire(model, distance=1)
            self.assertEqual(len(measure_cosines(wrapped, tokenizer, torch.device("cpu"), frame, 2)), 2)

    def test_cosine_uses_only_unmasked_tokens(self) -> None:
        hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
        mask = torch.tensor([[True, True, False]])
        self.assertAlmostEqual(sentence_cosines(hidden, mask)[0], 0.0, places=6)

    def test_local_report_smoke_writes_artifacts(self) -> None:
        from sacrebleu.significance import PairedTest

        frame = pd.read_csv(ROOT / "data" / "processed" / "test.csv").fillna("")
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            for run in (*MAIN_RUNS, *NEW_RUNS):
                predictions = frame.copy()
                predictions.insert(0, "index", range(len(frame)))
                predictions["prediction"] = predictions["target"]
                if run in MAIN_RUNS:
                    path = output_root / "analysis" / "predictions" / f"{run}.csv"
                else:
                    path = output_root / run / "test_predictions.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                predictions.to_csv(path, index=False)

            def short_paired_test(*args, **kwargs):
                kwargs["n_samples"] = 10
                return PairedTest(*args, **kwargs)

            with patch("followup_analysis.PairedTest", side_effect=short_paired_test), \
                 patch("followup_analysis.HfApi"), \
                 patch.dict("os.environ", {"HF_TOKEN": "local-smoke"}):
                report("local/smoke", output_root, ROOT / "data" / "processed")
            for name in ("all_scores.csv", "paired_bootstrap.csv", "case_candidates.csv", "analysis_notes.txt"):
                self.assertTrue((output_root / "analysis" / name).exists())

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
