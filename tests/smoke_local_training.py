"""Run tiny local training jobs through the real CLI without Hugging Face access."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import ByT5Tokenizer, T5Config, T5ForConditionalGeneration, T5TokenizerFast


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        pretrained = root / "pretrained"
        pretrained.mkdir()
        vocab = {"[PAD]": 0, "[UNK]": 1, "</s>": 2, "a": 3, "b": 4, "x": 5, "y": 6}
        backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.save(str(pretrained / "tokenizer.json"))
        tokenizer = T5TokenizerFast(
            tokenizer_file=str(pretrained / "tokenizer.json"), extra_ids=0,
            pad_token="[PAD]", unk_token="[UNK]", eos_token="</s>",
        )
        tokenizer.save_pretrained(pretrained)
        model = T5ForConditionalGeneration(T5Config(
            vocab_size=len(tokenizer), d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1,
            num_heads=2, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, decoder_start_token_id=tokenizer.pad_token_id,
        ))
        model.save_pretrained(pretrained)
        byt5_pretrained = root / "byt5_pretrained"
        byt5_pretrained.mkdir()
        byt5_tokenizer = ByT5Tokenizer()
        byt5_tokenizer.save_pretrained(byt5_pretrained)
        T5ForConditionalGeneration(T5Config(
            vocab_size=len(byt5_tokenizer), d_model=16, d_ff=32, num_layers=1,
            num_decoder_layers=1, num_heads=2,
            pad_token_id=byt5_tokenizer.pad_token_id,
            eos_token_id=byt5_tokenizer.eos_token_id,
            decoder_start_token_id=byt5_tokenizer.pad_token_id,
        )).save_pretrained(byt5_pretrained)

        data_dir = root / "data"
        data_dir.mkdir()
        examples = pd.DataFrame({"source": ["a b", "b a", "a a", "b b"], "target": ["x y", "y x", "x x", "y y"]})
        for split, rows in (("train", 4), ("validation", 2), ("test", 2)):
            examples.iloc[:rows].to_csv(data_dir / f"{split}.csv", index=False)

        experiments = (
            ("mt5-small", pretrained, "baseline", "encoder"),
            ("mt5-small", pretrained, "jepa-clrr", "both"),
            ("byt5-small", byt5_pretrained, "jepa-clrr", "encoder"),
        )
        for model_type, model_path, method, stack in experiments:
            run = f"tiny-{model_type}-{method}-{stack}"
            command = [
                sys.executable, "-m", "amis_rewire.train",
                "--model", model_type, "--model-name", str(model_path),
                "--method", method, "--rewire-stack", stack,
                "--data-dir", str(data_dir), "--output-dir", str(root / "outputs"),
                "--backup-dir", str(root / "backups"), "--run-name", run,
                "--num-train-epochs", "1", "--early-stopping-patience", "1",
                "--per-device-train-batch-size", "2", "--per-device-eval-batch-size", "2",
                "--max-source-length", "8", "--max-target-length", "8",
                "--num-beams", "1", "--eval-beams", "1",
                "--dataloader-num-workers", "0", "--no-bf16", "--no-gradient-checkpointing",
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"{run} failed:\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
            output = root / "outputs" / run
            archive = root / "backups" / run / f"{run}-best.zip"
            assert (output / "best_model" / "training_args.bin").exists(), run
            assert (output / "metrics.json").exists(), run
            assert len(pd.read_csv(output / "test_predictions.csv")) == 2, run
            assert archive.exists(), run
            with zipfile.ZipFile(archive) as zipped:
                assert "pytorch_model.bin" in zipped.namelist(), run
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            assert "test_bleu" in metrics and "test_chrf++" in metrics, run
            print(f"[smoke] {run}: checkpoint, ZIP, metrics, and predictions OK")


if __name__ == "__main__":
    main()
