"""Offline CPU checks of four training branches using a synthetic parallel fixture."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import pandas as pd
import torch
from sacrebleu.metrics import BLEU, CHRF
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (AutoModelForSeq2SeqLM, MBartConfig, MBartForConditionalGeneration,
                          M2M100Config, M2M100ForConditionalGeneration, PreTrainedTokenizerFast)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from amis_rewire import train as mbart_train
from src.nllb_suite import train_nllb


class MBartTinyTokenizer(PreTrainedTokenizerFast):
    lang_code_to_id = {"es_XX": 4}


def main():
    torch.set_num_threads(2)
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "spa_Latn": 4,
             "fuente": 5, "texto": 6, "hola": 7, "mundo": 8}
    vocab.update({f"word{i}": i for i in range(9, 64)})
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data = root / "parallel_fixture"
        data.mkdir()
        frame = pd.DataFrame({"source": ["fuente texto", "texto fuente"],
                              "target": ["hola mundo", "mundo hola"]})
        for split in ("train", "validation", "test"):
            frame.to_csv(data / f"{split}.csv", index=False)
        for backbone, method in (("mbart", "baseline"), ("mbart", "jepa-clrr-enc"), ("mbart", "validation-only"),
                                 ("nllb", "baseline"), ("nllb", "clrr_enc")):
            cls = MBartTinyTokenizer if backbone == "mbart" else PreTrainedTokenizerFast
            tokenizer = cls(tokenizer_object=backend, bos_token="<s>", pad_token="<pad>",
                            eos_token="</s>", unk_token="<unk>",
                            model_input_names=["input_ids", "attention_mask"])
            config_cls = MBartConfig if backbone == "mbart" else M2M100Config
            model_cls = MBartForConditionalGeneration if backbone == "mbart" else M2M100ForConditionalGeneration
            config = config_cls(vocab_size=64, d_model=16, encoder_layers=3, decoder_layers=1,
                                encoder_attention_heads=2, decoder_attention_heads=2,
                                encoder_ffn_dim=32, decoder_ffn_dim=32,
                                pad_token_id=1, bos_token_id=0, eos_token_id=2,
                                decoder_start_token_id=2, forced_bos_token_id=4)
            def factory(*args, **kwargs):
                return model_cls(config)
            run_name = f"tiny-{backbone}-{method}"
            validation_only = method == "validation-only"
            if validation_only:
                (data / "test.csv").unlink()
            argv = ["train", "--method", "jepa-clrr-enc" if validation_only else method, "--model-name", "tiny-offline",
                    "--data-dir", str(data), "--output-dir", str(root / "results"),
                    "--run-name", run_name, "--num-train-epochs", "1",
                    "--early-stopping-patience", "4", "--learning-rate", "5e-5",
                    "--weight-decay", "0.0", "--per-device-train-batch-size", "2",
                    "--per-device-eval-batch-size", "2", "--gradient-accumulation-steps", "1",
                    "--max-source-length", "8", "--max-target-length", "8",
                    "--dataloader-num-workers", "0", "--hf-backup-repo", "", "--no-bf16",
                    "--bleu-tokenizer", "13a"]
            if backbone == "mbart":
                argv += ["--model", "mbart-large-50", "--backup-dir", str(root / "backups"),
                         "--src-lang", "es_XX", "--tgt-lang", "es_XX", "--eval-beams", "1",
                         "--num-beams", "4", "--test-reference", "original_csv"]
                module = mbart_train
                model_patch = "amis_rewire.modeling.AutoModelForSeq2SeqLM.from_pretrained"
                token_patch = patch.object(module.AutoTokenizer, "from_pretrained", return_value=tokenizer)
                if validation_only:
                    argv += ["--validation-only"]
            else:
                argv += ["--src-lang", "spa_Latn", "--tgt-lang", "spa_Latn", "--eval-beams", "4"]
                module = train_nllb
                model_patch = "src.nllb_suite.modeling_nllb.AutoModelForSeq2SeqLM.from_pretrained"
                token_patch = patch.object(module, "get_nllb_tokenizer", return_value=tokenizer)
            with patch.object(sys, "argv", argv), patch(model_patch, side_effect=factory), token_patch:
                module.main()
            run = root / "results" / run_name
            metrics = json.loads((run / "metrics.json").read_text())
            prediction_name = "validation_predictions.csv" if validation_only else "test_predictions.csv"
            predictions = pd.read_csv(run / prediction_name, keep_default_na=False)
            assert predictions[["source", "target"]].equals(frame)
            if validation_only:
                assert metrics["test_evaluated"] is False
                assert not any(key.startswith("test_") and key != "test_evaluated" for key in metrics)
                assert not (run / "test_predictions.csv").exists()
                frame.to_csv(data / "test.csv", index=False)
            if not validation_only:
                assert metrics["test_reference"] == "original_csv"
            preds = predictions["prediction"].tolist()
            refs = [frame["target"].tolist()]
            metric_prefix = "eval" if validation_only else "test"
            assert abs(metrics[metric_prefix + "_bleu"] - BLEU(tokenize="13a").corpus_score(preds, refs).score) < 1e-8
            assert abs(metrics[metric_prefix + "_chrf++"] - CHRF(word_order=2).corpus_score(preds, refs).score) < 1e-8
            best = run / "best_model" if backbone == "mbart" else Path(metrics["best_model_checkpoint"])
            restored = AutoModelForSeq2SeqLM.from_pretrained(best)
            assert restored.generation_config.forced_bos_token_id == 4
            saved_args = torch.load(best / "training_args.bin", weights_only=False)
            assert saved_args.weight_decay == 0.0
            assert saved_args.learning_rate == 5e-5
            print(f"PASS {run_name}: train/backward, best checkpoint, original refs, Spanish generation, reload", flush=True)


if __name__ == "__main__":
    main()
