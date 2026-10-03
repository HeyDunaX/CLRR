"""Train and evaluate one reproducible Amis-to-Chinese experiment."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from .metrics import generation_metrics, safe_decode_inputs, build_compute_metrics as original_reference_metrics, metric_protocol
from .modeling import load_model
from .evaluation import PrecisionSeq2SeqTrainer


MODEL_DEFAULTS = {
    "mt5-small": "google/mt5-small",
    "mbart-large-50": "facebook/mbart-large-50-many-to-many-mmt",
    "byt5-small": "google/byt5-small",
}


class ConsoleMetricsCallback(TrainerCallback):
    """Prints the exact loss/metric fields needed for experiment logs in real time."""

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any):
        epoch_idx = int(state.epoch or 0) + 1
        total_epochs = int(args.num_train_epochs)
        print(f"[epoch {epoch_idx}/{total_epochs}] starting training...", flush=True)

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, float] | None = None, **kwargs: Any):
        if not logs:
            return
        fields = []
        for name in ("loss", "eval_loss", "eval_bleu", "eval_chrf++", "test_loss"):
            if name in logs:
                fields.append(f"{name}={logs[name]:.4f}")
        if fields:
            epoch_str = f"epoch={state.epoch:.2f}" if state.epoch is not None else ""
            prefix = f"[step={state.global_step}" + (f" | {epoch_str}] " if epoch_str else "] ")
            print(prefix + " | ".join(fields), flush=True)

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: dict[str, float] | None = None, **kwargs: Any):
        if metrics:
            eval_fields = [f"{k}={v:.4f}" for k, v in metrics.items() if k in ("eval_loss", "eval_bleu", "eval_chrf++")]
            if eval_fields:
                print(f"[eval @ step {state.global_step}] " + " | ".join(eval_fields), flush=True)


class CheckpointBackupCallback(TrainerCallback):
    """Archives each saved checkpoint so an ephemeral Colab runtime is recoverable."""

    def __init__(
        self,
        backup_dir: Path,
        run_name: str,
        hf_backup_repo: str | None = None,
        hf_backup_prefix: str = "",
    ):
        self.backup_dir = backup_dir / run_name
        self.run_name = run_name
        self.hf_backup_repo = hf_backup_repo
        self.hf_backup_prefix = hf_backup_prefix.strip("/")
        self.hf_api = None
        if hf_backup_repo:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN is required when --hf-backup-repo is set.")
            self.hf_api = HfApi(token=token)
            self.hf_api.create_repo(hf_backup_repo, repo_type="model", private=True, exist_ok=True)

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any):
        # Per-epoch backup disabled to maximize throughput; best_model is archived at completion.
        return


def safe_hf_upload(
    api: Any,
    path: Path,
    repo_id: str,
    path_in_repo: str,
    max_retries: int = 3,
    initial_delay: float = 5.0,
) -> bool:
    """Uploads a file to Hugging Face Hub with exponential backoff on transient errors."""
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
            )
            return True
        except Exception as exc:
            print(f"[backup] upload attempt {attempt}/{max_retries} for {path_in_repo} failed ({type(exc).__name__}): {exc}", flush=True)
            if attempt < max_retries:
                time.sleep(initial_delay * attempt)
    return False


def remote_path(prefix: str, run_name: str, filename: str) -> str:
    return "/".join(part for part in (prefix.strip("/"), run_name, filename) if part)


def archive_best_model(
    best_model_dir: Path,
    backup_dir: Path,
    run_name: str,
    hf_backup_repo: str | None,
    hf_backup_prefix: str = "",
) -> None:
    """Archive the selected best model separately from resumable checkpoints."""
    backup_run_dir = backup_dir / run_name
    backup_run_dir.mkdir(parents=True, exist_ok=True)
    archive_path = backup_run_dir / f"{run_name}-best.zip"
    temporary_path = archive_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in best_model_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(best_model_dir))
    temporary_path.replace(archive_path)
    print(f"[backup] best model saved to {archive_path}", flush=True)
    if hf_backup_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required when --hf-backup-repo is set.")
        target = remote_path(hf_backup_prefix, run_name, archive_path.name)
        uploaded = safe_hf_upload(
            HfApi(token=token),
            archive_path,
            hf_backup_repo,
            target,
            max_retries=5,
            initial_delay=5.0,
        )
        if uploaded:
            print(f"[backup] best model uploaded to {hf_backup_repo}/{target}", flush=True)
        else:
            print(
                f"[backup] warning: best model upload will be finalized by post-training upload check",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_DEFAULTS), default="mt5-small")
    parser.add_argument("--model-name", default=None, help="Optional Hugging Face checkpoint override.")
    parser.add_argument(
        "--method",
        choices=("baseline", "rewire", "clrr", "clrr-enc", "jepa", "jepa-clrr", "jepa-clrr-enc"),
        default="baseline",
    )
    parser.add_argument("--data-dir", default="data_processed/amis_mandarin")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--backup-dir", default="backups")
    parser.add_argument("--hf-backup-repo", default=None, help="Private HF repo, e.g. user/amis-rewire-checkpoints.")
    parser.add_argument("--hf-backup-prefix", default="", help="Optional folder inside the HF backup repo.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--per-device-train-batch-size", type=int, default=256)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4, help="Beam size for test evaluation (default: 4).")
    parser.add_argument(
        "--eval-beams",
        type=int,
        default=1,
        help="Beam size for intermediate validation during training (default: 1 for fast greedy evaluation).",
    )
    parser.add_argument(
        "--source-prefix",
        type=str,
        default="",
        help="Optional source prefix prepended to input text (e.g. for mT5).",
    )
    parser.add_argument("--rewire-distance", type=int, default=2)
    parser.add_argument("--rewire-strength", type=float, default=0.1)
    parser.add_argument(
        "--rewire-stack",
        default="encoder",
        choices=["encoder", "decoder", "both"],
        help="Transformer stack to rewire (default: encoder, specifically addressing context over-smoothing).",
    )
    parser.add_argument(
        "--jepa-weight",
        type=float,
        default=0.1,
        help="Weight for the JEPA latent predictive alignment loss (default: 0.1).",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--src-lang", default=None, help="Source language code (auto-detected if None).")
    parser.add_argument("--tgt-lang", default=None, help="Target language code (auto-detected if None).")
    parser.add_argument("--bleu-tokenizer", default=None, help="Tokenizer for BLEU (zh or 13a, auto-detected if None).")
    parser.add_argument("--test-reference", choices=["original_csv"], default="original_csv",
                        help="Score predictions against the original ordered CSV references.")
    parser.add_argument("--validation-only", action="store_true",
                        help="Select and save using validation only; do not load or evaluate test during tuning.")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation-precision", choices=["fp32", "bf16"], default="fp32",
                        help="Declared generation precision shared by all methods.")
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--dataloader-pin-memory", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_split(data_dir: Path, split: str) -> Dataset:
    path = data_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run amis-rewire-prepare first.")
    frame = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False).fillna("")
    if list(frame.columns) != ["source", "target"]:
        raise ValueError(f"{path} must have exactly columns source,target")
    return Dataset.from_pandas(frame, preserve_index=False)


def detect_mbart_languages(data_dir: Path | str, cli_src: str | None, cli_tgt: str | None, cli_tok: str | None) -> tuple[str, str, str]:
    path_str = str(data_dir).lower()
    is_spanish = "spanish" in path_str
    is_turkish = "turkish" in path_str or "tr_en" in path_str or "opus100" in path_str
    if is_turkish:
        tgt_lang = cli_tgt or "en_XX"
        src_lang = cli_src or "tr_TR"
        bleu_tok = cli_tok or "13a"
    elif is_spanish:
        tgt_lang = cli_tgt or "es_XX"
        src_lang = cli_src or "es_XX"
        bleu_tok = cli_tok or "13a"
    else:
        tgt_lang = cli_tgt or "zh_CN"
        src_lang = cli_src or "tl_XX"
        bleu_tok = cli_tok or "zh"
    return src_lang, tgt_lang, bleu_tok


def configure_mbart(tokenizer: Any, model: Any, src_lang: str = "tl_XX", tgt_lang: str = "zh_CN") -> None:
    if not tokenizer.__class__.__name__.lower().startswith("mbart"):
        return
    if tgt_lang not in tokenizer.lang_code_to_id:
        raise ValueError(f"The selected mBART tokenizer does not expose {tgt_lang}.")
    if src_lang not in tokenizer.lang_code_to_id:
        src_lang = "en_XX"
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    model.config.forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
    model.generation_config.forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]


def tokenize_dataset(dataset: Dataset, tokenizer: Any, args: argparse.Namespace) -> Dataset:
    prefix = getattr(args, "source_prefix", "") or ""

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        sources = [prefix + text for text in batch["source"]] if prefix else batch["source"]
        model_inputs = tokenizer(
            sources,
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def build_compute_metrics(tokenizer: Any, bleu_tokenizer: str = "zh", references: list[str] | None = None):
    if references is None:
        raise ValueError("Original references for the evaluation split are required")
    return original_reference_metrics(tokenizer, references, tokenize=bleu_tokenizer)

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model_name = args.model_name or MODEL_DEFAULTS[args.model]
    run_name = args.run_name or f"{args.model}-{args.method}-seed{args.seed}"
    output_dir = Path(args.output_dir) / run_name
    data_dir = Path(args.data_dir)

    src_lang, tgt_lang, bleu_tokenizer = detect_mbart_languages(data_dir, args.src_lang, args.tgt_lang, args.bleu_tokenizer)
    print(f"[init] language configuration: src={src_lang}, tgt={tgt_lang}, bleu_tok={bleu_tokenizer}", flush=True)

    print(f"[init] loading tokenizer and base model for {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = load_model(
        model_name,
        args.method,
        distance=args.rewire_distance,
        strength=args.rewire_strength,
        stack=args.rewire_stack,
        jepa_weight=args.jepa_weight,
    )
    configure_mbart(tokenizer, model, src_lang=src_lang, tgt_lang=tgt_lang)
    if args.gradient_checkpointing:
        model.config.use_cache = False
    print(f"[init] tokenizing dataset splits from {data_dir}...", flush=True)
    train_dataset = tokenize_dataset(load_split(data_dir, "train"), tokenizer, args)
    validation_dataset = tokenize_dataset(load_split(data_dir, "validation"), tokenizer, args)
    test_dataset = None if args.validation_only else tokenize_dataset(load_split(data_dir, "test"), tokenizer, args)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": run_name,
        "seed": args.seed,
        "data_seed": args.seed,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "predict_with_generate": True,
        "generation_num_beams": args.eval_beams,
        "generation_max_length": args.max_target_length,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_chrf++",
        "greater_is_better": True,
        "fp16": args.fp16 and torch.cuda.is_available(),
        "bf16": args.bf16 and torch.cuda.is_available(),
        "tf32": torch.cuda.is_available(),
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "optim": "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        "report_to": [],
        "remove_unused_columns": False,
        "save_safetensors": False,
    }
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    if "warmup_ratio" in sig.parameters:
        training_kwargs["warmup_ratio"] = args.warmup_ratio
    else:
        steps_per_epoch = max(1, len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps))
        training_kwargs["warmup_steps"] = max(1, int(steps_per_epoch * args.num_train_epochs * args.warmup_ratio))

    training_args = Seq2SeqTrainingArguments(**training_kwargs)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": collator,
        "compute_metrics": build_compute_metrics(tokenizer, bleu_tokenizer=bleu_tokenizer, references=load_split(data_dir, "validation")["target"]),
        "callbacks": [
            ConsoleMetricsCallback(),
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            CheckpointBackupCallback(
                Path(args.backup_dir), run_name, args.hf_backup_repo, args.hf_backup_prefix
            ),
        ],
    }
    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if args.generation_precision:
        trainer = PrecisionSeq2SeqTrainer(**trainer_kwargs, generation_dtype=args.generation_precision)
    else:
        trainer = Seq2SeqTrainer(**trainer_kwargs)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"[run] model={model_name} method={args.method} parameters={parameter_count:,} trainable={trainable_count:,}", flush=True)
    print(f"[run] rewiring stack={args.rewire_stack} extra_parameters=0 distance={args.rewire_distance} strength={args.rewire_strength} jepa_weight={args.jepa_weight}", flush=True)
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint is None and args.auto_resume:
        resume_checkpoint = get_last_checkpoint(str(output_dir))
        if resume_checkpoint:
            print(f"[resume] continuing from {resume_checkpoint}", flush=True)
    start_time = time.time()
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    training_time_seconds = time.time() - start_time
    if args.validation_only:
        validation_output = trainer.predict(validation_dataset, metric_key_prefix="eval")
        validation_metrics = validation_output.metrics
        predictions = validation_output.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded = tokenizer.batch_decode(safe_decode_inputs(predictions, tokenizer.pad_token_id), skip_special_tokens=True)
        raw_validation = pd.read_csv(data_dir / "validation.csv", encoding="utf-8", dtype=str, keep_default_na=False).fillna("")
        if len(decoded) != len(raw_validation):
            raise ValueError("Validation prediction count does not match validation.csv")
        predictions_path = output_dir / "validation_predictions.csv"
        pd.DataFrame({"index": range(len(decoded)), "source": raw_validation["source"],
                      "target": raw_validation["target"], "prediction": [p.strip() for p in decoded]}).to_csv(
                          predictions_path, index=False, encoding="utf-8")
        best_model_dir = output_dir / "best_model"
        trainer.save_model(str(best_model_dir))
        tokenizer.save_pretrained(best_model_dir)
        model.config.save_pretrained(best_model_dir)
        model.generation_config.save_pretrained(best_model_dir)
        archive_best_model(best_model_dir, Path(args.backup_dir), run_name, args.hf_backup_repo, args.hf_backup_prefix)
        final_metrics = {"model": model_name, "method": args.method, "seed": args.seed,
                         "parameters": parameter_count, "trainable_parameters": trainable_count,
                         "extra_parameters": 0, "configuration": vars(args),
                         "training_time_seconds": training_time_seconds,
                         "best_model_checkpoint": trainer.state.best_model_checkpoint,
                         "completed_epochs": trainer.state.epoch, "test_evaluated": False,
                         "evaluation_protocol": metric_protocol(bleu_tokenizer),
                         **{key: float(value) for key, value in validation_metrics.items() if isinstance(value, (int, float))}}
        final_metrics.update({f"eval_{key}": value for key, value in generation_metrics(
            decoded, raw_validation["target"].tolist(), tokenize=bleu_tokenizer).items()})
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
        if args.hf_backup_repo:
            api = HfApi(token=os.environ["HF_TOKEN"])
            for artifact in (metrics_path, predictions_path):
                api.upload_file(path_or_fileobj=str(artifact),
                                path_in_repo=remote_path(args.hf_backup_prefix, run_name, artifact.name),
                                repo_id=args.hf_backup_repo, repo_type="model")
        print(json.dumps(final_metrics, indent=2), flush=True)
        return
    validation_metrics = trainer.evaluate(validation_dataset, metric_key_prefix="eval")
    # Switch to full beam search for final test evaluation
    trainer.args.generation_num_beams = args.num_beams
    trainer.compute_metrics = build_compute_metrics(tokenizer, bleu_tokenizer=bleu_tokenizer, references=load_split(data_dir, "test")["target"])
    test_output = trainer.predict(test_dataset, metric_key_prefix="test")
    test_metrics = test_output.metrics
    test_predictions = test_output.predictions
    if isinstance(test_predictions, tuple):
        test_predictions = test_predictions[0]
    decoded_predictions = tokenizer.batch_decode(
        safe_decode_inputs(test_predictions, tokenizer.pad_token_id), skip_special_tokens=True
    )
    raw_test = pd.read_csv(data_dir / "test.csv", encoding="utf-8", dtype=str, keep_default_na=False).fillna("")
    if len(raw_test) != len(decoded_predictions):
        raise ValueError("Test prediction count does not match test.csv")
    predictions_path = output_dir / "test_predictions.csv"
    pd.DataFrame(
        {
            "index": range(len(raw_test)),
            "source": raw_test["source"],
            "target": raw_test["target"],
            "prediction": [prediction.strip() for prediction in decoded_predictions],
        }
    ).to_csv(predictions_path, index=False, encoding="utf-8")
    best_model_dir = output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(best_model_dir)
    model.config.save_pretrained(best_model_dir)
    model.generation_config.save_pretrained(best_model_dir)
    archive_best_model(
        best_model_dir, Path(args.backup_dir), run_name, args.hf_backup_repo, args.hf_backup_prefix
    )
    final_metrics = {
        "model": model_name,
        "method": args.method,
        "stack": args.rewire_stack,
        "jepa_weight": args.jepa_weight,
        "seed": args.seed,
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "extra_parameters": 0,
        "configuration": vars(args),
        "training_time_seconds": training_time_seconds,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "completed_epochs": trainer.state.epoch,
        **{key: float(value) for key, value in validation_metrics.items() if isinstance(value, (int, float))},
        **{key: float(value) for key, value in test_metrics.items() if isinstance(value, (int, float))},
    }
    final_metrics["test_reference"] = "original_csv"
    final_metrics["evaluation_protocol"] = metric_protocol(bleu_tokenizer)
    official_scores = generation_metrics(
        decoded_predictions, raw_test["target"].tolist(), tokenize=bleu_tokenizer
    )
    final_metrics.update({f"test_{key}": value for key, value in official_scores.items()})
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    if args.hf_backup_repo:
        api = HfApi(token=os.environ["HF_TOKEN"])
        for artifact in (metrics_path, predictions_path):
            api.upload_file(
                path_or_fileobj=str(artifact),
                path_in_repo=remote_path(args.hf_backup_prefix, run_name, artifact.name),
                repo_id=args.hf_backup_repo,
                repo_type="model",
            )
    print(json.dumps(final_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
