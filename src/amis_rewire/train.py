"""Train and evaluate one reproducible Amis-to-Chinese experiment."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
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

from .metrics import generation_metrics, safe_decode_inputs
from .modeling import load_model


MODEL_DEFAULTS = {
    "mt5-small": "google/mt5-small",
    "mbart-large-50": "facebook/mbart-large-50-many-to-many-mmt",
    "byt5-small": "google/byt5-small",
}


class ConsoleMetricsCallback(TrainerCallback):
    """Prints the exact loss/metric fields needed for experiment logs."""

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, float] | None = None, **kwargs: Any):
        if not logs:
            return
        fields = []
        for name in ("loss", "eval_loss", "eval_bleu", "eval_chrf++", "test_loss"):
            if name in logs:
                fields.append(f"{name}={logs[name]:.4f}")
        if fields:
            print(f"[step={state.global_step}] " + " | ".join(fields), flush=True)


class CheckpointBackupCallback(TrainerCallback):
    """Archives each saved checkpoint so an ephemeral Colab runtime is recoverable."""

    def __init__(self, backup_dir: Path, run_name: str, hf_backup_repo: str | None = None):
        self.backup_dir = backup_dir / run_name
        self.run_name = run_name
        self.hf_backup_repo = hf_backup_repo
        self.hf_api = None
        if hf_backup_repo:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN is required when --hf-backup-repo is set.")
            self.hf_api = HfApi(token=token)
            self.hf_api.create_repo(hf_backup_repo, repo_type="model", private=True, exist_ok=True)

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self.backup_dir / f"checkpoint-{state.global_step}.zip"
        temporary_path = archive_path.with_suffix(".tmp.zip")
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in checkpoint_dir.rglob("*"):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(checkpoint_dir))
        temporary_path.replace(archive_path)
        print(f"[backup] saved {archive_path}", flush=True)
        if self.hf_api is not None:
            self.hf_api.upload_file(
                path_or_fileobj=str(archive_path),
                path_in_repo=f"{self.run_name}/{archive_path.name}",
                repo_id=self.hf_backup_repo,
                repo_type="model",
            )
            print(f"[backup] uploaded {self.run_name}/{archive_path.name}", flush=True)


def archive_best_model(best_model_dir: Path, backup_dir: Path, run_name: str, hf_backup_repo: str | None) -> None:
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
        HfApi(token=token).upload_file(
            path_or_fileobj=str(archive_path),
            path_in_repo=f"{run_name}/{archive_path.name}",
            repo_id=hf_backup_repo,
            repo_type="model",
        )
        print(f"[backup] best model uploaded to {hf_backup_repo}/{run_name}/{archive_path.name}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_DEFAULTS), default="mt5-small")
    parser.add_argument("--model-name", default=None, help="Optional Hugging Face checkpoint override.")
    parser.add_argument(
        "--method",
        choices=("baseline", "rewire", "clrr", "clrr-enc", "jepa", "jepa-clrr", "jepa-clrr-enc"),
        default="baseline",
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--backup-dir", default="backups")
    parser.add_argument("--hf-backup-repo", default=None, help="Private HF repo, e.g. user/amis-rewire-checkpoints.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--per-device-train-batch-size", type=int, default=256)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
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
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--dataloader-pin-memory", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_split(data_dir: Path, split: str) -> Dataset:
    path = data_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run amis-rewire-prepare first.")
    frame = pd.read_csv(path, encoding="utf-8").fillna("")
    if list(frame.columns) != ["source", "target"]:
        raise ValueError(f"{path} must have exactly columns source,target")
    return Dataset.from_pandas(frame, preserve_index=False)


def configure_mbart(tokenizer: Any, model: Any) -> None:
    if not tokenizer.__class__.__name__.lower().startswith("mbart"):
        return
    if "zh_CN" not in tokenizer.lang_code_to_id:
        raise ValueError("The selected mBART tokenizer does not expose zh_CN.")
    tokenizer.src_lang = "zh_CN"
    tokenizer.tgt_lang = "zh_CN"
    model.config.forced_bos_token_id = tokenizer.lang_code_to_id["zh_CN"]


def tokenize_dataset(dataset: Dataset, tokenizer: Any, args: argparse.Namespace) -> Dataset:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        model_inputs = tokenizer(
            batch["source"],
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


def build_compute_metrics(tokenizer: Any):
    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_predictions = tokenizer.batch_decode(safe_decode_inputs(predictions), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(safe_decode_inputs(labels), skip_special_tokens=True)
        return generation_metrics(decoded_predictions, decoded_labels)

    return compute_metrics


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

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = load_model(
        model_name,
        args.method,
        distance=args.rewire_distance,
        strength=args.rewire_strength,
        stack=args.rewire_stack,
        jepa_weight=args.jepa_weight,
    )
    configure_mbart(tokenizer, model.base_model if hasattr(model, "base_model") else model)
    if args.gradient_checkpointing:
        model.config.use_cache = False
    train_dataset = tokenize_dataset(load_split(data_dir, "train"), tokenizer, args)
    validation_dataset = tokenize_dataset(load_split(data_dir, "validation"), tokenizer, args)
    test_dataset = tokenize_dataset(load_split(data_dir, "test"), tokenizer, args)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        generation_max_length=args.max_target_length,
        load_best_model_at_end=True,
        metric_for_best_model="eval_chrf++",
        greater_is_better=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        bf16=args.bf16 and torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        save_safetensors=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[
            ConsoleMetricsCallback(),
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            CheckpointBackupCallback(Path(args.backup_dir), run_name, args.hf_backup_repo),
        ],
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"[run] model={model_name} method={args.method} parameters={parameter_count:,} trainable={trainable_count:,}")
    print(f"[run] rewiring stack={args.rewire_stack} extra_parameters=0 distance={args.rewire_distance} strength={args.rewire_strength} jepa_weight={args.jepa_weight}")
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint is None and args.auto_resume:
        resume_checkpoint = get_last_checkpoint(str(output_dir))
        if resume_checkpoint:
            print(f"[resume] continuing from {resume_checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    validation_metrics = trainer.evaluate(validation_dataset, metric_key_prefix="eval")
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    best_model_dir = output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(best_model_dir)
    archive_best_model(best_model_dir, Path(args.backup_dir), run_name, args.hf_backup_repo)
    final_metrics = {
        "model": model_name,
        "method": args.method,
        "stack": args.rewire_stack,
        "jepa_weight": args.jepa_weight,
        "seed": args.seed,
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "extra_parameters": 0,
        **{key: float(value) for key, value in validation_metrics.items() if isinstance(value, (int, float))},
        **{key: float(value) for key, value in test_metrics.items() if isinstance(value, (int, float))},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(json.dumps(final_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()