"""Train and evaluate NLLB-200 under strict fairness protocols."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from .modeling_nllb import DEFAULT_NLLB_MODEL, DEFAULT_TGT_LANG, get_nllb_tokenizer, load_nllb_model


DEFAULT_LR_BY_METHOD = {
    "baseline": 5e-5,
    "bitfit": 1e-4,
    "lora": 2e-4,
    "layerskip": 5e-5,
    "middle_align": 5e-5,
    "clrr_enc": 5e-5,
    "clrr_dec": 5e-5,
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


def safe_decode_inputs(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions)
    return np.where(predictions != -100, predictions, 0)


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
            print(f"[backup] upload attempt {attempt}/{max_retries} for {path_in_repo} failed: {exc}", flush=True)
            if attempt < max_retries:
                time.sleep(initial_delay * (2 ** (attempt - 1)))
    return False


def load_split(data_dir: Path, split: str) -> Dataset:
    path = data_dir / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split CSV at {path}")
    dataframe = pd.read_csv(path)
    return Dataset.from_pandas(dataframe, preserve_index=False)


def tokenize_dataset(dataset: Dataset, tokenizer: Any, args: argparse.Namespace) -> Dataset:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            text=batch["source"],
            text_target=batch["target"],
            max_length=args.max_source_length,
            max_target_length=args.max_target_length,
            truncation=True,
        )

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def build_compute_metrics(tokenizer: Any):
    bleu_metric = BLEU(tokenize="zh")
    chrfpp_metric = CHRF(word_order=2)

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_predictions = tokenizer.batch_decode(safe_decode_inputs(predictions), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(safe_decode_inputs(labels), skip_special_tokens=True)
        preds_clean = [p.strip() for p in decoded_predictions]
        refs_clean = [[r.strip() for r in decoded_labels]]
        return {
            "bleu": float(bleu_metric.corpus_score(preds_clean, refs_clean).score),
            "chrf++": float(chrfpp_metric.corpus_score(preds_clean, refs_clean).score),
        }

    return compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=["baseline", "bitfit", "lora", "layerskip", "middle_align", "clrr_enc", "clrr_dec"],
        help="Experimental method to run on NLLB-200.",
    )
    parser.add_argument("--model-name", default=DEFAULT_NLLB_MODEL, help="NLLB-200 model checkpoint.")
    parser.add_argument("--data-dir", default="data/processed", help="Path to processed dataset directory.")
    parser.add_argument("--output-dir", default="results/nllb-200", help="Root directory for outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fairness.")
    parser.add_argument("--num-train-epochs", type=float, default=20.0, help="Total training epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate override.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--eval-beams", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--hf-backup-repo", default="FiveC/amis-rewire-checkpoints")
    parser.add_argument("--hf-backup-prefix", default="nllb-200")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    lr = args.learning_rate or DEFAULT_LR_BY_METHOD[args.method]
    run_name = args.method
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print(f"[init] NLLB-200 Suite: method={args.method}, lr={lr}, seed={args.seed}", flush=True)
    tokenizer = get_nllb_tokenizer(args.model_name, src_lang="zho_Hant", tgt_lang="zho_Hant")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_TGT_LANG)

    model, method_metadata = load_nllb_model(args.method, model_name=args.model_name)
    # Configure generation target language
    if hasattr(model, "config"):
        model.config.forced_bos_token_id = forced_bos_token_id
    elif hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.forced_bos_token_id = forced_bos_token_id

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[init] Trainable params: {trainable_params:,} / {total_params:,} ({100.0 * trainable_params / total_params:.4f}%)", flush=True)

    print(f"[init] Tokenizing dataset splits from {data_dir}...", flush=True)
    train_dataset = tokenize_dataset(load_split(data_dir, "train"), tokenizer, args)
    validation_dataset = tokenize_dataset(load_split(data_dir, "validation"), tokenizer, args)
    test_dataset = tokenize_dataset(load_split(data_dir, "test"), tokenizer, args)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": run_name,
        "seed": args.seed,
        "data_seed": args.seed,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": lr,
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
        "dataloader_num_workers": args.dataloader_num_workers,
        "report_to": "none",
    }

    import inspect
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    valid_params = set(sig.parameters.keys())

    if "eval_strategy" in valid_params:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in valid_params:
        training_kwargs["evaluation_strategy"] = "epoch"

    if "warmup_ratio" in valid_params:
        training_kwargs["warmup_ratio"] = args.warmup_ratio
    elif "warmup_steps" in valid_params:
        eff_bs = args.per_device_train_batch_size * args.gradient_accumulation_steps
        steps_per_epoch = max(1, len(train_dataset) // eff_bs)
        total_steps = int(steps_per_epoch * args.num_train_epochs)
        training_kwargs["warmup_steps"] = max(1, int(total_steps * args.warmup_ratio))

    final_kwargs = {k: v for k, v in training_kwargs.items() if k in valid_params}
    training_args = Seq2SeqTrainingArguments(**final_kwargs)
    callbacks: list[TrainerCallback] = [
        ConsoleMetricsCallback(),
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
    ]

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=callbacks,
    )

    last_checkpoint = get_last_checkpoint(str(output_dir))
    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    training_time_seconds = time.time() - start_time

    print("[eval] Evaluating best checkpoint on validation set...", flush=True)
    val_metrics = trainer.evaluate(eval_dataset=validation_dataset, metric_key_prefix="eval")

    print("[eval] Evaluating best checkpoint on test set...", flush=True)
    test_predictions = trainer.predict(test_dataset=test_dataset, metric_key_prefix="test")
    test_metrics = test_predictions.metrics or {}

    # Extract test predictions and calculate full metrics (including TokenizerZh)
    raw_preds = test_predictions.predictions
    if isinstance(raw_preds, tuple):
        raw_preds = raw_preds[0]
    decoded_preds = tokenizer.batch_decode(safe_decode_inputs(raw_preds), skip_special_tokens=True)
    clean_preds = [p.strip() for p in decoded_preds]

    test_raw_df = pd.read_csv(data_dir / "test.csv")
    test_targets = [str(t).strip() for t in test_raw_df["target"].tolist()]

    # Save test predictions CSV
    preds_df = pd.DataFrame({
        "source": test_raw_df["source"].tolist(),
        "target": test_targets,
        "prediction": clean_preds,
    })
    preds_path = output_dir / "test_predictions.csv"
    preds_df.to_csv(preds_path, index=False, encoding="utf-8")
    print(f"[eval] Saved predictions to {preds_path}", flush=True)

    # Compute official scores
    bleu_score = float(BLEU(tokenize="zh").corpus_score(clean_preds, [test_targets]).score)
    chrf_score = float(CHRF(word_order=2).corpus_score(clean_preds, [test_targets]).score)
    tok_zh = TokenizerZh()
    preds_zh = [tok_zh(p) for p in clean_preds]
    refs_zh = [[tok_zh(t) for t in test_targets]]
    chrf_zh_score = float(CHRF(word_order=2).corpus_score(preds_zh, refs_zh).score)

    report: dict[str, Any] = {
        "model": args.model_name,
        "method": args.method,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_percent": (100.0 * trainable_params / total_params) if total_params > 0 else 0.0,
        "training_time_seconds": training_time_seconds,
        "eval_bleu": val_metrics.get("eval_bleu", 0.0),
        "eval_chrf++": val_metrics.get("eval_chrf++", 0.0),
        "test_bleu": bleu_score,
        "test_chrf++": chrf_score,
        "test_chrf++_zh": chrf_zh_score,
        "metadata": method_metadata,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[eval] Saved metrics report to {metrics_path}", flush=True)
    print(f"=== RESULT for {args.method} ===")
    print(f"Test BLEU:        {bleu_score:.4f}")
    print(f"Test chrF++:      {chrf_score:.4f}")
    print(f"Test chrF++ (Zh): {chrf_zh_score:.4f}")

    # Archive best checkpoint and optionally upload to HF
    best_dir = Path(trainer.state.best_model_checkpoint) if trainer.state.best_model_checkpoint else output_dir
    archive_path = output_dir / f"nllb-200-{args.method}-best.zip"
    print(f"[archive] Archiving best checkpoint from {best_dir} to {archive_path}...", flush=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file in best_dir.rglob("*"):
            if file.is_file() and not file.name.endswith(".zip"):
                zipf.write(file, file.relative_to(best_dir))
        if preds_path.is_file():
            zipf.write(preds_path, preds_path.name)
        if metrics_path.is_file():
            zipf.write(metrics_path, metrics_path.name)

    if args.hf_backup_repo and os.environ.get("HF_TOKEN"):
        api = HfApi(token=os.environ["HF_TOKEN"])
        target_path = f"{args.hf_backup_prefix}/{archive_path.name}"
        print(f"[upload] Uploading {archive_path} to {args.hf_backup_repo}/{target_path}...", flush=True)
        safe_hf_upload(api, archive_path, args.hf_backup_repo, target_path)

    print(f"[complete] Finished run for method: {args.method}!", flush=True)


if __name__ == "__main__":
    main()
