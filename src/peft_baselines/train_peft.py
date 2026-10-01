"""Training script for PEFT Baselines (LoRA & BitFit) on Amis-to-Mandarin Translation.

Strictly follows the project's Fair Comparison Protocol:
- Backbone: facebook/mbart-large-50-many-to-many-mmt
- Data: 4,600 train / 576 val / 575 test
- Seed: 42
- Effective batch size: 128 (per_device=4, grad_accum=32 on single A100)
- Max source/target length: 256
- Early stopping: patience 4 on validation chrF++
- Generation: greedy during validation, beam_size=4 on final test set
- Metrics: SacreBLEU (tokenize='zh') + chrF++ (word_order=2)
- Artifacts: metrics.json, test_predictions.csv, best_model.zip saved to results/
"""

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
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    MBartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)

try:
    from .lora_adapter import apply_lora_to_model
    from .bitfit_adapter import apply_bitfit_to_model
except ImportError:
    from src.peft_baselines.lora_adapter import apply_lora_to_model
    from src.peft_baselines.bitfit_adapter import apply_bitfit_to_model


BLEU_METRIC = BLEU(tokenize="zh")
CHRFPP_METRIC = CHRF(word_order=2)


def generation_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    predictions = [p.strip() for p in predictions]
    references = [[r.strip() for r in references]]
    return {
        "bleu": float(BLEU_METRIC.corpus_score(predictions, references).score),
        "chrf++": float(CHRFPP_METRIC.corpus_score(predictions, references).score),
    }


def safe_decode_inputs(predictions: Any) -> Any:
    preds = np.asarray(predictions)
    return np.where(preds != -100, preds, 0)


class ConsoleMetricsCallback(TrainerCallback):
    def __init__(self, method_name: str):
        self.method_name = method_name

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any):
        epoch_idx = int(state.epoch or 0) + 1
        total_epochs = int(args.num_train_epochs)
        print(f"[{self.method_name}] [Epoch {epoch_idx}/{total_epochs}] starting training...", flush=True)

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, float] | None = None, **kwargs: Any):
        if not logs:
            return
        fields = []
        for name in ("loss", "eval_loss", "eval_bleu", "eval_chrf++"):
            if name in logs:
                fields.append(f"{name}={logs[name]:.4f}")
        if fields:
            epoch_str = f"epoch={state.epoch:.2f}" if state.epoch is not None else ""
            prefix = f"[{self.method_name}] [step={state.global_step}" + (f" | {epoch_str}] " if epoch_str else "] ")
            print(prefix + " | ".join(fields), flush=True)

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: dict[str, float] | None = None, **kwargs: Any):
        if metrics:
            eval_fields = [f"{k}={v:.4f}" for k, v in metrics.items() if k in ("eval_loss", "eval_bleu", "eval_chrf++")]
            if eval_fields:
                print(f"[{self.method_name} Eval @ step {state.global_step}] " + " | ".join(eval_fields), flush=True)


def configure_mbart(tokenizer: Any, model: Any) -> None:
    """Configures tokenizer languages and forced_bos_token_id for mBART translation."""
    if "zh_CN" not in tokenizer.lang_code_to_id:
        raise ValueError("The selected mBART tokenizer does not expose zh_CN.")
    src_lang = "tl_XX" if "tl_XX" in tokenizer.lang_code_to_id else ("id_XX" if "id_XX" in tokenizer.lang_code_to_id else "en_XX")
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = "zh_CN"
    zh_id = tokenizer.lang_code_to_id["zh_CN"]

    # Support both raw model and PEFT-wrapped model
    if hasattr(model, "config"):
        model.config.forced_bos_token_id = zh_id
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.forced_bos_token_id = zh_id


def load_split(data_dir: Path, split: str) -> Dataset:
    path = data_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run amis-rewire-prepare first.")
    frame = pd.read_csv(path, encoding="utf-8").fillna("")
    return Dataset.from_pandas(frame, preserve_index=False)


def tokenize_dataset(dataset: Dataset, tokenizer: Any, args: argparse.Namespace) -> Dataset:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        model_inputs = tokenizer(batch["source"], max_length=args.max_source_length, truncation=True)
        labels = tokenizer(text_target=batch["target"], max_length=args.max_target_length, truncation=True)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("lora", "bitfit"), required=True, help="PEFT method to apply.")
    parser.add_argument(
        "--model-name",
        default="facebook/mbart-large-50-many-to-many-mmt",
        help="Hugging Face model checkpoint (default: mbart-large-50).",
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=20.0)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=None, help="Auto-selected based on method if None.")
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4, help="Beam size for test evaluation.")
    parser.add_argument("--eval-beams", type=int, default=1, help="Beam size for validation during training.")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-backup-repo", default="FiveC/amis-rewire-checkpoints")
    # LoRA specific arguments
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank r (default: 8).")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha scaling (default: 16).")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--target-modules", type=str, default="q_proj,v_proj", help="Comma-separated module names for LoRA.")
    parser.add_argument("--unfreeze-embeddings", action=argparse.BooleanOptionalAction, default=False, help="Whether to unfreeze and train token embeddings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    short_name = args.model_name.split("/")[-1].replace("-many-to-many-mmt", "")
    run_name = args.run_name or f"{short_name}-{args.method}"
    output_dir = Path(args.output_dir) / run_name
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Learning rate selection: LoRA standard is 2e-4, BitFit standard is 1e-4
    if args.learning_rate is None:
        learning_rate = 2e-4 if args.method == "lora" else 1e-4
    else:
        learning_rate = args.learning_rate

    print(f"\n=======================================================", flush=True)
    print(f"Starting PEFT Evaluation: {args.method.upper()}", flush=True)
    print(f"Model: {args.model_name} | Run name: {run_name}", flush=True)
    print(f"Learning rate: {learning_rate} | Seed: {args.seed} | Batch: {args.per_device_train_batch_size}x{args.gradient_accumulation_steps}=128", flush=True)
    print(f"=======================================================\n", flush=True)

    # Load tokenizer & raw base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    raw_model = MBartForConditionalGeneration.from_pretrained(args.model_name)

    # Apply PEFT method
    peft_stats = {}
    if args.method == "lora":
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        modules_to_save = ["shared", "lm_head"] if args.unfreeze_embeddings else None
        model, peft_stats = apply_lora_to_model(
            raw_model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
    elif args.method == "bitfit":
        model, peft_stats = apply_bitfit_to_model(raw_model, bias_components=["bias"])
    else:
        raise ValueError(f"Unsupported PEFT method: {args.method}")

    configure_mbart(tokenizer, model)

    # Load & tokenize dataset splits
    print(f"[{args.method.upper()}] Tokenizing dataset splits from {data_dir}...", flush=True)
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
        learning_rate=learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        generation_num_beams=args.eval_beams,
        generation_max_length=args.max_target_length,
        load_best_model_at_end=True,
        metric_for_best_model="eval_chrf++",
        greater_is_better=True,
        fp16=False,
        bf16=args.bf16 and torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
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
            ConsoleMetricsCallback(args.method.upper()),
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
        ],
    )

    # Resume handling
    has_checkpoints = any(output_dir.glob("checkpoint-*"))
    resume_checkpoint = None
    if args.auto_resume and has_checkpoints:
        print(f"[{args.method.upper()}] Checkpoints detected in {output_dir}. Resuming training...", flush=True)
        resume_checkpoint = True

    print(f"[{args.method.upper()}] Starting training on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}...", flush=True)
    start_time = time.time()
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    training_time = time.time() - start_time

    # Best model validation
    val_metrics = trainer.evaluate(validation_dataset, metric_key_prefix="eval")

    # Final test evaluation with beam size 4
    print(f"[{args.method.upper()}] Running final test evaluation with beam_size={args.num_beams}...", flush=True)
    trainer.args.generation_num_beams = args.num_beams
    test_output = trainer.predict(test_dataset, metric_key_prefix="test")
    test_metrics = test_output.metrics

    test_predictions = test_output.predictions
    if isinstance(test_predictions, tuple):
        test_predictions = test_predictions[0]
    decoded_predictions = tokenizer.batch_decode(safe_decode_inputs(test_predictions), skip_special_tokens=True)

    raw_test = pd.read_csv(data_dir / "test.csv", encoding="utf-8").fillna("")
    pred_df = pd.DataFrame({
        "index": range(len(raw_test)),
        "source": raw_test["source"],
        "target": raw_test["target"],
        "prediction": decoded_predictions,
    })
    pred_path = output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    all_metrics = {
        "model": run_name,
        "method": args.method,
        "reference_paper": "Hu et al. (ICLR 2022)" if args.method == "lora" else "Ben-Zaken et al. (ACL 2022)",
        "trainable_parameters": peft_stats.get("trainable_params"),
        "total_parameters": peft_stats.get("all_params"),
        "trainable_percent": peft_stats.get("trainable_percent"),
        "training_time_seconds": training_time,
        "eval_bleu": val_metrics.get("eval_bleu"),
        "eval_chrf++": val_metrics.get("eval_chrf++"),
        "test_bleu": test_metrics.get("test_bleu"),
        "test_chrf++": test_metrics.get("test_chrf++"),
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n=======================================================", flush=True)
    print(f"[{args.method.upper()}] Final Test Results:", flush=True)
    print(f"BLEU (zh): {test_metrics.get('test_bleu', 0):.4f}", flush=True)
    print(f"chrF++ (w=2): {test_metrics.get('test_chrf++', 0):.4f}", flush=True)
    print(f"Trainable params: {peft_stats.get('trainable_params', 0):,} ({peft_stats.get('trainable_percent', 0):.4f}%)", flush=True)
    print(f"=======================================================\n", flush=True)

    # Push metrics and test predictions immediately so they are never lost
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and args.hf_backup_repo:
        try:
            print(f"[{args.method.upper()}] Uploading metrics & predictions to Hugging Face ({args.hf_backup_repo})...", flush=True)
            api = HfApi(token=hf_token)
            api.create_repo(args.hf_backup_repo, repo_type="model", private=True, exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(metrics_path),
                path_in_repo=f"peft_baselines/{run_name}/metrics.json",
                repo_id=args.hf_backup_repo,
                repo_type="model",
            )
            api.upload_file(
                path_or_fileobj=str(pred_path),
                path_in_repo=f"peft_baselines/{run_name}/test_predictions.csv",
                repo_id=args.hf_backup_repo,
                repo_type="model",
            )
            print(f"[{args.method.upper()}] Metrics & predictions uploaded to HF successfully!", flush=True)
        except Exception as e:
            print(f"[{args.method.upper()}] Warning during early HF upload: {e}", flush=True)

    # Save best model to zip
    best_model_dir = output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    zip_path = output_dir / "best_model.zip"
    print(f"[{args.method.upper()}] Archiving best model to {zip_path}...", flush=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_p in best_model_dir.rglob("*"):
            if file_p.is_file():
                archive.write(file_p, file_p.relative_to(best_model_dir))

    # Push weights archive to Hugging Face
    if hf_token and args.hf_backup_repo:
        try:
            print(f"[{args.method.upper()}] Uploading best model archive to Hugging Face...", flush=True)
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=str(zip_path),
                path_in_repo=f"peft_baselines/{run_name}/best_model.zip",
                repo_id=args.hf_backup_repo,
                repo_type="model",
            )
            print(f"[{args.method.upper()}] Best model archive uploaded to HF!", flush=True)
        except Exception as e:
            print(f"[{args.method.upper()}] Warning during zip upload: {e}", flush=True)



if __name__ == "__main__":
    main()
