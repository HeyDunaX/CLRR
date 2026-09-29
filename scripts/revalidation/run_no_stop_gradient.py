"""Empirical Stop-Gradient ablation on mT5-small.

Compares CLRR with stop_gradient (standard) vs. without stop_gradient (no-sg).
Records layer-wise gradient norms (||grad_W||_2) and training loss across 5 epochs
to empirically prove gradient instability and representation collapse without sg(.).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add src to sys.path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

import pandas as pd
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from amis_rewire.train import load_split, tokenize_dataset, build_compute_metrics
from amis_rewire.modeling import _find_stack_layers, _hidden_and_repack


class CrossLayerResidualRewireNoStopGrad(nn.Module):
    """CLRR variant WITHOUT stop_gradient (removes .detach()).

    Leads to backward gradients propagating backwards through cross-layer skip
    connections, empirically demonstrating gradient accumulation and representation collapse.
    """

    def __init__(
        self,
        base_model: nn.Module,
        distance: int = 2,
        strength: float = 0.1,
        stack: str = "encoder",
    ):
        super().__init__()
        self.base_model = base_model
        self.distance = distance
        self.strength = strength
        self.stack = stack
        self._cache: dict[str, dict[int, torch.Tensor]] = {}
        self._handles: list[Any] = []
        self._install_hooks()

    def _install_hooks(self) -> None:
        layers = _find_stack_layers(self.base_model, self.stack)
        self._cache[self.stack] = {}
        for index, layer in enumerate(layers):
            self._handles.append(
                layer.register_forward_hook(self._make_hook(self.stack, index))
            )

    def _make_hook(self, stack_name: str, index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden, torch.Tensor):
                return output
            source = self._cache[stack_name].get(index - self.distance)
            if source is not None and source.shape == hidden.shape:
                # Crucial difference: source has computation graph attached (NO detach)
                hidden = hidden + self.strength * source.to(dtype=hidden.dtype)
            # Store without detach()!
            self._cache[stack_name][index] = hidden
            return _hidden_and_repack(output, hidden)

        return hook

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    @property
    def config(self) -> Any:
        return self.base_model.config

    def get_encoder(self) -> nn.Module:
        return self.base_model.get_encoder()

    def get_decoder(self) -> nn.Module:
        return self.base_model.get_decoder()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("num_items_in_batch", None)
        self._cache.clear()
        self._cache[self.stack] = {}
        return self.base_model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("num_items_in_batch", None)
        self._cache.clear()
        self._cache[self.stack] = {}
        return self.base_model.generate(*args, **kwargs)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return self.base_model.prepare_decoder_input_ids_from_labels(labels=labels)



class GradientNormTracker(TrainerCallback):
    """Tracks L2 norm of gradients per layer during training."""

    def __init__(self, log_every_steps: int = 25):
        self.log_every_steps = log_every_steps
        self.trace: list[dict[str, Any]] = []

    def on_step_end(self, args: Any, state: Any, control: Any, model: nn.Module = None, **kwargs: Any):
        if state.global_step % self.log_every_steps == 0 and model is not None:
            total_norm = 0.0
            layer_norms = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    if "block" in name and "layer" in name:
                        parts = name.split(".")
                        # e.g., encoder.block.0
                        block_name = ".".join(parts[:3])
                        layer_norms[block_name] = layer_norms.get(block_name, 0.0) + param_norm ** 2

            total_norm = total_norm ** 0.5
            layer_norms = {k: v ** 0.5 for k, v in layer_norms.items()}

            self.trace.append({
                "step": state.global_step,
                "epoch": round(state.epoch, 2) if state.epoch else 0.0,
                "total_grad_norm": round(total_norm, 4),
                "layer_grad_norms": {k: round(v, 4) for k, v in layer_norms.items()},
            })


def train_no_stop_grad(epochs: float = 5.0, lr: float = 3e-4) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "processed"
    out_dir = repo_root / "outputs_revalidation" / "mt5-small-no-sg"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[init] Loading google/mt5-small for No-Stop-Gradient experiment...")
    model_name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = MT5ForConditionalGeneration.from_pretrained(model_name)

    model = CrossLayerResidualRewireNoStopGrad(
        base_model, distance=2, strength=0.1, stack="encoder"
    )

    args = argparse.Namespace(
        max_source_length=256,
        max_target_length=256,
        source_prefix="",
    )

    train_dataset = tokenize_dataset(load_split(data_dir, "train"), tokenizer, args)
    val_dataset = tokenize_dataset(load_split(data_dir, "validation"), tokenizer, args)
    test_dataset = tokenize_dataset(load_split(data_dir, "test"), tokenizer, args)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    grad_tracker = GradientNormTracker(log_every_steps=10)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        run_name="mt5-small-clrr-no-sg",
        seed=42,
        data_seed=42,
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_ratio=0.06,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=8,  # Effective batch = 128
        eval_strategy="epoch",
        save_strategy="no",  # Don't waste disk space, we only want convergence trace
        logging_strategy="steps",
        logging_steps=10,
        predict_with_generate=True,
        generation_num_beams=1,
        generation_max_length=256,
        fp16=False,
        bf16=torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
        gradient_checkpointing=False,  # Keep graph explicit for grad tracking
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[grad_tracker],
    )

    print("[train] Starting No-Stop-Gradient training (5 epochs)...")
    train_result = trainer.train()

    print("[eval] Evaluating on validation and test set...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    # Save gradient trace and summary
    summary = {
        "model": "google/mt5-small",
        "method": "CLRR-Enc (NO Stop-Gradient)",
        "epochs": epochs,
        "learning_rate": lr,
        "train_loss": train_result.training_loss,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "gradient_trace": grad_tracker.trace,
    }

    trace_file = repo_root / "outputs_revalidation" / "no_sg_gradient_trace.json"
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] Gradient trace and metrics saved to: {trace_file}")
    print(f"Validation chrF++: {val_metrics.get('val_chrf++', 'N/A')}")
    print(f"Test chrF++: {test_metrics.get('test_chrf++', 'N/A')}")


if __name__ == "__main__":
    train_no_stop_grad()
