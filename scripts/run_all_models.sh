#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/processed}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
BACKUP_DIR="${BACKUP_DIR:-backups}"
HF_BACKUP_REPO="${HF_BACKUP_REPO:-}"
HF_ARGS=()
if [[ -n "$HF_BACKUP_REPO" ]]; then
  HF_ARGS+=(--hf-backup-repo "$HF_BACKUP_REPO")
fi

RESTORE_ARGS=(--backup-dir "$BACKUP_DIR" --output-dir "$OUTPUT_DIR" "${HF_ARGS[@]}")
python scripts/restore_backups.py "${RESTORE_ARGS[@]}"

# Focused 6-run protocol (Option A):
# - 1-4: Full ablation on central backbone mt5-small (baseline, clrr-enc, jepa, jepa-clrr-enc)
# - 5-6: Cross-architecture validation on mbart-large-50 (baseline, jepa-clrr-enc)
EXPERIMENTS=(
  "mt5-small:baseline"
  "mt5-small:clrr-enc"
  "mt5-small:jepa"
  "mt5-small:jepa-clrr-enc"
  "mbart-large-50:baseline"
  "mbart-large-50:jepa-clrr-enc"
)

for EXP in "${EXPERIMENTS[@]}"; do
  MODEL="${EXP%%:*}"
  METHOD="${EXP##*:}"
  RUN_NAME="${MODEL}-ami-cmn-${METHOD}"
  if [[ -f "$OUTPUT_DIR/$RUN_NAME/metrics.json" ]]; then
    echo "[skip] $RUN_NAME is complete"
    continue
  fi
  # Safe batch size per model to prevent CUDA OOM while maintaining effective batch size of 256
  if [[ "$MODEL" == *"mbart"* ]]; then
    TRAIN_BS="${MBART_TRAIN_BATCH_SIZE:-32}"
    EVAL_BS="${MBART_EVAL_BATCH_SIZE:-32}"
    GRAD_ACC="${MBART_GRAD_ACC:-8}"
  else
    TRAIN_BS="${PER_DEVICE_TRAIN_BATCH_SIZE:-128}"
    EVAL_BS="${PER_DEVICE_EVAL_BATCH_SIZE:-64}"
    GRAD_ACC="${GRADIENT_ACCUMULATION_STEPS:-2}"
  fi

  python -m amis_rewire.train \
    --model "$MODEL" \
    --method "$METHOD" \
    --rewire-stack "${REWIRE_STACK:-encoder}" \
    --jepa-weight "${JEPA_WEIGHT:-0.1}" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --seed 42 \
    --run-name "$RUN_NAME" \
    --backup-dir "$BACKUP_DIR" \
    --num-train-epochs 5 \
    --early-stopping-patience 2 \
    --learning-rate 5e-5 \
    --warmup-ratio 0.06 \
    --per-device-train-batch-size "$TRAIN_BS" \
    --per-device-eval-batch-size "$EVAL_BS" \
    --gradient-accumulation-steps "$GRAD_ACC" \
    --max-source-length 256 \
    --max-target-length 256 \
    --num-beams 4 \
    --rewire-distance 2 \
    --rewire-strength 0.1 \
    --bf16 \
    --gradient-checkpointing \
    --dataloader-num-workers 4 \
    --dataloader-pin-memory \
    "${HF_ARGS[@]}"
done