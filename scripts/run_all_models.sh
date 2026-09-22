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
  # Optimized batch size 128 for A100 GPU (40GB VRAM) with gradient_accumulation 1
  TRAIN_BS="${PER_DEVICE_TRAIN_BATCH_SIZE:-128}"
  EVAL_BS="${PER_DEVICE_EVAL_BATCH_SIZE:-128}"
  GRAD_ACC="${GRADIENT_ACCUMULATION_STEPS:-1}"

  # Model-specific learning rate (mT5 requires 3e-4 with AdamW, mBART standard 5e-5)
  if [[ "$MODEL" == *"mbart"* ]]; then
    LR="${MBART_LEARNING_RATE:-5e-5}"
  else
    LR="${MT5_LEARNING_RATE:-3e-4}"
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
    --num-train-epochs "${NUM_TRAIN_EPOCHS:-20}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-4}" \
    --learning-rate "$LR" \
    --warmup-ratio 0.06 \
    --per-device-train-batch-size "$TRAIN_BS" \
    --per-device-eval-batch-size "$EVAL_BS" \
    --gradient-accumulation-steps "$GRAD_ACC" \
    --max-source-length 256 \
    --max-target-length 256 \
    --num-beams "${NUM_BEAMS:-4}" \
    --eval-beams "${EVAL_BEAMS:-1}" \
    --rewire-distance 2 \
    --rewire-strength 0.1 \
    --bf16 \
    --no-gradient-checkpointing \
    --dataloader-num-workers 4 \
    --dataloader-pin-memory \
    "${HF_ARGS[@]}"
done