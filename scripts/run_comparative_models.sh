#!/usr/bin/env bash
# Pure Bash runner script for Comparative Baseline Experiments on Colab A100 VM.
set -e

echo "================================================================="
echo "COLAB SSH RUNNER: ACL 2024 & ACL 2025 BASELINE EXPERIMENTS"
echo "================================================================="

# Navigate to project repository
if [ -d "/content/CLRR" ]; then
    cd /content/CLRR
else
    cd "$(dirname "$0")/.."
fi
echo "[Runner] Current working directory: $(pwd)"

# Verify GPU
echo "[Runner] Checking GPU specifications:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "No nvidia-smi found"

# Ensure runtime dependencies matching pyproject.toml
echo "[Runner] Ensuring required Python packages are installed..."
pip install --quiet sacrebleu protobuf pypdf huggingface_hub 'transformers<5.0'



# Step 1: Preflight Smoke Test
echo ""
echo "================================================================="
echo "STEP 1: RUNNING MANDATORY PREFLIGHT SMOKE TEST"
echo "================================================================="
python -u scripts/smoke_test_comparative.py

# Step 2: Full Training & Evaluation
echo ""
echo "================================================================="
echo "STEP 2: LAUNCHING FULL COMPARATIVE TRAINING PIPELINE"
echo "================================================================="
python -u scripts/run_comparative_models.py

echo ""
echo "================================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY! CHECKPOINTS UPLOADED."
echo "================================================================="
