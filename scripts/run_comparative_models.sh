#!/usr/bin/env bash
# Pure Bash runner script for Comprehensive Comparative Experiments on Colab A100 VM.
# Detached execution ready: logs to /content/run.log, uploads to HF, and automatically unassigns VM upon completion.
set -e

echo "================================================================="
echo "COLAB SSH RUNNER: ACL MULTI-BACKBONE & COMPARATIVE EXPERIMENTS"
echo "Start time: $(date)"
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
pip install --quiet sacrebleu protobuf pypdf huggingface_hub pandas 'transformers<5.0'

export PYTHONUNBUFFERED=1

# Step 1: Preflight Smoke Test for all backbones
echo ""
echo "================================================================="
echo "STEP 1: RUNNING MANDATORY PREFLIGHT SMOKE TEST (ALL BACKBONES)"
echo "================================================================="
python -u scripts/smoke_test_comparative.py --backbone all

# Step 2: Full Training & Evaluation across all 6 extension models
echo ""
echo "================================================================="
echo "STEP 2: LAUNCHING FULL 6-MODEL EXTENSION PIPELINE"
echo "================================================================="
python -u scripts/run_comparative_models.py --run-group all_extensions

echo ""
echo "================================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY! TABLE 5 GENERATED."
echo "Checkpoints & Table 5 pushed to Hugging Face Hub (FiveC/amis-rewire-checkpoints)."
echo "End time: $(date)"
echo "================================================================="

# Step 3: Automatically terminate Colab instance to preserve compute units
echo "[Runner] Terminating Colab instance to preserve Compute Units..."
python3 -c "import google.colab; google.colab.runtime.unassign()" 2>/dev/null || sudo poweroff 2>/dev/null || true
