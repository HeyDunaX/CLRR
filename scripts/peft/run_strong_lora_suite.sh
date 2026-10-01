#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "    STARTING CLRR STRONG LORA SUITE (PRIORITY 1 REBUTTAL)"
echo "=========================================================="
date

cd /content/CLRR || cd "$(dirname "$0")/../.."

export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
pip uninstall -y torchao 2>/dev/null || true

# 0. Ensure peft & required packages are installed
echo "[Step 0/4] Checking dependencies..."
python3 -c "import peft" 2>/dev/null || pip install peft -q
python3 -c "import sacrebleu" 2>/dev/null || pip install sacrebleu -q

# 1. Run Preflight Smoke Test (< 25s)
echo ""
echo "[Step 1/4] Running Preflight Smoke Test..."
python -u scripts/peft/smoke_test_peft.py
python -u scripts/peft/smoke_test_strong_lora.py

# 2. Run Strong LoRA Suite (Run A: All-Linear, Run B: All-Linear + Unfrozen Embeddings)
echo ""
echo "[Step 2/4] Running mBART-50 Strong LoRA Training (Run A & Run B)..."
python -u scripts/peft/run_strong_lora_mbart.py

# 3. Packaging and Uploading results to Hugging Face
echo ""
echo "[Step 3/4] Packaging Strong LoRA Results..."
mkdir -p outputs_rebuttal
zip -r strong_lora_results.zip \
  results/mbart-large-50-lora-all-linear/metrics.json \
  results/mbart-large-50-lora-all-linear/test_predictions.csv \
  results/mbart-large-50-lora-all-linear-unfreeze-embed/metrics.json \
  results/mbart-large-50-lora-all-linear-unfreeze-embed/test_predictions.csv \
  outputs_rebuttal/strong_lora_scores.csv 2>/dev/null || true

if [ -n "$HF_TOKEN" ]; then
    echo "[hf] Uploading artifacts to Hugging Face Hub..."
    python -c "
import os
from huggingface_hub import HfApi
token = os.environ.get('HF_TOKEN')
if token:
    api = HfApi(token=token)
    if os.path.exists('strong_lora_results.zip'):
        api.upload_file(
            path_or_fileobj='strong_lora_results.zip',
            path_in_repo='peft_baselines/strong_lora_results.zip',
            repo_id='FiveC/amis-rewire-checkpoints',
            repo_type='model'
        )
    if os.path.exists('outputs_rebuttal/strong_lora_scores.csv'):
        api.upload_file(
            path_or_fileobj='outputs_rebuttal/strong_lora_scores.csv',
            path_in_repo='peft_baselines/strong_lora_scores.csv',
            repo_id='FiveC/amis-rewire-checkpoints',
            repo_type='model'
        )
    print('[hf] Successfully uploaded Strong LoRA artifacts to FiveC/amis-rewire-checkpoints')
"
fi

echo ""
echo "=========================================================="
echo "    STRONG LORA SUITE COMPLETED SUCCESSFULLY"
echo "=========================================================="
date

# 4. Optional shutdown (uncomment if running unattended to save Compute Units)
# echo "[colab] Shutting down instance to protect Compute Units..."
# python -c "
# try:
#     from google.colab import runtime
#     runtime.unassign()
# except Exception as e:
#     print(f'Shutdown notice: {e}')
# " || true
