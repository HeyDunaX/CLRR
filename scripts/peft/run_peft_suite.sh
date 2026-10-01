#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "    STARTING CLRR PEFT BASELINE SUITE (LORA & BITFIT)"
echo "=========================================================="
date

cd /content/CLRR || cd "$(dirname "$0")/../.."

export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
pip uninstall -y torchao 2>/dev/null || true

# 0. Ensure peft is installed in Colab environment
echo "[Step 0/4] Checking PEFT library..."
python3 -c "import peft" 2>/dev/null || pip install peft -q


# 1. Run Preflight Smoke Test (< 25s)
echo ""
echo "[Step 1/4] Running Preflight Smoke Test..."
python -u scripts/peft/smoke_test_peft.py

# 2. Run PEFT Suite (LoRA & BitFit on mBART-large-50)
echo ""
echo "[Step 2/4] Running mBART-50 PEFT Training (LoRA & BitFit)..."
python -u scripts/peft/run_peft_mbart.py

# 3. Packaging and Uploading results to Hugging Face
echo ""
echo "[Step 3/4] Packaging PEFT Results..."
zip -r peft_results.zip results/mbart-large-50/lora/ results/mbart-large-50/bitfit/ results/peft_scores.csv

if [ -n "$HF_TOKEN" ]; then
    echo "[hf] Uploading artifacts to Hugging Face Hub..."
    python -c "
import os
from huggingface_hub import HfApi
token = os.environ.get('HF_TOKEN')
if token:
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj='peft_results.zip',
        path_in_repo='peft_baselines/peft_results.zip',
        repo_id='FiveC/amis-rewire-checkpoints',
        repo_type='model'
    )
    if os.path.exists('results/peft_scores.csv'):
        api.upload_file(
            path_or_fileobj='results/peft_scores.csv',
            path_in_repo='peft_baselines/peft_scores.csv',
            repo_id='FiveC/amis-rewire-checkpoints',
            repo_type='model'
        )
    print('[hf] Successfully uploaded PEFT artifacts to FiveC/amis-rewire-checkpoints')
"
fi

echo ""
echo "=========================================================="
echo "    PEFT BASELINE SUITE COMPLETED SUCCESSFULLY"
echo "=========================================================="
date

# 4. Automatically shut down Colab instance to preserve Compute Units
echo "[colab] Shutting down instance to protect Compute Units..."
python -c "
try:
    from google.colab import runtime
    runtime.unassign()
except Exception as e:
    print(f'Shutdown notice: {e}')
" || true
