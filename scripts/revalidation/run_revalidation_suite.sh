#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "    STARTING CLRR REVALIDATION SUITE (COLAB A100)"
echo "=========================================================="
date

cd /content/CLRR || cd "$(dirname "$0")/../.."

# 1. Run Preflight Smoke Test
echo ""
echo "[Step 1/4] Running Preflight Smoke Test..."
python -u scripts/revalidation/smoke_test_revalidation.py

# 2. Run Task 3: mT5 No-Stop-Gradient
if [ ! -f "outputs_revalidation/no_sg_gradient_trace.json" ]; then
    echo ""
    echo "[Step 2/4] Task 3: Running mT5 No-Stop-Gradient Experiment (5 epochs)..."
    python -u scripts/revalidation/run_no_stop_gradient.py

    if [ -n "$HF_TOKEN" ] && [ -f "outputs_revalidation/no_sg_gradient_trace.json" ]; then
        echo "[hf] Uploading Task 3 gradient trace..."
        python -c "
import os
from huggingface_hub import HfApi
token = os.environ.get('HF_TOKEN')
if token:
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj='outputs_revalidation/no_sg_gradient_trace.json',
        path_in_repo='revalidation/no_sg_gradient_trace.json',
        repo_id='FiveC/amis-rewire-checkpoints',
        repo_type='model'
    )
    print('[hf] Uploaded no_sg_gradient_trace.json')
" || true
    fi
else
    echo ""
    echo "[Step 2/4] Task 3: no_sg_gradient_trace.json already exists! Skipping."
fi


# 3. Run Task 2: mBART-50 Ablations (CLRR-only and LSR-only)
echo ""
echo "[Step 3/4] Task 2: Running mBART-50 Ablations (CLRR-only & LSR-only)..."
python -u scripts/revalidation/run_ablation_mbart.py

# 4. Packaging and Push to Hugging Face
echo ""
echo "[Step 4/4] Packaging Revalidation Outputs..."
zip -r outputs_revalidation.zip outputs_revalidation/

if [ -n "$HF_TOKEN" ]; then
    echo "[hf] Uploading artifacts to Hugging Face Hub..."
    python -c "
import os
from huggingface_hub import HfApi
token = os.environ.get('HF_TOKEN')
if token:
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj='outputs_revalidation.zip',
        path_in_repo='revalidation/outputs_revalidation.zip',
        repo_id='FiveC/amis-rewire-checkpoints',
        repo_type='model'
    )
    print('[hf] Successfully uploaded outputs_revalidation.zip to FiveC/amis-rewire-checkpoints')
"
fi

echo ""
echo "=========================================================="
echo "    CLRR REVALIDATION SUITE COMPLETED SUCCESSFULLY"
echo "=========================================================="
date

# 5. Automatically shut down Colab instance to preserve Compute Units
echo "[colab] Shutting down instance to protect Compute Units..."
python -c "
try:
    from google.colab import runtime
    print('[colab] Triggering runtime.unassign()...')
    runtime.unassign()
except Exception as e:
    print(f'[colab] runtime.unassign failed: {e}')
" || true

if command -v colab &> /dev/null; then
    colab stop || colab stop -s colab || colab stop -s 780099 || true
fi

