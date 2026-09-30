#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "    STARTING NLLB-200 SOTA EXPERIMENTAL SUITE (7 RUNS)"
echo "=========================================================="
date

cd /content/CLRR || cd "$(dirname "$0")/../.."

export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
pip uninstall -y torchao 2>/dev/null || true

# 0. Dependencies
echo "[Step 0] Checking dependencies..."
python3 -c "import peft" 2>/dev/null || pip install peft -q
python3 -c "import sacrebleu" 2>/dev/null || pip install sacrebleu -q

# 1. Preflight smoke test
echo ""
echo "[Step 1] Running Preflight Fast Smoke Test..."
python -u scripts/nllb/smoke_test_nllb.py

# 2. Run all 7 methods under strict fairness
METHODS=("baseline" "bitfit" "lora" "layerskip" "middle_align" "clrr_enc" "clrr_dec")

for method in "${METHODS[@]}"; do
    echo ""
    echo "=========================================================="
    echo "  RUNNING NLLB-200: $method"
    echo "=========================================================="
    date
    python -u -m src.nllb_suite.train_nllb --method "$method"
done

# 3. Consolidation
echo ""
echo "[Step 3] Packaging and Uploading NLLB Results..."
zip -r nllb_results.zip results/nllb-200/

if [ -n "$HF_TOKEN" ]; then
    echo "[hf] Uploading artifacts to Hugging Face Hub..."
    python -c "
import os
from huggingface_hub import HfApi
token = os.environ.get('HF_TOKEN')
if token:
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj='nllb_results.zip',
        path_in_repo='nllb-200/nllb_results.zip',
        repo_id='FiveC/amis-rewire-checkpoints',
        repo_type='model'
    )
    print('[hf] Successfully uploaded NLLB-200 artifacts to FiveC/amis-rewire-checkpoints')
"
fi

echo ""
echo "=========================================================="
echo "  NLLB-200 EXPERIMENTAL SUITE FINISHED SUCCESSFULLY!"
echo "=========================================================="
date
