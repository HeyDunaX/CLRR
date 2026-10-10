#!/usr/bin/env bash
set -euo pipefail
if [ ! -x /content/miniforge3/bin/conda ]; then
 curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /content/miniforge.sh
 bash /content/miniforge.sh -b -p /content/miniforge3
fi
if [ ! -x /content/miniforge3/envs/clrr/bin/python ]; then
 /content/miniforge3/bin/conda create -y -n clrr python=3.13 pip
fi
PY=/content/miniforge3/envs/clrr/bin/python
if [ ! -f /content/STANDARDIZED_ENV_READY ]; then
 "$PY" -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
 "$PY" -m pip install --no-cache-dir transformers==4.57.6 tokenizers==0.22.2 huggingface_hub==0.36.2 accelerate==1.15.0 peft==0.21.0 sentencepiece==0.2.2 datasets==3.6.0 sacrebleu==2.6.0 pandas==3.0.6 numpy==2.5.3 scipy==1.17.1 protobuf==7.36.2 safetensors==0.8.0
 touch /content/STANDARDIZED_ENV_READY
fi
mkdir -p /content/CLRR_controls
cd /content/CLRR_controls
unzip -q -o /content/standardized_controls_bundle.zip
export PYTHONPATH=/content/CLRR_controls/src
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_TELEMETRY=1
chmod 600 /content/.standardized_hf_token
export HF_TOKEN=$(cat /content/.standardized_hf_token)
unzip -q -o /content/strong_mbart_supplement_bundle.zip
"$PY" -u /content/controller_b_restart.py
