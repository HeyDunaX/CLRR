#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python - <<'PY'
import torch
if not torch.cuda.is_available():
	raise SystemExit("CUDA is unavailable. Select an NVIDIA A100 runtime before continuing.")
major, minor = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name(0)
print(f"Using {name}; compute capability {major}.{minor}; torch {torch.__version__}; CUDA {torch.version.cuda}")
if major < 8:
	raise SystemExit("This setup expects an Ampere-or-newer GPU; A100 has compute capability 8.0.")
PY
python -m pip install -r requirements-colab.txt
python -m pip install -e . --no-deps
python -m amis_rewire.train --help >/dev/null
echo "Environment ready. Download data from Google Drive (https://drive.google.com/drive/folders/1W-glGBpCz9R16Oy-P96jdK7YGSVuvdg2) and unpack using scripts/unpack_parallel_data.py."