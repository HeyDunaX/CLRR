"""Frozen runner generation retained for historical regression checks only.

New execution uses cli.py and a recipe; this module never starts a session.
"""
from .colab import get_session_name

def get_run_plan(account, partition="3gpu_safe"):
    acc_str = str(account)
    if partition == "3gpu_safe":
        if acc_str == '1':
            return [('baseline', s) for s in range(45, 53)] + [('clrr', s) for s in range(45, 49)]
        elif acc_str == '2':
            return [('clrr', s) for s in range(49, 53)] + [('lsr', s) for s in range(45, 53)] + [('full', s) for s in range(45, 47)]
        elif acc_str == '3':
            return [('full', s) for s in range(47, 53)]
        else:
            raise ValueError(f"Unknown account {account} for 3gpu_safe partition")
    elif partition == "2gpu":
        if acc_str == '1':
            return [('baseline', s) for s in range(45, 53)] + [('clrr', s) for s in range(45, 53)]
        elif acc_str == '2':
            return [('lsr', s) for s in range(45, 53)] + [('full', s) for s in range(45, 53)]
        else:
            raise ValueError(f"Unknown account {account} for 2gpu partition")
    else:
        raise ValueError(f"Unknown partition {partition}")


def generate_remote_runner(account, stage="train", partition="3gpu_safe", custom_runs=None):
    session = get_session_name(account)
    run_plan = custom_runs if custom_runs is not None else get_run_plan(account, partition=partition)

    commands = []
    if stage in ('train', 'test'):
        for fam, seed in run_plan:
            commands.append(f'"$PY" -u scripts/revalidation/run_fresh_seed_confirmation.py --family {fam} --seed {seed} --stage {stage}')
    elif stage == 'probe':
        for fam, seed in run_plan:
            commands.append(f'"$PY" -u scripts/revalidation/measure_fresh_confirmation_mechanisms.py --family {fam} --seed {seed} --output-dir outputs_rebuttal/amis_fresh_confirmation_mechanisms_20261007')

    stage_actions = []
    if stage in ('gate', 'gate_and_benchmark', 'train'):
        stage_actions.append('''
echo "=== RUNNING GPU PREFLIGHT GATE TESTS ==="
mkdir -p outputs_rebuttal/amis_fresh_confirmation_20261005/gpu_gate
"$PY" tests/smoke_rerun_optimization.py --cuda --result-dir outputs_rebuttal/amis_fresh_confirmation_20261005/gpu_gate
"$PY" tests/smoke_precision_training.py --cuda --audit-rerun --result-dir outputs_rebuttal/amis_fresh_confirmation_20261005/gpu_gate
touch /content/GPU_GATE_COMPLETE
echo "=== GPU PREFLIGHT GATE COMPLETE ==="
''')

    if stage in ('benchmark', 'gate_and_benchmark'):
        stage_actions.append('''
echo "=== RUNNING BENCHMARK V2 ==="
mkdir -p outputs_rebuttal/amis_cost_benchmark_20261006_v2
"$PY" scripts/revalidation/benchmark_corrected_components_v2.py --output-dir outputs_rebuttal/amis_cost_benchmark_20261006_v2 --cache-dir /content/benchmark_cache
touch /content/BENCHMARK_COMPLETE
echo "=== BENCHMARK V2 COMPLETE ==="
''')

    if stage == 'gate_and_benchmark':
        stage_actions.append(f'''
touch /content/GATE_AND_BENCHMARK_COMPLETE
echo "=== [$(date -u)] ACCOUNT {account}: GATE AND BENCHMARK COMPLETED SUCCESSFULLY ==="
''')
    elif stage == 'gate':
        stage_actions.append(f'''
touch /content/GATE_COMPLETE
echo "=== [$(date -u)] ACCOUNT {account}: GATE COMPLETED SUCCESSFULLY ==="
''')
    elif stage == 'benchmark':
        stage_actions.append(f'''
touch /content/BENCHMARK_COMPLETE
echo "=== [$(date -u)] ACCOUNT {account}: BENCHMARK COMPLETED SUCCESSFULLY ==="
''')
    elif stage in ('train', 'test', 'probe'):
        stage_actions.append("\n".join(commands))
        stage_actions.append(f'''
echo "=== [$(date -u)] ACCOUNT {account}: ALL {len(run_plan)} RUNS COMPLETED SUCCESSFULLY ==="
touch /content/CONFIRMATION_ACCOUNT_{account}_{stage}_COMPLETE
''')

    stage_block = "\n".join(stage_actions)

    script_content = f'''#!/usr/bin/env bash
set -euo pipefail
cd /content

echo "=== [$(date -u)] ACCOUNT {account}: INITIALIZING CONFIRMATION RUNNER ({stage.upper()}) ==="

if [ ! -f /content/ENV_READY ]; then
  if [ ! -x /content/miniforge3/bin/conda ]; then
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /content/miniforge.sh
    bash /content/miniforge.sh -b -p /content/miniforge3
  fi
  /content/miniforge3/bin/conda create -y -n clrr python=3.13 pip
  PY=/content/miniforge3/envs/clrr/bin/python
  "$PY" -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  "$PY" -m pip install --no-cache-dir transformers==4.57.6 tokenizers==0.22.2 huggingface_hub==0.36.0 accelerate==1.15.0 numpy==2.5.3 sentencepiece==0.2.2 protobuf==7.36.2 safetensors==0.8.0 pandas==3.0.6 datasets==3.6.0 sacrebleu==2.6.0 scipy==1.17.1
  "$PY" -c 'import torch, transformers, sacrebleu; assert torch.cuda.is_available(); assert "A100" in torch.cuda.get_device_name(0); print("ENV VERIFIED:", torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))'
  touch /content/ENV_READY
fi

mkdir -p /content/CLRR
cd /content/CLRR
unzip -q -o /content/fresh_confirmation_bundle.zip -d /content/CLRR
if [ -f /content/fresh_confirmation_metadata.zip ]; then
  mkdir -p /content/CLRR/outputs_rebuttal/amis_fresh_confirmation_20261005
  unzip -q -o /content/fresh_confirmation_metadata.zip -d /content/CLRR/outputs_rebuttal/amis_fresh_confirmation_20261005
fi

export PYTHONPATH=/content/CLRR/src
chmod 600 /content/.hf_token
export HF_TOKEN=$(cat /content/.hf_token)
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1
PY=/content/miniforge3/envs/clrr/bin/python

# Gate receipts always come from this exact bundle.
"$PY" -c 'import hashlib,json; from pathlib import Path; m=json.loads(Path("fresh_bundle_files.json").read_text()); assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in m.items())'

{stage_block}
'''
    return script_content
