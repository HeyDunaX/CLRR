"""Shared Colab account transport; existing credentials remain in place."""
import ast
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PY_EXE = sys.executable

def get_hf_token():
    if os.environ.get('HF_TOKEN'):
        return os.environ['HF_TOKEN']
    tree = ast.parse((ROOT / 'scratch/launch_peft_colab.py').read_text(encoding='utf-8'))
    values = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith('hf_') and len(n.value.split()) == 1 and len(n.value) < 100}
    assert len(values) == 1
    return values.pop()

def get_session_name(account):
    return 'colab' if str(account) == '1' else f'colab{account}'

def call_pilot(account, args, timeout=120):
    cmd = [PY_EXE, '-X', 'utf8', str(ROOT / 'scripts/colab.py'), '--account', str(account)] + args
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    return res.returncode, res.stdout, res.stderr

def ensure_session(account):
    session = get_session_name(account)
    code, stdout, stderr = call_pilot(account, ['status', '-s', session])
    if code != 0 or 'not found' in stdout.lower() or 'no active sessions' in stdout.lower() or 'gpu-a100-s-' not in stdout:
        print(f"Starting new A100 session for Account {account} ({session})...")
        code, stdout, stderr = call_pilot(account, ['new', '-s', session, '--gpu', 'A100'], timeout=300)
        if code != 0:
            raise RuntimeError(f"Failed to start A100 on Account {account}: {stderr}\n{stdout}")

    verified = False
    for attempt in range(8):
        time.sleep(10)
        code, stdout, stderr = call_pilot(account, ['status', '-s', session])
        if code == 0 and 'gpu-a100-s-' in stdout and 'Standard' in stdout:
            verified = True
            break
        print(f"Waiting for Account {account} session to become ready (attempt {attempt+1}/8)...")
    assert verified, f"Instance on Account {account} is not ready or not A100 Standard: {stdout}"
    print(f"Account {account} A100 verified: {stdout.strip()}")
