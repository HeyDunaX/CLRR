"""Master runner for all follow-up & rebuttal experiments on GPU.

Execution order prioritized by user request:
1. Priority 1 (NLLB Amis): Strong LoRA Run A & B on NLLB-200 (Amis -> Mandarin)
2. Priority 2 (NLLB Asháninka): Baseline & CLRR-Enc on NLLB-200 (Asháninka -> Spanish)
3. Priority 3 (mBART Asháninka): Baseline & CLRR-Enc on mBART-50 (Asháninka -> Spanish)
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def run_step(step_name: str, cmd: list[str], cwd: Path) -> None:
    print(f"\n{'='*70}", flush=True)
    print(f"STARTING SUITE STEP: {step_name}", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)
    print(f"{'='*70}\n", flush=True)
    start_time = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    duration = time.time() - start_time
    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_name} failed with exit code {result.returncode} after {duration/60:.1f}m!", flush=True)
        sys.exit(result.returncode)
    print(f"\n[SUCCESS] Step {step_name} completed in {duration/60:.1f}m.", flush=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    print("=================================================================", flush=True)
    print("MASTER REBUTTAL & ACL EXPANSION EXPERIMENTAL SUITE", flush=True)
    print("Order of execution: NLLB Suite (Amis) -> NLLB (Asháninka) -> mBART (Asháninka)", flush=True)
    print("=================================================================", flush=True)

    # 1. NLLB Strong LoRA Suite on Amis -> Mandarin
    run_step(
        "Priority 1: NLLB-200 Strong LoRA Suite (Amis -> Mandarin)",
        [sys.executable, "-u", "scripts/peft/run_strong_lora_nllb.py"],
        repo_root,
    )

    # 2. NLLB on Asháninka -> Spanish
    run_step(
        "Priority 2: NLLB-200 Core Matrix (Asháninka -> Spanish: Baseline + CLRR-Enc)",
        [sys.executable, "-u", "scripts/run_ashaninka_experiments.py", "--runs", "nllb_baseline", "nllb_clrr"],
        repo_root,
    )

    # 3. mBART on Asháninka -> Spanish
    run_step(
        "Priority 3: mBART-50 Core Matrix (Asháninka -> Spanish: Baseline + CLRR-Enc)",
        [sys.executable, "-u", "scripts/run_ashaninka_experiments.py", "--runs", "mbart_baseline", "mbart_clrr"],
        repo_root,
    )

    print("\n=================================================================", flush=True)
    print("ALL EXPERIMENTAL SUITES COMPLETED SUCCESSFULLY!", flush=True)
    print("=================================================================", flush=True)


if __name__ == "__main__":
    main()
