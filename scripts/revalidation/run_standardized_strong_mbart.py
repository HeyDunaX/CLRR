"""Historical import compatibility; new runs use scripts/experiment.py."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from experiment_runner import engine as worker
worker.OUT = ROOT / "outputs_rebuttal/amis_standardized_strong_mbart"
if __name__ == "__main__":
    worker.main()
