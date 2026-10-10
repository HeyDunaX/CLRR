"""Compatibility entry point; use scripts/experiment.py for new runs."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from experiment_runner import engine
engine.OUT = ROOT / "outputs_rebuttal/amis_standardized_controls"
engine.engine = engine
if __name__ == "__main__":
    engine.main()
else:
    sys.modules[__name__] = engine
