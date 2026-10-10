"""Compatibility import for historical integrity checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from experiment_runner import integrity
sys.modules[__name__] = integrity
