"""Single entry point for shared experiment execution and Colab orchestration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from experiment_runner.cli import main

if __name__ == '__main__':
    main()
