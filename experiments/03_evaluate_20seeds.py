"""
TrustShield AI — Experiment Step 03: 20-Seed Canonical Benchmark & Statistical Testing.
Evaluates 20 temporal holdout seeds (42-1910) with Holm-Bonferroni correction and plain-count control.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "scripts" / "run_all_experiments.py"

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, str(TARGET)] + sys.argv[1:]))
