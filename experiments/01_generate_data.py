"""
TrustShield AI — Experiment Step 01: Realistic Synthetic Data Generation (v2.1).
Executes causal transaction graph and multi-hop entity generation.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "scripts" / "generate_realistic_synthetic_data_v2_1.py"

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, str(TARGET)] + sys.argv[1:]))
