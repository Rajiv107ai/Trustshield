"""
TrustShield AI — Experiment Step 02: Model Training & Artifact Persistence.
Trains Phase 3 tabular+graph models, specialized fake listing, and return abuse models.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "scripts" / "train_and_save_models.py"

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, str(TARGET)] + sys.argv[1:]))
