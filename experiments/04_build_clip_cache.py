"""
TrustShield AI — Experiment Step 04: Build Offline CLIP Embedding Cache.
Requires requirements-clip.txt (torch, transformers, Pillow).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "scripts" / "build_clip_embeddings.py"

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, str(TARGET)] + sys.argv[1:]))
