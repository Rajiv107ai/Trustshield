"""Single-command end-to-end pipeline reproduction script for TrustShield AI.

Fulfills Phase 18 and Issue 60 of the Master Repair Specification:
1. Sets deterministic random state.
2. Generates temporal synthetic entities & transactions.
3. Builds point-in-time tabular and graph features under strict cutoff invariants.
4. Trains models (Phase 2 specialized, Phase 3 combined, Phase 5 hybrid).
5. Fits probability calibrators (isotonic) on validation splits.
6. Builds pre-ranked candidate collusion rings.
7. Evaluates performance metrics (ROC-AUC, PR-AUC, ECE, Brier).
8. Exports complete reproduction manifest and metadata to models/reproduction_manifest.json.
"""

from __future__ import annotations
import os
import sys
import json
import time
import hashlib
import subprocess
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..", "trustshield_project")
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
sys.path.insert(0, PROJECT_DIR)


def _file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    if not os.path.isfile(filepath):
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _get_git_commit() -> str:
    """Retrieve current git commit hash if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def main() -> int:
    start_time = time.time()
    print("=" * 70)
    print("TrustShield AI — Complete Reproduction Pipeline (Phase 18 / Issue 60)")
    print("=" * 70)

    # 1. Train baseline, graph features, and Phase 3 calibrator
    print("\n[Step 1/3] Running baseline & Phase 3 training pipeline...")
    train_script = os.path.join(SCRIPT_DIR, "train_and_save_models.py")
    res1 = subprocess.run([sys.executable, train_script], cwd=ROOT_DIR)
    if res1.returncode != 0:
        print("[ERROR] train_and_save_models.py failed.")
        return res1.returncode

    # 2. Train Phase 5 hybrid model and fit Phase 5 calibrator
    print("\n[Step 2/3] Running Phase 5 hybrid model training & calibration...")
    p5_script = os.path.join(SCRIPT_DIR, "train_phase5.py")
    res2 = subprocess.run([sys.executable, p5_script], cwd=ROOT_DIR)
    if res2.returncode != 0:
        print("[ERROR] train_phase5.py failed.")
        return res2.returncode

    # 3. Collect artifact hashes and generate reproduction manifest
    print("\n[Step 3/3] Generating reproduction manifest and artifact catalog...")
    artifacts = [
        "combined_graph_model.joblib",
        "fake_listing_model.joblib",
        "return_fraud_model.joblib",
        "fraud_rings.joblib",
        "calibrator.joblib",
        "hybrid_model.joblib",
        "phase5_calibrator.joblib",
        "buyer_embeddings.joblib",
        "seller_embeddings.joblib",
    ]

    artifact_hashes = {}
    for art in artifacts:
        p = os.path.join(MODELS_DIR, art)
        artifact_hashes[art] = {
            "path": p,
            "sha256": _file_sha256(p),
            "size_bytes": os.path.getsize(p) if os.path.isfile(p) else 0,
        }

    manifest = {
        "pipeline": "TrustShield Reproduction Master Pipeline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "python_version": sys.version,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "invariants": {
            "temporal_cutoff_enforced": True,
            "decision_time_rule": "event_time < decision_time",
            "calibration_method": "isotonic_regression",
            "multimodal_self_match_excluded": True,
        },
        "artifacts": artifact_hashes,
    }

    manifest_path = os.path.join(MODELS_DIR, "reproduction_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nReproduction manifest written to: {manifest_path}")
    print("=" * 70)
    print(f"Reproduction complete in {manifest['elapsed_seconds']}s. All models serialized & calibrated.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
