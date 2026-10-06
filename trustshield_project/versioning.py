"""Artifact hashing, traceability, and schema versioning for TrustShield."""

import hashlib
from pathlib import Path
from typing import Dict, Any


FEATURE_SCHEMA_VERSION = "1.0.0"
MODEL_METADATA_VERSION = "1.0.0"
TRUST_ENGINE_VERSION = "1.0.0"


def sha256_file(path: str | Path) -> str:
    """Calculate SHA-256 cryptographic digest of a model artifact or data file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found for hashing: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_model_metadata(
    model_name: str,
    model_version: str,
    feature_cols: list[str],
    train_cutoff: str,
    val_cutoff: str,
    artifact_path: str | Path | None = None,
    extra_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Generate standardized traceable metadata for any persisted model."""
    meta = {
        "model_name": model_name,
        "model_version": model_version,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "n_features": len(feature_cols),
        "feature_cols": list(feature_cols),
        "train_cutoff": train_cutoff,
        "val_cutoff": val_cutoff,
    }
    if artifact_path and Path(artifact_path).exists():
        meta["artifact_sha256"] = sha256_file(artifact_path)
    if extra_meta:
        meta.update(extra_meta)
    return meta
