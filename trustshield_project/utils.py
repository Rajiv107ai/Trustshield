"""
TrustShield AI — Consolidated Common Utilities.

Consolidates artifact versioning & cryptographic digests, missingness audit & indicators,
environment seed reproducibility, and dataset split validation & temporal boundary guards.
"""

import hashlib
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

# ==============================================================================
# 0. Evaluation & Cost-Optimal Threshold Sweeps
# ==============================================================================

def evaluate(y_true, y_pred, y_score, label):
    """
    Prints precision / recall / F1 / ROC-AUC / PR-AUC / confusion matrix
    for a binary classifier.
    """
    print(f"\n=== {label} ===")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.3f}")
    if y_score is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_score):.3f}")
        print(f"PR-AUC:    {average_precision_score(y_true, y_score):.3f}"
              f"  (more informative than ROC-AUC at ~7% base rate)")
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{cm}")


def find_cost_optimal_threshold(y_val, score_val, amount_val, fp_cost=50.0):
    """
    Sweeps thresholds in [0.01, 0.99] on the VALIDATION set only (never test)
    and returns the threshold that minimises total business cost.
    """
    y_arr = np.asarray(y_val)
    amt_arr = np.asarray(amount_val)
    thresholds = np.arange(0.01, 1.00, 0.01)
    costs = []
    for t in thresholds:
        pred = (score_val >= t).astype(int)
        fn_mask = (y_arr == 1) & (pred == 0)
        fp_mask = (y_arr == 0) & (pred == 1)
        costs.append(amt_arr[fn_mask].sum() + fp_cost * fp_mask.sum())
    costs = np.array(costs)
    best_idx = costs.argmin()

    default_pred = (score_val >= 0.5).astype(int)
    default_fn = (y_arr == 1) & (default_pred == 0)
    default_fp = (y_arr == 0) & (default_pred == 1)
    default_cost = amt_arr[default_fn].sum() + fp_cost * default_fp.sum()

    return thresholds[best_idx], costs[best_idx], default_cost


# ==============================================================================
# 1. Versioning & Lineage Metadata
# ==============================================================================

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


# ==============================================================================
# 2. Missingness Handling & Indicators
# ==============================================================================

def add_missingness_indicators(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Add explicit binary indicator columns '{col}__missing' for columns with nulls or specified columns.
    In fraud detection, whether a field (e.g. device_id or tracking number) is missing
    is often an informative signal in itself.
    """
    out = df.copy()
    target_cols = columns if columns is not None else [c for c in out.columns if bool(out[c].isna().any())]
    for col in target_cols:
        if col in out.columns:
            out[f"{col}__missing"] = out[col].isna().astype("int8")
    return out


def audit_missingness(df: pd.DataFrame) -> Dict[str, float]:
    """Return dictionary of missingness fraction for each column with missing values."""
    null_counts = df.isna().mean()
    return {str(col): float(rate) for col, rate in null_counts.items() if rate > 0.0}


# ==============================================================================
# 3. Seed & Environment Reproducibility
# ==============================================================================

def seed_everything(seed: int = 42) -> int:
    """
    Set random seeds across os, Python random, NumPy, and PyTorch (if available)
    to ensure deterministic experimental runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


# ==============================================================================
# 4. Split Isolation & Temporal Boundary Guards
# ==============================================================================

def assert_disjoint_ids(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str,
) -> None:
    """Verify that identity keys (e.g. order_id) never overlap across train, validation, and test splits."""
    train_ids = set(train[id_col].dropna())
    val_ids = set(validation[id_col].dropna())
    test_ids = set(test[id_col].dropna())

    pairs = [
        ("train", "validation", train_ids & val_ids),
        ("train", "test", train_ids & test_ids),
        ("validation", "test", val_ids & test_ids),
    ]

    for name1, name2, overlap in pairs:
        if overlap:
            raise AssertionError(
                f"Data leakage detected: {name1} and {name2} share {len(overlap)} IDs on '{id_col}'."
            )


def validate_temporal_boundaries(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    time_col: str = "order_date",
) -> None:
    """Verify strict chronological sequence: max(train) <= min(val) and max(val) <= min(test)."""
    train_max = pd.to_datetime(train[time_col]).max()
    val_min = pd.to_datetime(val[time_col]).min()
    val_max = pd.to_datetime(val[time_col]).max()
    test_min = pd.to_datetime(test[time_col]).min()

    if train_max > val_min:
        raise AssertionError(
            f"Temporal boundary violation: train max ({train_max}) > val min ({val_min})"
        )
    if val_max > test_min:
        raise AssertionError(
            f"Temporal boundary violation: val max ({val_max}) > test min ({test_min})"
        )
