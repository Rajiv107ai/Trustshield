"""
Global Pytest fixtures and configuration for TrustShield test suite.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure repository root and core directories are on sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
for p in [str(_ROOT_DIR), str(_ROOT_DIR / "core"), str(_ROOT_DIR / "backend"), str(_ROOT_DIR / "trustshield_project")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.entity_generator import build_base_entities
from core.product_listing_generator import build_catalog_and_listings
from core.order_return_generator import build_orders_and_returns
from core.fraud_injection import inject_all_fraud
from core.baseline_model import build_features, TRAIN_END, VAL_END


@pytest.fixture(scope="session")
def pipeline():
    """Generates synthetic dataset and feature matrices once per test session."""
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(
        base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"]
    )
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"], catalog["products"],
        base["address_sharing_log"], base["device_sharing_log"],
    )

    df, feature_cols = build_features(
        result["orders"], result["listings"], result["returns"],
        txn["buyers"], catalog["sellers"], catalog["products"]
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["y"] = df["is_fraudulent"].astype(int)

    return {
        "base": base,
        "catalog": catalog,
        "txn": txn,
        "result": result,
        "df": df,
        "feature_cols": feature_cols,
        "train": df[df["order_date"] <= TRAIN_END],
        "val": df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)],
        "test": df[df["order_date"] > VAL_END],
    }
