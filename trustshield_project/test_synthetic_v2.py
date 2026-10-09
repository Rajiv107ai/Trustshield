"""Unit and integration tests for Synthetic Data Generator v2 (Stage 3.1)."""

import os
import pytest
import pandas as pd
import numpy as np

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_v2"))


@pytest.fixture(scope="module")
def v2_data():
    assert os.path.isdir(DATA_DIR), f"Missing synthetic v2 directory: {DATA_DIR}"
    tables = {}
    for name in ["orders", "listings", "returns", "buyers", "sellers", "products"]:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        assert os.path.isfile(path), f"Missing CSV: {path}"
        tables[name] = pd.read_csv(path)
    return tables


def test_date_boundary_clamping(v2_data):
    """Verify that NO transactions or returns occur outside the 2025 calendar year."""
    orders = v2_data["orders"].copy()
    returns = v2_data["returns"].copy()

    orders["order_date"] = pd.to_datetime(orders["order_date"])
    returns["return_date"] = pd.to_datetime(returns["return_date"])

    sim_start = pd.Timestamp("2025-01-01")
    sim_end = pd.Timestamp("2025-12-31")

    assert orders["order_date"].min() >= sim_start
    assert orders["order_date"].max() <= sim_end
    assert returns["return_date"].min() >= sim_start
    assert returns["return_date"].max() <= sim_end

    # Explicit check: Zero orders in 2026 (fixing the 27-order overflow artifact)
    overflow_2026 = (orders["order_date"] > sim_end).sum()
    assert overflow_2026 == 0, f"Found {overflow_2026} orders leaking into 2026!"


def test_no_id_label_leakage(v2_data):
    """Verify that no entity IDs or return IDs contain label leak strings."""
    leak_words = ["FRAUD", "ABUSE", "COLLUSION", "RING"]
    for table_name in ["orders", "listings", "returns", "buyers", "sellers"]:
        df = v2_data[table_name]
        id_col = f"{table_name[:-1] if table_name != 'returns' else 'return'}_id"
        if id_col not in df.columns:
            id_col = f"{table_name}_id"
        matches = df[id_col].str.contains("|".join(leak_words), case=False).sum()
        assert matches == 0, f"Found {matches} label-revealing IDs in {table_name}.{id_col}!"


def test_chronological_ordering_invariants(v2_data):
    """Verify order_date >= listing_date and return_date >= order_date."""
    orders = v2_data["orders"].copy()
    listings = v2_data["listings"].copy()
    returns = v2_data["returns"].copy()

    orders["order_date"] = pd.to_datetime(orders["order_date"])
    listings["listing_date"] = pd.to_datetime(listings["listing_date"])
    returns["return_date"] = pd.to_datetime(returns["return_date"])

    ol = orders.merge(listings[["listing_id", "listing_date"]], on="listing_id")
    assert (ol["order_date"] < ol["listing_date"]).sum() == 0, "Orders placed before listing creation!"

    ro = returns.merge(orders[["order_id", "order_date"]], on="order_id")
    assert (ro["return_date"] < ro["order_date"]).sum() == 0, "Returns filed before order placement!"


def test_return_delay_overlap(v2_data):
    """Verify that legitimate and abusive return delays overlap substantially (AUC < 0.70)."""
    orders = v2_data["orders"].copy()
    returns = v2_data["returns"].copy()
    from sklearn.metrics import roc_auc_score

    orders["order_date"] = pd.to_datetime(orders["order_date"])
    returns["return_date"] = pd.to_datetime(returns["return_date"])

    ro = returns.merge(orders[["order_id", "order_date"]], on="order_id")
    ro["delay_days"] = (ro["return_date"] - ro["order_date"]).dt.days

    auc = roc_auc_score(ro["is_fraudulent"].astype(int), -ro["delay_days"])
    # In v1 baseline, AUC was 0.8441 due to [1, 5] vs [1, 21] separation
    # In v2, AUC must be well below 0.70 (demonstrating genuine overlap)
    assert auc < 0.70, f"Return delay is still trivially separable! AUC = {auc:.4f}"
