"""Regression tests for CODE-001: baseline_model sort_values bug fix."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Ensure trustshield_project is in python path
_project_dir = os.path.dirname(os.path.abspath(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from baseline_model import _device_shared_buyer_count_asof, build_features


def test_device_shared_buyer_count_asof_chronological():
    """Verify distinct buyers on a device strictly before current order date are counted."""
    df = pd.DataFrame({
        "device_id": ["dev_1", "dev_1", "dev_2", "dev_1"],
        "buyer_id": ["b_1", "b_2", "b_3", "b_1"],
        "order_date": pd.to_datetime([
            "2025-01-01 10:00:00",
            "2025-01-01 12:00:00",
            "2025-01-02 09:00:00",
            "2025-01-03 10:00:00",
        ])
    })

    result = _device_shared_buyer_count_asof(df)
    
    # Row 0: first order on dev_1 -> 0 prior distinct buyers
    assert result.iloc[0] == 0.0
    # Row 1: second order on dev_1 by b_2 -> 1 prior distinct buyer (b_1)
    assert result.iloc[1] == 1.0
    # Row 2: first order on dev_2 -> 0 prior distinct buyers
    assert result.iloc[2] == 0.0
    # Row 3: third order on dev_1 by b_1 -> 2 prior distinct buyers (b_1, b_2)
    assert result.iloc[3] == 2.0


def test_device_shared_buyer_count_asof_empty():
    """Empty DataFrame returns an empty Series of zeros without error."""
    df = pd.DataFrame(columns=["device_id", "buyer_id", "order_date"])
    result = _device_shared_buyer_count_asof(df)
    assert len(result) == 0


def test_device_shared_buyer_count_asof_missing_columns():
    """Missing required columns return all zeros without exception."""
    df = pd.DataFrame({
        "buyer_id": ["b_1"],
        "order_date": pd.to_datetime(["2025-01-01"])
    })
    result = _device_shared_buyer_count_asof(df)
    assert len(result) == 1
    assert result.iloc[0] == 0.0


def test_build_features_with_device_id_completes_successfully():
    """build_features must complete end-to-end when device_id is present."""
    listings = pd.DataFrame({
        "listing_id": [1], "seller_id": ["s1"], "category": ["electronics"],
        "listing_date": pd.to_datetime(["2025-01-01"]), "product_id": [1]
    })
    products = pd.DataFrame({"product_id": [1], "base_price": [100.0]})
    sellers = pd.DataFrame({"seller_id": ["s1"], "signup_date": pd.to_datetime(["2024-01-01"])})
    buyers = pd.DataFrame({"buyer_id": ["buyer_1"], "signup_date": pd.to_datetime(["2024-01-01"])})
    returns = pd.DataFrame(columns=["buyer_id", "return_date"])
    orders = pd.DataFrame({
        "order_id": [1], "buyer_id": ["buyer_1"], "seller_id": ["s1"], "listing_id": [1],
        "amount": [100.0], "order_date": pd.to_datetime(["2025-01-01"]), "device_id": ["dev_1"]
    })

    feat_df, cols = build_features(orders, listings, returns, buyers, sellers, products)
    assert "device_shared_buyer_count" in cols
    assert "device_shared_buyer_count" in feat_df.columns
    assert feat_df["device_shared_buyer_count"].iloc[0] == 0.0


def test_device_shared_buyer_count_asof_equal_timestamps_deterministic():
    """Verify deterministic handling and zero leakage for orders with equal timestamps."""
    df1 = pd.DataFrame({
        "device_id": ["dev_1", "dev_1", "dev_1"],
        "buyer_id": ["b_1", "b_2", "b_3"],
        "order_date": pd.to_datetime([
            "2025-01-01 10:00:00",
            "2025-01-01 10:00:00",
            "2025-01-01 12:00:00",
        ])
    })
    res1 = _device_shared_buyer_count_asof(df1)
    assert res1.iloc[0] == 0.0
    assert res1.iloc[1] == 0.0
    assert res1.iloc[2] == 2.0

    # Reversed order of tied rows must produce identical deterministic results
    df2 = pd.DataFrame({
        "device_id": ["dev_1", "dev_1", "dev_1"],
        "buyer_id": ["b_2", "b_1", "b_3"],
        "order_date": pd.to_datetime([
            "2025-01-01 10:00:00",
            "2025-01-01 10:00:00",
            "2025-01-01 12:00:00",
        ])
    })
    res2 = _device_shared_buyer_count_asof(df2)
    assert res2.iloc[0] == 0.0
    assert res2.iloc[1] == 0.0
    assert res2.iloc[2] == 2.0

