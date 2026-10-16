"""
TrustShield — Stage 3.3 Controlled Model Evaluation Unit Test Suite.

Covers:
1. Gate 0: Observation maturity verification and unobserved return handling.
2. Feature meta and decision thresholds integrity.
3. Versioned artifact persistence and loadability in models/v2_1/.
"""

from __future__ import annotations

import os
import sys

import joblib
import pandas as pd
import pytest

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_DATA_DIR = os.path.join(_ROOT_DIR, "data", "synthetic_v2_1")
_MODELS_V2_1_DIR = os.path.join(_ROOT_DIR, "models", "v2_1")

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


@pytest.fixture(scope="module")
def feature_meta() -> dict:
    """Loads models/v2_1/feature_meta.joblib."""
    p = os.path.join(_MODELS_V2_1_DIR, "feature_meta.joblib")
    assert os.path.isfile(p), f"Missing feature_meta: {p}"
    return joblib.load(p)


class TestGate0ObservationMaturity:
    """Gate 0: Reconcile exact observation cutoffs and prevent false negative labeling."""

    def test_exact_21_day_maturity_cutoff(self):
        """Verifies that an order has a guaranteed complete 21-day observation window

        iff order_date <= 2025-12-10 00:00:00 (41,049 orders).
        """
        orders_path = os.path.join(_DATA_DIR, "orders.csv")
        orders = pd.read_csv(orders_path, parse_dates=["order_date"])

        n_mature = int((orders["order_date"] <= "2025-12-10 00:00:00").sum())
        assert n_mature == 41049
        n_truncated = int((orders["order_date"] > "2025-12-10 00:00:00").sum())
        assert n_truncated == 8951

    def test_unobserved_returns_not_imputed_as_negatives(self):
        """Verifies that returns table strictly contains observed returns

        (5,537 records) without synthetic negative imputation across orders.
        """
        returns_path = os.path.join(_DATA_DIR, "returns.csv")
        returns = pd.read_csv(returns_path)
        assert len(returns) == 5537
        assert returns["return_id"].is_unique
        assert (returns["is_fraudulent"].isin([0, 1])).all()


class TestGate1FeatureMetaAndThresholds:
    """Gate 1: Verification of frozen feature metadata and decision thresholds."""

    def test_feature_meta_thresholds_defined(self, feature_meta):
        """Verifies that decision thresholds exist and are bounded in (0, 1)."""
        thresholds = feature_meta["thresholds"]
        assert "fake_listing" in thresholds
        assert "transaction_fraud" in thresholds
        assert "return_fraud" in thresholds
        for t_name, t_val in thresholds.items():
            if isinstance(t_val, list):
                assert all(0.0 < x < 1.0 for x in t_val), f"Threshold list {t_name}={t_val} out of bounds"
            else:
                assert 0.0 < t_val < 1.0, f"Threshold {t_name}={t_val} out of bounds (0, 1)"


class TestVersionedArtifacts:
    """Verifies that all Stage 3.3 models are properly persisted in models/v2_1/."""

    def test_versioned_model_files_exist(self):
        expected_files = [
            "fake_listing_model.joblib",
            "fake_listing_scaler.joblib",
            "fake_listing_calibrator.joblib",
            "return_fraud_model.joblib",
            "return_fraud_scaler.joblib",
            "return_fraud_calibrator.joblib",
            "tabular_baseline_model.joblib",
            "combined_graph_model.joblib",
            "combined_graph_calibrator.joblib",
            "feature_meta.joblib",
        ]
        for fname in expected_files:
            fpath = os.path.join(_MODELS_V2_1_DIR, fname)
            assert os.path.isfile(fpath), f"Missing model file: {fpath}"
            # Verify loadable
            obj = joblib.load(fpath)
            assert obj is not None
