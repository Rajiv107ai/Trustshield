"""
TrustShield — Stage 3.3 Controlled Model Evaluation Unit Test Suite.

Covers:
1. Gate 0: Observation maturity verification and unobserved return handling.
2. Gate 1: Task-specific model evaluation and baseline comparisons.
3. Gate 2: Calibration, threshold freezing, and honest metric reporting.
4. Gate 3: Combined trust engine evaluation and multimodal ablation verification.
5. Versioned artifact persistence in models/v2_1/.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_DATA_DIR = os.path.join(_ROOT_DIR, "data", "synthetic_v2_1")
_MODELS_V2_1_DIR = os.path.join(_ROOT_DIR, "models", "v2_1")
_REPORTS_DIR = os.path.join(_ROOT_DIR, "reports")

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


@pytest.fixture(scope="module")
def stage33_metrics() -> dict:
    """Loads reports/phase3_stage33_metrics.json."""
    p = os.path.join(_REPORTS_DIR, "phase3_stage33_metrics.json")
    assert os.path.isfile(p), f"Missing metrics file: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


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

    def test_unobserved_returns_not_imputed_as_negatives(self, stage33_metrics):
        """Verifies that return-abuse detection was evaluated strictly on observed returns

        (2,446 test samples) and did NOT impute unobserved orders as negatives.
        """
        ret_metrics = stage33_metrics["models"]["return_fraud_detector"]["xgboost_calibrated"]["test"]
        assert ret_metrics["support"]["total"] == 2446
        assert ret_metrics["support"]["positives"] == 785
        assert ret_metrics["support"]["negatives"] == 1661


class TestGate1TaskBaselines:
    """Gate 1: Task-specific baselines and model performance."""

    def test_fake_listing_models_evaluated(self, stage33_metrics):
        """Verifies fake-listing detector metrics on frozen test cohort (2,686 listings)."""
        fl_models = stage33_metrics["models"]["fake_listing_detector"]
        assert "baseline_heuristic" in fl_models
        assert "logistic_regression" in fl_models
        assert "random_forest" in fl_models
        assert "xgboost_calibrated" in fl_models

        # Test cohort sizing
        test_m = fl_models["xgboost_calibrated"]["test"]
        assert test_m["support"]["total"] == 2686
        assert test_m["support"]["positives"] == 56
        assert test_m["roc_auc"] >= 0.85
        assert test_m["pr_auc"] >= 0.35

    def test_transaction_fraud_models_and_subtypes(self, stage33_metrics):
        """Verifies transaction fraud detector metrics and subtype analyses."""
        tx_models = stage33_metrics["models"]["transaction_fraud_detector"]
        assert "tabular_logistic_regression" in tx_models
        assert "tabular_xgboost_no_graph" in tx_models
        assert "full_graph_xgboost_calibrated" in tx_models
        assert "degraded_zero_graph" in tx_models

        test_m = tx_models["full_graph_xgboost_calibrated"]["test"]
        assert test_m["support"]["total"] == 20919
        assert test_m["support"]["positives"] == 1835

        # Subtype breakdown verification
        subtypes = tx_models["subtype_analysis"]
        assert subtypes["fake_listing"]["support"] == 473
        assert subtypes["return_abuse"]["support"] == 467
        assert subtypes["coordinated_fraud"]["support"] == 445
        assert subtypes["seller_buyer_collusion"]["support"] == 450


class TestGate2EvaluationMethodology:
    """Gate 2: Calibration, threshold freezing, and honest reporting."""

    def test_thresholds_tuned_on_validation(self, feature_meta, stage33_metrics):
        """Verifies that decision thresholds were frozen from validation tuning."""
        thresholds = feature_meta["thresholds"]
        assert "fake_listing" in thresholds
        assert "transaction_fraud" in thresholds
        assert "return_fraud" in thresholds

        fl_val_t = stage33_metrics["models"]["fake_listing_detector"]["xgboost_calibrated"]["validation"]["threshold"]
        fl_test_t = stage33_metrics["models"]["fake_listing_detector"]["xgboost_calibrated"]["test"]["threshold"]
        assert fl_val_t == fl_test_t == thresholds["fake_listing"]

    def test_calibration_metrics_present(self, stage33_metrics):
        """Verifies that Brier score and 10-bin ECE are reported for all models."""
        cal_keys = {
            "fake_listing_detector": "xgboost_calibrated",
            "transaction_fraud_detector": "full_graph_xgboost_calibrated",
            "return_fraud_detector": "xgboost_calibrated",
        }
        for m_group, model_key in cal_keys.items():
            cal_m = stage33_metrics["models"][m_group][model_key]["test"]
            assert "brier_score" in cal_m
            assert "ece" in cal_m
            assert 0.0 <= cal_m["ece"] <= 0.20


class TestGate3CombinedTrustEngine:
    """Gate 3: Combined trust engine evaluation and ablations."""

    def test_multimodal_max_risk_synergy(self, stage33_metrics):
        """Verifies that multimodal listing risk provides positive synergy

        over transaction-only modeling.
        """
        abl = stage33_metrics["ablations"]
        assert abl["multimodal_incremental_pr_auc"] > 0.05
        comb = stage33_metrics["combined_trust_engine"]
        assert comb["max_risk_ensemble"]["pr_auc"] > comb["full_graph_model"]["pr_auc"]

    def test_trust_engine_decision_distribution(self, stage33_metrics):
        """Verifies that the TrustEngine produces structured decisions across all orders."""
        decisions = stage33_metrics["combined_trust_engine"]["trust_engine_decisions"]
        assert "ALLOW" in decisions
        assert "REVIEW" in decisions
        assert sum(decisions.values()) == 20919


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
