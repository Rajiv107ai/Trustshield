"""Targeted P0 Remediation Regression Test Suite.

Verifies:
1. Conformal Score Semantics (Step 1):
   - Exact finite-sample quantile calculation: k = ceil((n + 1) * (1 - alpha))
   - Prediction-set inclusion logic under Mondrian conformal prediction
   - Missing calibration artifacts handling and safe fallback
   - Score-source consistency: conformal evaluation strictly uses calibrated model probability,
     NEVER the composite policy_risk_score.

2. Point-in-Time Edge-Weight Serving (Step 2):
   - First transaction between buyer and seller = 0.0
   - Repeated transactions monotonically increment prior count
   - Future transactions are strictly excluded
   - Equal timestamps handling (strictly before; allow_exact_matches=False)
   - Unknown pairs safely default to 0.0 with honest provenance
   - Client-supplied mismatch detection and authoritative server value enforcement
   - Historical replay consistency matches canonical orders.csv replay logic
"""

import math
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure trustshield_project and root are in path
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_curr_dir)
_trust_dir = os.path.join(_root_dir, "trustshield_project")
for p in [_root_dir, _trust_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from trustshield_project.advanced_trust_engine import (
    ConformalPredictor,
    AdvancedTrustEngine,
    OperationalDecision,
)
from trustshield_project.point_in_time_history import PointInTimeHistoryStore
from backend.main import app, store

AUTH_HEADERS = {"X-API-Key": "dev-operator-api-key-32chars-long"}


@pytest.fixture(scope="module")
def api_client():
    store.load()
    with TestClient(app, headers=AUTH_HEADERS) as c:
        yield c


# ===========================================================================
# Step 1 Tests: Conformal Score Semantics & Mondrian Guarantees
# ===========================================================================

class TestConformalQuantileCalculation:
    """Tests for exact finite-sample quantile calculation: k = ceil((n + 1) * (1 - alpha))."""

    def test_exact_finite_sample_formula_toy_sizes(self):
        """Verifies k = ceil((n + 1) * (1 - alpha)) order statistic across various sample sizes."""
        alpha = 0.05

        # Case 1: n = 19 -> (19 + 1) * 0.95 = 19.0 -> k = 19 (the maximum item)
        scores_19 = np.linspace(0.01, 0.95, 19)
        q_19 = ConformalPredictor.compute_finite_sample_quantile(scores_19, alpha)
        assert q_19 == pytest.approx(scores_19[-1])

        # Case 2: n = 99 -> (99 + 1) * 0.95 = 95.0 -> k = 95 (index 94)
        scores_99 = np.linspace(0.01, 0.99, 99)
        q_99 = ConformalPredictor.compute_finite_sample_quantile(scores_99, alpha)
        assert q_99 == pytest.approx(scores_99[94])

        # Case 3: n = 100 -> (100 + 1) * 0.95 = 95.95 -> ceil = 96 (index 95)
        scores_100 = np.linspace(0.01, 1.0, 100)
        q_100 = ConformalPredictor.compute_finite_sample_quantile(scores_100, alpha)
        assert q_100 == pytest.approx(scores_100[95])

    def test_empty_or_degenerate_arrays(self):
        """Edge case: empty or single-element arrays."""
        empty_scores = np.array([])
        assert ConformalPredictor.compute_finite_sample_quantile(empty_scores, 0.05) == 1.0

        single_score = np.array([0.42])
        # n = 1 -> (1 + 1) * 0.95 = 1.9 -> ceil = 2 > n -> returns 1.0
        assert ConformalPredictor.compute_finite_sample_quantile(single_score, 0.05) == 1.0


class TestConformalPredictionSetInclusion:
    """Tests prediction-set inclusion logic under Mondrian conformal prediction."""

    def test_prediction_set_inclusion_rules(self):
        """Verifies {0}, {1}, and {0, 1} sets based on non-conformity cutoffs.

        Class 0 non-conformity: s0 = p <= q_hat_0 -> include 0
        Class 1 non-conformity: s1 = 1 - p <= q_hat_1 -> include 1 (i.e. p >= 1 - q_hat_1)
        """
        cp = ConformalPredictor(alpha=0.05)
        # Manually calibrate with clear quantiles
        # q_hat_0 = 0.20 (scores above 0.20 are not conforming to class 0)
        # q_hat_1 = 0.90 (scores below 1 - 0.90 = 0.10 are not conforming to class 1)
        cp.q_hat_0 = 0.20
        cp.q_hat_1 = 0.90
        cp.is_calibrated = True

        # Case 1: p = 0.05 -> p <= 0.20 (include 0), p < 0.10 (exclude 1) -> {0}
        assert cp.predict_set(0.05) == {0}

        # Case 2: p = 0.15 -> p <= 0.20 (include 0), p >= 0.10 (include 1) -> {0, 1}
        assert cp.predict_set(0.15) == {0, 1}

        # Case 3: p = 0.85 -> p > 0.20 (exclude 0), p >= 0.10 (include 1) -> {1}
        assert cp.predict_set(0.85) == {1}

    def test_non_empty_fallback(self):
        """If gap between bounds theoretically produces empty set, fallback produces argmax label."""
        cp = ConformalPredictor(alpha=0.05)
        # Gap where q_hat_0 < 1 - q_hat_1
        cp.q_hat_0 = 0.30
        cp.q_hat_1 = 0.60  # 1 - q_hat_1 = 0.40
        cp.is_calibrated = True

        # p = 0.35: p > 0.30 (not 0) and p < 0.40 (not 1) -> empty set prevented by fallback
        res_low = cp.predict_set(0.35)
        assert res_low == {0}  # argmax(0.35 < 0.5)

        res_high = cp.predict_set(0.39)
        assert len(res_high) >= 1


class TestConformalMissingArtifacts:
    """Verifies graceful degradation when calibration files are absent."""

    def test_uncalibrated_graceful_fallback(self):
        """When not calibrated, ConformalPredictor provides safe default heuristic sets."""
        cp = ConformalPredictor(alpha=0.05)
        assert cp.is_calibrated is False

        # In uncalibrated mode:
        # p < 0.35 -> {0}
        # 0.35 <= p <= 0.65 -> {0, 1}
        # p > 0.65 -> {1}
        assert cp.predict_set(0.10) == {0}
        assert cp.predict_set(0.50) == {0, 1}
        assert cp.predict_set(0.90) == {1}

    def test_missing_calibration_artifacts_handling(self):
        """When calibration files cannot be found, ConformalPredictor remains uncalibrated safely."""
        cp = ConformalPredictor(alpha=0.05)
        # Attempt calibrate with empty/zero inputs
        cp.calibrate(np.array([]), np.array([]))
        assert cp.is_calibrated is False
        assert cp.q_hat_0 == 0.5
        assert cp.q_hat_1 == 0.5
        # predict_set still returns valid set without raising exception
        s = cp.predict_set(0.42)
        assert isinstance(s, set)
        assert len(s) >= 1


class TestScoreSourceConsistency:
    """Verifies that conformal set is evaluated on model probability, NOT policy_risk_score."""

    def test_engine_evaluates_conformal_on_model_probability(self):
        """Crucial correctness test:

        Create a scenario where composite policy_risk_score is high (e.g. 0.85)
        due to non-model heuristic detector scores, but calibrated model_probability is low (0.01).
        The conformal predictor calibrated on model probabilities must receive 0.01 and predict {0}.
        """
        engine = AdvancedTrustEngine()
        # Calibrate conformal predictor on realistic model probabilities
        cal_probs = np.concatenate([np.random.beta(0.5, 20, 1000), np.random.beta(5, 2, 100)])
        cal_labels = np.concatenate([np.zeros(1000, dtype=int), np.ones(100, dtype=int)])
        engine.conformal.calibrate(cal_probs, cal_labels)

        # High heuristic risk signals drive composite policy risk score high
        detector_scores = {
            "tabular_risk": 0.01,
            "graph_risk": 0.95,
            "device_velocity_risk": 0.90,
        }

        # Case A: Passing model_probability = 0.01 explicitly
        result_with_prob = engine.score(
            detector_scores=detector_scores,
            model_probability=0.01,
        )

        # The composite risk is high (> 0.50)
        assert result_with_prob.calibrated_risk > 0.50
        # But the conformal prediction set strictly evaluated on 0.01 must NOT include fraud (1)
        # Class 1 requires prob >= 1 - q_hat_1, which for 0.01 is false.
        assert 0 in result_with_prob.conformal_prediction_set
        assert result_with_prob.conformal_prediction_set == {0}

    def test_api_passes_calibrated_probability_to_engine(self, api_client):
        """Verifies through the API endpoint that conformal prediction set aligns with model prob."""
        payload = {
            "order_id": "ORD_SCORE_SOURCE_TEST",
            "amount": 25.0,
            "base_price": 25.0,
            "category_median_price": 25.0,
            "buyer_orders_before": 20,
            "buyer_returns_before": 0,
            "buyer_age_days": 180.0,
            "seller_age_days": 365.0,
            "seller_total_listings_before": 50,
        }
        resp = api_client.post("/transaction/score", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["calibrated_fraud_probability"] is not None
        assert data["policy_risk_score"] is not None
        assert data["conformal_is_calibrated"] is True
        assert isinstance(data["conformal_prediction_set"], list)
        assert len(data["conformal_prediction_set"]) >= 1


# ===========================================================================
# Step 2 Tests: Point-in-Time Edge-Weight Serving
# ===========================================================================

class TestPointInTimeEdgeWeightLogic:
    """Direct algorithmic tests on PointInTimeHistoryStore."""

    @pytest.fixture
    def sample_store(self):
        """Creates a standalone store with controlled transactions."""
        store = PointInTimeHistoryStore()
        # Seed transactions for pair ('BUYER_A', 'SELLER_X')
        # t1: 2025-01-10 10:00:00
        # t2: 2025-01-10 10:00:00 (exact duplicate timestamp)
        # t3: 2025-02-01 12:00:00
        # t4: 2025-03-01 09:00:00
        store._history[("BUYER_A", "SELLER_X")] = [
            pd.Timestamp("2025-01-10 10:00:00"),
            pd.Timestamp("2025-01-10 10:00:00"),
            pd.Timestamp("2025-02-01 12:00:00"),
            pd.Timestamp("2025-03-01 09:00:00"),
        ]
        store.is_loaded = True
        return store

    def test_first_transaction_is_zero(self, sample_store):
        """Query before any recorded transaction returns exactly 0.0."""
        weight, src, _ = sample_store.get_prior_edge_weight(
            "BUYER_A", "SELLER_X", order_date="2025-01-01 00:00:00"
        )
        assert weight == 0.0
        assert src == "server_point_in_time"

    def test_repeated_transactions_increment(self, sample_store):
        """Query between transactions reflects strictly prior occurrences."""
        # After t1, t2 but before t3
        weight, src, _ = sample_store.get_prior_edge_weight(
            "BUYER_A", "SELLER_X", order_date="2025-01-15 00:00:00"
        )
        assert weight == 2.0
        assert src == "server_point_in_time"

        # After t3 but before t4
        weight, src, _ = sample_store.get_prior_edge_weight(
            "BUYER_A", "SELLER_X", order_date="2025-02-15 00:00:00"
        )
        assert weight == 3.0

        # After all transactions
        weight, src, _ = sample_store.get_prior_edge_weight(
            "BUYER_A", "SELLER_X", order_date="2025-04-01 00:00:00"
        )
        assert weight == 4.0

    def test_future_transactions_strictly_excluded(self, sample_store):
        """Transactions occurring at or after order_date are never counted."""
        weight, _, _ = sample_store.get_prior_edge_weight(
            "BUYER_A", "SELLER_X", order_date="2025-02-01 11:59:59"
        )
        # t3 is at 12:00:00, so only t1 and t2 (count=2) are included
        assert weight == 2.0

    def test_equal_timestamps_do_not_count_each_other(self, sample_store):
        """Transactions sharing identical timestamps (allow_exact_matches=False) exclude each other."""
        # Querying at exact duplicate timestamp 2025-01-10 10:00:00
        # Both t1 and t2 are at 10:00:00, so strictly before (< target_dt) yields 0
        weight, _, _ = sample_store.get_prior_edge_weight(
            "BUYER_A", "SELLER_X", order_date="2025-01-10 10:00:00"
        )
        assert weight == 0.0

    def test_unknown_pairs_safely_return_zero(self, sample_store):
        """Unknown buyer-seller pair returns 0.0 with honest fallback provenance."""
        weight, src, note = sample_store.get_prior_edge_weight(
            "BUYER_UNKNOWN", "SELLER_UNKNOWN", order_date="2025-02-01 00:00:00"
        )
        assert weight == 0.0
        assert src == "fallback_unobserved_zero"
        assert "First transaction between buyer and seller" in note

    def test_unloaded_store_safe_failure(self):
        """When transaction history is unavailable, returns safe 0.0 fallback (no cumulative snapshot)."""
        empty_store = PointInTimeHistoryStore()
        weight, src, note = empty_store.get_prior_edge_weight(
            "BUYER_A", "SELLER_X", order_date="2025-02-01 00:00:00"
        )
        assert weight == 0.0
        assert src == "fallback_history_unavailable"
        assert "Server point-in-time history unavailable" in note


class TestPointInTimeHistoricalReplayConsistency:
    """Verifies that PointInTimeHistoryStore matches canonical pandas cumcount on orders.csv."""

    def test_replay_exact_match_against_canonical_cumcount(self):
        """Verifies point-in-time calculation matches cumcount on orders.csv sample."""
        orders_csv_path = os.path.join(_root_dir, "data", "synthetic_v2_1", "orders.csv")
        if not os.path.isfile(orders_csv_path):
            orders_csv_path = os.path.join(_root_dir, "data", "synthetic_v2", "orders.csv")
        if not os.path.isfile(orders_csv_path):
            pytest.skip("orders.csv not found for historical replay consistency test")

        df = pd.read_csv(orders_csv_path, usecols=["order_id", "buyer_id", "seller_id", "order_date"])
        df["order_date"] = pd.to_datetime(df["order_date"])
        df = df.sort_values(by="order_date", kind="mergesort").reset_index(drop=True)
        # Canonical point-in-time feature computation
        df["expected_edge_weight"] = df.groupby(["buyer_id", "seller_id"]).cumcount().astype(float)

        # Test against PointInTimeHistoryStore loaded from same orders.csv
        pit_store = PointInTimeHistoryStore(orders_csv_path)
        assert pit_store.is_loaded is True

        # Sample 500 rows across temporal distribution
        sample_indices = np.linspace(0, len(df) - 1, 500, dtype=int)
        mismatches = 0
        for idx in sample_indices:
            row = df.iloc[idx]
            calc_weight, src, _ = pit_store.get_prior_edge_weight(
                row["buyer_id"], row["seller_id"], order_date=row["order_date"]
            )
            # Both cumcount and PointInTimeHistoryStore (with mergesort index ordering) must match
            if calc_weight != row["expected_edge_weight"]:
                mismatches += 1

        assert mismatches == 0, f"Found {mismatches} historical replay mismatches vs canonical cumcount"


class TestPointInTimeApiServing:
    """API integration tests for point-in-time edge weight and client mismatch enforcement."""

    def test_api_honors_server_point_in_time_over_client_tampering(self, api_client):
        """If client passes fabricated buyer_seller_edge_weight_before, server overrides it."""
        tampered_payload = {
            "order_id": "ORD_TAMPERED_EDGE_WEIGHT",
            "buyer_id": "BUYER_00001",
            "seller_id": "SELLER_00189",
            "order_date": "2025-05-01 12:00:00",
            "amount": 100.0,
            "base_price": 100.0,
            "category_median_price": 100.0,
            "buyer_seller_edge_weight_before": 999.0,  # Deliberately spoofed client value
        }
        resp = api_client.post("/transaction/score", json=tampered_payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["graph_features_source"] == "server"
        assert data["client_mismatch"] is True
        assert "buyer_seller_edge_weight_before" in data["note"]
        assert "Server snapshot value enforced" in data["note"]

    def test_api_live_scoring_without_order_date(self, api_client):
        """Live scoring request without order_date uses decision-time confirmed history."""
        live_payload = {
            "order_id": "ORD_LIVE_DECISION_TIME",
            "buyer_id": "BUYER_00001",
            "seller_id": "SELLER_00189",
            "amount": 75.0,
            "base_price": 75.0,
            "category_median_price": 75.0,
        }
        resp = api_client.post("/transaction/score", json=live_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["graph_features_source"] == "server"
        assert data["client_mismatch"] is False
