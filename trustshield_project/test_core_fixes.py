"""Comprehensive test suite for TrustShield Phase 1 Core Fixes:
- Unified Trust Engine (decisions, confidence, disagreement, feedback loop guard)
- Probability Calibration & ECE & Cost-sensitive threshold optimization
- Entity ID split isolation
- Missingness indicator generation
- Reproducibility seed locking
- Multi-seed summaries and bootstrap confidence intervals
- Artifact versioning & SHA256 hashing
- API /ready probe endpoint
"""

import math
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from trustshield_project.advanced_trust_engine import (
    CanonicalTrustEngine as TrustEngine,
    OperationalDecision as Decision,
    ConformalPredictor,
    AdvancedTrustEngine,
)

from trustshield_project.calibration import (
    ProbabilityCalibrator,
    calculate_brier_score,
    calculate_expected_calibration_error,
    optimize_cost_sensitive_threshold,
)
from trustshield_project.splits import assert_disjoint_ids, validate_temporal_boundaries
from trustshield_project.missingness import add_missingness_indicators, audit_missingness
from trustshield_project.reproducibility import seed_everything
from trustshield_project.robustness import (
    multiseed_summary,
    bootstrap_ci,
    simulate_prevalence_shift,
)
from trustshield_project.versioning import sha256_file, create_model_metadata, FEATURE_SCHEMA_VERSION


# ===========================================================================
# 1. Trust Engine Tests
# ===========================================================================

class TestTrustEngine:
    def test_default_initialization(self):
        engine = TrustEngine()
        assert math.isclose(sum(engine.norm_weights.values()), 1.0, rel_tol=1e-5)
        assert engine.allow_cutoff == 0.25 and engine.review_cutoff == 0.55 and engine.hold_cutoff == 0.85

    def test_invalid_weights_raise_error(self):
        with pytest.raises(ValueError):
            TrustEngine(weights={"a": -1.0, "b": 0.5})
        with pytest.raises(ValueError):
            TrustEngine(weights={})

    def test_invalid_thresholds_raise_error(self):
        with pytest.raises(ValueError):
            TrustEngine(thresholds=(0.8, 0.4, 0.9))

    def test_decision_tiers(self):
        engine = TrustEngine()
        res_low = engine.score({"tabular_risk": 0.05, "graph_risk": 0.05, "gnn_risk": 0.05})
        assert res_low.decision == Decision.ALLOW and res_low.trust_score > 90.0
        assert engine.score({"tabular_risk": 0.45, "graph_risk": 0.35, "gnn_risk": 0.40}).decision == Decision.REVIEW
        assert engine.score({"tabular_risk": 0.70, "graph_risk": 0.75, "gnn_risk": 0.65}).decision == Decision.HOLD
        assert engine.score({"tabular_risk": 0.95, "graph_risk": 0.90, "gnn_risk": 0.92}).decision == Decision.BLOCK

    def test_shannon_entropy_confidence(self):
        assert TrustEngine.calculate_confidence(0.0) > 0.99 and TrustEngine.calculate_confidence(1.0) > 0.99
        assert TrustEngine.calculate_confidence(0.5) < 0.05

    def test_cold_start_confidence_penalty(self):
        engine = TrustEngine()
        comp = {"tabular_risk": 0.1, "graph_risk": 0.1}
        res_cold = engine.score(comp, is_cold_start=True)
        assert res_cold.confidence < engine.score(comp, is_cold_start=False).confidence
        assert "COLD_START_INSUFFICIENT_HISTORY" in res_cold.reason_codes

    def test_model_disagreement_detection(self):
        res = TrustEngine().score({"tabular_risk": 0.10, "graph_risk": 0.90})
        assert res.model_disagreement == 0.80 and any("HIGH_DETECTOR_DISAGREEMENT" in r for r in res.reason_codes)

    def test_feedback_loop_guard(self):
        with pytest.raises(ValueError, match="Feedback loop violation"):
            TrustEngine.validate_no_feedback_loop(["amount", "risk_score", "seller_age"])
        TrustEngine.validate_no_feedback_loop(["amount", "buyer_orders_before", "seller_age"])


class TestCalibration:
    def test_isotonic_calibrator_fit_and_predict(self):
        np.random.seed(42)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        raw_scores = np.array([0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.85, 0.95])
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(raw_scores, y)
        preds = calibrator.predict_proba(raw_scores)
        assert len(preds) == len(raw_scores) and np.all((preds >= 0.0) & (preds <= 1.0)) and np.all(np.diff(preds) >= -1e-6)

    def test_sigmoid_calibrator(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        cal = ProbabilityCalibrator(method="sigmoid")
        cal.fit(scores, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        preds = cal.predict_proba(scores)
        assert len(preds) == len(scores) and np.all((preds >= 0.0) & (preds <= 1.0))

    def test_brier_score_and_ece(self):
        y = np.array([0, 0, 1, 1])
        assert calculate_brier_score(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
        assert calculate_brier_score(y, np.array([0.9, 0.8, 0.1, 0.2])) > 0.5
        assert calculate_expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) < 0.05

    def test_cost_sensitive_threshold_optimization(self):
        opt = optimize_cost_sensitive_threshold(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), np.linspace(0.1, 0.9, 10), fn_cost=10.0, fp_cost=1.0)
        assert "threshold" in opt and "cost" in opt and 0.0 < opt["threshold"] < 1.0


class TestSplits:
    def test_assert_disjoint_ids_passes(self):
        assert_disjoint_ids(pd.DataFrame({"order_id": ["O1", "O2"]}), pd.DataFrame({"order_id": ["O3"]}), pd.DataFrame({"order_id": ["O4"]}), "order_id")

    def test_assert_disjoint_ids_catches_leakage(self):
        with pytest.raises(AssertionError, match="leakage"):
            assert_disjoint_ids(pd.DataFrame({"order_id": ["O1", "O2"]}), pd.DataFrame({"order_id": ["O2", "O3"]}), pd.DataFrame({"order_id": ["O4"]}), "order_id")

    def test_validate_temporal_boundaries(self):
        train = pd.DataFrame({"order_date": ["2024-01-01", "2024-01-15"]})
        val = pd.DataFrame({"order_date": ["2024-01-16", "2024-01-31"]})
        test = pd.DataFrame({"order_date": ["2024-02-01", "2024-02-15"]})
        validate_temporal_boundaries(train, val, test, "order_date")
        with pytest.raises(AssertionError, match="Temporal boundary violation"):
            validate_temporal_boundaries(train, pd.DataFrame({"order_date": ["2024-01-10", "2024-01-20"]}), test, "order_date")


class TestMissingness:
    def test_add_missingness_indicators(self):
        out = add_missingness_indicators(pd.DataFrame({"amount": [10.0, None, 30.0], "device_id": ["D1", "D2", None]}))
        assert list(out["amount__missing"]) == [0, 1, 0] and list(out["device_id__missing"]) == [0, 0, 1]

    def test_audit_missingness(self):
        rates = audit_missingness(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, None, 3.0]}))
        assert "a" not in rates and math.isclose(rates["b"], 1.0 / 3.0, rel_tol=1e-3)


class TestReproducibilityAndRobustness:
    def test_seed_everything(self):
        seed_everything(123)
        v1 = np.random.rand()
        seed_everything(123)
        assert v1 == np.random.rand()

    def test_multiseed_summary(self):
        summary = multiseed_summary([0.82, 0.85, 0.84, 0.83, 0.86])
        assert summary["n_seeds"] == 5 and math.isclose(summary["mean"], 0.84, rel_tol=1e-3)
        assert summary["min"] == 0.82 and summary["max"] == 0.86

    def test_bootstrap_ci(self):
        y = np.array([0] * 50 + [1] * 50)
        p = np.array([0.1] * 45 + [0.8] * 5 + [0.2] * 5 + [0.9] * 45)
        res = bootstrap_ci(y, p, lambda yt, yp: float(np.mean(yt == (yp >= 0.5))), n_bootstraps=100, seed=42)
        assert res["ci_lower"] <= res["estimate"] <= res["ci_upper"]

    def test_simulate_prevalence_shift(self):
        y_shifted, _ = simulate_prevalence_shift(np.array([0] * 900 + [1] * 100), np.random.rand(1000), target_prevalence=0.02)
        assert np.mean(y_shifted) < 0.05


class TestVersioning:
    def test_sha256_file(self, tmp_path):
        f = tmp_path / "model.bin"
        f.write_bytes(b"TrustShield Model Weight Bytes")
        assert len(sha256_file(f)) == 64 and isinstance(sha256_file(f), str)

    def test_create_model_metadata(self, tmp_path):
        f = tmp_path / "model.bin"
        f.write_bytes(b"Artifact")
        meta = create_model_metadata("Hybrid_GNN_XGBoost", "1.0.0", ["amount", "buyer_orders_before"], "2024-01-15", "2024-01-31", f)
        assert meta["schema_version"] == FEATURE_SCHEMA_VERSION and meta["n_features"] == 2 and "artifact_sha256" in meta


class TestReadyEndpoint:
    def test_ready_endpoint_status(self):
        from backend.main import app
        res = TestClient(app).get("/ready")
        assert res.status_code == 200 and res.json()["status"] in ("ready", "degraded", "not_ready")


class TestAuditFixes:
    def test_derived_return_rate_capped_at_one(self):
        from backend.features import _build_feature_row
        from backend.schemas import TransactionScoreRequest
        row = _build_feature_row(TransactionScoreRequest(buyer_orders_before=2, buyer_returns_before=10), ["buyer_return_rate_before"])
        assert row["buyer_return_rate_before"] == 1.0

    def test_derived_return_rate_normal_ratio(self):
        from backend.features import _build_feature_row
        from backend.schemas import TransactionScoreRequest
        row = _build_feature_row(TransactionScoreRequest(buyer_orders_before=10, buyer_returns_before=2), ["buyer_return_rate_before"])
        assert row["buyer_return_rate_before"] == pytest.approx(0.20)

    def test_zero_base_price_produces_no_infs(self):
        from trustshield_project.baseline_model import build_features
        orders_df = pd.DataFrame([{"order_id": "O1", "listing_id": "L1", "order_date": pd.Timestamp("2025-05-01"), "amount": 50.0, "buyer_id": "B1", "seller_id": "S1", "device_id": "D1"}])
        listings_df = pd.DataFrame([{"listing_id": "L1", "seller_id": "S1", "product_id": "P1", "category": "Electronics", "listing_date": pd.Timestamp("2025-04-01")}])
        returns_df = pd.DataFrame(columns=["buyer_id", "return_date"])
        buyers_df = pd.DataFrame([{"buyer_id": "B1", "signup_date": pd.Timestamp("2025-01-01")}])
        sellers_df = pd.DataFrame([{"seller_id": "S1", "signup_date": pd.Timestamp("2025-01-01")}])
        products_df = pd.DataFrame([{"product_id": "P1", "base_price": 0.0}])
        df, _ = build_features(orders_df, listings_df, returns_df, buyers_df, sellers_df, products_df)
        val = df.loc[0, "price_vs_base_price_ratio"]
        assert not np.isinf(val) and not np.isnan(val) and val == 0.0


class TestConformalPredictor:
    def test_exact_finite_sample_formula_toy_sizes(self):
        scores_19 = np.linspace(0.01, 0.95, 19)
        assert ConformalPredictor.compute_finite_sample_quantile(scores_19, 0.05) == pytest.approx(scores_19[-1])
        scores_99 = np.linspace(0.01, 0.99, 99)
        assert ConformalPredictor.compute_finite_sample_quantile(scores_99, 0.05) == pytest.approx(scores_99[94])
        scores_100 = np.linspace(0.01, 1.0, 100)
        assert ConformalPredictor.compute_finite_sample_quantile(scores_100, 0.05) == pytest.approx(scores_100[95])

    def test_empty_or_degenerate_arrays(self):
        assert ConformalPredictor.compute_finite_sample_quantile(np.array([]), 0.05) == 1.0
        assert ConformalPredictor.compute_finite_sample_quantile(np.array([0.42]), 0.05) == 1.0

    def test_prediction_set_inclusion_rules(self):
        cp = ConformalPredictor(alpha=0.05)
        cp.q_hat_0, cp.q_hat_1, cp.is_calibrated = 0.20, 0.90, True
        assert cp.predict_set(0.05) == {0} and cp.predict_set(0.15) == {0, 1} and cp.predict_set(0.85) == {1}

    def test_non_empty_fallback(self):
        cp = ConformalPredictor(alpha=0.05)
        cp.q_hat_0, cp.q_hat_1, cp.is_calibrated = 0.30, 0.60, True
        assert cp.predict_set(0.35) == {0} and len(cp.predict_set(0.39)) >= 1

    def test_uncalibrated_graceful_fallback(self):
        cp = ConformalPredictor(alpha=0.05)
        assert cp.is_calibrated is False
        assert cp.predict_set(0.10) == {0} and cp.predict_set(0.50) == {0, 1} and cp.predict_set(0.90) == {1}

    def test_missing_calibration_artifacts_handling(self):
        cp = ConformalPredictor(alpha=0.05)
        cp.calibrate(np.array([]), np.array([]))
        assert cp.is_calibrated is False and cp.q_hat_0 == 0.5 and cp.q_hat_1 == 0.5 and len(cp.predict_set(0.42)) >= 1

    def test_engine_evaluates_conformal_on_model_probability(self):
        engine = AdvancedTrustEngine()
        np.random.seed(42)
        engine.conformal.calibrate(np.concatenate([np.random.beta(0.5, 20, 1000), np.random.beta(5, 2, 100)]), np.concatenate([np.zeros(1000, dtype=int), np.ones(100, dtype=int)]))
        result = engine.score(detector_scores={"tabular_risk": 0.01, "graph_risk": 0.95, "device_velocity_risk": 0.90}, model_probability=0.01)
        assert result.calibrated_risk > 0.50 and result.conformal_prediction_set == {0}

