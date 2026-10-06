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

from trustshield_project.trust_engine import TrustEngine, Decision, TrustResult
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
        assert math.isclose(sum(engine.weights.values()), 1.0, rel_tol=1e-5)
        assert engine.allow_cutoff == 0.25
        assert engine.review_cutoff == 0.60
        assert engine.hold_cutoff == 0.85

    def test_invalid_weights_raise_error(self):
        with pytest.raises(ValueError):
            TrustEngine(weights={"a": -1.0, "b": 0.5})
        with pytest.raises(ValueError):
            TrustEngine(weights={})

    def test_invalid_thresholds_raise_error(self):
        with pytest.raises(ValueError):
            TrustEngine(thresholds=(0.8, 0.4, 0.9))  # Not monotonic

    def test_decision_tiers(self):
        engine = TrustEngine()
        # Clean low risk -> ALLOW
        res_low = engine.score({"tabular_risk": 0.05, "graph_risk": 0.05, "gnn_risk": 0.05})
        assert res_low.decision == Decision.ALLOW
        assert res_low.trust_score > 90.0

        # Moderate risk -> REVIEW
        res_med = engine.score({"tabular_risk": 0.45, "graph_risk": 0.35, "gnn_risk": 0.40})
        assert res_med.decision == Decision.REVIEW

        # High risk -> HOLD
        res_high = engine.score({"tabular_risk": 0.70, "graph_risk": 0.75, "gnn_risk": 0.65})
        assert res_high.decision == Decision.HOLD

        # Extreme risk -> BLOCK
        res_block = engine.score({"tabular_risk": 0.95, "graph_risk": 0.90, "gnn_risk": 0.92})
        assert res_block.decision == Decision.BLOCK

    def test_shannon_entropy_confidence(self):
        # Extreme predictions should have near-perfect confidence
        conf_0 = TrustEngine.calculate_confidence(0.0)
        conf_1 = TrustEngine.calculate_confidence(1.0)
        assert conf_0 > 0.99
        assert conf_1 > 0.99

        # Maximum uncertainty at 0.5
        conf_mid = TrustEngine.calculate_confidence(0.5)
        assert conf_mid < 0.05

    def test_cold_start_confidence_penalty(self):
        engine = TrustEngine()
        comp = {"tabular_risk": 0.1, "graph_risk": 0.1}
        res_warm = engine.score(comp, is_cold_start=False)
        res_cold = engine.score(comp, is_cold_start=True)
        assert res_cold.confidence < res_warm.confidence
        assert "COLD_START_INSUFFICIENT_HISTORY" in res_cold.reason_codes

    def test_model_disagreement_detection(self):
        engine = TrustEngine()
        # High disagreement between tabular (0.1) and graph (0.9)
        res = engine.score({"tabular_risk": 0.10, "graph_risk": 0.90})
        assert res.model_disagreement == 0.80
        assert "HIGH_DETECTOR_DISAGREEMENT" in res.reason_codes

    def test_feedback_loop_guard(self):
        # Forbidden model outputs must trigger validation error
        with pytest.raises(ValueError, match="Feedback loop violation"):
            TrustEngine.validate_no_feedback_loop(["amount", "risk_score", "seller_age"])

        # Legitimate features must pass
        TrustEngine.validate_no_feedback_loop(["amount", "buyer_orders_before", "seller_age"])


# ===========================================================================
# 2. Probability Calibration Tests
# ===========================================================================

class TestCalibration:
    def test_isotonic_calibrator_fit_and_predict(self):
        # Raw uncalibrated probabilities (skewed high)
        np.random.seed(42)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        raw_scores = np.array([0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.85, 0.95])

        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(raw_scores, y)
        preds = calibrator.predict_proba(raw_scores)

        assert len(preds) == len(raw_scores)
        assert np.all((preds >= 0.0) & (preds <= 1.0))
        # Monotonicity check
        assert np.all(np.diff(preds) >= -1e-6)

    def test_sigmoid_calibrator(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        cal = ProbabilityCalibrator(method="sigmoid")
        cal.fit(scores, y)
        preds = cal.predict_proba(scores)
        assert len(preds) == len(scores)
        assert np.all((preds >= 0.0) & (preds <= 1.0))

    def test_brier_score_and_ece(self):
        y = np.array([0, 0, 1, 1])
        p_perfect = np.array([0.0, 0.0, 1.0, 1.0])
        p_poor = np.array([0.9, 0.8, 0.1, 0.2])

        assert calculate_brier_score(y, p_perfect) == 0.0
        assert calculate_brier_score(y, p_poor) > 0.5

        ece = calculate_expected_calibration_error(y, p_perfect)
        assert ece < 0.05

    def test_cost_sensitive_threshold_optimization(self):
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.linspace(0.1, 0.9, 10)

        # High FN cost (fraud is expensive) pushes threshold lower
        opt = optimize_cost_sensitive_threshold(y, scores, fn_cost=10.0, fp_cost=1.0)
        assert "threshold" in opt
        assert "cost" in opt
        assert 0.0 < opt["threshold"] < 1.0


# ===========================================================================
# 3. Splits & Entity Isolation Tests
# ===========================================================================

class TestSplits:
    def test_assert_disjoint_ids_passes(self):
        train = pd.DataFrame({"order_id": ["O1", "O2", "O3"]})
        val = pd.DataFrame({"order_id": ["O4", "O5"]})
        test = pd.DataFrame({"order_id": ["O6", "O7"]})
        assert_disjoint_ids(train, val, test, "order_id")  # Should not raise

    def test_assert_disjoint_ids_catches_leakage(self):
        train = pd.DataFrame({"order_id": ["O1", "O2", "O3"]})
        val = pd.DataFrame({"order_id": ["O3", "O4"]})  # O3 leaked!
        test = pd.DataFrame({"order_id": ["O5"]})
        with pytest.raises(AssertionError, match="leakage"):
            assert_disjoint_ids(train, val, test, "order_id")

    def test_validate_temporal_boundaries(self):
        train = pd.DataFrame({"order_date": ["2024-01-01", "2024-01-15"]})
        val = pd.DataFrame({"order_date": ["2024-01-16", "2024-01-31"]})
        test = pd.DataFrame({"order_date": ["2024-02-01", "2024-02-15"]})
        validate_temporal_boundaries(train, val, test, "order_date")  # Should pass

        # Overlapping boundary
        bad_val = pd.DataFrame({"order_date": ["2024-01-10", "2024-01-20"]})
        with pytest.raises(AssertionError, match="Temporal boundary violation"):
            validate_temporal_boundaries(train, bad_val, test, "order_date")


# ===========================================================================
# 4. Missingness Tests
# ===========================================================================

class TestMissingness:
    def test_add_missingness_indicators(self):
        df = pd.DataFrame({
            "amount": [10.0, None, 30.0],
            "device_id": ["D1", "D2", None],
        })
        out = add_missingness_indicators(df)
        assert "amount__missing" in out.columns
        assert "device_id__missing" in out.columns
        assert list(out["amount__missing"]) == [0, 1, 0]
        assert list(out["device_id__missing"]) == [0, 0, 1]

    def test_audit_missingness(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [1.0, None, 3.0],
        })
        rates = audit_missingness(df)
        assert "a" not in rates
        assert "b" in rates
        assert math.isclose(rates["b"], 1.0 / 3.0, rel_tol=1e-3)


# ===========================================================================
# 5. Reproducibility & Robustness Tests
# ===========================================================================

class TestReproducibilityAndRobustness:
    def test_seed_everything(self):
        seed_everything(123)
        v1 = np.random.rand()
        seed_everything(123)
        v2 = np.random.rand()
        assert v1 == v2

    def test_multiseed_summary(self):
        scores = [0.82, 0.85, 0.84, 0.83, 0.86]
        summary = multiseed_summary(scores)
        assert summary["n_seeds"] == 5
        assert math.isclose(summary["mean"], 0.84, rel_tol=1e-3)
        assert summary["min"] == 0.82
        assert summary["max"] == 0.86

    def test_bootstrap_ci(self):
        y = np.array([0] * 50 + [1] * 50)
        p = np.array([0.1] * 45 + [0.8] * 5 + [0.2] * 5 + [0.9] * 45)

        def mock_acc(y_true, y_prob):
            return float(np.mean(y_true == (y_prob >= 0.5)))

        res = bootstrap_ci(y, p, mock_acc, n_bootstraps=100, seed=42)
        assert "estimate" in res
        assert "ci_lower" in res
        assert "ci_upper" in res
        assert res["ci_lower"] <= res["estimate"] <= res["ci_upper"]

    def test_simulate_prevalence_shift(self):
        y = np.array([0] * 900 + [1] * 100)  # 10% prevalence
        p = np.random.rand(1000)
        y_shifted, p_shifted = simulate_prevalence_shift(y, p, target_prevalence=0.02)
        new_prev = np.mean(y_shifted)
        # Should be much closer to 2%
        assert new_prev < 0.05


# ===========================================================================
# 6. Versioning Tests
# ===========================================================================

class TestVersioning:
    def test_sha256_file(self, tmp_path):
        test_file = tmp_path / "model.bin"
        test_file.write_bytes(b"TrustShield Model Weight Bytes")
        digest = sha256_file(test_file)
        assert len(digest) == 64
        assert isinstance(digest, str)

    def test_create_model_metadata(self, tmp_path):
        test_file = tmp_path / "model.bin"
        test_file.write_bytes(b"Artifact")
        meta = create_model_metadata(
            model_name="Hybrid_GNN_XGBoost",
            model_version="1.0.0",
            feature_cols=["amount", "buyer_orders_before"],
            train_cutoff="2024-01-15",
            val_cutoff="2024-01-31",
            artifact_path=test_file,
        )
        assert meta["schema_version"] == FEATURE_SCHEMA_VERSION
        assert meta["n_features"] == 2
        assert "artifact_sha256" in meta


# ===========================================================================
# 7. Backend /ready Probe Test
# ===========================================================================

class TestReadyEndpoint:
    def test_ready_endpoint_status(self):
        from backend.main import app
        client = TestClient(app)
        res = client.get("/ready")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ("ready", "degraded", "not_ready")
        assert "models_ready" in data
        assert "details" in data
