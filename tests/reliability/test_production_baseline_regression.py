"""
TrustShield Phase 4.3B — Production Baseline Regression Test Suite.

Verifies complete regression protection against the frozen post-Phase-4.3B
golden baseline fixture located at:
    backend/fixtures/phase3_golden_baseline.json

Assertions:
1. Environment and Artifact Integrity:
   - Model artifact SHA-256 hashes match frozen metadata.
   - 18 feature columns and exact order match feature_meta.joblib.
   - TransactionScoreResponse schema has exactly 29 response fields.
2. Comprehensive 29-field assertion for all 4 scenarios:
   - low_risk_standard
   - high_risk_anomalous
   - historical_point_in_time
   - cold_start_new_entities
   - Deterministic categorical/structural fields: exact equality (==).
   - Numeric probabilities/scores: documented tolerance (abs_tol=1e-4).
   - Local TreeSHAP attributions and top risk drivers: verified within tolerance.
3. Negative Sensitivity / Mutation Testing:
   - Verifies that deliberate mutations (decision, probability, cold_start,
     conformal set, reason codes, missing field) are reliably caught.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pytest
from fastapi.testclient import TestClient

from backend.main import app, store
from backend.schemas import TransactionScoreResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "backend" / "fixtures" / "phase3_golden_baseline.json"
AUTH_HEADERS = {"X-API-Key": "dev-admin-secret-key-32chars-long!"}

# Documented comparison tolerances
PROBABILITY_TOLERANCE = 1e-4
TRUST_SCORE_TOLERANCE = 1e-2
SHAP_TOLERANCE = 1e-3


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def golden_data() -> Dict[str, Any]:
    assert FIXTURE_PATH.is_file(), f"Golden baseline fixture missing at: {FIXTURE_PATH}"
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def client():
    store.load()
    with TestClient(app, headers=AUTH_HEADERS) as c:
        yield c


# ---------------------------------------------------------------------------
# Comparator Helper for Full 29-Field Assertions
# ---------------------------------------------------------------------------
def compare_response_against_golden(
    actual: Dict[str, Any],
    expected: Dict[str, Any],
    allow_timestamp_mismatch: bool = False,
) -> None:
    """
    Compares every one of the 29 TransactionScoreResponse fields against expected golden response.
    Raises AssertionError on any discrepancy with field name and values.
    """
    schema_fields = list(TransactionScoreResponse.model_fields.keys())
    assert len(schema_fields) == 29, f"Expected 29 schema fields, found {len(schema_fields)}"

    # Check for missing fields
    for field in schema_fields:
        assert field in actual, f"Regression: actual response missing schema field '{field}'"
        assert field in expected, f"Golden fixture missing schema field '{field}'"

    # 1. Deterministic Categorical & Structural Fields (Exact Equality)
    exact_match_fields = [
        "order_id",
        "risk_label",
        "model_used",
        "model_version",
        "graph_features_source",
        "client_mismatch",
        "note",
        "decision",
        "conformal_is_calibrated",
        "cold_start",
        "scoring_mode",
        "is_simulation",
    ]
    for field in exact_match_fields:
        act_val = actual[field]
        exp_val = expected[field]
        assert act_val == exp_val, (
            f"Regression in categorical field '{field}': expected {exp_val!r}, got {act_val!r}"
        )

    # 2. Conformal Prediction Set (Exact List Equality)
    assert actual["conformal_prediction_set"] == expected["conformal_prediction_set"], (
        f"Regression in 'conformal_prediction_set': expected {expected['conformal_prediction_set']}, "
        f"got {actual['conformal_prediction_set']}"
    )

    # 3. Reason Codes (Exact Set & Order Equality)
    assert actual["reason_codes"] == expected["reason_codes"], (
        f"Regression in 'reason_codes': expected {expected['reason_codes']}, got {actual['reason_codes']}"
    )

    # 4. Evidence Availability & Infrastructure Sources (Exact Dict Equality)
    assert actual["evidence_availability"] == expected["evidence_availability"], (
        f"Regression in 'evidence_availability': expected {expected['evidence_availability']}, "
        f"got {actual['evidence_availability']}"
    )
    assert actual["infrastructure_sources"] == expected["infrastructure_sources"], (
        f"Regression in 'infrastructure_sources': expected {expected['infrastructure_sources']}, "
        f"got {actual['infrastructure_sources']}"
    )

    # 5. Numeric Fields with Documented Tolerances
    float_fields = [
        ("overall_fraud_probability", PROBABILITY_TOLERANCE),
        ("raw_fraud_probability", PROBABILITY_TOLERANCE),
        ("calibrated_fraud_probability", PROBABILITY_TOLERANCE),
        ("policy_risk_score", PROBABILITY_TOLERANCE),
        ("canonical_trust_engine_risk", PROBABILITY_TOLERANCE),
        ("conformal_coverage_guarantee", PROBABILITY_TOLERANCE),
        ("conformal_q_hat", PROBABILITY_TOLERANCE),
        ("trust_score", TRUST_SCORE_TOLERANCE),
        ("confidence", PROBABILITY_TOLERANCE),
        ("predictive_uncertainty", PROBABILITY_TOLERANCE),
        ("model_disagreement", PROBABILITY_TOLERANCE),
    ]
    for field, tol in float_fields:
        act_val = actual.get(field)
        exp_val = expected.get(field)
        if exp_val is None:
            assert act_val is None, f"Expected None for '{field}', got {act_val}"
            continue
        assert act_val is not None, f"Expected numeric value for '{field}', got None"
        assert math.isclose(act_val, exp_val, abs_tol=tol), (
            f"Regression in numeric field '{field}': expected {exp_val:.6f} +/- {tol}, got {act_val:.6f}"
        )

    # 6. SHAP Attributions (Key matching & Float tolerance)
    act_shap = actual.get("shap_attributions")
    exp_shap = expected.get("shap_attributions")
    if exp_shap is not None:
        assert isinstance(act_shap, dict), "shap_attributions must be a dictionary"
        assert sorted(act_shap.keys()) == sorted(exp_shap.keys()), (
            f"SHAP feature set mismatch: expected {list(exp_shap.keys())}, got {list(act_shap.keys())}"
        )
        for k in exp_shap:
            assert math.isclose(act_shap[k], exp_shap[k], abs_tol=SHAP_TOLERANCE), (
                f"Regression in SHAP value for '{k}': expected {exp_shap[k]:.4f}, got {act_shap[k]:.4f}"
            )

    # 7. Top Risk Drivers (List matching & Float tolerance)
    act_drivers = actual.get("top_risk_drivers")
    exp_drivers = expected.get("top_risk_drivers")
    if exp_drivers is not None:
        assert isinstance(act_drivers, list), "top_risk_drivers must be a list"
        assert len(act_drivers) == len(exp_drivers), (
            f"Top risk drivers count mismatch: expected {len(exp_drivers)}, got {len(act_drivers)}"
        )
        for act_d, exp_d in zip(act_drivers, exp_drivers):
            assert act_d["feature_name"] == exp_d["feature_name"]
            assert math.isclose(act_d["shap_value"], exp_d["shap_value"], abs_tol=SHAP_TOLERANCE)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
class TestProductionBaselineIntegrity:
    """Validates schema, artifacts, and feature order against frozen metadata."""

    def test_schema_field_count_is_29(self):
        """Confirm actual TransactionScoreResponse field count is exactly 29."""
        schema_fields = list(TransactionScoreResponse.model_fields.keys())
        assert len(schema_fields) == 29, f"Expected 29 schema fields, found {len(schema_fields)}"

    def test_artifact_hashes_match_baseline(self, golden_data):
        """Model artifacts on disk must match recorded SHA-256 hashes bit-for-bit."""
        recorded_hashes = golden_data["metadata"]["model_artifact_hashes"]
        for fname, expected_hash in recorded_hashes.items():
            fpath = REPO_ROOT / "models" / fname
            assert fpath.is_file(), f"Model artifact file missing: {fpath}"
            computed_hash = _compute_sha256(fpath)
            assert computed_hash == expected_hash, (
                f"Artifact hash mismatch for {fname}: expected {expected_hash}, got {computed_hash}"
            )

    def test_feature_columns_and_order_match_baseline(self, golden_data):
        """Feature names and exact order must match the 18 frozen baseline features."""
        meta_path = REPO_ROOT / "models" / "feature_meta.joblib"
        feature_meta = joblib.load(meta_path)
        active_features = feature_meta["all_feature_cols"]
        recorded_features = golden_data["metadata"]["feature_order"]

        assert len(active_features) == 18
        assert active_features == recorded_features, (
            f"Feature column order mismatch:\nActive:   {active_features}\nBaseline: {recorded_features}"
        )


class TestProductionScoringRegression:
    """Executes live requests and asserts full 29-field parity against golden fixture."""

    @pytest.mark.parametrize("scenario_key", [
        "low_risk_standard",
        "high_risk_anomalous",
        "historical_point_in_time",
        "cold_start_new_entities",
    ])
    def test_scenario_regression_invariance(self, client, golden_data, scenario_key):
        """Each scenario must reproduce all 29 fields of the post-Phase-4.3B frozen baseline."""
        scenario = golden_data["scenarios"][scenario_key]
        req_payload = scenario["request_payload"]
        expected_response = scenario["expected_response"]

        # Reset any live history to guarantee clean state
        if getattr(store, "transaction_history", None) is not None:
            store.transaction_history.reset_live_history()

        resp = client.post("/transaction/score", json=req_payload)
        assert resp.status_code == 200, f"Scoring failed for '{scenario_key}': {resp.text}"

        actual_response = resp.json()
        compare_response_against_golden(actual_response, expected_response)


class TestRegressionSensitivityAndMutationDetection:
    """Proves that compare_response_against_golden detects deliberate mutations in representative fields."""

    @pytest.fixture
    def base_response(self, golden_data) -> Dict[str, Any]:
        return copy.deepcopy(golden_data["scenarios"]["low_risk_standard"]["expected_response"])

    def test_detects_deliberate_decision_change(self, base_response):
        mutated = copy.deepcopy(base_response)
        mutated["decision"] = "BLOCK" if base_response["decision"] != "BLOCK" else "ALLOW"
        with pytest.raises(AssertionError, match="Regression in categorical field 'decision'"):
            compare_response_against_golden(mutated, base_response)

    def test_detects_deliberate_probability_change(self, base_response):
        mutated = copy.deepcopy(base_response)
        mutated["overall_fraud_probability"] += 0.05
        with pytest.raises(AssertionError, match="Regression in numeric field 'overall_fraud_probability'"):
            compare_response_against_golden(mutated, base_response)

    def test_detects_deliberate_cold_start_change(self, base_response):
        mutated = copy.deepcopy(base_response)
        mutated["cold_start"] = not base_response["cold_start"]
        with pytest.raises(AssertionError, match="Regression in categorical field 'cold_start'"):
            compare_response_against_golden(mutated, base_response)

    def test_detects_deliberate_conformal_set_change(self, base_response):
        mutated = copy.deepcopy(base_response)
        mutated["conformal_prediction_set"] = [1] if base_response["conformal_prediction_set"] != [1] else [0]
        with pytest.raises(AssertionError, match="Regression in 'conformal_prediction_set'"):
            compare_response_against_golden(mutated, base_response)

    def test_detects_deliberate_reason_codes_change(self, base_response):
        mutated = copy.deepcopy(base_response)
        mutated["reason_codes"] = ["MUTATED_REASON_CODE"]
        with pytest.raises(AssertionError, match="Regression in 'reason_codes'"):
            compare_response_against_golden(mutated, base_response)

    def test_detects_missing_schema_field(self, base_response):
        mutated = copy.deepcopy(base_response)
        del mutated["model_version"]
        with pytest.raises(AssertionError, match="Regression: actual response missing schema field 'model_version'"):
            compare_response_against_golden(mutated, base_response)
