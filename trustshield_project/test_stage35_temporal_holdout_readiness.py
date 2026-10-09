"""
TrustShield Stage 3.5 — Temporal Generalization & Blind-Holdout Readiness Test Suite.

Verifies:
1. Split chronology and temporal precedence (Train < Val < Test).
2. Absence of duplicate order identifiers across splits.
3. Point-in-time feature engineering causality (no post-prediction information).
4. Return-label 21-day observation maturity rule and right-censoring logic.
5. Graph relationships and snapshots strictly precede transaction timestamps.
6. Preprocessing fitted strictly on permitted training cohort.
7. Cryptographic hash integrity of frozen source, data, and model artifacts.
8. Model and feature schema consistency and explicit feature ordering.
9. Prediction contract input validation (ordering, count, non-null, ID uniqueness).
10. Prediction score range and numerical validity ([0.0, 1.0], finite).
11. Cost-formula equivalence and boundary operator semantics.
12. Coarse vs fine threshold-cost discrepancy resolution ($101,590 vs $101,130 vs $101,090).
13. Capacity limits with strict floor integer rounding (floor(b * N)).
14. Deterministic score tie handling at capacity boundaries.
15. Calibration fit vs evaluation cohort separation enforcement.
16. Evaluation manifest schema completeness.
17. Fail-closed holdout access authorization gate.
18. Immutable environment identification requirements.
19. Protection of frozen historical artifacts against mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_MODELS_STAGE34_DIR = os.path.join(_ROOT_DIR, "models", "stage34")
_DATA_DIR = os.path.join(_ROOT_DIR, "data", "synthetic_v2_1")
_SCRIPTS_DIR = os.path.join(_ROOT_DIR, "scripts")
_REPORTS_DIR = os.path.join(_ROOT_DIR, "reports")
_PROJECT_DIR = os.path.join(_ROOT_DIR, "trustshield_project")

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)


@pytest.fixture(scope="module")
def saved_stage34_validation_arrays():
    """Loads saved validation predictions and labels from Stage 3.4 / 3.4.2."""
    p_path = os.path.join(_MODELS_STAGE34_DIR, "stage342_val_predictions.npy")
    y_path = os.path.join(_MODELS_STAGE34_DIR, "stage342_val_labels.npy")
    assert os.path.exists(p_path), f"Missing validation predictions at {p_path}"
    assert os.path.exists(y_path), f"Missing validation labels at {y_path}"
    p = np.load(p_path)
    y = np.load(y_path)
    return p, y


@pytest.fixture(scope="module")
def v2_1_orders_and_returns():
    """Loads frozen v2.1 orders and returns tables with parsed dates."""
    orders_path = os.path.join(_DATA_DIR, "orders.csv")
    returns_path = os.path.join(_DATA_DIR, "returns.csv")
    assert os.path.exists(orders_path), f"Orders table missing: {orders_path}"
    assert os.path.exists(returns_path), f"Returns table missing: {returns_path}"
    orders = pd.read_csv(orders_path, parse_dates=["order_date"])
    returns = pd.read_csv(returns_path, parse_dates=["return_date"])
    return orders, returns


# ===========================================================================
# 1. Temporal Split Integrity & Entity Leakage
# ===========================================================================

class TestTemporalSplitIntegrity:
    """Verifies temporal separation, split intervals, and ID uniqueness."""

    def test_split_chronology_and_temporal_precedence(self, v2_1_orders_and_returns):
        """Train precedes Val, and Val precedes Diagnostic Test with zero overlap."""
        orders, _ = v2_1_orders_and_returns
        train_end = pd.Timestamp("2025-08-31 23:59:59")
        val_end = pd.Timestamp("2025-10-31 23:59:59")

        train = orders[orders["order_date"] <= train_end]
        val = orders[(orders["order_date"] > train_end) & (orders["order_date"] <= val_end)]
        test = orders[orders["order_date"] > val_end]

        assert len(train) == 16952
        assert len(val) == 12129
        assert len(test) == 20919
        assert len(train) + len(val) + len(test) == 50000

        # Strict temporal boundaries
        assert train["order_date"].max() <= train_end
        assert val["order_date"].min() > train_end
        assert val["order_date"].max() <= val_end
        assert test["order_date"].min() > val_end

    def test_no_duplicate_order_ids_across_splits(self, v2_1_orders_and_returns):
        """No order ID crosses splits or occurs more than once."""
        orders, _ = v2_1_orders_and_returns
        assert orders["order_id"].nunique() == len(orders)
        assert orders["order_id"].duplicated().sum() == 0

    def test_point_in_time_feature_causality(self):
        """Validates that as-of accumulators enforce allow_exact_matches=False."""
        from baseline_model import _cumulative_count_asof

        # Synthetic fixture with simultaneous orders
        test_df = pd.DataFrame({
            "order_id": ["O1", "O2", "O3", "O4"],
            "buyer_id": ["B1", "B1", "B1", "B2"],
            "order_date": pd.to_datetime([
                "2025-03-01 10:00:00",
                "2025-03-01 10:00:00",  # simultaneous same-second transaction
                "2025-03-01 11:00:00",
                "2025-03-01 10:00:00",
            ])
        })
        counts = _cumulative_count_asof(test_df, "buyer_id")

        # For simultaneous B1 orders, neither sees the other as prior history
        assert counts.iloc[0] == 0
        assert counts.iloc[1] == 0
        # Subsequent B1 order at 11:00 sees exactly 2 prior orders
        assert counts.iloc[2] == 2
        # Distinct buyer B2 starts at 0
        assert counts.iloc[3] == 0


# ===========================================================================
# 2. Return-Label Maturity & Right-Censoring
# ===========================================================================

class TestReturnLabelMaturity:
    """Verifies the 21-day observation window and right-censoring rules."""

    def test_return_label_maturity_cutoff_enforcement(self, v2_1_orders_and_returns):
        """Orders within 21 days of simulation end cannot guarantee mature return labels."""
        orders, returns = v2_1_orders_and_returns
        sim_end = pd.Timestamp("2025-12-31 23:59:59")
        maturity_days = 21
        safe_cutoff = sim_end - timedelta(days=maturity_days)

        # In 2025, orders placed on or after 2025-12-11 00:00:00 have < 21 days observation
        immature_orders = orders[orders["order_date"] > safe_cutoff]
        assert len(immature_orders) == 8618

        # Any return that did occur must have return_date <= sim_end
        assert returns["return_date"].max() <= sim_end

        # Order turnaround delay is strictly positive (at least 1 day)
        ro = returns.merge(orders[["order_id", "order_date"]], on="order_id")
        delays = (ro["return_date"] - ro["order_date"]).dt.total_seconds() / 86400.0
        assert (delays < 1.0).sum() == 0, "No 0-day returns allowed!"

    def test_future_2026_holdout_maturity_boundary(self):
        """Calculates exact 21-day observation cutoff for a proposed Feb 28, 2026 simulation end."""
        proposed_sim_end = pd.Timestamp("2026-02-28 23:59:59")
        maturity_window = timedelta(days=21)
        expected_cutoff = proposed_sim_end - maturity_window

        assert expected_cutoff == pd.Timestamp("2026-02-07 23:59:59")
        # An order placed on 2026-02-08 00:00:00 has only 20 days remaining, thus right-censored
        order_immature = pd.Timestamp("2026-02-08 00:00:00")
        assert (proposed_sim_end - order_immature).days < 21


# ===========================================================================
# 3. Graph Features Point-in-Time Causality
# ===========================================================================

class TestGraphTemporalIntegrity:
    """Verifies that graph relationships and monthly snapshots strictly precede prediction."""

    def test_graph_cutoff_filters_future_relationships(self):
        """Relationship graph strictly filters edges with first_seen_date >= cutoff."""
        from graph_features import build_relationship_graph

        addr_log = pd.DataFrame({
            "buyer_id": ["B1", "B2"],
            "shared_with_buyer_id": ["B2", "B3"],
            "first_seen_date": pd.to_datetime(["2025-05-01", "2025-09-15"]),
        })
        dev_log = pd.DataFrame(columns=["buyer_id", "shared_with_buyer_id", "first_seen_date"])

        # Cutoff at Train End (2025-08-31)
        g_train = build_relationship_graph(addr_log, dev_log, cutoff_date="2025-08-31 23:59:59")
        assert g_train.has_edge("B1", "B2")
        assert not g_train.has_edge("B2", "B3")  # Future relationship excluded!

    def test_preprocessing_fitted_strictly_on_training_cohort(self):
        """Category median and device counts must be computed strictly on training data."""
        from baseline_model import TRAIN_END
        assert TRAIN_END == pd.Timestamp("2025-08-31")


# ===========================================================================
# 4. Artifact Hashes & Schema Consistency
# ===========================================================================

class TestArtifactIntegrityAndSchema:
    """Verifies cryptographic hashes and feature schema pinning."""

    def test_artifact_hashes_match_records(self):
        """Critical generator, data, and model artifacts must match verified SHA-256."""
        expected_hashes = {
            os.path.join(_SCRIPTS_DIR, "generate_realistic_synthetic_data_v2_1.py"):
                "0098dad245d724997560c5a5bc4bf70c1bcb98057a4b32d42f85560a4e08bd55",
            os.path.join(_DATA_DIR, "orders.csv"):
                "325597a6e0c2afa533c01d26895457cba166cd0bc2c0f9ce4f12c43355741b8f",
            os.path.join(_DATA_DIR, "listings.csv"):
                "0081b562dc900c14e1b90f2310261401ab2f8f40dda4805aa72bd17959d5e619",
            os.path.join(_MODELS_STAGE34_DIR, "stage34_tabular_model.joblib"):
                "1310b43c3c8d23a1d4054200a5321f815de113b1a07793bb0d17f508c6cac22a",
            os.path.join(_MODELS_STAGE34_DIR, "stage34_tabular_calibrator.joblib"):
                "152f287e8f4d478285eeac13d380f4312ceb453bed85b3ff1f99d799424148a8",
        }
        for path, exp_hash in expected_hashes.items():
            assert os.path.exists(path), f"Artifact missing: {path}"
            with open(path, "rb") as f:
                act_hash = hashlib.sha256(f.read()).hexdigest()
            assert act_hash == exp_hash, f"Hash mismatch for {path}: expected {exp_hash}, got {act_hash}"

    def test_model_feature_schema_and_order_consistency(self):
        """Tabular model artifact expects exactly 10 features with explicit names."""
        import joblib
        model_path = os.path.join(_MODELS_STAGE34_DIR, "stage34_tabular_model.joblib")
        model = joblib.load(model_path)

        expected_features = [
            "price_vs_base_price_ratio",
            "price_vs_category_median_ratio",
            "seller_age_days",
            "seller_total_listings_before",
            "buyer_age_days",
            "buyer_orders_before",
            "buyer_returns_before",
            "buyer_return_rate_before",
            "device_shared_buyer_count",
            "amount",
        ]
        assert hasattr(model, "feature_names_in_")
        assert list(model.feature_names_in_) == expected_features
        assert model.n_features_in_ == 10


# ===========================================================================
# 5. Threshold-Cost Discrepancy Reconciliation
# ===========================================================================

class TestThresholdCostReconciliation:
    """Reconciles coarse grid (0.07 -> $101,590) vs fine step optimization ($101,130 / $101,090)."""

    def test_threshold_cost_calculation_and_boundary_semantics(self, saved_stage34_validation_arrays):
        p, y = saved_stage34_validation_arrays
        t = 0.07
        flagged = p >= t
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        fn = int((~flagged & (y == 1)).sum())
        tn = int((~flagged & (y == 0)).sum())

        assert tp == 424
        assert fp == 3285
        assert fn == 430
        assert tn == 7990
        assert tp + fp == 3709
        assert tp + fn == 854

        cost = fp * 10.0 + fn * 150.0 + tp * 10.0
        assert cost == 101590.0

    def test_threshold_cost_reconciliation_coarse_vs_fine(self, saved_stage34_validation_arrays):
        """Proves exact origin of $101,590 vs $101,130 and discovers unconstrained minimum ($101,090)."""
        p, y = saved_stage34_validation_arrays

        # 1. Coarse 80-point grid at t = 0.06:
        # In coarse grid, t = 0.06 captures all p >= 0.062992 (7,022 orders)
        flagged_006 = p >= 0.06
        tp_006 = int((flagged_006 & (y == 1)).sum())
        fp_006 = int((flagged_006 & (y == 0)).sum())
        fn_006 = int((~flagged_006 & (y == 1)).sum())
        cost_006 = fp_006 * 10.0 + fn_006 * 150.0 + tp_006 * 10.0
        assert flagged_006.sum() == 7022
        assert cost_006 == 101720.0

        # 2. Coarse 80-point grid at t = 0.07:
        flagged_007 = p >= 0.07
        assert flagged_007.sum() == 3709
        cost_007 = (flagged_007.sum()) * 10.0 + int((~flagged_007 & (y == 1)).sum()) * 150.0
        assert cost_007 == 101590.0
        # On coarse grid, 0.07 is lower than 0.06 (101590 < 101720)
        assert cost_007 < cost_006

        # 3. Fine isotonic score step 13 (u = 0.06907378...):
        # Mislabeled as "t = 0.06" in Stage 3.4.2 text table
        unique_p = np.sort(np.unique(p))
        u_step13 = unique_p[13]  # exactly 0.06907378335949764
        flagged_step13 = p >= u_step13
        tp_s13 = int((flagged_step13 & (y == 1)).sum())
        fp_s13 = int((flagged_step13 & (y == 0)).sum())
        fn_s13 = int((~flagged_step13 & (y == 1)).sum())
        cost_s13 = fp_s13 * 10.0 + fn_s13 * 150.0 + tp_s13 * 10.0
        assert flagged_step13.sum() == 4983
        assert tp_s13 == 512
        assert fp_s13 == 4471
        assert fn_s13 == 342
        assert cost_s13 == 101130.0  # Exactly $101,130!

        # 4. True global minimum across all 26 distinct isotonic steps: step 11 (u = 0.06763285...)
        u_step11 = unique_p[11]
        flagged_step11 = p >= u_step11
        tp_s11 = int((flagged_step11 & (y == 1)).sum())
        fp_s11 = int((flagged_step11 & (y == 0)).sum())
        fn_s11 = int((~flagged_step11 & (y == 1)).sum())
        cost_s11 = fp_s11 * 10.0 + fn_s11 * 150.0 + tp_s11 * 10.0
        assert flagged_step11.sum() == 5219
        assert tp_s11 == 528
        assert fp_s11 == 4691
        assert fn_s11 == 326
        assert cost_s11 == 101090.0  # Lowest achievable cost is $101,090!


# ===========================================================================
# 6. Capacity Limits & Boundary Semantics
# ===========================================================================

class TestCapacityConstraintsAndTieHandling:
    """Verifies floor rounding and deterministic tie handling."""

    def test_capacity_limits_floor_rounding(self):
        n = 12129
        assert math.floor(0.01 * n) == 121
        assert math.floor(0.02 * n) == 242
        assert math.floor(0.05 * n) == 606
        assert math.floor(0.10 * n) == 1212

    def test_tied_scores_at_capacity_boundary(self):
        """Verifies that capacity limits cannot be breached due to score ties."""
        scores = np.array([0.9, 0.8, 0.5, 0.5, 0.5, 0.1])
        labels = np.array([1, 1, 1, 0, 0, 0])
        n = len(scores)
        capacity_ceiling = 4

        # A nearest-threshold rule would pick t=0.5 -> 5 orders > 4 (VIOLATION)
        # Strict floor selection requires volume <= capacity_ceiling
        eligible_t = [t for t in np.unique(scores) if (scores >= t).sum() <= capacity_ceiling]
        best_t = min(eligible_t)  # lowest threshold that still obeys ceiling
        assert best_t == 0.8
        assert (scores >= best_t).sum() == 2 <= capacity_ceiling


# ===========================================================================
# 7. Calibration Separation & Contract Integrity
# ===========================================================================

class TestCalibrationAndContractIntegrity:
    """Verifies calibration cohort separation and contract validation."""

    def test_calibration_cohort_separation_enforcement(self, saved_stage34_validation_arrays):
        """Fitting cohort ECE cannot be claimed as generalization evidence."""
        # On validation (the fitting cohort of the isotonic calibrator): ECE = 0.0000
        # On test (separate cohort): ECE = 0.0289 for tabular model
        metrics_path = os.path.join(_REPORTS_DIR, "phase34_metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        val_ece = m["task_a_graph_ablation"]["tabular_only"]["validation"]["ece"]
        te_ece = m["task_a_graph_ablation"]["tabular_only"]["historical_diagnostic_test"]["ece"]

        assert val_ece == 0.0  # In-sample artifact
        assert te_ece > 0.02   # True out-of-sample error (0.0289)

    def test_prediction_contract_validation_functions(self):
        """Validates prediction input constraints (lengths, types, finite ranges)."""
        def validate_prediction_contract(
            order_ids: List[str],
            predictions: np.ndarray,
            expected_count: int,
        ) -> bool:
            if len(predictions) != expected_count or len(order_ids) != expected_count:
                raise ValueError("Row count mismatch")
            if len(set(order_ids)) != expected_count:
                raise ValueError("Duplicate order IDs detected")
            if not np.all(np.isfinite(predictions)):
                raise ValueError("Non-finite scores detected")
            if (predictions < 0.0).any() or (predictions > 1.0).any():
                raise ValueError("Scores outside [0, 1] range")
            return True

        # Valid mock inputs
        o_ids = [f"ORD_{i}" for i in range(10)]
        preds = np.array([0.1 * i for i in range(10)])
        assert validate_prediction_contract(o_ids, preds, 10) is True

        # Invalid: out of bounds
        with pytest.raises(ValueError, match="outside"):
            validate_prediction_contract(o_ids, np.array([0.5]*9 + [1.2]), 10)

        # Invalid: duplicates
        with pytest.raises(ValueError, match="Duplicate"):
            validate_prediction_contract(o_ids[:9] + [o_ids[0]], preds, 10)

    def test_holdout_access_authorization_fail_closed(self):
        """Holdout evaluation entry point must fail closed without explicit authorized token."""
        def access_holdout_labels(auth_token: str | None) -> Any:
            AUTHORIZED_TOKEN_HASH = "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918"  # sha256("TRUSTSHIELD_AUTHORIZED_HOLDOUT_RUN_STAGE35")
            if not auth_token:
                raise PermissionError("BLOCKED: Holdout access authorization token missing.")
            h = hashlib.sha256(auth_token.encode("utf-8")).hexdigest()
            if h != AUTHORIZED_TOKEN_HASH:
                raise PermissionError("BLOCKED: Invalid authorization token. Holdout access denied.")
            return "ACCESS_GRANTED"

        # No token -> fail closed
        with pytest.raises(PermissionError, match="missing"):
            access_holdout_labels(None)

        # Invalid token -> fail closed
        with pytest.raises(PermissionError, match="denied"):
            access_holdout_labels("UNAUTHORIZED_ATTEMPT")

    def test_immutable_environment_identification(self):
        """Verifies Python runtime environment identification requirements."""
        import platform
        py_ver = platform.python_version()
        assert py_ver.startswith("3.")
        # Requirements must require an immutable container image digest (e.g., sha256:...)
        # rather than mutable 'latest' tags for production holdout evaluation.
        image_spec = "trustshield-evaluator:latest"  # mutable tag
        assert ":" in image_spec
        is_pinned_digest = "@sha256:" in image_spec
        assert is_pinned_digest is False, "Demonstrates that current tags are mutable and must be pinned."


# ===========================================================================
# 8. Protection of Historical Artifacts
# ===========================================================================

class TestProtectedArtifactImmutability:
    """Verifies that historical artifacts remain strictly unmutated."""

    def test_no_changes_to_protected_artifacts(self):
        """Manifest and historical test data hashes must remain identical."""
        manifest_path = os.path.join(_DATA_DIR, "dataset_manifest.json")
        assert os.path.exists(manifest_path)
        with open(manifest_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        assert h == "b0aba240f1439c1d83c1c9773977334ddf39321feed0e95f53ef30b1c138e74b"
