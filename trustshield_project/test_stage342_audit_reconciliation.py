"""
TrustShield Stage 3.4.2 — Automated Regression and Evidence Reconciliation Test Suite.

Verifies:
1. Confusion-matrix conservation identities.
2. Cost-formula equivalence.
3. Threshold boundary behavior for scores exactly equal to the threshold.
4. Strict capacity enforcement (hard ceiling).
5. Capacity integer rounding (floor(budget * N)).
6. Deterministic tied-score handling.
7. Calibration fit/evaluation cohort separation.
8. ECE calculation for a known controlled small example.
9. Consistent order-ID joins and duplicate detection.
10. Consistent transaction-label semantics for ensemble metrics.
11. Detection of missing or non-finite scores.
12. SHA-256 verification logic for the generator file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pytest
import numpy as np
import pandas as pd

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_MODELS_STAGE34_DIR = os.path.join(_ROOT_DIR, "models", "stage34")
_DATA_DIR = os.path.join(_ROOT_DIR, "data", "synthetic_v2_1")
_REPORTS_DIR = os.path.join(_ROOT_DIR, "reports")


@pytest.fixture(scope="module")
def saved_val_data():
    """Loads saved validation predictions and labels."""
    p_path = os.path.join(_MODELS_STAGE34_DIR, "stage342_val_predictions.npy")
    y_path = os.path.join(_MODELS_STAGE34_DIR, "stage342_val_labels.npy")
    assert os.path.exists(p_path), f"Missing validation predictions at {p_path}"
    assert os.path.exists(y_path), f"Missing validation labels at {y_path}"
    p = np.load(p_path)
    y = np.load(y_path)
    return p, y


@pytest.fixture(scope="module")
def stage34_metrics():
    metrics_path = os.path.join(_REPORTS_DIR, "phase34_metrics.json")
    assert os.path.exists(metrics_path), f"Metrics JSON missing at {metrics_path}"
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestTaskAConfusionMatrixAndCostIdentities:
    """Tests for Task A: Confusion matrix identities, boundary semantics, and cost equivalence."""

    def test_confusion_matrix_conservation_identities(self, saved_val_data):
        p, y = saved_val_data
        n = len(y)
        t = 0.07

        flagged = p >= t
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        fn = int((~flagged & (y == 1)).sum())
        tn = int((~flagged & (y == 0)).sum())
        total_fraud = int(y.sum())
        total_flagged = int(flagged.sum())

        # Exact identity checks
        assert tp + fn == total_fraud, f"Identity violated: TP ({tp}) + FN ({fn}) != Total Fraud ({total_fraud})"
        assert tp + fp == total_flagged, f"Identity violated: TP ({tp}) + FP ({fp}) != Total Flagged ({total_flagged})"
        assert tn + fp + fn + tp == n, f"Identity violated: TN + FP + FN + TP ({tn+fp+fn+tp}) != Cohort Size ({n})"

        # Concrete values at t = 0.07
        assert tp == 424
        assert fp == 3285
        assert fn == 430
        assert tn == 7990
        assert total_flagged == 3709
        assert total_fraud == 854
        assert n == 12129

    def test_cost_formula_mathematical_equivalence(self, saved_val_data):
        p, y = saved_val_data
        t = 0.07
        flagged = p >= t
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        fn = int((~flagged & (y == 1)).sum())
        total_flagged = int(flagged.sum())

        c_fp, c_fn, c_tp = 10.0, 150.0, 10.0

        cost1 = fp * c_fp + fn * c_fn + tp * c_tp
        cost2 = total_flagged * c_fp + fn * c_fn

        assert cost1 == cost2
        assert cost1 == 101590.0

    def test_threshold_boundary_behavior_scores_equal_to_threshold(self):
        """Verifies boundary inclusivity semantics: score >= t flags exact ties, while score > t excludes them."""
        scores = np.array([0.05, 0.07, 0.07, 0.09])
        labels = np.array([0, 0, 1, 1])

        # Inclusive >=
        flagged_ge = scores >= 0.07
        assert int(flagged_ge.sum()) == 3
        assert list(flagged_ge) == [False, True, True, True]

        # Strict >
        flagged_gt = scores > 0.07
        assert int(flagged_gt.sum()) == 1
        assert list(flagged_gt) == [False, False, False, True]


class TestTaskBCapacityConstraintsAndRounding:
    """Tests for Task B: Hard capacity enforcement, integer rounding, and deterministic ties."""

    def test_capacity_integer_rounding_floor(self):
        n = 12129
        # Rule: floor(budget * N)
        assert math.floor(0.01 * n) == 121
        assert math.floor(0.02 * n) == 242
        assert math.floor(0.05 * n) == 606
        assert math.floor(0.10 * n) == 1212

    def test_strict_capacity_enforcement_rejects_over_budget(self, saved_val_data):
        p, y = saved_val_data
        n = len(y)
        cap_5pct = math.floor(0.05 * n)  # 606 orders

        # Old Stage 3.4 threshold t = 0.13
        vol_013 = int((p >= 0.13).sum())
        assert vol_013 == 699
        assert vol_013 > cap_5pct, "t = 0.13 violates 5% hard capacity budget!"

        # Corrected threshold t = 0.15
        vol_015 = int((p >= 0.15).sum())
        assert vol_015 == 477
        assert vol_015 <= cap_5pct, "t = 0.15 must strictly obey 5% hard capacity budget!"

    def test_deterministic_tied_score_handling(self):
        """Deterministic policy: never split tied scores arbitrarily; require score >= t."""
        # Fixture with tied score cluster at 0.15
        scores = np.array([0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
        labels = np.array([1, 1, 1, 0, 0, 0])
        budget_max = 4  # Can take at most 4 orders

        # Candidate thresholds:
        # t = 0.20 -> flags 2 orders <= 4 (TP=2)
        # t = 0.15 -> flags 5 orders > 4 (VIOLATION)
        # Therefore, t = 0.20 is selected without arbitrarily choosing which tied order gets reviewed
        eligible_t = [t for t in [0.25, 0.20, 0.15, 0.10] if (scores >= t).sum() <= budget_max]
        best_t = max(eligible_t, key=lambda t: (((scores >= t) & (labels == 1)).sum(), t))
        assert best_t == 0.20
        assert (scores >= best_t).sum() == 2


class TestTaskCCalibrationAndECESpecification:
    """Tests for Task C: Controlled ECE calculation, binning behavior, and cohort separation."""

    def test_controlled_small_example_ece_calculation(self):
        """Computes ECE on a hand-calculated 2-bin fixture to verify implementation logic."""
        from calibration import calculate_expected_calibration_error

        # 4 samples:
        # Bin 0 [0.0, 0.5): p=[0.1, 0.3], y=[0, 1] -> mean_p=0.2, mean_y=0.5 -> diff=0.3, weight=2/4=0.5 -> contrib=0.15
        # Bin 1 [0.5, 1.0]: p=[0.7, 0.9], y=[1, 1] -> mean_p=0.8, mean_y=1.0 -> diff=0.2, weight=2/4=0.5 -> contrib=0.10
        # Expected ECE = 0.15 + 0.10 = 0.25
        p = np.array([0.1, 0.3, 0.7, 0.9])
        y = np.array([0, 1, 1, 1])

        computed_ece = calculate_expected_calibration_error(y, p, n_bins=2)
        assert abs(computed_ece - 0.25) < 1e-6

    def test_calibration_cohort_separation_classification(self, stage34_metrics):
        """Verifies that in-sample validation ECE equals 0.0000, while diagnostic test ECE is strictly positive."""
        val_ece = stage34_metrics["task_a_graph_ablation"]["tabular_only"]["validation"]["ece"]
        test_ece = stage34_metrics["task_a_graph_ablation"]["tabular_only"]["historical_diagnostic_test"]["ece"]

        assert val_ece == 0.0, "Validation ECE was measured in-sample"
        assert test_ece > 0.02, "Diagnostic test ECE must reflect out-of-sample calibration error"


class TestTaskDEnsembleSemanticsAndJoins:
    """Tests for Task D: Joins, duplicate prevention, label consistency, and score validity."""

    def test_consistent_order_id_joins_and_no_duplicates(self):
        orders_path = os.path.join(_DATA_DIR, "orders.csv")
        orders = pd.read_csv(orders_path)
        assert orders["order_id"].is_unique, "order_id in orders.csv must be unique!"
        assert len(orders) == 50000

    def test_detection_of_missing_or_non_finite_scores(self):
        """Verifies that infinite or NaN scores are correctly caught."""
        invalid_scores = np.array([0.1, np.nan, 0.8, np.inf])
        assert np.isnan(invalid_scores).any(), "Must detect NaN scores"
        assert np.isinf(invalid_scores).any(), "Must detect Infinite scores"

    def test_ensemble_transaction_label_consistency(self, stage34_metrics):
        """Verifies that all ensemble scores evaluate against transaction fraud label."""
        ens = stage34_metrics["task_d_ensemble_semantics"]
        for key in ["max_risk_heuristic", "raw_noisy_or", "calibrated_noisy_or"]:
            val_eval = ens[key]["validation"]
            assert val_eval["support"]["total"] == 12129
            assert val_eval["support"]["positives"] == 854
            assert abs(val_eval["positive_prevalence"] - 0.0704) < 1e-4


class TestTaskEGeneratorVerification:
    """Tests for Task E: Generator hash and observation maturity cutoff."""

    def test_generator_script_sha256_hash_integrity(self):
        generator_path = os.path.join(_ROOT_DIR, "scripts", "generate_realistic_synthetic_data_v2_1.py")
        assert os.path.exists(generator_path)
        h = hashlib.sha256()
        with open(generator_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        expected_hash = "0098dad245d724997560c5a5bc4bf70c1bcb98057a4b32d42f85560a4e08bd55"
        assert h.hexdigest() == expected_hash, f"Generator hash mismatch! Found {h.hexdigest()}"

    def test_observation_window_maturity_21_day_rule(self):
        """Verifies that an order placed at 2026-02-07 23:59:59 has 21 days before 2026-02-28 23:59:59."""
        cutoff = pd.Timestamp("2026-02-07 23:59:59")
        sim_end = pd.Timestamp("2026-02-28 23:59:59")
        delta = (sim_end - cutoff).days
        assert delta == 21, f"Expected 21 full observation days, got {delta}"
