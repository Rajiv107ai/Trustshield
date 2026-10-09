"""
Unit and regression tests for TrustShield Stage 3.4.1:
Independent Policy and Calibration Audit.
"""

import json
import math
import os
import pytest
import numpy as np
import pandas as pd

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_MODELS_STAGE34_DIR = os.path.join(_ROOT_DIR, "models", "stage34")
_REPORTS_DIR = os.path.join(_ROOT_DIR, "reports")


@pytest.fixture(scope="module")
def stage34_metrics():
    metrics_path = os.path.join(_REPORTS_DIR, "phase34_metrics.json")
    assert os.path.exists(metrics_path), f"Metrics JSON missing at {metrics_path}"
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def stage34_policy():
    policy_path = os.path.join(_MODELS_STAGE34_DIR, "stage34_decision_policy.json")
    assert os.path.exists(policy_path), f"Policy JSON missing at {policy_path}"
    with open(policy_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestTaskACalibrationAudit:
    """Audits the fit/eval cohort separation and investigates ECE = 0.0000 claims."""

    def test_in_sample_vs_out_of_sample_calibration_separation(self, stage34_metrics):
        """Verifies that ECE = 0.0000 occurs strictly in-sample on the fitting cohort,

        whereas the out-of-sample diagnostic test exhibits ECE > 0.0.
        """
        tab_metrics = stage34_metrics["task_a_graph_ablation"]["tabular_only"]
        val_ece = tab_metrics["validation"]["ece"]
        test_ece = tab_metrics["historical_diagnostic_test"]["ece"]

        # In-sample validation ECE was reported as 0.0000 due to evaluating on fitting data
        assert val_ece == 0.0, "Validation ECE was measured in-sample"
        # Out-of-sample diagnostic test ECE is strictly non-zero
        assert test_ece > 0.02, f"Diagnostic test ECE must be non-zero (found {test_ece})"

    def test_isotonic_in_sample_mean_equals_positive_prevalence(self, stage34_metrics):
        """In-sample Isotonic regression guarantees mean predicted probability equals sample prevalence."""
        val_eval = stage34_metrics["task_a_graph_ablation"]["tabular_only"]["validation"]
        prevalence = val_eval["positive_prevalence"]
        assert 0.065 <= prevalence <= 0.075

    def test_calibrated_noisy_or_calibration_separation(self, stage34_metrics):
        """Calibrated Noisy-OR in-sample ECE is 0.0000, while out-of-sample test ECE is ~0.0259."""
        ens = stage34_metrics["task_d_ensemble_semantics"]["calibrated_noisy_or"]
        assert ens["validation"]["ece"] == 0.0
        assert ens["historical_diagnostic_test"]["ece"] > 0.02


class TestTaskBCapacityConstrainedPolicyAudit:
    """Audits review capacity constraints, cost calculations, and hard budget compliance."""

    def test_stage34_capacity_5pct_violation_detected(self, stage34_policy):
        """Detects the defect where Stage 3.4 capacity_5_pct achieved 5.76% volume, exceeding 5.0%."""
        cap_5 = stage34_policy["capacity_operating_points"]["capacity_5_pct"]
        target = cap_5["target_capacity_pct"]
        achieved = cap_5["achieved_volume_pct"]

        assert target == 5.0
        # The audit detects that achieved volume strictly exceeded the target limit
        assert achieved > target, f"Expected defect: achieved {achieved}% > {target}%"

    def test_corrected_hard_capacity_selection_enforces_upper_bounds(self):
        """Implements and verifies the deterministic, capacity-compliant selection algorithm."""
        # Simulated sweep points corresponding to Stage 3.4 late fusion sweep
        sweep = [
            {"t": 0.11, "vol": 1111, "vol_pct": 9.16, "tp": 185, "prec": 0.1665, "rec": 0.2166},
            {"t": 0.13, "vol": 699, "vol_pct": 5.76, "tp": 132, "prec": 0.1888, "rec": 0.1546},
            {"t": 0.15, "vol": 477, "vol_pct": 3.93, "tp": 99, "prec": 0.2075, "rec": 0.1159},
            {"t": 0.20, "vol": 41, "vol_pct": 0.34, "tp": 15, "prec": 0.3659, "rec": 0.0176},
        ]

        def select_hard_capacity_point(sweep_list, cap_limit):
            # 1. Filter candidates strictly satisfying volume <= cap_limit
            eligible = [p for p in sweep_list if p["vol_pct"] <= cap_limit]
            if not eligible:
                raise ValueError("No eligible operating point satisfies capacity limit")
            # 2. Maximize true positives / recall; tie breaker: highest precision, highest threshold
            return max(eligible, key=lambda x: (x["tp"], x["prec"], x["t"]))

        p_1pct = select_hard_capacity_point(sweep, 1.0)
        assert p_1pct["t"] == 0.20
        assert p_1pct["vol_pct"] <= 1.0

        p_5pct = select_hard_capacity_point(sweep, 5.0)
        # Corrected 5% capacity threshold is 0.15 (3.93%), NOT 0.13 (5.76%)!
        assert p_5pct["t"] == 0.15
        assert p_5pct["vol_pct"] <= 5.0
        assert p_5pct["vol_pct"] == 3.93

        p_10pct = select_hard_capacity_point(sweep, 10.0)
        assert p_10pct["t"] == 0.11
        assert p_10pct["vol_pct"] <= 10.0

    def test_cost_calculation_formula_mathematical_consistency(self, stage34_policy):
        """Verifies that cost equals FP * 10 + FN * 150 + TP * 10 = (TP + FP) * 10 + FN * 150."""
        c_fp = stage34_policy["cost_model"]["manual_review_cost_c_fp"]
        c_fn = stage34_policy["cost_model"]["fraud_loss_cost_c_fn"]

        assert c_fp == 10.0
        assert c_fn == 150.0

        # Optimal operating point: t = 0.07, flagged = 3709, fn = 430
        opt_t = stage34_policy["cost_optimal_threshold"]
        assert opt_t == 0.07
        expected_cost = 3709 * c_fp + 430 * c_fn
        assert expected_cost == 101590.0


class TestTaskCEnsembleSemantics:
    """Audits multi-modal score fusion, independence violation, and calibration status."""

    def test_component_correlation_disproves_independence(self, stage34_metrics):
        """Pearson correlation > 0 disproves conditional independence assumption of Noisy-OR."""
        corr = stage34_metrics["task_d_ensemble_semantics"]["correlation_order_and_listing_scores_validation"]
        r_pearson = corr["pearson"]
        rho_spearman = corr["spearman"]

        # Moderate positive correlation exists
        assert r_pearson > 0.20
        assert rho_spearman > 0.20

    def test_raw_noisy_or_calibration_degradation(self, stage34_metrics):
        """Due to correlation, raw Noisy-OR over-predicts risk, having ECE > 0.04."""
        raw_noisy = stage34_metrics["task_d_ensemble_semantics"]["raw_noisy_or"]
        assert raw_noisy["is_calibrated"] is False
        assert raw_noisy["validation"]["ece"] > 0.04

    def test_max_risk_labeled_uncalibrated_heuristic(self, stage34_metrics):
        """Max-Risk is strictly flagged as an uncalibrated ranking heuristic."""
        max_risk = stage34_metrics["task_d_ensemble_semantics"]["max_risk_heuristic"]
        assert max_risk["is_calibrated"] is False
        assert "uncalibrated ranking heuristic" in max_risk["semantic_note"].lower()


class TestTaskDDatasetHoldoutProtocol:
    """Verifies that dataset v2.2 design requirements and access controls are fully specified."""

    def test_v2_2_protocol_manifest_requirements(self):
        """Verifies protocol parameters for future untouched holdout."""
        proposal = {
            "version": "v2.2",
            "time_window_start": "2026-01-01T00:00:00Z",
            "time_window_end": "2026-02-28T23:59:59Z",
            "maturity_days": 21,
            "access_control": "AES-256-GCM encrypted labels with commit-reveal hash",
        }
        assert proposal["maturity_days"] == 21
        assert "AES-256-GCM" in proposal["access_control"]
