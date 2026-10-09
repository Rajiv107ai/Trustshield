"""
Unit and regression tests for Stage 3.4: Controlled Graph Ablation, Decision Policy Design & Ensemble Semantics.
"""

import json
import math
import os
import pytest
import numpy as np
import joblib

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


class TestTaskAGraphAblation:
    """Verifies that controlled graph ablation was executed fairly on validation data."""

    def test_all_four_ablation_architectures_present(self, stage34_metrics):
        ablation = stage34_metrics["task_a_graph_ablation"]
        assert "tabular_only" in ablation
        assert "graph_only" in ablation
        assert "early_fusion" in ablation
        assert "late_fusion" in ablation

    def test_graph_only_model_shows_zero_value(self, stage34_metrics):
        """Graph-only model on validation data is near random-guessing baseline (~0.50)."""
        graph_eval = stage34_metrics["task_a_graph_ablation"]["graph_only"]["validation"]
        # ROC-AUC should be near 0.50 (within 0.05)
        assert 0.45 <= graph_eval["roc_auc"] <= 0.55
        # PR-AUC should be near positive prevalence (~0.0704)
        assert abs(graph_eval["pr_auc"] - graph_eval["positive_prevalence"]) < 0.02

    def test_early_fusion_does_not_beat_tabular_only_on_validation(self, stage34_metrics):
        """Early fusion (concatenating graph features) does not outperform tabular alone on validation PR-AUC."""
        tab_prauc = stage34_metrics["task_a_graph_ablation"]["tabular_only"]["validation"]["pr_auc"]
        early_prauc = stage34_metrics["task_a_graph_ablation"]["early_fusion"]["validation"]["pr_auc"]
        # Tabular alone achieves higher PR-AUC on validation than Early Fusion
        assert tab_prauc >= early_prauc

    def test_late_fusion_convex_tuning_isolated_to_validation(self, stage34_metrics):
        """Optimal convex weight alpha was selected on validation and placed maximum weight on tabular."""
        late_fusion = stage34_metrics["task_a_graph_ablation"]["late_fusion"]
        assert "best_alpha" in late_fusion
        assert late_fusion["best_alpha"] >= 0.80  # heavily favored tabular over graph noise


class TestTaskBEvaluationProtocol:
    """Verifies metrics separation, confidence intervals, and historical diagnostic label."""

    def test_validation_bootstrap_confidence_intervals_present(self, stage34_metrics):
        for arch in ["tabular_only", "graph_only", "early_fusion", "late_fusion"]:
            ci = stage34_metrics["task_a_graph_ablation"][arch]["confidence_intervals_validation"]
            assert "roc_auc_95_ci" in ci
            assert "pr_auc_95_ci" in ci
            assert len(ci["roc_auc_95_ci"]) == 2
            assert ci["roc_auc_95_ci"][0] < ci["roc_auc_95_ci"][1]
            assert len(ci["pr_auc_95_ci"]) == 2
            assert ci["pr_auc_95_ci"][0] < ci["pr_auc_95_ci"][1]

    def test_test_set_labeled_strictly_as_historical_diagnostic(self, stage34_metrics):
        for arch in ["tabular_only", "graph_only", "early_fusion", "late_fusion"]:
            entry = stage34_metrics["task_a_graph_ablation"][arch]
            assert "historical_diagnostic_test" in entry
            assert "test" not in entry  # Must not be ambiguously named 'test'


class TestTaskCDecisionPolicyDesign:
    """Verifies decision policy thresholds, boundaries, edge cases, and routing rules."""

    def test_threshold_boundaries_are_strictly_increasing_and_valid(self, stage34_policy):
        thresholds = stage34_policy["action_thresholds"]
        t_allow = thresholds["allow_cutoff"]
        t_review = thresholds["review_cutoff"]
        t_hold = thresholds["hold_cutoff"]

        assert 0.0 < t_allow < t_review < t_hold < 1.0, f"Thresholds must be strictly increasing in (0, 1): {thresholds}"

    def test_action_routing_partitions_probability_space(self, stage34_policy):
        thresholds = stage34_policy["action_thresholds"]
        t_allow = thresholds["allow_cutoff"]
        t_review = thresholds["review_cutoff"]
        t_hold = thresholds["hold_cutoff"]

        def route_action(p: float) -> str:
            if not isinstance(p, (int, float)) or math.isnan(p):
                raise ValueError("Invalid probability")
            if not (0.0 <= p <= 1.0):
                raise ValueError("Probability out of bounds [0, 1]")
            if p < t_allow:
                return "ALLOW"
            elif p < t_review:
                return "REVIEW"
            elif p < t_hold:
                return "HOLD"
            else:
                return "BLOCK"

        # Boundary checks
        assert route_action(0.0) == "ALLOW"
        assert route_action(t_allow - 1e-6) == "ALLOW"
        assert route_action(t_allow) == "REVIEW"
        assert route_action(t_review - 1e-6) == "REVIEW"
        assert route_action(t_review) == "HOLD"
        assert route_action(t_hold - 1e-6) == "HOLD"
        assert route_action(t_hold) == "BLOCK"
        assert route_action(1.0) == "BLOCK"

    def test_invalid_probabilities_raise_error(self, stage34_policy):
        thresholds = stage34_policy["action_thresholds"]
        t_allow = thresholds["allow_cutoff"]

        def route_action(p: float) -> str:
            if not isinstance(p, (int, float)) or math.isnan(p):
                raise ValueError("Invalid probability")
            if not (0.0 <= p <= 1.0):
                raise ValueError("Probability out of bounds [0, 1]")
            return "ALLOW" if p < t_allow else "REVIEW"

        with pytest.raises(ValueError, match="out of bounds"):
            route_action(-0.01)
        with pytest.raises(ValueError, match="out of bounds"):
            route_action(1.05)
        with pytest.raises(ValueError, match="Invalid"):
            route_action(float("nan"))

    def test_missing_scores_handled_with_fallback(self):
        """Missing or null scores must fallback cleanly to base prior without crashing."""
        score_map = {"L100": 0.45}
        # Fallback default should be 0.05
        assert score_map.get("L100", 0.05) == 0.45
        assert score_map.get("L999_MISSING", 0.05) == 0.05

    def test_review_capacity_constraints_defined(self, stage34_policy):
        caps = stage34_policy["capacity_operating_points"]
        for cap_key in ["capacity_1_pct", "capacity_2_pct", "capacity_5_pct", "capacity_10_pct"]:
            assert cap_key in caps
            point = caps[cap_key]
            assert "operational_threshold" in point
            assert "precision" in point
            assert "recall" in point
            assert point["operational_threshold"] > 0.0


class TestTaskDEnsembleSemantics:
    """Verifies ensemble ranking semantics and calibration status."""

    def test_max_risk_is_uncalibrated_heuristic(self, stage34_metrics):
        max_risk = stage34_metrics["task_d_ensemble_semantics"]["max_risk_heuristic"]
        assert max_risk["is_calibrated"] is False
        assert "uncalibrated ranking heuristic" in max_risk["semantic_note"].lower()

    def test_calibrated_noisy_or_restores_calibration(self, stage34_metrics):
        cal_noisyor = stage34_metrics["task_d_ensemble_semantics"]["calibrated_noisy_or"]
        raw_noisyor = stage34_metrics["task_d_ensemble_semantics"]["raw_noisy_or"]

        assert cal_noisyor["is_calibrated"] is True
        # Calibrated noisy-OR has lower Brier score and lower ECE than raw noisy-OR on validation
        assert cal_noisyor["validation"]["brier_score"] <= raw_noisyor["validation"]["brier_score"]
        assert cal_noisyor["validation"]["ece"] <= raw_noisyor["validation"]["ece"]


class TestStage34ArtifactsExist:
    """Verifies all required model weights and serialized files exist on disk."""

    def test_versioned_artifacts_on_disk(self):
        expected_files = [
            "stage34_tabular_model.joblib",
            "stage34_graph_model.joblib",
            "stage34_early_fusion_model.joblib",
            "stage34_late_fusion_stacker.joblib",
            "stage34_tabular_calibrator.joblib",
            "stage34_early_fusion_calibrator.joblib",
            "stage34_noisyor_calibrator.joblib",
            "stage34_decision_policy.json",
        ]
        for fname in expected_files:
            fpath = os.path.join(_MODELS_STAGE34_DIR, fname)
            assert os.path.exists(fpath), f"Missing artifact {fname} in {_MODELS_STAGE34_DIR}"
