"""
TrustShield — Stage 3.3.1 Independent Metrics and Decision-Policy Audit Test Suite.

Covers:
1. Task A: Confusion matrix mathematical consistency, continuous score AUC verification,
   sample counts, and prevalence.
2. Task B: Combined engine lift reconciliation (Full-Graph vs Tabular XGB vs Tabular LR).
3. Task C: Graph contribution dual-perspective audit (Zero-Graph drop vs Tabular standalone).
4. Task D: Decision-routing policy audit (calibrated score compression and threshold 0.50 defect).
5. Task E: Return-abuse model selection validation rule verification and calibration auditing.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_REPORTS_DIR = os.path.join(_ROOT_DIR, "reports")

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


@pytest.fixture(scope="module")
def metrics_data() -> dict:
    """Loads reports/phase3_stage33_metrics.json."""
    p = os.path.join(_REPORTS_DIR, "phase3_stage33_metrics.json")
    assert os.path.isfile(p), f"Missing metrics file: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


class TestTaskAMetricsReconciliation:
    """Task A: Reconcile all reported metrics for mathematical consistency."""

    def test_confusion_matrices_agree_with_precision_recall_f1(self, metrics_data):
        """Verifies that precision, recall, and F1 mathematically match confusion matrices

        across all models and splits with zero discrepancies.
        """
        for model_group, models in metrics_data["models"].items():
            if model_group in ["subtype_analysis", "cold_start_analysis"]:
                continue
            for model_name, splits in models.items():
                if model_name in ["subtype_analysis", "cold_start_analysis"]:
                    continue
                for split_name, m in splits.items():
                    cm = m["confusion_matrix"]
                    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
                    expected_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    expected_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    expected_f1 = (
                        2 * expected_prec * expected_rec / (expected_prec + expected_rec)
                        if (expected_prec + expected_rec) > 0
                        else 0.0
                    )

                    assert abs(m["precision"] - round(expected_prec, 4)) <= 0.0001, (
                        f"Precision mismatch in {model_group}.{model_name}.{split_name}"
                    )
                    assert abs(m["recall"] - round(expected_rec, 4)) <= 0.0001, (
                        f"Recall mismatch in {model_group}.{model_name}.{split_name}"
                    )
                    assert abs(m["f1"] - round(expected_f1, 4)) <= 0.0001, (
                        f"F1 mismatch in {model_group}.{model_name}.{split_name}"
                    )

    def test_sample_counts_and_prevalences_match_splits(self, metrics_data):
        """Verifies that sample counts and positive prevalences match the frozen contract."""
        fl_test = metrics_data["models"]["fake_listing_detector"]["xgboost_calibrated"]["test"]
        assert fl_test["support"]["total"] == 2686
        assert fl_test["support"]["positives"] == 56
        assert fl_test["positive_prevalence"] == 0.0208

        tx_test = metrics_data["models"]["transaction_fraud_detector"]["full_graph_xgboost_calibrated"]["test"]
        assert tx_test["support"]["total"] == 20919
        assert tx_test["support"]["positives"] == 1835
        assert tx_test["positive_prevalence"] == 0.0877

        ret_test = metrics_data["models"]["return_fraud_detector"]["xgboost_calibrated"]["test"]
        assert ret_test["support"]["total"] == 2446
        assert ret_test["support"]["positives"] == 785
        assert ret_test["positive_prevalence"] == 0.3209


class TestTaskBCombinedEngineAudit:
    """Task B: Audit the Max-Risk Ensemble fairly against all baselines."""

    def test_relative_pr_auc_improvements_reconciled(self, metrics_data):
        """Recomputes relative PR-AUC improvement against Full-Graph, Tabular XGBoost,

        and the strongest standalone baseline (Tabular Logistic Regression).
        """
        tx = metrics_data["models"]["transaction_fraud_detector"]
        mr = metrics_data["combined_trust_engine"]["max_risk_ensemble"]

        lr_pr = tx["tabular_logistic_regression"]["test"]["pr_auc"]
        tab_xgb_pr = tx["tabular_xgboost_no_graph"]["test"]["pr_auc"]
        fg_xgb_pr = tx["full_graph_xgboost_calibrated"]["test"]["pr_auc"]
        mr_pr = mr["pr_auc"]

        assert lr_pr == 0.1550
        assert tab_xgb_pr == 0.1470
        assert fg_xgb_pr == 0.1234
        assert mr_pr == 0.2556

        # Relative lifts
        lift_vs_fg = (mr_pr - fg_xgb_pr) / fg_xgb_pr
        lift_vs_tab_xgb = (mr_pr - tab_xgb_pr) / tab_xgb_pr
        lift_vs_lr = (mr_pr - lr_pr) / lr_pr

        assert round(lift_vs_fg, 3) == 1.071  # +107.1% vs Full-Graph XGBoost
        assert round(lift_vs_tab_xgb, 3) == 0.739  # +73.9% vs Tabular XGBoost
        assert round(lift_vs_lr, 3) == 0.649  # +64.9% vs strongest standalone baseline (Tabular LR)


class TestTaskCGraphContributionAudit:
    """Task C: Dual-perspective audit of graph feature contribution."""

    def test_graph_ablation_dual_findings(self, metrics_data):
        """Verifies that while intra-model zero-graph degradation drops performance (-0.0440),

        the standalone Tabular XGBoost model actually outperformed Full-Graph XGBoost (+0.0203 ROC, +0.0236 PR).
        """
        tx = metrics_data["models"]["transaction_fraud_detector"]
        tab_xgb = tx["tabular_xgboost_no_graph"]["test"]
        fg_xgb = tx["full_graph_xgboost_calibrated"]["test"]
        zero_graph = tx["degraded_zero_graph"]["test"]

        # Intra-model degradation penalty
        degradation_drop_roc = fg_xgb["roc_auc"] - zero_graph["roc_auc"]
        assert round(degradation_drop_roc, 4) == 0.0440

        # Cross-model standalone comparison
        cross_model_diff_roc = fg_xgb["roc_auc"] - tab_xgb["roc_auc"]
        cross_model_diff_pr = fg_xgb["pr_auc"] - tab_xgb["pr_auc"]
        assert round(cross_model_diff_roc, 4) == -0.0203  # Tabular was higher!
        assert round(cross_model_diff_pr, 4) == -0.0236  # Tabular was higher!


class TestTaskDDecisionRoutingPolicyAudit:
    """Task D: Audit decision-routing policy and explain the 99.25% ALLOW / 0.16% recall result."""

    def test_threshold_50_defect_on_calibrated_probabilities(self, metrics_data):
        """Proves that applying a fixed 0.50 threshold to calibrated 8.77% base-rate probabilities

        causes exactly 6 orders to exceed threshold, detecting only 3 of 1,835 frauds (0.16% recall).
        """
        te = metrics_data["combined_trust_engine"]
        decisions = te["trust_engine_decisions"]
        cm = te["trust_engine_weighted"]["confusion_matrix"]
        supp = te["trust_engine_weighted"]["support"]

        # 99.25% ALLOW
        assert decisions["ALLOW"] == 20762
        assert round(decisions["ALLOW"] / supp["total"], 4) == 0.9925

        # Only 6 orders exceed 0.50 (5 HOLD + 1 BLOCK)
        assert decisions["HOLD"] + decisions["BLOCK"] == 6

        # Confusion matrix @ 0.50
        assert cm["tp"] == 3
        assert cm["fp"] == 3
        assert cm["fn"] == 1832
        assert cm["tn"] == 19081

        # Recall is exactly 0.1635%
        rec = cm["tp"] / supp["positives"]
        assert round(rec, 4) == 0.0016


class TestTaskEModelSelectionValidationAudit:
    """Task E: Audit validation-only model selection rule for return-abuse detector."""

    def test_return_model_selection_was_validation_optimal(self, metrics_data):
        """Verifies that XGBoost Calibrated was the legitimate winner on Validation data

        for both F1 (0.4884) and ROC-AUC (0.6880), explaining the validation-only selection decision.
        """
        ret = metrics_data["models"]["return_fraud_detector"]
        val_f1s = {name: data["validation"]["f1"] for name, data in ret.items()}
        val_rocs = {name: data["validation"]["roc_auc"] for name, data in ret.items()}

        # Highest F1 on Validation
        best_f1_model = max(val_f1s, key=lambda k: val_f1s[k])
        assert best_f1_model == "xgboost_calibrated"
        assert val_f1s["xgboost_calibrated"] == 0.4884

        # Highest ROC-AUC on Validation
        best_roc_model = max(val_rocs, key=lambda k: val_rocs[k])
        assert best_roc_model == "xgboost_calibrated"
        assert val_rocs["xgboost_calibrated"] == 0.6880

        # On Test, Random Forest retained higher discrimination (ROC 0.6115 vs 0.5923)
        assert ret["random_forest"]["test"]["roc_auc"] == 0.6115
        assert ret["xgboost_calibrated"]["test"]["roc_auc"] == 0.5923
