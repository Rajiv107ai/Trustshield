"""Deterministic tests for monitoring metrics, contracts, and alert routing."""

import pandas as pd
import pytest

from backend.contracts.monitoring import MonitoringWindowV1, classify_threshold
from backend.services.calibration_service import (
    CalibrationInputError,
    CalibrationService,
    CalibrationThresholds,
    calculate_brier_score,
    expected_calibration_error,
    fixed_budget_capture,
)
from backend.services.drift_service import (
    DriftService,
    DriftThresholds,
    calculate_category_frequencies,
    calculate_decision_volume,
    calculate_js_divergence,
    calculate_ks_statistic,
    calculate_missingness,
    calculate_psi,
    calculate_quantile_shifts,
    calculate_unseen_categories,
)


WINDOW = MonitoringWindowV1(
    window_id="test-window",
    window_start="2025-01-01T00:00:00+00:00",
    window_end="2025-01-02T00:00:00+00:00",
)


def test_numeric_metrics_are_zero_for_identical_windows() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert calculate_psi(values, values, bin_edges=[0.0, 2.0, 4.0]) == pytest.approx(0.0)
    assert calculate_ks_statistic(values, values) == pytest.approx(0.0)
    shifts = calculate_quantile_shifts(values, values)
    assert all(item["delta"] == pytest.approx(0.0) for item in shifts.values())
    assert calculate_missingness([1.0, None]) == pytest.approx(0.5)


def test_categorical_metrics_include_frequency_js_and_unseen_rate() -> None:
    assert calculate_category_frequencies(["a", "a", "b"]) == {
        "a": pytest.approx(2 / 3),
        "b": pytest.approx(1 / 3),
    }
    assert calculate_js_divergence(["a", "b"], ["a", "b"]) == pytest.approx(0.0)
    unseen = calculate_unseen_categories(["a", "b"], ["b", "c", "c"])
    assert unseen["count"] == 2
    assert unseen["rate"] == pytest.approx(2 / 3)


def test_decision_volume_is_aggregate_and_deterministic() -> None:
    result = calculate_decision_volume(["ALLOW", "ALLOW", "BLOCK"], ["ALLOW", "BLOCK", "BLOCK", "BLOCK"])
    assert result["reference_total"] == 3
    assert result["current_total"] == 4
    assert result["volume_delta"] == 1
    assert result["current_counts"] == {"ALLOW": 1, "BLOCK": 3}


def test_alert_boundary_equality_is_inclusive() -> None:
    assert classify_threshold(0.10, 0.10, 0.25) == "WARNING"
    assert classify_threshold(0.25, 0.10, 0.25) == "CRITICAL"
    assert classify_threshold(0.50, 0.50, 0.25, direction="lower") == "WARNING"
    assert classify_threshold(0.25, 0.50, 0.25, direction="lower") == "CRITICAL"


def test_drift_report_routes_critical_alert_to_shadow_evaluation() -> None:
    reference = pd.DataFrame({"amount": [0.0, 1.0, 2.0, 3.0]})
    current = pd.DataFrame({"amount": [100.0, 101.0, 102.0, 103.0]})
    thresholds = DriftThresholds(
        warning_psi=0.01,
        critical_psi=0.02,
        warning_ks=1.0,
        critical_ks=1.0,
        warning_quantile_shift=1.0,
        critical_quantile_shift=1.0,
        warning_missingness_delta=1.0,
        critical_missingness_delta=1.0,
    )
    report = DriftService(thresholds).monitor(
        reference,
        current,
        numeric_features=["amount"],
        window=WINDOW,
    )
    assert report["status"] == "CRITICAL"
    assert report["recommended_action"] == "shadow_evaluation"
    assert report["automatic_retraining"] is False
    assert all(alert["recommended_action"] == "shadow_evaluation" for alert in report["alerts"])
    assert "amount" not in str(report["metrics"].get("rows", ""))


def test_calibration_uses_only_matured_labels_and_fixed_budgets() -> None:
    labels = [0, 1, 0, 1]
    scores = [0.0, 1.0, 0.0, 1.0]
    assert expected_calibration_error(labels, scores) == pytest.approx(0.0)
    assert calculate_brier_score(labels, scores) == pytest.approx(0.0)
    capture = fixed_budget_capture(labels, scores, budgets=[0.5])
    assert capture[0]["review_count"] == 2
    assert capture[0]["captured_positive_count"] == 2
    assert capture[0]["capture_rate"] == pytest.approx(1.0)

    frame = pd.DataFrame(
        {
            "event_time": [
                "2025-01-01T00:00:00+00:00",
                "2025-01-01T00:00:00+00:00",
                "2025-01-01T00:00:00+00:00",
            ],
            "label_time": [
                "2025-01-10T00:00:00+00:00",
                "2025-01-01T12:00:00+00:00",
                "2025-01-10T00:00:00+00:00",
            ],
            "outcome_window_days": [1, 1, 1],
            "score": [0.0, 0.9, 1.0],
            "label": [0, 1, 1],
            "region": ["west", "west", "east"],
        }
    )
    report = CalibrationService(min_subgroup_size=1).monitor(
        frame,
        as_of="2025-01-11T00:00:00+00:00",
        subgroup_fields=["region"],
        fixed_budgets=[0.5],
        window=WINDOW,
    )
    assert report["matured_label_count"] == 2
    assert report["immature_label_count"] == 1
    assert report["metrics"]["brier_score"] == pytest.approx(0.0)
    assert {item["value"] for item in report["subgroup_slices"]} == {"west", "east"}


def test_calibration_threshold_boundary_routes_promotion_review() -> None:
    thresholds = CalibrationThresholds(
        warning_ece=0.10,
        critical_ece=0.20,
        warning_brier=1.0,
        critical_brier=1.0,
        warning_precision=0.0,
        critical_precision=0.0,
        warning_recall=0.0,
        critical_recall=0.0,
        warning_capture=0.0,
        critical_capture=0.0,
    )
    # Scores all equal 0.5 with labels all zero gives ECE exactly 0.5.  The
    # critical threshold equality is checked independently above; this case
    # verifies the calibration action and no-training invariant.
    frame = pd.DataFrame(
        {
            "event_time": ["2025-01-01T00:00:00+00:00"] * 4,
            "label_time": ["2025-01-03T00:00:00+00:00"] * 4,
            "outcome_window_days": [1] * 4,
            "score": [0.5] * 4,
            "label": [0] * 4,
        }
    )
    report = CalibrationService(thresholds).monitor(
        frame,
        as_of="2025-01-04T00:00:00+00:00",
        fixed_budgets=[0.5],
        window=WINDOW,
    )
    assert report["status"] == "CRITICAL"
    assert report["recommended_action"] == "promotion_review"
    assert report["automatic_retraining"] is False


def test_unapproved_subgroup_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "event_time": ["2025-01-01T00:00:00+00:00"],
            "label_time": ["2025-01-03T00:00:00+00:00"],
            "outcome_window_days": [1],
            "score": [0.1],
            "label": [0],
        }
    )
    with pytest.raises(CalibrationInputError, match="not approved"):
        CalibrationService().monitor(
            frame,
            as_of="2025-01-04T00:00:00+00:00",
            subgroup_fields=["email"],
            window=WINDOW,
        )
