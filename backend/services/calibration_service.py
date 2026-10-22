"""Matured-label calibration monitoring with aggregate, privacy-safe output.

Calibration metrics are computed only after the declared outcome window has
elapsed.  This prevents an early-labeling bias from being presented as model
quality.  The service reports ECE, Brier score, operating-point precision and
recall, and deterministic fixed-review-budget capture.  Approved subgroup
slices are aggregate-only and are rejected for unapproved dimensions.

Alerts route to investigation or promotion review.  They never invoke model
training or automatic retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import math
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_score, recall_score

from backend.contracts.monitoring import (
    APPROVED_SUBGROUP_FIELDS,
    CalibrationReportV1,
    MonitoringAlertV1,
    MonitoringSubgroupV1,
    MonitoringWindowV1,
    classify_threshold,
)


POSITIVE_LABELS = frozenset({"1", "true", "fraud", "confirmed_fraud", "positive", "yes"})
NEGATIVE_LABELS = frozenset({"0", "false", "legitimate", "cleared", "negative", "no"})
POSITIVE_DECISIONS = frozenset(
    {"1", "true", "fraud", "block", "blocked", "decline", "declined", "hold", "held", "review", "flag", "flagged"}
)
DEFAULT_CALIBRATION_BUDGETS = (0.01, 0.05, 0.10)


class CalibrationInputError(ValueError):
    """Raised when labels or scores cannot be evaluated safely."""


@dataclass(frozen=True)
class CalibrationThresholds:
    """Warning/critical boundaries for matured-label calibration metrics."""

    warning_ece: float = 0.05
    critical_ece: float = 0.10
    warning_brier: float = 0.15
    critical_brier: float = 0.25
    warning_precision: float = 0.50
    critical_precision: float = 0.25
    warning_recall: float = 0.50
    critical_recall: float = 0.25
    warning_capture: float = 0.50
    critical_capture: float = 0.25

    def __post_init__(self) -> None:
        higher_pairs = (
            ("warning_ece", "critical_ece"),
            ("warning_brier", "critical_brier"),
        )
        lower_pairs = (
            ("warning_precision", "critical_precision"),
            ("warning_recall", "critical_recall"),
            ("warning_capture", "critical_capture"),
        )
        for warning_name, critical_name in (*higher_pairs, *lower_pairs):
            warning = float(getattr(self, warning_name))
            critical = float(getattr(self, critical_name))
            if not math.isfinite(warning) or not math.isfinite(critical):
                raise ValueError("calibration thresholds must be finite")
            if not 0.0 <= critical <= 1.0 or not 0.0 <= warning <= 1.0:
                raise ValueError("calibration thresholds must be in [0, 1]")
            if warning_name in dict(higher_pairs):
                if critical < warning:
                    raise ValueError(f"{critical_name} must be >= {warning_name}")
            elif critical > warning:
                raise ValueError(f"{critical_name} must be <= {warning_name} for a lower-is-worse metric")


def _as_array(values: Iterable[Any], *, name: str) -> np.ndarray:
    try:
        array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    except (TypeError, ValueError) as exc:
        raise CalibrationInputError(f"{name} must be one-dimensional") from exc
    if array.ndim != 1:
        raise CalibrationInputError(f"{name} must be one-dimensional")
    return array


def _normalize_labels(values: Iterable[Any]) -> np.ndarray:
    labels: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            labels.append(int(value))
            continue
        if isinstance(value, (int, np.integer, float, np.floating)) and not isinstance(value, bool):
            number = float(value)
            if number in (0.0, 1.0):
                labels.append(int(number))
                continue
        text = str(value).strip().lower()
        if text in POSITIVE_LABELS:
            labels.append(1)
        elif text in NEGATIVE_LABELS:
            labels.append(0)
        else:
            raise CalibrationInputError(f"unsupported binary label value: {value!r}")
    return np.asarray(labels, dtype=int)


def _validate_scores(values: Iterable[Any]) -> np.ndarray:
    try:
        scores = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise CalibrationInputError("scores must be numeric probabilities") from exc
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise CalibrationInputError("scores must be finite probabilities in [0, 1]")
    return scores


def expected_calibration_error(
    labels: Iterable[Any],
    scores: Iterable[Any],
    *,
    bins: int = 10,
    n_bins: Optional[int] = None,
) -> float:
    """Return equal-width expected calibration error in ``[0, 1]``."""

    if n_bins is not None:
        bins = n_bins
    y = _normalize_labels(labels)
    p = _validate_scores(scores)
    if len(y) != len(p):
        raise CalibrationInputError("labels and scores must have equal lengths")
    if bins < 1:
        raise ValueError("bins must be positive")
    if not len(y):
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        if not mask.any():
            continue
        result += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(result)


calculate_ece = expected_calibration_error
calculate_expected_calibration_error = expected_calibration_error


def calculate_brier_score(labels: Iterable[Any], scores: Iterable[Any]) -> float:
    """Return the binary Brier score (lower is better)."""

    y = _normalize_labels(labels)
    p = _validate_scores(scores)
    if len(y) != len(p):
        raise CalibrationInputError("labels and scores must have equal lengths")
    if not len(y):
        return 0.0
    return float(brier_score_loss(y, p))


brier_score = calculate_brier_score
calculate_brier = calculate_brier_score
ece = expected_calibration_error


def _decision_array(scores: np.ndarray, decisions: Optional[Iterable[Any]], threshold: float) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if decisions is None:
        return (scores >= threshold).astype(int)
    values = _as_array(decisions, name="decisions")
    result: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            result.append(int(value))
            continue
        if isinstance(value, (int, np.integer, float, np.floating)) and float(value) in (0.0, 1.0):
            result.append(int(value))
            continue
        text = str(value).strip().lower()
        if text in POSITIVE_DECISIONS:
            result.append(1)
        elif text in NEGATIVE_LABELS or text in {"allow", "allowed", "pass", "passed", "none", "0"}:
            result.append(0)
        else:
            raise CalibrationInputError(f"unsupported decision value: {value!r}")
    return np.asarray(result, dtype=int)


def precision_recall_metrics(
    labels: Iterable[Any],
    scores: Iterable[Any],
    *,
    decisions: Optional[Iterable[Any]] = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return aggregate precision/recall at a supplied decision operating point."""

    y = _normalize_labels(labels)
    p = _validate_scores(scores)
    if len(y) != len(p):
        raise CalibrationInputError("labels and scores must have equal lengths")
    predicted = _decision_array(p, decisions, threshold)
    if len(predicted) != len(y):
        raise CalibrationInputError("decisions and labels must have equal lengths")
    if not len(y):
        return {
            "threshold": float(threshold),
            "precision": None,
            "recall": None,
            "predicted_positive_count": 0,
            "positive_count": 0,
            "sample_count": 0,
        }
    predicted_positive_count = int(predicted.sum())
    positive_count = int(y.sum())
    return {
        "threshold": float(threshold),
        "precision": (
            float(precision_score(y, predicted, zero_division=0))
            if predicted_positive_count
            else None
        ),
        "recall": float(recall_score(y, predicted, zero_division=0)) if positive_count else None,
        "predicted_positive_count": predicted_positive_count,
        "positive_count": positive_count,
        "sample_count": int(len(y)),
    }


calculate_precision_recall = precision_recall_metrics


def calculate_precision(labels: Iterable[Any], scores: Iterable[Any], **kwargs: Any) -> Optional[float]:
    """Return only the aggregate precision from an operating point."""

    return precision_recall_metrics(labels, scores, **kwargs)["precision"]


def calculate_recall(labels: Iterable[Any], scores: Iterable[Any], **kwargs: Any) -> Optional[float]:
    """Return only the aggregate recall from an operating point."""

    return precision_recall_metrics(labels, scores, **kwargs)["recall"]


def _budget_count(sample_count: int, budget: float | int) -> tuple[float, int]:
    numeric = float(budget)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("fixed budgets must be positive")
    if numeric <= 1.0:
        count = max(1, int(math.ceil(sample_count * numeric))) if sample_count else 0
        return numeric, min(count, sample_count)
    if numeric != int(numeric):
        raise ValueError("budgets above 1 must be whole review counts")
    return numeric, min(int(numeric), sample_count)


def fixed_budget_capture(
    labels: Iterable[Any],
    scores: Iterable[Any],
    budgets: Sequence[float | int] = DEFAULT_CALIBRATION_BUDGETS,
    *,
    budget: Optional[float | int] = None,
    amounts: Optional[Iterable[Any]] = None,
) -> list[dict[str, Any]]:
    """Return top-score fraud capture for fixed review-rate/count budgets."""

    if budget is not None:
        budgets = (budget,)
    y = _normalize_labels(labels)
    p = _validate_scores(scores)
    if len(y) != len(p):
        raise CalibrationInputError("labels and scores must have equal lengths")
    amount_array: Optional[np.ndarray] = None
    if amounts is not None:
        try:
            amount_array = np.asarray(amounts, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise CalibrationInputError("amounts must be numeric") from exc
        if len(amount_array) != len(y) or not np.isfinite(amount_array).all() or (amount_array < 0.0).any():
            raise CalibrationInputError("amounts must be finite non-negative values matching labels")
    total_positive = int(y.sum())
    order = np.argsort(-p, kind="stable")
    output: list[dict[str, Any]] = []
    for raw_budget in budgets:
        budget, count = _budget_count(len(y), raw_budget)
        selected = order[:count]
        captured = int(y[selected].sum())
        selected_positive_amount = (
            float(amount_array[selected][y[selected] == 1].sum()) if amount_array is not None else None
        )
        output.append(
            {
                "budget": float(budget),
                "review_count": int(count),
                "review_rate": float(count / len(y)) if len(y) else 0.0,
                "captured_positive_count": captured,
                "positive_count": total_positive,
                "capture_rate": float(captured / total_positive) if total_positive else None,
                "precision": float(captured / count) if count else None,
                "score_cutoff": float(p[selected[-1]]) if count else None,
                "captured_positive_amount": selected_positive_amount,
            }
        )
    return output


calculate_fixed_budget_capture = fixed_budget_capture
fixed_budget_capture_metrics = fixed_budget_capture


def _utc(value: datetime | str, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationInputError(f"{name} must be a valid timestamp") from exc
    if timestamp.tzinfo is None:
        raise CalibrationInputError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def matured_label_mask(
    frame: pd.DataFrame,
    *,
    as_of: datetime | str,
    event_time_column: str = "event_time",
    label_time_column: str = "label_time",
    outcome_window_column: str = "outcome_window_days",
) -> np.ndarray:
    """Return rows whose labels are available and whose outcome window matured."""

    required = {event_time_column, label_time_column, outcome_window_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CalibrationInputError(f"maturity columns are missing: {missing}")
    cutoff = _utc(as_of, name="as_of")
    event_times = pd.to_datetime(frame[event_time_column], utc=True, errors="coerce")
    label_times = pd.to_datetime(frame[label_time_column], utc=True, errors="coerce")
    windows = pd.to_numeric(frame[outcome_window_column], errors="coerce")
    if event_times.isna().any() or label_times.isna().any() or windows.isna().any() or (windows < 0).any():
        raise CalibrationInputError("maturity columns contain invalid values")
    maturity = event_times + pd.to_timedelta(windows, unit="D")
    return ((maturity <= cutoff) & (label_times >= maturity) & (label_times <= cutoff)).to_numpy(dtype=bool)


def is_matured_label(*args: Any, **kwargs: Any) -> np.ndarray:
    """Compatibility alias for :func:`matured_label_mask`."""

    return matured_label_mask(*args, **kwargs)


def _coerce_window(
    window: Optional[MonitoringWindowV1 | Mapping[str, Any]],
    *,
    sample_count: int,
    matured_count: int,
) -> MonitoringWindowV1:
    if isinstance(window, MonitoringWindowV1):
        values = window.model_dump(mode="python")
    elif window is not None:
        values = dict(window)
    else:
        start = datetime(1970, 1, 1, tzinfo=timezone.utc)
        values = {"window_id": "default", "window_start": start, "window_end": start + timedelta(days=1)}
    if values.get("sample_count", 0) == 0:
        values["sample_count"] = sample_count
    if values.get("current_sample_count", 0) == 0:
        values["current_sample_count"] = sample_count
    values["matured_label_count"] = matured_count
    return MonitoringWindowV1(**values)


def _subgroup_token(field: str, value: Any) -> str:
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    text = "__MISSING__" if missing else str(value)
    # Approved identifier-like dimensions are represented by a stable short
    # token rather than persisting a raw merchant/device identifier.
    lowered = text.lower()
    if (
        field.endswith("_id")
        or "@" in text
        or len(text) > 64
        or any(marker in lowered for marker in ("email", "phone", "address", "token"))
    ):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return text[:128]


def _alert(
    metric_name: str,
    metric_type: str,
    value: Optional[float],
    warning: Optional[float],
    critical: Optional[float],
    *,
    direction: Literal["higher", "lower"],
    sample_count: int,
    min_samples: int,
) -> Optional[dict[str, Any]]:
    if value is None or sample_count < min_samples:
        return None
    status = classify_threshold(value, warning, critical, direction=direction)
    if status == "OK":
        return None
    action = "promotion_review" if status == "CRITICAL" else "investigation"
    return MonitoringAlertV1(
        metric_name=metric_name,
        metric_type=metric_type,
        status=status,
        observed_value=value,
        warning_threshold=warning,
        critical_threshold=critical,
        direction=direction,
        recommended_action=action,
        sample_count=sample_count,
        reason=(
            "critical calibration result requires promotion review; retraining is not automatic"
            if status == "CRITICAL"
            else "warning calibration result requires investigation"
        ),
    ).model_dump(mode="json")


def _report_action(alerts: Sequence[Mapping[str, Any]]) -> str:
    if any(alert.get("status") == "CRITICAL" for alert in alerts):
        return "promotion_review"
    if any(alert.get("status") == "WARNING" for alert in alerts):
        return "investigation"
    return "none"


def _calibration_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    decisions: Optional[Iterable[Any]],
    threshold: float,
    budgets: Sequence[float | int],
    amounts: Optional[Iterable[Any]],
    bins: int,
) -> dict[str, Any]:
    operating = precision_recall_metrics(labels, scores, decisions=decisions, threshold=threshold)
    budgets_result = fixed_budget_capture(labels, scores, budgets, amounts=amounts)
    return {
        "sample_count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
        "ece": expected_calibration_error(labels, scores, bins=bins),
        "brier_score": calculate_brier_score(labels, scores),
        "precision": operating["precision"],
        "recall": operating["recall"],
        "threshold": operating["threshold"],
        "predicted_positive_count": operating["predicted_positive_count"],
        "fixed_budget_capture": budgets_result,
    }


def calculate_calibration_metrics(
    labels: Iterable[Any],
    scores: Iterable[Any],
    *,
    decisions: Optional[Iterable[Any]] = None,
    threshold: float = 0.5,
    fixed_budgets: Sequence[float | int] = DEFAULT_CALIBRATION_BUDGETS,
    amounts: Optional[Iterable[Any]] = None,
    bins: int = 10,
) -> dict[str, Any]:
    """Compute aggregate calibration metrics without a monitoring frame."""

    y = _normalize_labels(labels)
    p = _validate_scores(scores)
    if len(y) != len(p):
        raise CalibrationInputError("labels and scores must have equal lengths")
    return _calibration_metrics(
        y,
        p,
        decisions=decisions,
        threshold=threshold,
        budgets=fixed_budgets,
        amounts=amounts,
        bins=bins,
    )


class CalibrationService:
    """Evaluate matured labels and return a versioned aggregate report."""

    def __init__(
        self,
        thresholds: Optional[CalibrationThresholds] = None,
        *,
        min_samples: int = 1,
        min_subgroup_size: int = 1,
    ) -> None:
        if min_samples < 1:
            raise ValueError("min_samples must be positive")
        if min_subgroup_size < 1:
            raise ValueError("min_subgroup_size must be positive")
        self.thresholds = thresholds or CalibrationThresholds()
        self.min_samples = min_samples
        self.min_subgroup_size = min_subgroup_size

    def monitor(
        self,
        frame: pd.DataFrame,
        *,
        as_of: datetime | str,
        score_column: str = "score",
        label_column: str = "label",
        decision_column: Optional[str] = None,
        threshold: float = 0.5,
        fixed_budgets: Sequence[float | int] = DEFAULT_CALIBRATION_BUDGETS,
        subgroup_fields: Sequence[str] = (),
        event_time_column: str = "event_time",
        label_time_column: str = "label_time",
        outcome_window_column: str = "outcome_window_days",
        amount_column: Optional[str] = None,
        window: Optional[MonitoringWindowV1 | Mapping[str, Any]] = None,
        bins: int = 10,
    ) -> dict[str, Any]:
        """Evaluate mature rows and return a ``monitoring.v1`` report."""

        if not isinstance(frame, pd.DataFrame):
            raise CalibrationInputError("frame must be a pandas DataFrame")
        if score_column not in frame or label_column not in frame:
            raise CalibrationInputError("frame must contain score and label columns")
        unapproved = sorted(set(subgroup_fields) - APPROVED_SUBGROUP_FIELDS)
        if unapproved:
            raise CalibrationInputError(f"subgroup field is not approved: {unapproved[0]}")
        if decision_column is not None and decision_column not in frame:
            raise CalibrationInputError(f"decision column is missing: {decision_column}")
        if amount_column is not None and amount_column not in frame:
            raise CalibrationInputError(f"amount column is missing: {amount_column}")

        work = frame.reset_index(drop=True).copy()
        mature = matured_label_mask(
            work,
            as_of=as_of,
            event_time_column=event_time_column,
            label_time_column=label_time_column,
            outcome_window_column=outcome_window_column,
        )
        matured = work.loc[mature].reset_index(drop=True)
        immature_count = int((~mature).sum())
        labels = _normalize_labels(matured[label_column]) if len(matured) else np.asarray([], dtype=int)
        scores = _validate_scores(matured[score_column].to_numpy()) if len(matured) else np.asarray([], dtype=float)
        decisions = matured[decision_column].to_numpy() if decision_column is not None and len(matured) else None
        amounts = matured[amount_column].to_numpy() if amount_column is not None and len(matured) else None

        metrics = _calibration_metrics(
            labels,
            scores,
            decisions=decisions,
            threshold=threshold,
            budgets=fixed_budgets,
            amounts=amounts,
            bins=bins,
        )
        alerts: list[dict[str, Any]] = []
        for result in (
            _alert(
                "ece",
                "calibration",
                metrics["ece"],
                self.thresholds.warning_ece,
                self.thresholds.critical_ece,
                direction="higher",
                sample_count=len(labels),
                min_samples=self.min_samples,
            ),
            _alert(
                "brier_score",
                "calibration",
                metrics["brier_score"],
                self.thresholds.warning_brier,
                self.thresholds.critical_brier,
                direction="higher",
                sample_count=len(labels),
                min_samples=self.min_samples,
            ),
            _alert(
                "precision",
                "calibration",
                metrics["precision"],
                self.thresholds.warning_precision,
                self.thresholds.critical_precision,
                direction="lower",
                sample_count=len(labels),
                min_samples=self.min_samples,
            ),
            _alert(
                "recall",
                "calibration",
                metrics["recall"],
                self.thresholds.warning_recall,
                self.thresholds.critical_recall,
                direction="lower",
                sample_count=len(labels),
                min_samples=self.min_samples,
            ),
        ):
            if result is not None:
                alerts.append(result)
        for capture in metrics["fixed_budget_capture"]:
            result = _alert(
                f"capture_at_budget_{capture['budget']:g}",
                "calibration",
                capture["capture_rate"],
                self.thresholds.warning_capture,
                self.thresholds.critical_capture,
                direction="lower",
                sample_count=len(labels),
                min_samples=self.min_samples,
            )
            if result is not None:
                alerts.append(result)

        subgroup_slices: list[MonitoringSubgroupV1] = []
        for field in subgroup_fields:
            grouped = matured.groupby(field, dropna=False, sort=True).groups
            for value, indices in grouped.items():
                index_array = np.asarray(list(indices), dtype=int)
                if len(index_array) < self.min_subgroup_size:
                    continue
                group_labels = labels[index_array]
                group_scores = scores[index_array]
                group_decisions = decisions[index_array] if decisions is not None else None
                group_amounts = amounts[index_array] if amounts is not None else None
                subgroup_slices.append(
                    MonitoringSubgroupV1(
                        field=field,
                        value=_subgroup_token(field, value),
                        sample_count=len(index_array),
                        metrics=_calibration_metrics(
                            group_labels,
                            group_scores,
                            decisions=group_decisions,
                            threshold=threshold,
                            budgets=fixed_budgets,
                            amounts=group_amounts,
                            bins=bins,
                        ),
                    )
                )

        report_window = _coerce_window(window, sample_count=len(work), matured_count=len(matured))
        status = "CRITICAL" if any(item["status"] == "CRITICAL" for item in alerts) else (
            "WARNING" if alerts else "OK"
        )
        report = CalibrationReportV1(
            monitoring_window=report_window,
            status=status,
            metrics=metrics,
            alerts=[MonitoringAlertV1(**item) for item in alerts],
            recommended_action=_report_action(alerts),
            matured_label_count=len(matured),
            immature_label_count=immature_count,
            subgroup_slices=subgroup_slices,
            subgroups=subgroup_slices,
        )
        return report.model_dump(mode="json")

    evaluate = monitor
    evaluate_frame = monitor


CalibrationMonitoringService = CalibrationService


def evaluate_calibration(frame: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    """Functional convenience wrapper around :class:`CalibrationService`."""

    return CalibrationService().monitor(frame, **kwargs)


monitor_calibration = evaluate_calibration
compute_calibration_report = evaluate_calibration


__all__ = [
    "APPROVED_SUBGROUP_FIELDS",
    "CalibrationInputError",
    "CalibrationMonitoringService",
    "CalibrationService",
    "CalibrationThresholds",
    "DEFAULT_CALIBRATION_BUDGETS",
    "brier_score",
    "calculate_brier_score",
    "calculate_brier",
    "calculate_calibration_metrics",
    "calculate_ece",
    "calculate_expected_calibration_error",
    "calculate_fixed_budget_capture",
    "calculate_precision_recall",
    "expected_calibration_error",
    "ece",
    "compute_calibration_report",
    "evaluate_calibration",
    "fixed_budget_capture",
    "fixed_budget_capture_metrics",
    "is_matured_label",
    "matured_label_mask",
    "precision_recall_metrics",
    "calculate_precision",
    "calculate_recall",
    "monitor_calibration",
]
