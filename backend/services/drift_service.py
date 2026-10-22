"""Aggregate drift monitoring for features, scores, and decisions.

The service compares two in-memory windows and returns only aggregate values.
It deliberately has no persistence, routing, model-training, or row-level
logging responsibilities.  A caller can persist the returned
``monitoring.v1`` report through its own governed store if desired.

Threshold alerts route to investigation or shadow evaluation.  They never
trigger retraining automatically; promotion of a candidate model is a
separate human/governance decision.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import math
import re
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from backend.contracts.monitoring import (
    APPROVED_SUBGROUP_FIELDS,
    DriftReportV1,
    MonitoringAlertV1,
    MonitoringWindowV1,
    classify_threshold,
)


DEFAULT_DRIFT_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
_MISSING_CATEGORY = "__MISSING__"
_UUID_LIKE = re.compile(r"^[0-9a-f]{8}-[0-9a-f-]{17,}$", re.IGNORECASE)


class DriftInputError(ValueError):
    """Raised when a drift comparison cannot be performed safely."""


@dataclass(frozen=True)
class DriftThresholds:
    """Warning/critical boundaries for aggregate drift signals.

    Threshold equality is included in the corresponding alert band.  PSI,
    KS, JS, quantile shift, missingness change, unseen-category rate, and
    decision-volume change are all non-negative magnitudes for alerting.
    """

    warning_psi: float = 0.10
    critical_psi: float = 0.25
    warning_ks: float = 0.10
    critical_ks: float = 0.20
    warning_quantile_shift: float = 0.10
    critical_quantile_shift: float = 0.25
    warning_missingness_delta: float = 0.05
    critical_missingness_delta: float = 0.10
    warning_js: float = 0.10
    critical_js: float = 0.20
    warning_unseen_category_rate: float = 0.01
    critical_unseen_category_rate: float = 0.05
    warning_decision_volume_change: float = 0.10
    critical_decision_volume_change: float = 0.25

    def __post_init__(self) -> None:
        names = (
            "warning_psi",
            "critical_psi",
            "warning_ks",
            "critical_ks",
            "warning_quantile_shift",
            "critical_quantile_shift",
            "warning_missingness_delta",
            "critical_missingness_delta",
            "warning_js",
            "critical_js",
            "warning_unseen_category_rate",
            "critical_unseen_category_rate",
            "warning_decision_volume_change",
            "critical_decision_volume_change",
        )
        values = {name: float(getattr(self, name)) for name in names}
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("drift thresholds must be finite and non-negative")
        pairs = (
            ("warning_psi", "critical_psi"),
            ("warning_ks", "critical_ks"),
            ("warning_quantile_shift", "critical_quantile_shift"),
            ("warning_missingness_delta", "critical_missingness_delta"),
            ("warning_js", "critical_js"),
            ("warning_unseen_category_rate", "critical_unseen_category_rate"),
            ("warning_decision_volume_change", "critical_decision_volume_change"),
        )
        for warning_name, critical_name in pairs:
            if values[critical_name] < values[warning_name]:
                raise ValueError(f"{critical_name} must be >= {warning_name}")


def _series(values: Any, *, name: str = "values") -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    if isinstance(values, pd.DataFrame):
        raise DriftInputError(f"{name} must be a one-dimensional sequence")
    try:
        return pd.Series(values).reset_index(drop=True)
    except (TypeError, ValueError) as exc:
        raise DriftInputError(f"{name} must be a one-dimensional sequence") from exc


def _numeric(values: Any, *, name: str = "values") -> tuple[np.ndarray, int, int]:
    series = _series(values, name=name)
    converted = pd.to_numeric(series, errors="coerce")
    missing = int(converted.isna().sum())
    array = converted.to_numpy(dtype=float)
    finite = np.isfinite(array)
    invalid = int((~finite & ~converted.isna().to_numpy()).sum())
    missing += invalid
    return array[finite], missing, int(len(array))


def _finite_values(values: Any, *, name: str = "values") -> np.ndarray:
    array, _, _ = _numeric(values, name=name)
    return array


def _round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def calculate_missingness(values: Any) -> float:
    """Return the fraction of missing/invalid values in a window."""

    series = _series(values)
    if len(series) == 0:
        return 0.0
    return float(series.isna().mean())


def missingness_metrics(reference: Any, current: Any) -> dict[str, Any]:
    """Return aggregate missingness rates and the absolute change."""

    reference_rate = calculate_missingness(reference)
    current_rate = calculate_missingness(current)
    return {
        "reference": _round(reference_rate),
        "current": _round(current_rate),
        "delta": _round(current_rate - reference_rate),
        "absolute_delta": _round(abs(current_rate - reference_rate)),
        "reference_count": int(len(_series(reference))),
        "current_count": int(len(_series(current))),
    }


calculate_missingness_rate = calculate_missingness


def _bin_edges(reference: np.ndarray, *, bins: int, bin_edges: Optional[Sequence[float]]) -> np.ndarray:
    if bin_edges is not None:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all():
            raise ValueError("bin_edges must be a finite, increasing sequence")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("bin_edges must be strictly increasing")
        return edges
    if bins < 1:
        raise ValueError("bins must be positive")
    if len(reference) == 0:
        return np.linspace(0.0, 1.0, bins + 1)
    quantile_edges = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(quantile_edges)
    if len(edges) >= 2:
        # Extend the outer edges so values equal to min/max always land in a
        # finite bin while retaining baseline quantile boundaries.
        spread = max(float(edges[-1] - edges[0]), 1.0)
        edges = edges.astype(float)
        edges[0] -= spread * 1e-9
        edges[-1] += spread * 1e-9
        return edges
    center = float(reference[0])
    width = max(abs(center) * 1e-6, 1e-6)
    return np.asarray([center - width, center, center + width], dtype=float)


def calculate_psi(
    reference: Any,
    current: Any,
    *,
    bins: int = 10,
    bin_edges: Optional[Sequence[float]] = None,
    epsilon: float = 1e-6,
) -> float:
    """Calculate population stability index (PSI) for numeric values.

    Bins are fitted to the reference window's quantiles unless explicit
    ``bin_edges`` are supplied.  Empty windows have no observable drift and
    return ``0.0``; the report retains their sample counts so callers can
    apply their own minimum-volume policy.
    """

    if epsilon <= 0.0 or not math.isfinite(float(epsilon)):
        raise ValueError("epsilon must be finite and positive")
    expected = _finite_values(reference, name="reference")
    actual = _finite_values(current, name="current")
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    edges = _bin_edges(expected, bins=bins, bin_edges=bin_edges)
    # The first/last reference bins are open-ended.  Without this extension,
    # a current value outside the reference range would disappear from the
    # histogram and the percentages would no longer sum to one.
    histogram_edges = np.concatenate(([-np.inf], edges, [np.inf]))
    expected_counts, _ = np.histogram(expected, bins=histogram_edges)
    actual_counts, _ = np.histogram(actual, bins=histogram_edges)
    expected_pct = expected_counts.astype(float) / len(expected)
    actual_pct = actual_counts.astype(float) / len(actual)
    expected_pct = np.clip(expected_pct, epsilon, None)
    actual_pct = np.clip(actual_pct, epsilon, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


population_stability_index = calculate_psi
calculate_population_stability_index = calculate_psi
calculate_numeric_psi = calculate_psi


def calculate_ks_statistic(reference: Any, current: Any) -> float:
    """Return the two-sample Kolmogorov-Smirnov statistic."""

    expected = _finite_values(reference, name="reference")
    actual = _finite_values(current, name="current")
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    return float(ks_2samp(expected, actual, alternative="two-sided", method="auto").statistic)


ks_statistic = calculate_ks_statistic
calculate_ks = calculate_ks_statistic
calculate_numeric_ks = calculate_ks_statistic


def calculate_ks_p_value(reference: Any, current: Any) -> float:
    """Return a KS p-value for diagnostics; the statistic drives alerting."""

    expected = _finite_values(reference, name="reference")
    actual = _finite_values(current, name="current")
    if len(expected) == 0 or len(actual) == 0:
        return 1.0
    return float(ks_2samp(expected, actual, alternative="two-sided", method="auto").pvalue)


def calculate_quantile_shifts(
    reference: Any,
    current: Any,
    *,
    quantiles: Sequence[float] = DEFAULT_DRIFT_QUANTILES,
) -> dict[str, dict[str, Optional[float]]]:
    """Return reference/current values and deltas at deterministic quantiles."""

    expected = _finite_values(reference, name="reference")
    actual = _finite_values(current, name="current")
    normalized = tuple(float(q) for q in quantiles)
    if any(q < 0.0 or q > 1.0 or not math.isfinite(q) for q in normalized):
        raise ValueError("quantiles must be finite values in [0, 1]")
    result: dict[str, dict[str, Optional[float]]] = {}
    for quantile in normalized:
        key = f"{quantile:g}"
        expected_value = float(np.quantile(expected, quantile)) if len(expected) else None
        actual_value = float(np.quantile(actual, quantile)) if len(actual) else None
        delta = None if expected_value is None or actual_value is None else actual_value - expected_value
        denominator = abs(expected_value) if expected_value is not None else 0.0
        relative = None if delta is None else (delta / denominator if denominator > 1e-12 else delta)
        result[key] = {
            "reference": _round(expected_value),
            "current": _round(actual_value),
            "delta": _round(delta),
            "relative_delta": _round(relative),
        }
    return result


quantile_shift_metrics = calculate_quantile_shifts
calculate_quantile_shift = calculate_quantile_shifts
calculate_numeric_quantile_shift = calculate_quantile_shifts


def _max_quantile_shift(shifts: Mapping[str, Mapping[str, Optional[float]]]) -> float:
    values = [
        abs(float(item["relative_delta"]))
        for item in shifts.values()
        if item.get("relative_delta") is not None
    ]
    return max(values, default=0.0)


def _category_token(value: Any) -> str:
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        return _MISSING_CATEGORY
    text = str(value)
    lowered = text.lower()
    # Preserve ordinary low-cardinality labels (``region=west``), while
    # replacing values that look like direct identifiers or contact data with
    # a stable non-reversible report token.
    identifier_marker = any(
        marker in lowered
        for marker in ("email", "phone", "address", "token", "buyer_", "seller_", "merchant_", "device_")
    )
    if "@" in text or len(text) > 64 or _UUID_LIKE.fullmatch(text) or identifier_marker:
        return f"hash:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
    return text


def calculate_category_frequencies(
    values: Iterable[Any],
    *,
    include_missing: bool = False,
    max_categories: Optional[int] = None,
) -> dict[str, float]:
    """Return a deterministic category frequency distribution.

    Missing values are excluded by default because their rate is reported by
    the dedicated missingness metric.  If ``max_categories`` is provided, the
    least frequent categories are folded into ``__OTHER__`` so reports remain
    bounded and aggregate.
    """

    series = _series(values)
    tokens = [_category_token(value) for value in series]
    if not include_missing:
        tokens = [token for token in tokens if token != _MISSING_CATEGORY]
    if not tokens:
        return {}
    counts = Counter(tokens)
    if max_categories is not None:
        if max_categories < 1:
            raise ValueError("max_categories must be positive")
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if len(ordered) > max_categories:
            kept = ordered[:max_categories]
            other_count = sum(count for _, count in ordered[max_categories:])
            counts = Counter(dict(kept))
            counts["__OTHER__"] = other_count
    denominator = float(sum(counts.values()))
    return {key: count / denominator for key, count in sorted(counts.items())}


frequency_distribution = calculate_category_frequencies
category_frequencies = calculate_category_frequencies
calculate_categorical_frequency = calculate_category_frequencies


def calculate_js_divergence(
    reference: Any,
    current: Any,
    *,
    include_missing: bool = False,
    max_categories: Optional[int] = None,
) -> float:
    """Return Jensen-Shannon divergence (not distance) in base two.

    The returned value is bounded by one.  The implementation is explicit so
    callers do not accidentally confuse SciPy's square-root JS distance with
    the divergence used for thresholds.
    """

    reference_frequencies = calculate_category_frequencies(
        reference, include_missing=include_missing, max_categories=max_categories
    )
    current_frequencies = calculate_category_frequencies(
        current, include_missing=include_missing, max_categories=max_categories
    )
    categories = sorted(set(reference_frequencies) | set(current_frequencies))
    if not categories:
        return 0.0
    p = np.asarray([reference_frequencies.get(category, 0.0) for category in categories], dtype=float)
    q = np.asarray([current_frequencies.get(category, 0.0) for category in categories], dtype=float)
    midpoint = 0.5 * (p + q)

    def _kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0.0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return float(0.5 * (_kl(p, midpoint) + _kl(q, midpoint)))


jensen_shannon_divergence = calculate_js_divergence
calculate_js = calculate_js_divergence
calculate_categorical_js = calculate_js_divergence


def calculate_js_distance(reference: Any, current: Any, **kwargs: Any) -> float:
    """Return the square-root Jensen-Shannon distance for SciPy compatibility."""

    return math.sqrt(max(0.0, calculate_js_divergence(reference, current, **kwargs)))


def calculate_unseen_category_rate(reference: Any, current: Any) -> float:
    """Return current observations whose non-missing category was not in reference."""

    reference_values = {_category_token(value) for value in _series(reference) if _category_token(value) != _MISSING_CATEGORY}
    current_values = [_category_token(value) for value in _series(current)]
    current_values = [value for value in current_values if value != _MISSING_CATEGORY]
    if not current_values:
        return 0.0
    return float(sum(value not in reference_values for value in current_values) / len(current_values))


def calculate_unseen_categories(reference: Any, current: Any) -> dict[str, Any]:
    """Return count/rate aggregates without returning category values."""

    reference_values = {_category_token(value) for value in _series(reference) if _category_token(value) != _MISSING_CATEGORY}
    current_values = [_category_token(value) for value in _series(current)]
    current_values = [value for value in current_values if value != _MISSING_CATEGORY]
    unseen_count = sum(value not in reference_values for value in current_values)
    unseen_category_count = len(set(current_values) - reference_values)
    return {
        "count": int(unseen_count),
        "unseen_observation_count": int(unseen_count),
        "unseen_category_count": int(unseen_category_count),
        "rate": _round(unseen_count / len(current_values)) if current_values else 0.0,
        "current_non_missing_count": int(len(current_values)),
    }


unseen_category_rate = calculate_unseen_category_rate
calculate_unseen_category = calculate_unseen_category_rate


def calculate_decision_volume(reference: Iterable[Any], current: Iterable[Any]) -> dict[str, Any]:
    """Return aggregate counts/frequencies and signed decision-volume change."""

    reference_values = [_category_token(value) for value in _series(reference)]
    current_values = [_category_token(value) for value in _series(current)]
    reference_counts = Counter(reference_values)
    current_counts = Counter(current_values)
    reference_total = len(reference_values)
    current_total = len(current_values)
    volume_delta = current_total - reference_total
    change_rate = None if reference_total == 0 else volume_delta / reference_total
    decisions = sorted(set(reference_counts) | set(current_counts))
    reference_frequency = {
        decision: _round(reference_counts.get(decision, 0) / reference_total) if reference_total else 0.0
        for decision in decisions
    }
    current_frequency = {
        decision: _round(current_counts.get(decision, 0) / current_total) if current_total else 0.0
        for decision in decisions
    }
    return {
        "reference_total": int(reference_total),
        "current_total": int(current_total),
        "volume_delta": int(volume_delta),
        "volume_change_rate": _round(change_rate),
        "absolute_volume_change_rate": _round(abs(change_rate)) if change_rate is not None else None,
        "reference_counts": {key: int(reference_counts[key]) for key in decisions},
        "current_counts": {key: int(current_counts[key]) for key in decisions},
        "reference_frequencies": reference_frequency,
        "current_frequencies": current_frequency,
    }


decision_volume = calculate_decision_volume


def _score_histogram(values: Any, *, bins: int = 10) -> dict[str, Any]:
    scores = _finite_values(values, name="scores")
    edges = np.linspace(0.0, 1.0, bins + 1)
    histogram_edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
    counts, _ = np.histogram(scores, bins=histogram_edges)
    total = int(len(scores))
    buckets = {
        f"{edges[index]:g}-{edges[index + 1]:g}": int(counts[index])
        for index in range(len(counts))
    }
    frequencies = {
        key: _round(count / total) if total else 0.0
        for key, count in buckets.items()
    }
    return {"counts": buckets, "frequencies": frequencies, "finite_count": total}


def calculate_score_distribution(reference: Any, current: Any) -> dict[str, Any]:
    """Return numeric distribution drift for model scores in aggregate form."""

    reference_values, reference_missing, reference_total = _numeric(reference, name="reference scores")
    current_values, current_missing, current_total = _numeric(current, name="current scores")
    shifts = calculate_quantile_shifts(reference_values, current_values)
    psi = calculate_psi(reference_values, current_values)
    ks = calculate_ks_statistic(reference_values, current_values)
    return {
        "reference_sample_count": reference_total,
        "current_sample_count": current_total,
        "reference_finite_count": int(len(reference_values)),
        "current_finite_count": int(len(current_values)),
        "reference_missing_count": reference_missing,
        "current_missing_count": current_missing,
        "score_histogram": {
            "reference": _score_histogram(reference),
            "current": _score_histogram(current),
        },
        "psi": _round(psi),
        "ks_statistic": _round(ks),
        "ks_p_value": _round(calculate_ks_p_value(reference_values, current_values)),
        "quantile_shifts": shifts,
        "max_absolute_relative_quantile_shift": _round(_max_quantile_shift(shifts)),
        "missingness": {
            "reference": _round(reference_missing / reference_total) if reference_total else 0.0,
            "current": _round(current_missing / current_total) if current_total else 0.0,
            "absolute_delta": _round(
                abs((current_missing / current_total if current_total else 0.0) - (reference_missing / reference_total if reference_total else 0.0))
            ),
        },
    }


score_distribution = calculate_score_distribution
calculate_score_drift = calculate_score_distribution


def _coerce_window(
    window: Optional[MonitoringWindowV1 | Mapping[str, Any]],
    *,
    current_count: int,
    reference_count: int = 0,
) -> MonitoringWindowV1:
    if isinstance(window, MonitoringWindowV1):
        values = window.model_dump(mode="python")
        if values.get("sample_count", 0) == 0:
            values["sample_count"] = current_count
        if values.get("reference_sample_count", 0) == 0:
            values["reference_sample_count"] = reference_count
        if values.get("current_sample_count", 0) == 0:
            values["current_sample_count"] = current_count
        return MonitoringWindowV1(**values)
    if window is not None:
        values = dict(window)
        values.setdefault("sample_count", current_count)
        values.setdefault("reference_sample_count", reference_count)
        values.setdefault("current_sample_count", current_count)
        return MonitoringWindowV1(**values)
    start = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return MonitoringWindowV1(
        window_id="default",
        window_start=start,
        window_end=start + timedelta(days=1),
        sample_count=current_count,
        reference_sample_count=reference_count,
        current_sample_count=current_count,
    )


def _alert(
    metric_name: str,
    metric_type: str,
    value: Optional[float],
    warning: Optional[float],
    critical: Optional[float],
    *,
    sample_count: int,
    min_samples: int,
    direction: Literal["higher", "lower"] = "higher",
) -> Optional[dict[str, Any]]:
    if sample_count < min_samples:
        return None
    status = classify_threshold(value, warning, critical, direction=direction)
    if status == "OK":
        return None
    action = "shadow_evaluation" if status == "CRITICAL" else "investigation"
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
            "critical drift requires shadow evaluation before any promotion decision"
            if status == "CRITICAL"
            else "warning drift requires investigation"
        ),
    ).model_dump(mode="json")


def _status(alerts: Sequence[Mapping[str, Any]]) -> str:
    if any(item.get("status") == "CRITICAL" for item in alerts):
        return "CRITICAL"
    if any(item.get("status") == "WARNING" for item in alerts):
        return "WARNING"
    return "OK"


def _feature_numeric_metrics(
    feature: str,
    reference: Any,
    current: Any,
    thresholds: DriftThresholds,
    *,
    min_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference_values, reference_missing, reference_total = _numeric(reference, name=f"{feature} reference")
    current_values, current_missing, current_total = _numeric(current, name=f"{feature} current")
    shifts = calculate_quantile_shifts(reference_values, current_values)
    psi = calculate_psi(reference_values, current_values)
    ks = calculate_ks_statistic(reference_values, current_values)
    reference_missing_rate = reference_missing / reference_total if reference_total else 0.0
    current_missing_rate = current_missing / current_total if current_total else 0.0
    missing_delta = abs(current_missing_rate - reference_missing_rate)
    quantile_shift = _max_quantile_shift(shifts)
    metric = {
        "feature": feature,
        "metric_type": "numeric",
        "reference_sample_count": reference_total,
        "current_sample_count": current_total,
        "reference_finite_count": int(len(reference_values)),
        "current_finite_count": int(len(current_values)),
        "reference_missing_count": reference_missing,
        "current_missing_count": current_missing,
        "psi": _round(psi),
        "ks_statistic": _round(ks),
        "ks_p_value": _round(calculate_ks_p_value(reference_values, current_values)),
        "quantile_shifts": shifts,
        "max_absolute_relative_quantile_shift": _round(quantile_shift),
        "missingness": {
            "reference": _round(reference_missing_rate),
            "current": _round(current_missing_rate),
            "delta": _round(current_missing_rate - reference_missing_rate),
            "absolute_delta": _round(missing_delta),
        },
    }
    alerts: list[dict[str, Any]] = []
    for result in (
        _alert(
            f"{feature}.psi",
            "numeric",
            psi,
            thresholds.warning_psi,
            thresholds.critical_psi,
            sample_count=min(len(reference_values), len(current_values)),
            min_samples=min_samples,
        ),
        _alert(
            f"{feature}.ks_statistic",
            "numeric",
            ks,
            thresholds.warning_ks,
            thresholds.critical_ks,
            sample_count=min(len(reference_values), len(current_values)),
            min_samples=min_samples,
        ),
        _alert(
            f"{feature}.quantile_shift",
            "numeric",
            quantile_shift,
            thresholds.warning_quantile_shift,
            thresholds.critical_quantile_shift,
            sample_count=min(len(reference_values), len(current_values)),
            min_samples=min_samples,
        ),
        _alert(
            f"{feature}.missingness",
            "numeric",
            missing_delta,
            thresholds.warning_missingness_delta,
            thresholds.critical_missingness_delta,
            sample_count=min(reference_total, current_total),
            min_samples=min_samples,
        ),
    ):
        if result is not None:
            alerts.append(result)
    metric["status"] = _status(alerts)
    return metric, alerts


def _feature_categorical_metrics(
    feature: str,
    reference: Any,
    current: Any,
    thresholds: DriftThresholds,
    *,
    min_samples: int,
    max_categories: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference_series = _series(reference, name=f"{feature} reference")
    current_series = _series(current, name=f"{feature} current")
    reference_frequencies = calculate_category_frequencies(reference_series, max_categories=max_categories)
    current_frequencies = calculate_category_frequencies(current_series, max_categories=max_categories)
    js = calculate_js_divergence(reference_series, current_series, max_categories=max_categories)
    unseen = calculate_unseen_categories(reference_series, current_series)
    missing = missingness_metrics(reference_series, current_series)
    metric = {
        "feature": feature,
        "metric_type": "categorical",
        "reference_sample_count": int(len(reference_series)),
        "current_sample_count": int(len(current_series)),
        "reference_frequencies": reference_frequencies,
        "current_frequencies": current_frequencies,
        "js_divergence": _round(js),
        "js_distance": _round(math.sqrt(max(0.0, js))),
        "unseen_categories": unseen,
        "missingness": missing,
    }
    alerts: list[dict[str, Any]] = []
    for result in (
        _alert(
            f"{feature}.js_divergence",
            "categorical",
            js,
            thresholds.warning_js,
            thresholds.critical_js,
            sample_count=min(len(reference_series), len(current_series)),
            min_samples=min_samples,
        ),
        _alert(
            f"{feature}.unseen_category_rate",
            "categorical",
            float(unseen["rate"]),
            thresholds.warning_unseen_category_rate,
            thresholds.critical_unseen_category_rate,
            sample_count=len(current_series),
            min_samples=min_samples,
        ),
        _alert(
            f"{feature}.missingness",
            "categorical",
            float(missing.get("absolute_delta") or 0.0),
            thresholds.warning_missingness_delta,
            thresholds.critical_missingness_delta,
            sample_count=min(len(reference_series), len(current_series)),
            min_samples=min_samples,
        ),
    ):
        if result is not None:
            alerts.append(result)
    metric["status"] = _status(alerts)
    return metric, alerts


def _report_action(alerts: Sequence[Mapping[str, Any]]) -> str:
    if any(alert.get("status") == "CRITICAL" for alert in alerts):
        return "shadow_evaluation"
    if any(alert.get("status") == "WARNING" for alert in alerts):
        return "investigation"
    return "none"


class DriftService:
    """Compute aggregate drift metrics for two tabular monitoring windows."""

    def __init__(
        self,
        thresholds: Optional[DriftThresholds] = None,
        *,
        min_samples: int = 1,
        max_categories: int = 50,
    ) -> None:
        if min_samples < 1:
            raise ValueError("min_samples must be positive")
        if max_categories < 1:
            raise ValueError("max_categories must be positive")
        self.thresholds = thresholds or DriftThresholds()
        self.min_samples = min_samples
        self.max_categories = max_categories

    def monitor(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        *,
        numeric_features: Optional[Sequence[str]] = None,
        categorical_features: Optional[Sequence[str]] = None,
        score_column: Optional[str] = None,
        decision_column: Optional[str] = None,
        window: Optional[MonitoringWindowV1 | Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Compare reference/current frames and return a ``monitoring.v1`` report."""

        if not isinstance(reference, pd.DataFrame) or not isinstance(current, pd.DataFrame):
            raise DriftInputError("reference and current must be pandas DataFrames")
        reference = reference.reset_index(drop=True)
        current = current.reset_index(drop=True)
        if numeric_features is None:
            numeric_features = [
                column
                for column in reference.columns.intersection(current.columns)
                if pd.api.types.is_numeric_dtype(reference[column])
                and pd.api.types.is_numeric_dtype(current[column])
            ]
        if categorical_features is None:
            categorical_features = [
                column
                for column in reference.columns.intersection(current.columns)
                if column not in set(numeric_features)
                and not pd.api.types.is_numeric_dtype(reference[column])
            ]
        numeric_features = tuple(numeric_features)
        categorical_features = tuple(categorical_features)
        missing_columns = [
            column
            for column in (*numeric_features, *categorical_features)
            if column not in reference.columns or column not in current.columns
        ]
        if score_column is not None and (score_column not in reference or score_column not in current):
            missing_columns.append(score_column)
        if decision_column is not None and (decision_column not in reference or decision_column not in current):
            missing_columns.append(decision_column)
        if missing_columns:
            raise DriftInputError(f"monitoring columns are missing from one or both windows: {sorted(set(missing_columns))}")

        numeric_metrics: dict[str, Any] = {}
        categorical_metrics: dict[str, Any] = {}
        alerts: list[dict[str, Any]] = []
        for feature in numeric_features:
            metric, feature_alerts = _feature_numeric_metrics(
                feature,
                reference[feature],
                current[feature],
                self.thresholds,
                min_samples=self.min_samples,
            )
            numeric_metrics[feature] = metric
            alerts.extend(feature_alerts)
        for feature in categorical_features:
            metric, feature_alerts = _feature_categorical_metrics(
                feature,
                reference[feature],
                current[feature],
                self.thresholds,
                min_samples=self.min_samples,
                max_categories=self.max_categories,
            )
            categorical_metrics[feature] = metric
            alerts.extend(feature_alerts)

        score_metrics: Optional[dict[str, Any]] = None
        if score_column is not None:
            score_metrics = calculate_score_distribution(reference[score_column], current[score_column])
            score_alerts: list[dict[str, Any]] = []
            for result in (
                _alert(
                    f"{score_column}.psi",
                    "score_distribution",
                    float(score_metrics["psi"]),
                    self.thresholds.warning_psi,
                    self.thresholds.critical_psi,
                    sample_count=min(score_metrics["reference_finite_count"], score_metrics["current_finite_count"]),
                    min_samples=self.min_samples,
                ),
                _alert(
                    f"{score_column}.ks_statistic",
                    "score_distribution",
                    float(score_metrics["ks_statistic"]),
                    self.thresholds.warning_ks,
                    self.thresholds.critical_ks,
                    sample_count=min(score_metrics["reference_finite_count"], score_metrics["current_finite_count"]),
                    min_samples=self.min_samples,
                ),
                _alert(
                    f"{score_column}.quantile_shift",
                    "score_distribution",
                    float(score_metrics["max_absolute_relative_quantile_shift"]),
                    self.thresholds.warning_quantile_shift,
                    self.thresholds.critical_quantile_shift,
                    sample_count=min(score_metrics["reference_finite_count"], score_metrics["current_finite_count"]),
                    min_samples=self.min_samples,
                ),
                _alert(
                    f"{score_column}.missingness",
                    "score_distribution",
                    float(score_metrics["missingness"]["absolute_delta"]),
                    self.thresholds.warning_missingness_delta,
                    self.thresholds.critical_missingness_delta,
                    sample_count=min(score_metrics["reference_sample_count"], score_metrics["current_sample_count"]),
                    min_samples=self.min_samples,
                ),
            ):
                if result is not None:
                    alerts.append(result)
                    score_alerts.append(result)
            score_metrics["status"] = _status(score_alerts)

        volume_metrics: Optional[dict[str, Any]] = None
        if decision_column is not None:
            volume_metrics = calculate_decision_volume(reference[decision_column], current[decision_column])
            volume_change = volume_metrics["absolute_volume_change_rate"]
            result = _alert(
                f"{decision_column}.volume",
                "decision_volume",
                float(volume_change) if volume_change is not None else None,
                self.thresholds.warning_decision_volume_change,
                self.thresholds.critical_decision_volume_change,
                sample_count=min(volume_metrics["reference_total"], volume_metrics["current_total"]),
                min_samples=self.min_samples,
            )
            if result is not None:
                alerts.append(result)
            volume_metrics["status"] = result["status"] if result is not None else "OK"

        report_window = _coerce_window(
            window,
            current_count=len(current),
            reference_count=len(reference),
        )
        metrics = {
            "numeric": numeric_metrics,
            "categorical": categorical_metrics,
            "score_distribution": score_metrics,
            "decision_volume": volume_metrics,
            "alert_count": len(alerts),
            "reference_sample_count": int(len(reference)),
            "current_sample_count": int(len(current)),
        }
        status = "CRITICAL" if any(item["status"] == "CRITICAL" for item in alerts) else (
            "WARNING" if alerts else "OK"
        )
        report = DriftReportV1(
            monitoring_window=report_window,
            status=status,
            metrics=metrics,
            alerts=[MonitoringAlertV1(**item) for item in alerts],
            recommended_action=_report_action(alerts),
        )
        return report.model_dump(mode="json")

    # Names used by batch-monitoring callers in earlier integrations.
    compute_drift = monitor
    compare = monitor


DriftMonitoringService = DriftService


def evaluate_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Functional convenience wrapper around :class:`DriftService`."""

    return DriftService().monitor(reference, current, **kwargs)


monitor_drift = evaluate_drift
compute_drift_report = evaluate_drift


__all__ = [
    "APPROVED_SUBGROUP_FIELDS",
    "DEFAULT_DRIFT_QUANTILES",
    "DriftInputError",
    "DriftMonitoringService",
    "DriftService",
    "DriftThresholds",
    "calculate_category_frequencies",
    "calculate_categorical_frequency",
    "calculate_categorical_js",
    "calculate_decision_volume",
    "calculate_js",
    "calculate_js_divergence",
    "calculate_js_distance",
    "calculate_ks",
    "calculate_ks_p_value",
    "calculate_ks_statistic",
    "calculate_numeric_ks",
    "calculate_missingness",
    "calculate_missingness_rate",
    "calculate_numeric_psi",
    "calculate_numeric_quantile_shift",
    "calculate_population_stability_index",
    "calculate_psi",
    "calculate_quantile_shift",
    "calculate_quantile_shifts",
    "calculate_score_distribution",
    "calculate_score_drift",
    "calculate_unseen_category",
    "calculate_unseen_categories",
    "calculate_unseen_category_rate",
    "category_frequencies",
    "decision_volume",
    "compute_drift_report",
    "evaluate_drift",
    "frequency_distribution",
    "jensen_shannon_divergence",
    "ks_statistic",
    "missingness_metrics",
    "monitor_drift",
    "population_stability_index",
    "quantile_shift_metrics",
    "score_distribution",
    "unseen_category_rate",
]
