"""Independent, delayed-label evaluation for governed data sources.

This service evaluates a prediction table supplied from private storage. It
returns aggregate metrics only; event and customer identifiers are never
included in the result. The synthetic research evaluation remains separate
from this production-data evaluation path.
"""

from __future__ import annotations

import ast
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


REQUIRED_COLUMNS = frozenset(
    {
        "event_id",
        "event_time",
        "score",
        "label",
        "label_time",
        "outcome_window_days",
        "amount",
    }
)
ALLOWED_SUBGROUP_FIELDS = frozenset(
    {
        "merchant_id",
        "merchant_age_band",
        "region",
        "account_age_band",
        "device_class",
        "listing_category",
        "payment_channel",
        "fraud_type",
    }
)
POSITIVE_LABELS = frozenset({"1", "true", "fraud", "confirmed_fraud", "positive"})
NEGATIVE_LABELS = frozenset({"0", "false", "legitimate", "cleared", "negative"})


class EvaluationInputError(ValueError):
    """Raised when an independent evaluation input is unsafe or incomplete."""


def _utc_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if values.isna().any():
        raise EvaluationInputError(f"{column} contains invalid timestamps")
    return values


def _normalize_labels(values: Iterable[Any]) -> np.ndarray:
    normalized: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            normalized.append(int(value))
            continue
        text = str(value).strip().lower()
        if text in POSITIVE_LABELS:
            normalized.append(1)
        elif text in NEGATIVE_LABELS:
            normalized.append(0)
        else:
            raise EvaluationInputError(f"unsupported label value: {value!r}")
    return np.asarray(normalized, dtype=int)


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    """Return expected calibration error using equal-width score bins."""

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    if total == 0:
        return 0.0
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (scores >= lower) & ((scores < upper) if index < bins - 1 else (scores <= upper))
        if not mask.any():
            continue
        error += float(mask.sum() / total) * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return float(error)


def _safe_metric(function, labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return round(float(function(labels, scores)), 6)


def _review_metrics(frame: pd.DataFrame, labels: np.ndarray, scores: np.ndarray, budget: float) -> dict[str, float | int | None]:
    count = max(1, int(math.ceil(len(frame) * budget)))
    order = np.argsort(-scores, kind="stable")[:count]
    selected = labels[order]
    total_positive = int(labels.sum())
    total_negative = int((labels == 0).sum())
    true_positive = int(selected.sum())
    false_positive = int(len(selected) - true_positive)
    amount = frame["amount"].to_numpy(dtype=float)
    fraud_dollars = float(amount[order][selected == 1].sum())
    return {
        "budget": float(budget),
        "review_count": count,
        "review_rate": round(float(count / len(frame)), 6),
        "recall": round(float(true_positive / total_positive), 6) if total_positive else None,
        "false_positive_rate": round(float(false_positive / total_negative), 6) if total_negative else None,
        "fraud_dollars_prevented_proxy": round(fraud_dollars, 2),
    }


def _parse_conformal_set(value: Any) -> set[int] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise EvaluationInputError("conformal_set contains invalid serialized data") from exc
    if not isinstance(parsed, (list, tuple, set)):
        raise EvaluationInputError("conformal_set must contain a list of class labels")
    return {int(item) for item in parsed}


def _aggregate_metrics(frame: pd.DataFrame, labels: np.ndarray, scores: np.ndarray, review_budgets: Sequence[float]) -> dict[str, Any]:
    conformal_coverage: float | None = None
    if "conformal_set" in frame:
        sets = [_parse_conformal_set(value) for value in frame["conformal_set"]]
        available = [item for item in sets if item is not None]
        if available:
            conformal_coverage = round(
                float(sum(int(labels[index] in item) for index, item in enumerate(sets) if item is not None) / len(available)),
                6,
            )

    return {
        "sample_count": int(len(frame)),
        "positive_count": int(labels.sum()),
        "positive_rate": round(float(labels.mean()), 6),
        "roc_auc": _safe_metric(roc_auc_score, labels, scores),
        "pr_auc": _safe_metric(average_precision_score, labels, scores),
        "ece": round(expected_calibration_error(labels, scores), 6),
        "brier_score": round(float(brier_score_loss(labels, scores)), 6),
        "conformal_coverage": conformal_coverage,
        "review_budgets": [_review_metrics(frame, labels, scores, budget) for budget in review_budgets],
    }


def _subgroup_metrics(
    frame: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    fields: Sequence[str],
    review_budget: float,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for field in fields:
        if field not in ALLOWED_SUBGROUP_FIELDS:
            raise EvaluationInputError(f"subgroup field is not approved: {field}")
        if field not in frame:
            continue
        groups: list[dict[str, Any]] = []
        for group_value, indices in frame.groupby(field, dropna=False).groups.items():
            index_array = np.asarray(list(indices), dtype=int)
            group_frame = frame.iloc[index_array]
            group_labels = labels[index_array]
            group_scores = scores[index_array]
            metrics = _aggregate_metrics(group_frame, group_labels, group_scores, [review_budget])
            groups.append({"value": str(group_value), **metrics})
        output[field] = groups
    return output


def evaluate_prediction_frame(
    frame: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
    evaluation_type: str,
    subgroup_fields: Sequence[str] = (),
    review_budgets: Sequence[float] = (0.02, 0.05),
) -> dict[str, Any]:
    """Evaluate mature predictions and return aggregate, non-row-level results."""

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise EvaluationInputError(f"missing required evaluation columns: {missing}")
    if not evaluation_type.strip():
        raise EvaluationInputError("evaluation_type must not be empty")
    if any(not 0.0 < float(budget) <= 1.0 for budget in review_budgets):
        raise EvaluationInputError("review budgets must be in the interval (0, 1]")

    work = frame.copy().reset_index(drop=True)
    event_times = _utc_series(work, "event_time")
    label_times = _utc_series(work, "label_time")
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise EvaluationInputError("as_of must be timezone-aware")
    cutoff = cutoff.tz_convert("UTC")
    if bool((event_times > cutoff).any()):
        raise EvaluationInputError("evaluation input contains events after as_of")
    maturity = event_times + pd.to_timedelta(work["outcome_window_days"], unit="D")
    mature_mask = (label_times >= maturity) & (cutoff >= maturity) & (label_times <= cutoff)
    if not bool(mature_mask.all()):
        raise EvaluationInputError("evaluation input contains labels that are not mature at as_of")

    scores = pd.to_numeric(work["score"], errors="coerce").to_numpy(dtype=float)
    amounts = pd.to_numeric(work["amount"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise EvaluationInputError("score must contain finite probabilities in [0, 1]")
    if not np.isfinite(amounts).all() or (amounts < 0.0).any():
        raise EvaluationInputError("amount must contain finite non-negative values")
    labels = _normalize_labels(work["label"])
    work["amount"] = amounts

    return {
        "evaluation_schema_version": "evaluation.v1",
        "evaluation_type": evaluation_type,
        "as_of": cutoff.isoformat(),
        "metrics": _aggregate_metrics(work, labels, scores, review_budgets),
        "subgroups": _subgroup_metrics(work, labels, scores, subgroup_fields, float(review_budgets[-1])) if subgroup_fields else {},
    }


__all__ = ["EvaluationInputError", "evaluate_prediction_frame", "expected_calibration_error"]
