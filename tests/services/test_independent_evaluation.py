"""Tests for delayed-label independent evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from backend.services.independent_evaluation import EvaluationInputError, evaluate_prediction_frame


UTC = timezone.utc


def _frame() -> pd.DataFrame:
    event = datetime(2026, 1, 1, tzinfo=UTC)
    return pd.DataFrame(
        [
            {"event_id": "e1", "event_time": event, "score": 0.95, "label": 1, "label_time": event + timedelta(days=31), "outcome_window_days": 30, "amount": 100.0, "merchant_age_band": "new", "conformal_set": "[1]"},
            {"event_id": "e2", "event_time": event + timedelta(hours=1), "score": 0.10, "label": 0, "label_time": event + timedelta(days=31), "outcome_window_days": 30, "amount": 80.0, "merchant_age_band": "established", "conformal_set": "[0]"},
            {"event_id": "e3", "event_time": event + timedelta(hours=2), "score": 0.80, "label": "confirmed_fraud", "label_time": event + timedelta(days=31), "outcome_window_days": 30, "amount": 50.0, "merchant_age_band": "new", "conformal_set": "[0, 1]"},
            {"event_id": "e4", "event_time": event + timedelta(hours=3), "score": 0.20, "label": "legitimate", "label_time": event + timedelta(days=31), "outcome_window_days": 30, "amount": 20.0, "merchant_age_band": "established", "conformal_set": "[0]"},
        ]
    )


def test_evaluation_reports_aggregate_metrics_and_subgroups():
    report = evaluate_prediction_frame(
        _frame(),
        as_of="2026-02-15T00:00:00Z",
        evaluation_type="real_historical_temporal",
        subgroup_fields=["merchant_age_band"],
    )

    assert report["evaluation_schema_version"] == "evaluation.v1"
    assert report["metrics"]["sample_count"] == 4
    assert report["metrics"]["pr_auc"] == 1.0
    assert report["metrics"]["conformal_coverage"] == 1.0
    assert set(report["subgroups"]) == {"merchant_age_band"}
    assert "event_id" not in json_text(report)


def test_evaluation_rejects_immature_labels():
    frame = _frame()
    frame.loc[0, "label_time"] = datetime(2026, 1, 15, tzinfo=UTC)
    with pytest.raises(EvaluationInputError, match="not mature"):
        evaluate_prediction_frame(frame, as_of="2026-02-15T00:00:00Z", evaluation_type="test")


def test_evaluation_rejects_events_after_cutoff():
    frame = _frame()
    frame.loc[0, "event_time"] = datetime(2026, 3, 1, tzinfo=UTC)
    with pytest.raises(EvaluationInputError, match="after as_of"):
        evaluate_prediction_frame(frame, as_of="2026-02-15T00:00:00Z", evaluation_type="test")


def json_text(value) -> str:
    """Small recursive serialization helper for the no-row-data assertion."""

    import json

    return json.dumps(value, sort_keys=True)
