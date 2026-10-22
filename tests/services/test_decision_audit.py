"""Focused tests for decision audit gating without running model inference."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

import backend.routers.scoring as scoring
from backend.schemas import TransactionScoreRequest, TransactionScoreResponse


def _response() -> TransactionScoreResponse:
    return TransactionScoreResponse(
        order_id="order_token_001",
        overall_fraud_probability=0.42,
        raw_fraud_probability=0.4,
        calibrated_fraud_probability=0.42,
        risk_label="medium",
        model_used="phase3",
        model_version="phase3",
        decision="REVIEW",
        trust_score=58.0,
        confidence=0.7,
        cold_start=False,
        model_disagreement=0.1,
        reason_codes=["REVIEW_THRESHOLD"],
        scoring_mode="production_model",
        is_simulation=False,
        graph_features_source="server",
    )


class _Audit:
    def __init__(self):
        self.calls = []

    def append_decision(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_decision_audit_writes_allowlisted_snapshot(monkeypatch):
    audit = _Audit()
    monkeypatch.setenv("TRUSTSHIELD_AUDIT_DECISIONS", "true")
    monkeypatch.setattr(scoring, "_decision_audit_initialized", True)
    monkeypatch.setattr(scoring, "_decision_audit_service", audit)

    scoring._record_decision_audit(
        TransactionScoreRequest(order_id="order_token_001", amount=10.0),
        _response(),
    )

    assert len(audit.calls) == 1
    assert audit.calls[0][1]["metadata"]["source"] == "transaction_score"
    assert "api_key" not in audit.calls[0][1]["metadata"]


def test_required_decision_audit_failure_fails_closed(monkeypatch):
    monkeypatch.setenv("TRUSTSHIELD_AUDIT_DECISIONS", "true")
    monkeypatch.setattr(scoring, "_decision_audit_initialized", True)
    monkeypatch.setattr(scoring, "_decision_audit_service", RuntimeError("database unavailable"))

    with pytest.raises(HTTPException) as exc_info:
        scoring._record_decision_audit(
            TransactionScoreRequest(order_id="order_token_001", amount=10.0),
            _response(),
        )

    assert exc_info.value.status_code == 503
