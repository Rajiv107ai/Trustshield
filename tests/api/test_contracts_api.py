"""Contract validation route tests; payloads are synthetic and non-sensitive."""

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from backend.main import app


KEY = "dev-operator-api-key-32chars-long"
UTC = timezone.utc


def _event():
    return {
        "event_id": "evt_route_001",
        "event_time": "2026-10-20T10:00:00Z",
        "order_id": "order_token_001",
        "buyer_id": "buyer_token_001",
        "seller_id": "seller_token_001",
        "listing_id": "listing_token_001",
        "amount": 12.5,
        "currency": "INR",
        "source": "marketplace_test",
        "schema_version": "events.v1",
    }


def test_event_contract_route_validates_without_echoing_payload():
    with TestClient(app, headers={"X-API-Key": KEY}) as client:
        response = client.post("/contracts/events/validate", json=_event())
    assert response.status_code == 200
    assert response.json() == {"valid": True, "schema_version": "events.v1", "mature": None, "validation_errors": []}
    assert "order_token_001" not in response.text


def test_event_contract_route_rejects_sensitive_extra_field():
    payload = {**_event(), "payment_credentials": "should-not-cross-boundary"}
    with TestClient(app, headers={"X-API-Key": KEY}) as client:
        response = client.post("/contracts/events/validate", json=payload)
    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert "payment_credentials" not in response.text


def test_label_contract_route_reports_maturity():
    payload = {
        "event_id": "evt_route_001",
        "event_time": "2026-09-01T10:00:00Z",
        "label": "confirmed_fraud",
        "label_time": "2026-10-20T10:00:00Z",
        "label_source": "chargeback",
        "label_confidence": 0.99,
        "outcome_window_days": 30,
    }
    with TestClient(app, headers={"X-API-Key": KEY}) as client:
        response = client.post("/contracts/labels/validate", json=payload)
    assert response.status_code == 200
    assert response.json()["valid"] is True
    assert response.json()["mature"] is True
