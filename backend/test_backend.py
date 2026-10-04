"""Unit and integration tests for FastAPI backend serving layer."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app, store


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert data["rings_loaded"] is True
    assert data["n_rings"] > 0


def test_fraud_rings_endpoint(client):
    response = client.get("/fraud-rings?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "rings" in data
    assert len(data["rings"]) <= 5
    assert data["total_rings"] > 0
    if data["rings"]:
        ring = data["rings"][0]
        assert "ring_id" in ring
        assert "avg_risk_score" in ring
        assert "risk_label" in ring


def test_fraud_rings_min_risk_filtering(client):
    response = client.get("/fraud-rings?min_risk_score=0.4&limit=10")
    assert response.status_code == 200
    data = response.json()
    for ring in data["rings"]:
        assert ring["avg_risk_score"] >= 0.4


def test_transaction_scoring_phase5(client):
    payload = {
        "order_id": "ORDER_TEST_P5",
        "buyer_id": "BUYER_000001",
        "seller_id": "SELLER_000001",
        "amount": 150.0,
        "base_price": 150.0,
        "category_median_price": 150.0,
        "order_date": "2025-09-01T12:00:00",
    }
    response = client.post("/transaction/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "ORDER_TEST_P5"
    assert 0.0 <= data["overall_fraud_probability"] <= 1.0
    assert data["risk_label"] in {"low", "medium", "high"}
    assert "phase5" in data["model_version"]


def test_transaction_scoring_phase3_fallback(client):
    was_p5_loaded = store._phase5_loaded
    try:
        store._phase5_loaded = False
        payload = {
            "order_id": "ORDER_TEST_P3",
            "buyer_id": "BUYER_000002",
            "seller_id": "SELLER_000002",
            "amount": 80.0,
            "base_price": 80.0,
            "category_median_price": 80.0,
            "order_date": "2025-09-01T12:00:00",
        }
        response = client.post("/transaction/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["order_id"] == "ORDER_TEST_P3"
        assert 0.0 <= data["overall_fraud_probability"] <= 1.0
        assert data["model_version"] == "phase3"
    finally:
        store._phase5_loaded = was_p5_loaded


def test_transaction_scoring_derives_ratios(client):
    payload = {
        "order_id": "ORDER_TEST_RATIOS",
        "buyer_id": "BUYER_000003",
        "seller_id": "SELLER_000003",
        "amount": 500.0,
        "base_price": 100.0,
        "category_median_price": 100.0,
        "order_date": "2025-09-01T12:00:00",
    }
    response = client.post("/transaction/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "ORDER_TEST_RATIOS"
    assert 0.0 <= data["overall_fraud_probability"] <= 1.0


def test_listing_analysis_multimodal(client):
    payload = {
        "listing_id": "LISTING_TEST_MM",
        "seller_id": "SELLER_TEST",
        "price": 35.0,
        "base_price": 120.0,
        "category_median_price": 100.0,
        "seller_age_days_at_listing": 10.0,
        "seller_listings_before": 1,
        "multimodal_similarity_score": 0.20,
    }
    response = client.post("/listing/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["listing_id"] == "LISTING_TEST_MM"
    assert 0.0 <= data["fake_listing_probability"] <= 1.0
    assert "investigator_narrative" in data
    narrative = data["investigator_narrative"]
    assert "summary" in narrative
    assert "key_evidence" in narrative
    assert "recommended_action" in narrative

