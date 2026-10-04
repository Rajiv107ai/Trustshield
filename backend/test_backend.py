"""Unit and integration tests for FastAPI backend serving layer."""

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

from backend.main import app, store, _build_feature_row
from backend.schemas import TransactionScoreRequest


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


# ---------------------------------------------------------------------------
# Issue #1 — Amount mapping unit tests (pure, no model required)
# ---------------------------------------------------------------------------

# Sentinel column list: the model only needs "order_amount" for these tests.
_COLS = ["order_amount", "amount", "price_vs_base_price_ratio"]


class TestAmountMappingUnit:
    """Direct unit tests for _build_feature_row amount / order_amount logic."""

    def test_amount_only_sets_order_amount(self):
        """Supplying only `amount` must populate both `amount` and `order_amount`."""
        req = TransactionScoreRequest(amount=99.0)
        row = _build_feature_row(req, _COLS)
        assert row["amount"] == pytest.approx(99.0), "amount field should equal supplied value"
        assert row["order_amount"] == pytest.approx(99.0), "order_amount must mirror amount"

    def test_order_amount_only_sets_amount(self):
        """Supplying only `order_amount` must populate both `order_amount` and `amount`."""
        req = TransactionScoreRequest(order_amount=250.0)
        row = _build_feature_row(req, _COLS)
        assert row["order_amount"] == pytest.approx(250.0), "order_amount field should equal supplied value"
        assert row["amount"] == pytest.approx(250.0), "amount must mirror order_amount"

    def test_both_equal_passes(self):
        """Supplying equal `amount` and `order_amount` must succeed and use that value."""
        req = TransactionScoreRequest(amount=150.0, order_amount=150.0)
        row = _build_feature_row(req, _COLS)
        assert row["amount"] == pytest.approx(150.0)
        assert row["order_amount"] == pytest.approx(150.0)

    def test_conflicting_raises_http_400(self):
        """Supplying different `amount` and `order_amount` must raise HTTP 400."""
        req = TransactionScoreRequest(amount=100.0, order_amount=200.0)
        with pytest.raises(HTTPException) as exc_info:
            _build_feature_row(req, _COLS)
        assert exc_info.value.status_code == 400
        assert "Conflicting" in exc_info.value.detail

    def test_neither_supplied_defaults_to_zero(self):
        """When neither field is supplied both aliases should default to 0.0."""
        req = TransactionScoreRequest()
        row = _build_feature_row(req, _COLS)
        assert row["amount"] == pytest.approx(0.0)
        assert row["order_amount"] == pytest.approx(0.0)

    def test_order_amount_zero_explicitly_supplied(self):
        """Explicitly passing order_amount=0.0 must not be confused with 'not set'."""
        req = TransactionScoreRequest(order_amount=0.0)
        row = _build_feature_row(req, _COLS)
        # 0.0 is a valid explicit amount; result should still be 0.0 without error.
        assert row["order_amount"] == pytest.approx(0.0)
        assert row["amount"] == pytest.approx(0.0)


class TestAmountMappingEndpoint:
    """Integration tests: the /transaction/score endpoint must reflect the fix."""

    @pytest.fixture(scope="class")
    def client(self):
        with TestClient(app) as c:
            yield c

    def test_endpoint_amount_only(self, client):
        """Endpoint must not silently score with amount=0 when only order_amount supplied."""
        # Score once with amount-only
        r_amt = client.post("/transaction/score", json={
            "order_id": "AMT_ONLY",
            "amount": 300.0,
            "base_price": 300.0,
            "category_median_price": 300.0,
        })
        assert r_amt.status_code == 200

        # Score once with order_amount-only — result must match (same resolved amount)
        r_ord = client.post("/transaction/score", json={
            "order_id": "ORD_ONLY",
            "order_amount": 300.0,
            "base_price": 300.0,
            "category_median_price": 300.0,
        })
        assert r_ord.status_code == 200
        # Both requests represent the same amount, so probabilities must be identical.
        assert r_amt.json()["overall_fraud_probability"] == pytest.approx(
            r_ord.json()["overall_fraud_probability"], abs=1e-4
        ), "amount-only and order_amount-only with same value must produce identical scores"

    def test_endpoint_conflicting_amounts_returns_400(self, client):
        """Endpoint must return 400 when amount != order_amount."""
        r = client.post("/transaction/score", json={
            "order_id": "CONFLICT",
            "amount": 100.0,
            "order_amount": 200.0,
        })
        assert r.status_code == 400
        assert "Conflicting" in r.json()["detail"]
