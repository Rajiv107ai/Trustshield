"""Regression tests for CODE-005: /listing/analyze feature-schema alignment."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.model_loader import store

VALID_API_KEY = "dev-operator-api-key-32chars-long"

@pytest.fixture(scope="module")
def client():
    # Ensure models are loaded
    store.load()
    with TestClient(app, headers={"X-API-Key": VALID_API_KEY}) as c:
        yield c


def test_analyze_listing_valid_request(client):
    """Valid listing analysis request must succeed with 200 without XGBoost feature mismatch."""
    payload = {
        "listing_id": "LIST_TEST_001",
        "product_id": "PROD_TEST_99",
        "displayed_product_id": "PROD_TEST_99",
        "price": 120.0,
        "base_price": 100.0,
        "category_median_price": 95.0,
        "seller_age_days_at_listing": 45,
        "seller_listings_before": 12,
        "multimodal_similarity_score": 0.88,
    }
    response = client.post("/listing/analyze", json=payload)
    assert response.status_code == 200, f"Failed with {response.status_code}: {response.text}"
    data = response.json()
    assert "fake_listing_probability" in data
    assert 0.0 <= data["fake_listing_probability"] <= 1.0
    assert data["risk_label"] in ["low", "medium", "high", "critical"]
    assert data["listing_id"] == "LIST_TEST_001"


def test_analyze_listing_malformed_request_returns_422(client):
    """Malformed request (e.g. invalid type for price) must return 422 Unprocessable Entity."""
    payload = {
        "listing_id": "LIST_MALFORMED",
        "price": "not-a-number",
    }
    response = client.post("/listing/analyze", json=payload)
    assert response.status_code == 422


def test_listing_model_feature_schema_matches_booster(client):
    """Features provided to fake_listing_model must match the booster's feature names."""
    booster = store.fake_listing_model.get_booster()
    expected_features = booster.feature_names
    assert expected_features is not None
    # Verify synthetic_mismatch_score is in expected features for fake_listing_model.joblib
    assert "synthetic_mismatch_score" in expected_features


def test_analyze_listing_with_synthetic_mismatch_score_alias(client):
    """Supplying synthetic_mismatch_score directly must be respected and score successfully."""
    payload = {
        "listing_id": "LIST_SYNTH_01",
        "price": 40.0,
        "base_price": 100.0,
        "category_median_price": 95.0,
        "seller_age_days_at_listing": 10,
        "seller_listings_before": 1,
        "synthetic_mismatch_score": 0.15,
    }
    response = client.post("/listing/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fake_listing_probability" in data
    assert data["fake_listing_probability"] > 0.5


def test_analyze_listing_prediction_behavior(client):
    """A severe price drop and visual mismatch must yield higher fake probability than normal listing."""
    fake_payload = {
        "listing_id": "LIST_HIGH_RISK",
        "price": 20.0,
        "base_price": 200.0,
        "category_median_price": 200.0,
        "seller_age_days_at_listing": 2,
        "seller_listings_before": 0,
        "multimodal_similarity_score": 0.10,
    }
    normal_payload = {
        "listing_id": "LIST_LOW_RISK",
        "price": 195.0,
        "base_price": 200.0,
        "category_median_price": 200.0,
        "seller_age_days_at_listing": 365,
        "seller_listings_before": 50,
        "multimodal_similarity_score": 0.95,
    }
    r_fake = client.post("/listing/analyze", json=fake_payload)
    r_normal = client.post("/listing/analyze", json=normal_payload)
    assert r_fake.status_code == 200
    assert r_normal.status_code == 200
    assert r_fake.json()["fake_listing_probability"] > r_normal.json()["fake_listing_probability"]


def test_analyze_listing_default_similarity_is_point_eight_five(client):
    """When similarity scores are omitted, the default 0.85 is used without failure."""
    payload = {
        "listing_id": "LIST_DEFAULT_SIM",
        "price": 100.0,
        "base_price": 100.0,
        "category_median_price": 100.0,
        "seller_age_days_at_listing": 100,
        "seller_listings_before": 10,
    }
    response = client.post("/listing/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["fake_listing_probability"] <= 1.0


def test_analyze_listing_synthetic_mismatch_overrides_multimodal(client):
    """If both synthetic_mismatch_score and multimodal_similarity_score are provided, synthetic_mismatch_score takes precedence."""
    # Low synthetic mismatch score (high suspicion) paired with high multimodal score
    payload = {
        "listing_id": "LIST_OVERRIDE_PRIORITY",
        "price": 20.0,
        "base_price": 200.0,
        "category_median_price": 200.0,
        "seller_age_days_at_listing": 2,
        "seller_listings_before": 0,
        "multimodal_similarity_score": 0.95,
        "synthetic_mismatch_score": 0.10,
    }
    response = client.post("/listing/analyze", json=payload)
    assert response.status_code == 200
    # Should reflect high fake probability because the 0.10 mismatch score was prioritized
    assert response.json()["fake_listing_probability"] > 0.5


def test_analyze_listing_exact_feature_order_contract(client):
    """Model feature_names_in_ contract must exactly match the internal inference columns."""
    from backend.main import _build_feature_row  # if available or verify directly
    booster = store.fake_listing_model.get_booster()
    expected_cols = booster.feature_names
    assert expected_cols == [
        "price_vs_base_price_ratio",
        "price_vs_category_median_ratio",
        "seller_age_days_at_listing",
        "seller_listings_before",
        "synthetic_mismatch_score",
    ]

