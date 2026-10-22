"""Tests for production ownership of derived scoring features."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend.features import enforce_server_feature_boundary, require_trusted_feature_service
from backend.schemas import TransactionScoreRequest


def test_development_legacy_feature_payload_remains_accepted(monkeypatch):
    monkeypatch.setenv("TRUSTSHIELD_ENV", "development")
    monkeypatch.delenv("TRUSTSHIELD_STRICT_SERVER_FEATURES", raising=False)
    request = TransactionScoreRequest(order_id="DEV", amount=10.0, share_degree=5.0)

    enforce_server_feature_boundary(request)


@pytest.mark.parametrize("field", [
    "buyer_orders_before",
    "buyer_return_rate_before",
    "price_vs_base_price_ratio",
    "share_degree",
    "buyer_pagerank",
    "multimodal_similarity_score",
])
def test_production_rejects_client_derived_features(monkeypatch, field):
    monkeypatch.setenv("TRUSTSHIELD_ENV", "production")
    monkeypatch.delenv("TRUSTSHIELD_STRICT_SERVER_FEATURES", raising=False)
    request = TransactionScoreRequest(order_id="PROD", amount=10.0, **{field: 1.0})

    with pytest.raises(HTTPException) as exc_info:
        enforce_server_feature_boundary(request)

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["code"] == "CLIENT_DERIVED_FEATURES_NOT_ALLOWED"
    assert field in exc_info.value.detail["fields"]


def test_explicit_strict_flag_rejects_derived_features_in_development(monkeypatch):
    monkeypatch.setenv("TRUSTSHIELD_ENV", "development")
    monkeypatch.setenv("TRUSTSHIELD_STRICT_SERVER_FEATURES", "true")
    request = TransactionScoreRequest(order_id="STRICT", amount=10.0, seller_age_days=2.0)

    with pytest.raises(HTTPException):
        enforce_server_feature_boundary(request)


def test_production_accepts_raw_event_assertions(monkeypatch):
    monkeypatch.setenv("TRUSTSHIELD_ENV", "production")
    monkeypatch.delenv("TRUSTSHIELD_STRICT_SERVER_FEATURES", raising=False)
    request = TransactionScoreRequest(
        order_id="RAW_ONLY",
        buyer_id="buyer_token",
        seller_id="seller_token",
        listing_id="listing_token",
        amount=10.0,
        base_price=12.0,
        category_median_price=15.0,
        order_date="2026-10-20T10:00:00+00:00",
    )

    enforce_server_feature_boundary(request)


def test_production_scoring_boundary_requires_governed_feature_service(monkeypatch):
    monkeypatch.setenv("TRUSTSHIELD_ENV", "production")
    monkeypatch.delenv("TRUSTSHIELD_FEATURE_SERVICE_READY", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        require_trusted_feature_service()

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "TRUSTED_FEATURE_SERVICE_UNAVAILABLE"


def test_development_can_use_synthetic_compatibility_path(monkeypatch):
    monkeypatch.setenv("TRUSTSHIELD_ENV", "development")
    monkeypatch.delenv("TRUSTSHIELD_FEATURE_SERVICE_READY", raising=False)

    require_trusted_feature_service()
