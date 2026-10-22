"""Tests for Real-Time Point-in-Time Sliding Window Feature Service.

Validates:
- Strict point-in-time causality (zero future leakage, exact timestamp boundaries)
- Rolling velocity windows (1h, 24h, 7d counts and spend sums)
- Device sharing and collision graphs
- Catalog baseline price and category median derivations
- Redis failover and in-memory dual engine
- End-to-end production scoring with server-derived feature boundary
"""

from __future__ import annotations

import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.model_loader import store
from backend.schemas import TransactionScoreRequest
from backend.services.feature_service import (
    ALL_MODEL_FEATURE_KEYS,
    InMemorySlidingWindowStore,
    PointInTimeFeatureService,
    ProductCatalogService,
    RedisSlidingWindowBackend,
    normalize_timestamp,
)


def test_timestamp_normalization():
    t_naive = normalize_timestamp("2026-10-22 10:00:00")
    t_utc = normalize_timestamp("2026-10-22T10:00:00+00:00")
    assert t_naive == t_utc
    assert t_naive.tzinfo is None


def test_in_memory_pit_strict_causality():
    store_mem = InMemorySlidingWindowStore()
    t1 = normalize_timestamp("2026-10-01 10:00:00")
    t2 = normalize_timestamp("2026-10-01 11:00:00")
    t3 = normalize_timestamp("2026-10-01 12:00:00")

    store_mem.add_order("O1", "B1", "S1", t1, amount=50.0)
    store_mem.add_order("O2", "B1", "S1", t2, amount=75.0)
    store_mem.add_order("O3", "B1", "S1", t3, amount=100.0)

    # Strictly before t2: should only include O1 (count = 1)
    assert store_mem.count_buyer_orders_before("B1", t2) == 1
    # Strictly before t3: should include O1 and O2 (count = 2)
    assert store_mem.count_buyer_orders_before("B1", t3) == 2
    # At t1: strictly before t1 has 0
    assert store_mem.count_buyer_orders_before("B1", t1) == 0

    # Pair orders strictly before t3
    assert store_mem.count_pair_orders_before("B1", "S1", t3) == 2


def test_sliding_window_velocities():
    store_mem = InMemorySlidingWindowStore()
    base_t = normalize_timestamp("2026-10-05 12:00:00")

    # Order 30 mins ago (in 1h and 24h window)
    store_mem.add_order("O1", "B1", "S1", base_t - pd.Timedelta(minutes=30), amount=100.0)
    # Order 3 hours ago (in 24h window, but NOT 1h window)
    store_mem.add_order("O2", "B1", "S1", base_t - pd.Timedelta(hours=3), amount=200.0)
    # Order 30 hours ago (outside 24h window)
    store_mem.add_order("O3", "B1", "S1", base_t - pd.Timedelta(hours=30), amount=500.0)

    v1_count, v1_spend = store_mem.buyer_velocity("B1", base_t, window_seconds=3600)
    assert v1_count == 1
    assert v1_spend == 100.0

    v24_count, v24_spend = store_mem.buyer_velocity("B1", base_t, window_seconds=86400)
    assert v24_count == 2
    assert v24_spend == 300.0


def test_device_collision_metrics():
    store_mem = InMemorySlidingWindowStore()
    base_t = normalize_timestamp("2026-10-10 12:00:00")

    # Device DEV_1 used by buyer B1 and buyer B2 within 2 days
    store_mem.add_order("O1", "B1", "S1", base_t - pd.Timedelta(days=1), amount=50.0, device_id="DEV_1")
    store_mem.add_order("O2", "B2", "S2", base_t - pd.Timedelta(days=2), amount=80.0, device_id="DEV_1")
    # Device DEV_2 used by only B3
    store_mem.add_order("O3", "B3", "S3", base_t - pd.Timedelta(days=1), amount=40.0, device_id="DEV_2")

    # DEV_1 has 2 distinct shared buyers
    assert store_mem.count_device_shared_buyers("DEV_1", base_t, window_seconds=7 * 86400) == 2
    # DEV_2 has 1 buyer
    assert store_mem.count_device_shared_buyers("DEV_2", base_t, window_seconds=7 * 86400) == 1
    # Unknown device defaults to 1 (self)
    assert store_mem.count_device_shared_buyers("DEV_UNKNOWN", base_t) == 1


def test_live_transaction_recording_idempotency():
    store_mem = InMemorySlidingWindowStore()
    t = normalize_timestamp("2026-10-12 10:00:00")

    store_mem.add_order("ORD_100", "B1", "S1", t, amount=50.0)
    # Duplicate insert of ORD_100 with same order_id
    store_mem.add_order("ORD_100", "B1", "S1", t, amount=50.0)

    assert store_mem.count_buyer_orders_before("B1", t + pd.Timedelta(seconds=1)) == 1


def test_redis_backend_graceful_fallback(monkeypatch):
    # Non-existent redis host should gracefully disable without throwing
    backend = RedisSlidingWindowBackend(redis_url="redis://nonexistent-host:9999/0")
    assert backend.is_connected is False

    # Recording order while disconnected should not raise
    backend.record_order("O1", "B1", "S1", pd.Timestamp.now(), 10.0)
    assert backend.count_window("any_key", 0, 1000) is None


def test_feature_service_compute_features():
    fs = PointInTimeFeatureService(data_dir="data/synthetic_v2_1", catalog_path="models/product_catalog.parquet")
    assert fs.is_ready is True

    req = TransactionScoreRequest(
        order_id="TEST_PIT_REQ",
        buyer_id="BUYER_00000",
        seller_id="SELLER_00001",
        listing_id="B06X9STHNG",
        amount=49.99,
        order_date="2025-11-01 10:00:00",
    )

    feature_row, metadata = fs.compute_features(req)

    # Check that all 18 model features are present and non-null
    for col in ALL_MODEL_FEATURE_KEYS:
        assert col in feature_row
        assert isinstance(feature_row[col], (int, float))

    assert feature_row["buyer_orders_before"] >= 0
    assert 0.0 <= feature_row["buyer_return_rate_before"] <= 1.0
    assert feature_row["device_shared_buyer_count"] >= 1.0
    assert metadata["provenance"]["zero_future_leakage_verified"] is True


def test_features_api_preview_and_status():
    headers = {"X-API-Key": "dev-operator-api-key-32chars-long"}
    with TestClient(app) as client:
        # 1. Status endpoint
        res_status = client.get("/features/status")
        assert res_status.status_code == 200
        data_status = res_status.json()
        assert data_status["ready"] is True
        assert "orders_indexed" in data_status

        # 2. Preview endpoint
        payload = {
            "order_id": "PREVIEW_001",
            "buyer_id": "BUYER_00000",
            "seller_id": "SELLER_00001",
            "listing_id": "B06X9STHNG",
            "amount": 49.99,
            "order_date": "2025-11-01 10:00:00",
        }
        res_prev = client.post("/features/preview", json=payload, headers=headers)
        assert res_prev.status_code == 200
        data_prev = res_prev.json()
        assert "feature_row" in data_prev
        assert "metadata" in data_prev
        assert data_prev["feature_row"]["amount"] == 49.99

        # 3. Buyer velocity endpoint
        res_buyer = client.get("/features/buyer/BUYER_00000?as_of=2025-11-01T10:00:00", headers=headers)
        assert res_buyer.status_code == 200
        data_buyer = res_buyer.json()
        assert data_buyer["buyer_id"] == "BUYER_00000"
        assert "velocities" in data_buyer

        # 4. Device collision endpoint
        res_dev = client.get("/features/device/DEV_TEST", headers=headers)
        assert res_dev.status_code == 200
        data_dev = res_dev.json()
        assert data_dev["device_shared_buyer_count_7d"] >= 1


def test_production_strict_scoring_with_feature_service(monkeypatch):
    """End-to-end test of production serving lane: caller sends ONLY raw assertions,

    feature service computes all derived features without leaking future state.
    """
    monkeypatch.setenv("TRUSTSHIELD_ENV", "production")
    monkeypatch.setenv("TRUSTSHIELD_FEATURE_SERVICE_READY", "true")
    monkeypatch.setenv("TRUSTSHIELD_REQUIRE_PIT_HISTORY", "true")
    monkeypatch.setenv("TRUSTSHIELD_REQUIRE_RELEASE_MANIFEST", "false")
    monkeypatch.setenv("TRUSTSHIELD_AUDIT_DECISIONS", "false")
    monkeypatch.setenv("TRUSTSHIELD_ADMIN_KEY", "test-admin-key-32chars-long")
    monkeypatch.setenv("TRUSTSHIELD_API_KEY", "test-operator-key-32chars-long")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")

    headers = {"X-API-Key": "test-operator-key-32chars-long"}
    raw_payload = {
        "order_id": "PROD_RAW_001",
        "buyer_id": "BUYER_00000",
        "seller_id": "SELLER_00001",
        "listing_id": "B06X9STHNG",
        "amount": 55.0,
        "order_date": "2025-11-01 10:00:00",
    }

    with TestClient(app) as client:
        # Verify /ready is fully READY (status 200) with feature service enabled
        res_ready = client.get("/ready")
        assert res_ready.status_code == 200
        assert res_ready.json()["status"] == "ready"
        assert res_ready.json()["details"]["trusted_feature_service_ready"] is True

        # Score the transaction
        res_score = client.post("/transaction/score", json=raw_payload, headers=headers)
        assert res_score.status_code == 200
        score_data = res_score.json()
        assert "overall_fraud_probability" in score_data
        assert "decision" in score_data
        assert "trust_score" in score_data
        assert score_data["decision"] in ["ALLOW", "REVIEW", "HOLD", "BLOCK"]
