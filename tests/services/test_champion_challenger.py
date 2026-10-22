"""Tests for Champion-Challenger Shadow Scoring & Conformal A/B Deployment.

Validates:
- Real-time shadow scoring across Champion (RF) and Challenger (XGBoost)
- Conformal prediction set comparisons ({0}, {1}, or ambiguous {0, 1})
- Deterministic A/B hash traffic routing and Canary circuit breakers
- REST API endpoints /models/champion-challenger/status, /configure, /score-dual, /disagreements
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.champion_challenger_service import (
    ChampionChallengerService,
    champion_challenger_service,
)
from backend.schemas import ChampionChallengerConfigureRequest


def test_service_initialization_and_status():
    champion_challenger_service.reset_metrics()
    status = champion_challenger_service.get_status()
    assert "deployment_mode" in status
    assert status["deployment_mode"] in ("SHADOW", "AB_SPLIT", "CANARY")
    assert "champion_spec" in status
    assert "challenger_spec" in status
    assert status["champion_spec"]["id"] == "phase3_champion_rf"
    assert "decision_concordance_rate" in status
    assert "conformal_concordance_rate" in status


def test_dual_model_scoring_and_conformal():
    champion_challenger_service.reset_metrics()
    feature_row = {
        "amount": 450.0,
        "price_vs_base_price_ratio": 1.05,
        "price_vs_category_median_ratio": 0.98,
        "seller_age_days": 120.0,
        "seller_total_listings_before": 25,
        "buyer_age_days": 45.0,
        "buyer_orders_before": 12,
        "buyer_returns_before": 1,
        "buyer_return_rate_before": 0.08,
        "device_shared_buyer_count": 1,
        "share_degree": 0.0,
        "share_component_size": 1.0,
        "buyer_seller_degree": 2.0,
        "buyer_pagerank": 0.001,
        "seller_buyer_degree": 15.0,
        "seller_pagerank": 0.005,
        "seller_buyer_concentration_hhi": 0.12,
        "buyer_seller_edge_weight_before": 2.0,
    }

    comparison = champion_challenger_service.score_dual(feature_row, order_id="TEST_ORD_001")
    assert comparison["order_id"] == "TEST_ORD_001"
    assert "champion" in comparison
    assert "challenger" in comparison
    assert comparison["champion"]["algorithm"] == "RandomForestClassifier"
    assert comparison["challenger"]["algorithm"] == "XGBClassifier"
    assert 0.0 <= comparison["champion"]["calibrated_score"] <= 1.0
    assert 0.0 <= comparison["challenger"]["calibrated_score"] <= 1.0
    assert comparison["champion"]["decision"] in ("ALLOW", "REVIEW", "HOLD", "BLOCK")
    assert comparison["challenger"]["decision"] in ("ALLOW", "REVIEW", "HOLD", "BLOCK")

    # Conformal Prediction Sets
    champ_set = comparison["champion"]["conformal_prediction_set"]
    chall_set = comparison["challenger"]["conformal_prediction_set"]
    assert isinstance(champ_set, list) and len(champ_set) in (1, 2)
    assert isinstance(chall_set, list) and len(chall_set) in (1, 2)
    assert all(c in (0, 1) for c in champ_set)
    assert all(c in (0, 1) for c in chall_set)

    assert "risk_delta" in comparison
    assert "conformal_divergence" in comparison
    assert comparison["routed_model"] in ("CHAMPION", "CHALLENGER")


def test_deployment_mode_configuration():
    # Configure to AB_SPLIT with 50% split
    status = champion_challenger_service.configure("AB_SPLIT", traffic_split_pct=50.0)
    assert status["deployment_mode"] == "AB_SPLIT"
    assert status["traffic_split_pct"] == 50.0

    # Configure back to SHADOW mode
    status_shadow = champion_challenger_service.configure("SHADOW", traffic_split_pct=0.0)
    assert status_shadow["deployment_mode"] == "SHADOW"

    with pytest.raises(ValueError):
        champion_challenger_service.configure("INVALID_MODE")


def test_circuit_breaker_detection():
    service = ChampionChallengerService()
    service.reset_metrics()
    service.configure("CANARY", traffic_split_pct=10.0)

    # Inject 25 high-disagreement evaluations
    feature_row = {"amount": 100.0}
    for idx in range(25):
        comp = service.score_dual(feature_row, order_id=f"CB_ORD_{idx}")
        # Artificially modify the stored comparison to force divergence
        comp["decision_matches"] = False

    status = service.get_status()
    assert status["circuit_breaker_tripped"] is True
    assert "exceeded safety threshold" in (status["circuit_breaker_reason"] or "")


def test_api_status_endpoint():
    client = TestClient(app)
    res = client.get("/models/champion-challenger/status")
    assert res.status_code == 200
    data = res.json()
    assert "deployment_mode" in data
    assert "champion_spec" in data
    assert "challenger_spec" in data
    assert "decision_concordance_rate" in data


def test_api_score_dual_endpoint():
    client = TestClient(app)
    headers = {"X-API-Key": "dev-operator-api-key-32chars-long"}
    res = client.post(
        "/models/champion-challenger/score-dual",
        headers=headers,
        json={
            "order_id": "ORD_API_DUAL_99",
            "amount": 250.0,
            "buyer_id": "BUYER_TEST",
            "seller_id": "SELLER_TEST",
            "buyer_orders_before": 8,
            "buyer_returns_before": 0,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["order_id"] == "ORD_API_DUAL_99"
    assert "champion" in data
    assert "challenger" in data
    assert data["champion"]["decision"] in ("ALLOW", "REVIEW", "HOLD", "BLOCK")
    assert "conformal_prediction_set" in data["champion"]
    assert "conformal_prediction_set" in data["challenger"]


def test_api_configure_and_disagreements():
    client = TestClient(app)
    headers = {"X-API-Key": "dev-operator-api-key-32chars-long"}

    # Configure
    conf_res = client.post(
        "/models/champion-challenger/configure",
        headers=headers,
        json={"deployment_mode": "CANARY", "traffic_split_pct": 5.0},
    )
    assert conf_res.status_code == 200
    assert conf_res.json()["deployment_mode"] == "CANARY"

    # Query disagreements
    dis_res = client.get(
        "/models/champion-challenger/disagreements",
        headers=headers,
    )
    assert dis_res.status_code == 200
    assert isinstance(dis_res.json(), list)

    # Reset metrics
    reset_res = client.post(
        "/models/champion-challenger/reset-metrics",
        headers=headers,
    )
    assert reset_res.status_code == 200
    assert reset_res.json()["total_evaluations"] == 0
