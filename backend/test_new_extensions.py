"""Unit & integration tests for TrustShield extensions: Dossier API, SSE, Metrics, and Temporal Safety."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.main import app, store


VALID_API_KEY = "dev-operator-api-key-32chars-long"


@pytest.fixture(scope="module")
def client():
    with TestClient(app, headers={"X-API-Key": VALID_API_KEY}) as test_client:
        yield test_client


# ==============================================================================
# 1. Forensic Dossier API Tests
# ==============================================================================

def test_generate_dossier_known_preset(client):
    """Test generating a forensic dossier for a known preset transaction."""
    payload = {
        "entity_type": "transaction",
        "entity_id": "ORD_78901",
        "include_graph_evidence": True,
    }
    response = client.post("/investigation/generate-dossier", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["case_id"] == "CASE_ORD_78901"
    assert data["entity_id"] == "ORD_78901"
    assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert data["decision"] in {"ALLOW", "REVIEW", "HOLD", "BLOCK"}
    assert data["grounding_verification_passed"] is True

    # Validate distinct evidence separation
    assert "observed_evidence" in data
    assert "model_inference" in data
    assert "graph_findings" in data
    assert "recommendation" in data
    assert "timeline" in data

    obs = data["observed_evidence"]
    assert obs["transaction_amount"] == 890.0
    assert len(obs["verified_facts"]) > 0

    inf = data["model_inference"]
    assert 0.0 <= inf["calibrated_risk_score"] <= 1.0
    assert inf["operational_decision"] == data["decision"]
    assert inf["conformal_prediction_set"] in {"{0}", "{1}", "{0, 1}"}

    graph = data["graph_findings"]
    assert graph["graph_source"] in {"neo4j", "disk_artifact", "unavailable"}


def test_generate_dossier_with_explicit_context(client):
    """Test generating a dossier by supplying explicit transaction features."""
    payload = {
        "entity_type": "transaction",
        "entity_id": "ORD_CUSTOM_999",
        "transaction_data": {
            "order_id": "ORD_CUSTOM_999",
            "buyer_id": "BUYER_CUSTOM_01",
            "seller_id": "SELLER_CUSTOM_01",
            "amount": 1250.0,
            "base_price": 500.0,
            "category_median_price": 450.0,
            "buyer_orders_before": 1,
            "buyer_returns_before": 0,
            "buyer_age_days": 1.0,
            "device_shared_buyer_count": 8.0,
            "share_degree": 6.0,
        },
        "include_graph_evidence": True,
    }
    response = client.post("/investigation/generate-dossier", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["case_id"] == "CASE_ORD_CUSTOM_999"
    assert data["observed_evidence"]["transaction_amount"] == 1250.0
    assert data["observed_evidence"]["hardware_collision_detected"] is True


def test_generate_dossier_unknown_entity_404(client):
    """Verify unknown entity ID with no explicit payload returns 404, not 500."""
    payload = {
        "entity_type": "transaction",
        "entity_id": "ORD_NONEXISTENT_000000",
    }
    response = client.post("/investigation/generate-dossier", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_generate_dossier_invalid_entity_type_422(client):
    """Verify unsupported entity type returns 422 unprocessable entity."""
    payload = {
        "entity_type": "invalid_entity_kind",
        "entity_id": "ORD_123",
    }
    response = client.post("/investigation/generate-dossier", json=payload)
    assert response.status_code == 422


# ==============================================================================
# 2. Prometheus Metrics & Observability Tests
# ==============================================================================

def test_metrics_endpoint(client):
    """Verify /metrics returns Prometheus format with required operational metrics."""
    # First make a scoring request to ensure decision counters increment
    client.post("/transaction/score", json={
        "order_id": "ORD_METRIC_PROBE",
        "buyer_id": "BUYER_000001",
        "seller_id": "SELLER_000001",
        "amount": 100.0,
        "base_price": 100.0,
        "category_median_price": 100.0,
    })

    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text

    # Core metrics must exist in scrape output
    assert "trustshield_requests_total" in content
    assert "trustshield_decisions_total" in content


def test_system_benchmark_endpoint(client):
    """Verify /system/benchmark computes empirical latencies without fake claims."""
    admin_headers = {"X-API-Key": "dev-admin-secret-key-32chars-long!"}
    response = client.get("/system/benchmark?iterations=5", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()

    assert "scoring_pipeline" in data
    assert "redis_feature_store" in data
    assert "neo4j_graph_engine" in data

    scoring = data["scoring_pipeline"]
    assert scoring["count"] == 5.0
    assert scoring["p50_ms"] > 0.0
    assert scoring["p95_ms"] >= scoring["p50_ms"]


def test_readiness_probe_exposes_infrastructure(client):
    """Verify /ready probe reports core models state."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()

    assert "models_ready" in data
    assert "details" in data
