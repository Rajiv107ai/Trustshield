"""Unit & integration tests for TrustShield extensions: Dossier API, SSE, Metrics, and Temporal Safety."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.main import app, store
from backend.services.neo4j_service import neo4j_service
from backend.services.redis_service import redis_service


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
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
    assert "trustshield_request_duration_seconds" in content
    assert "trustshield_decisions_total" in content
    assert "trustshield_model_inference_seconds" in content


def test_system_benchmark_endpoint(client):
    """Verify /system/benchmark computes empirical latencies without fake claims."""
    response = client.get("/system/benchmark?iterations=5")
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
    """Verify /ready probe distinguishes models, Redis, and Neo4j states."""
    response = client.get("/ready")
    assert response.status_code in {200, 503}
    data = response.json()

    assert "redis_ready" in data
    assert "neo4j_ready" in data
    assert "models_ready" in data
    assert "details" in data
    assert "redis_status" in data["details"]
    assert "neo4j_status" in data["details"]


# ==============================================================================
# 3. Temporal Graph Safety Pipeline Test (Rule #30)
# ==============================================================================

def test_pipeline_temporal_graph_safety():
    """
    Formally tests temporal graph isolation:
    Event A occurred at T1 (2025-01-01)
    Event B occurred at T2 (2025-01-10)
    Decision is evaluated at T_cutoff (2025-01-05)

    Verifies that Event B is strictly filtered out and cannot influence the decision.
    """
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    # Mock Neo4j returning records filtered by cutoff
    def mock_run(query, params):
        cutoff = params.get("cutoff_date")
        # Simulates Cypher WHERE r.first_seen_date < $cutoff_date
        records = []
        # Event A: first seen 2025-01-01
        if "2025-01-01" < cutoff:
            records.append({
                "related_buyer_id": "BUYER_EVENT_A",
                "shared_device_id": "DEV_A",
                "first_seen": "2025-01-01",
            })
        # Event B: first seen 2025-01-10
        if "2025-01-10" < cutoff:
            records.append({
                "related_buyer_id": "BUYER_EVENT_B",
                "shared_device_id": "DEV_B",
                "first_seen": "2025-01-10",
            })
        return records

    mock_session.run.side_effect = mock_run

    with patch.object(neo4j_service, "_driver", mock_driver), \
         patch.object(neo4j_service, "_connected", True), \
         patch.object(neo4j_service, "enabled", True):

        # Evaluate at T1.5 (2025-01-05)
        res = neo4j_service.find_shared_devices(
            buyer_id="BUYER_TARGET",
            decision_time="2025-01-05T12:00:00Z",
        )

        assert res.graph_source == "neo4j"
        related_ids = [n["id"] for n in res.nodes]

        # Event A must be included
        assert "BUYER_EVENT_A" in related_ids
        # Future Event B MUST be excluded
        assert "BUYER_EVENT_B" not in related_ids


# ==============================================================================
# 4. Real-Time SSE Stream Test
# ==============================================================================

@pytest.mark.anyio
async def test_sse_stream_event_generator():
    """Test SSE generator producing connected and transaction events with disconnect handling."""
    from backend.services.stream_service import transaction_event_generator
    from backend.main import score_transaction

    class MockRequest:
        def __init__(self):
            self._disconnected = False

        async def is_disconnected(self):
            return self._disconnected

    mock_req = MockRequest()
    gen = transaction_event_generator(mock_req, scoring_fn=score_transaction, interval_seconds=0.01)  # type: ignore

    # 1. First event: connected
    first_evt = await gen.__anext__()
    assert "event: connected" in first_evt
    assert "stream_active" in first_evt

    # 2. Second event: transaction with real score
    second_evt = await gen.__anext__()
    assert "event: transaction" in second_evt
    assert "order_id" in second_evt
    assert "risk_score" in second_evt
    assert "calibrated_risk" in second_evt
    assert "simulation_stream" in second_evt

    # 3. Disconnect cleanup
    mock_req._disconnected = True
    try:
        await gen.__anext__()
    except StopAsyncIteration:
        pass
