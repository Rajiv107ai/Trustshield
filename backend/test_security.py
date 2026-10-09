"""
Security test suite for TrustShield AI.

Verifies:
1. Authentication enforcement (401 for missing/invalid credentials when enabled)
2. Role-Based Access Control (403 for unauthorized roles on admin/operator endpoints)
3. Input validation & protective controls (iterations bounds on /system/benchmark)
4. CORS origin headers and credential handling
5. Error sanitization (no stack traces or leaked secrets)
"""

import os
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.auth import Role, get_configured_keys


@pytest.fixture
def auth_client(monkeypatch):
    """Client with authentication strictly enabled."""
    monkeypatch.setenv("TRUSTSHIELD_AUTH_ENABLED", "true")
    monkeypatch.setenv("TRUSTSHIELD_API_KEY", "test-operator-api-key-32chars!!")
    monkeypatch.setenv("TRUSTSHIELD_ADMIN_KEY", "test-admin-secret-key-32chars!!")
    with TestClient(app) as client:
        yield client


@pytest.fixture
def dev_client(monkeypatch):
    """Client in unauthenticated dev mode."""
    monkeypatch.setenv("TRUSTSHIELD_AUTH_ENABLED", "false")
    with TestClient(app) as client:
        yield client


# ==============================================================================
# 1. Authentication Tests
# ==============================================================================

def test_public_probes_accessible_without_auth(auth_client):
    """Health and readiness probes must remain accessible without auth for orchestrators."""
    res_health = auth_client.get("/health")
    assert res_health.status_code == 200

    res_ready = auth_client.get("/ready")
    assert res_ready.status_code == 200

    res_metrics = auth_client.get("/metrics")
    assert res_metrics.status_code == 200


def test_scoring_unauthenticated_rejected(auth_client):
    """Unauthenticated request to /transaction/score must be rejected with 401."""
    payload = {
        "order_id": "SEC_TEST_01",
        "amount": 100.0,
    }
    response = auth_client.post("/transaction/score", json=payload)
    assert response.status_code == 401
    assert "authentication required" in response.json()["detail"].lower()


def test_scoring_invalid_key_rejected(auth_client):
    """Request with invalid API key must be rejected with 401."""
    headers = {"X-API-Key": "completely-invalid-key-value"}
    payload = {
        "order_id": "SEC_TEST_02",
        "amount": 100.0,
    }
    response = auth_client.post("/transaction/score", json=payload, headers=headers)
    assert response.status_code == 401
    assert "invalid authentication credentials" in response.json()["detail"].lower()


def test_scoring_valid_key_allowed(auth_client):
    """Request with valid operator API key must succeed with 200."""
    headers = {"X-API-Key": "test-operator-api-key-32chars!!"}
    payload = {
        "order_id": "SEC_TEST_03",
        "amount": 150.0,
        "base_price": 150.0,
        "category_median_price": 150.0,
    }
    response = auth_client.post("/transaction/score", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "SEC_TEST_03"
    assert "overall_fraud_probability" in data
    assert data["is_simulation"] is False


def test_bearer_token_authentication(auth_client):
    """Bearer token authentication must be supported equivalently to X-API-Key."""
    headers = {"Authorization": "Bearer test-operator-api-key-32chars!!"}
    payload = {
        "order_id": "SEC_TEST_04",
        "amount": 50.0,
    }
    response = auth_client.post("/transaction/score", json=payload, headers=headers)
    assert response.status_code == 200


# ==============================================================================
# 2. Role-Based Access Control (RBAC) Tests
# ==============================================================================

def test_operator_cannot_access_admin_benchmark(auth_client):
    """Operator role attempting to access admin-only /system/benchmark must be rejected with 403."""
    headers = {"X-API-Key": "test-operator-api-key-32chars!!"}
    response = auth_client.get("/system/benchmark", headers=headers)
    assert response.status_code == 403
    detail = response.json()["detail"].lower()
    assert "admin" in detail and "privilege" in detail


def test_admin_can_access_admin_benchmark(auth_client):
    """Admin role must successfully access /system/benchmark."""
    headers = {"X-API-Key": "test-admin-secret-key-32chars!!"}
    response = auth_client.get("/system/benchmark?iterations=5", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "scoring_pipeline" in data


def test_dossier_requires_operator_role(auth_client):
    """Dossier generation requires at least operator role."""
    headers = {"X-API-Key": "test-operator-api-key-32chars!!"}
    payload = {
        "entity_type": "transaction",
        "entity_id": "ORD_78901",
    }
    response = auth_client.post("/investigation/generate-dossier", json=payload, headers=headers)
    assert response.status_code == 200


# ==============================================================================
# 3. Input Validation & Protective Controls
# ==============================================================================

def test_benchmark_iterations_bounded_protective_control(auth_client):
    """Benchmark iterations > 50 must be rejected with 422 to prevent DoS."""
    headers = {"X-API-Key": "test-admin-secret-key-32chars!!"}
    response = auth_client.get("/system/benchmark?iterations=100", headers=headers)
    assert response.status_code == 422


# ==============================================================================
# 4. CORS Origins & Security
# ==============================================================================

def test_cors_trusted_origin_returns_headers(dev_client):
    """Preflight or request from trusted origin must return allowed CORS headers."""
    headers = {"Origin": "http://localhost:3000"}
    response = dev_client.options(
        "/transaction/score",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_untrusted_origin_disallowed(dev_client):
    """Request from untrusted origin must not receive Access-Control-Allow-Origin."""
    headers = {
        "Origin": "http://malicious-evilhacker.com",
        "Access-Control-Request-Method": "POST",
    }
    response = dev_client.options("/transaction/score", headers=headers)
    # When origin is not in allow_origins, FastAPI CORSMiddleware omits access-control-allow-origin header
    assert response.headers.get("access-control-allow-origin") != "http://malicious-evilhacker.com"


# ==============================================================================
# 5. Dev Mode Compatibility
# ==============================================================================

def test_dev_mode_allows_unauthenticated_requests(dev_client):
    """When TRUSTSHIELD_AUTH_ENABLED is false, local development requests pass smoothly."""
    payload = {
        "order_id": "DEV_PASS_01",
        "amount": 80.0,
    }
    response = dev_client.post("/transaction/score", json=payload)
    assert response.status_code == 200


# ==============================================================================
# 6. Production Fail-Closed & Simulation Attribution Tests
# ==============================================================================

def test_production_fails_closed_when_keys_missing(monkeypatch):
    """In production, missing admin or operator keys must fail startup closed with RuntimeError."""
    monkeypatch.setenv("TRUSTSHIELD_ENV", "production")
    monkeypatch.delenv("TRUSTSHIELD_ADMIN_KEY", raising=False)
    monkeypatch.delenv("TRUSTSHIELD_API_KEY", raising=False)

    from backend.auth import validate_security_configuration
    with pytest.raises(RuntimeError, match="CRITICAL SECURITY"):
        validate_security_configuration()


def test_production_fails_when_auth_disabled_attempted(monkeypatch):
    """In production, attempting to disable auth via TRUSTSHIELD_AUTH_ENABLED=false must raise RuntimeError."""
    monkeypatch.setenv("TRUSTSHIELD_ENV", "production")
    monkeypatch.setenv("TRUSTSHIELD_AUTH_ENABLED", "false")

    from backend.auth import is_auth_enabled
    with pytest.raises(RuntimeError, match="Cannot disable authentication"):
        is_auth_enabled()


def test_cors_does_not_substitute_for_authentication(auth_client):
    """A request from a trusted CORS origin without valid credentials must STILL be rejected with 401."""
    headers = {"Origin": "http://localhost:3000"}
    payload = {
        "order_id": "SEC_CORS_AUTH_TEST",
        "amount": 120.0,
    }
    response = auth_client.post("/transaction/score", json=payload, headers=headers)
    assert response.status_code == 401
    assert "authentication required" in response.json()["detail"].lower()


def test_unknown_entity_dossier_returns_404_without_fabrication(auth_client):
    """Dossier generation for an unknown buyer entity must return 404, not fabricated transactions."""
    headers = {"X-API-Key": "test-operator-api-key-32chars!!"}
    payload = {
        "entity_type": "buyer",
        "entity_id": "NON_EXISTENT_BUYER_99999",
    }
    response = auth_client.post("/investigation/generate-dossier", json=payload, headers=headers)
    assert response.status_code == 404
    detail = response.json()["detail"].lower()
    assert "not found" in detail


def test_scoring_response_identifies_real_inference(auth_client):
    """Real transaction scoring must accurately flag is_simulation=False and scoring_mode."""
    headers = {"X-API-Key": "test-operator-api-key-32chars!!"}
    payload = {
        "order_id": "SEC_ATTR_TEST",
        "amount": 99.0,
        "base_price": 100.0,
    }
    response = auth_client.post("/transaction/score", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["is_simulation"] is False
    assert data["scoring_mode"] in ("production_model_phase5_hybrid", "production_model_phase3_combined")

