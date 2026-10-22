"""Readiness behavior for trusted point-in-time history."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from backend.model_loader import store


def _set_loaded_without_pit(monkeypatch):
    """Keep the probe focused on readiness policy instead of model loading."""

    monkeypatch.setattr(store, "_loaded", True)
    monkeypatch.setattr(store, "combined_graph_model", object())
    monkeypatch.setattr(store, "rings_df", pd.DataFrame())
    monkeypatch.setattr(store, "transaction_history", None)
    monkeypatch.setattr(store, "load", lambda: None)


def test_production_readiness_fails_closed_without_pit_history(monkeypatch):
    _set_loaded_without_pit(monkeypatch)
    monkeypatch.setenv("TRUSTSHIELD_ENV", "production")
    monkeypatch.setenv("TRUSTSHIELD_REQUIRE_PIT_HISTORY", "true")
    monkeypatch.setenv("TRUSTSHIELD_REQUIRE_RELEASE_MANIFEST", "false")
    monkeypatch.setenv("TRUSTSHIELD_ADMIN_KEY", "test-admin-key-32chars-long")
    monkeypatch.setenv("TRUSTSHIELD_API_KEY", "test-operator-key-32chars-long")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["pit_history_ready"] is False
    assert payload["details"]["pit_history_required"] is True


def test_development_readiness_reports_degraded_without_pit_history(monkeypatch):
    _set_loaded_without_pit(monkeypatch)
    monkeypatch.setenv("TRUSTSHIELD_ENV", "development")
    monkeypatch.setenv("TRUSTSHIELD_REQUIRE_PIT_HISTORY", "false")
    monkeypatch.setenv("TRUSTSHIELD_REQUIRE_RELEASE_MANIFEST", "false")

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["pit_history_ready"] is False
    assert payload["details"]["pit_history_required"] is False
