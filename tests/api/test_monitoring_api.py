"""Unit tests for the monitoring router."""

from fastapi.testclient import TestClient
import pytest

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get_drift_report(client):
    response = client.get("/monitoring/drift?refresh=true")
    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == "monitoring.v1"
    assert data["report_type"] == "drift"
    assert data["status"] in ("OK", "WARNING", "CRITICAL")
    assert "metrics" in data
    assert "monitoring_window" in data


def test_get_calibration_report(client):
    response = client.get("/monitoring/calibration?refresh=true")
    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == "monitoring.v1"
    assert data["report_type"] == "calibration"
    assert data["status"] in ("OK", "WARNING", "CRITICAL")
    assert "ece" in data["metrics"]
    assert "brier_score" in data["metrics"]


def test_get_monitoring_overview(client):
    response = client.get("/monitoring/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("OK", "WARNING", "CRITICAL")
    assert "drift" in data
    assert "calibration" in data
    assert "active_alerts" in data
    assert isinstance(data["active_alerts_count"], int)


def test_list_monitoring_reports(client):
    # Ensure at least one report exists
    client.get("/monitoring/drift")
    response = client.get("/monitoring/reports?limit=5")
    assert response.status_code == 200
    reports = response.json()
    assert isinstance(reports, list)
    assert len(reports) >= 1
