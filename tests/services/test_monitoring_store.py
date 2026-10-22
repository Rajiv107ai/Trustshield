"""Durable aggregate monitoring report storage tests."""

from datetime import datetime, timezone

import pytest

from backend.services.monitoring_store import MonitoringStore, MonitoringStoreConfigurationError


def _report():
    return {
        "schema_version": "monitoring.v1",
        "report_type": "drift",
        "monitoring_window": {
            "schema_version": "monitoring.v1",
            "window_id": "window_001",
            "window_start": "2026-10-19T00:00:00Z",
            "window_end": "2026-10-20T00:00:00Z",
        },
        "metrics": {"psi": 0.02},
        "alerts": [],
        "recommended_action": "none",
        "automatic_retraining": False,
        "aggregate_only": True,
    }


def test_monitoring_store_round_trip_is_hash_checked(tmp_path):
    store = MonitoringStore(tmp_path / "monitoring.sqlite3")
    report_id = store.save(_report())
    loaded = store.get(report_id)
    store.close()

    assert loaded is not None
    assert loaded["metrics"]["psi"] == 0.02


def test_monitoring_store_latest_and_list_reports(tmp_path):
    store = MonitoringStore(tmp_path / "monitoring.sqlite3")
    rep1 = _report()
    rep1["monitoring_window"]["window_id"] = "win1"
    rep1["metrics"]["psi"] = 0.05
    id1 = store.save(rep1)

    rep2 = _report()
    rep2["monitoring_window"]["window_id"] = "win2"
    rep2["report_type"] = "calibration"
    rep2["metrics"]["ece"] = 0.03
    id2 = store.save(rep2)

    latest_drift = store.latest("drift")
    assert latest_drift is not None
    assert latest_drift["report_type"] == "drift"
    assert latest_drift["metrics"]["psi"] == 0.05

    latest_cal = store.latest("calibration")
    assert latest_cal is not None
    assert latest_cal["report_type"] == "calibration"
    assert latest_cal["metrics"]["ece"] == 0.03

    all_reports = store.list_reports(limit=5)
    assert len(all_reports) == 2

    drift_reports = store.list_reports(limit=5, report_type="drift")
    assert len(drift_reports) == 1
    assert drift_reports[0]["report_type"] == "drift"

    store.close()


def test_monitoring_store_rejects_production_sqlite(tmp_path):
    with pytest.raises(MonitoringStoreConfigurationError):
        MonitoringStore(tmp_path / "monitoring.sqlite3", environment="production")

