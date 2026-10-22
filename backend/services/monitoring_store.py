"""Durable storage for aggregate monitoring reports only.

This local SQLite implementation is for development and reproducible tests.
Production monitoring persistence must use an externally managed PostgreSQL
store; the service never stores row-level events or labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

from backend.contracts.monitoring import CalibrationReportV1, DriftReportV1


class MonitoringStoreConfigurationError(RuntimeError):
    """Raised when monitoring persistence is not safely configured."""


class MonitoringStore:
    def __init__(self, path: str | Path = ".runtime/monitoring.sqlite3", *, environment: Optional[str] = None):
        env = (environment or os.getenv("TRUSTSHIELD_ENV", "development")).strip().lower()
        if env in {"production", "prod"}:
            raise MonitoringStoreConfigurationError(
                "production monitoring persistence requires an externally managed PostgreSQL store"
            )
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS monitoring_reports (
                report_id TEXT PRIMARY KEY,
                report_type TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                window_start TEXT NOT NULL,
                window_end TEXT NOT NULL,
                report_json TEXT NOT NULL,
                report_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self._connection.commit()

    def save(self, report: Mapping[str, Any] | DriftReportV1 | CalibrationReportV1) -> str:
        if isinstance(report, (DriftReportV1, CalibrationReportV1)):
            payload = report.model_dump(mode="json")
        else:
            payload = dict(report)
        if payload.get("aggregate_only") is not True:
            raise ValueError("monitoring reports must be aggregate_only=true")
        report_id = str(payload.get("record_id") or payload.get("monitoring_window", {}).get("window_id") or hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24])
        window = payload.get("monitoring_window") or {}
        report_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(report_json.encode("utf-8")).hexdigest()
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO monitoring_reports (report_id, report_type, schema_version, window_start, window_end, report_json, report_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (report_id, str(payload.get("report_type", "monitoring")), str(payload.get("schema_version", "monitoring.v1")), str(window.get("window_start", "")), str(window.get("window_end", "")), report_json, digest),
            )
            self._connection.commit()
        return report_id

    def get(self, report_id: str) -> Optional[dict[str, Any]]:
        row = self._connection.execute("SELECT report_json, report_sha256 FROM monitoring_reports WHERE report_id = ?", (report_id,)).fetchone()
        if row is None:
            return None
        payload = json.loads(row["report_json"])
        digest = hashlib.sha256(row["report_json"].encode("utf-8")).hexdigest()
        if digest != row["report_sha256"]:
            raise RuntimeError("monitoring report integrity check failed")
        return payload

    def latest(self, report_type: Optional[str] = None) -> Optional[dict[str, Any]]:
        with self._lock:
            if report_type is not None:
                row = self._connection.execute(
                    "SELECT report_json, report_sha256 FROM monitoring_reports WHERE report_type = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                    (report_type,),
                ).fetchone()
            else:
                row = self._connection.execute(
                    "SELECT report_json, report_sha256 FROM monitoring_reports ORDER BY created_at DESC, rowid DESC LIMIT 1"
                ).fetchone()
            if row is None:
                return None
            payload = json.loads(row["report_json"])
            digest = hashlib.sha256(row["report_json"].encode("utf-8")).hexdigest()
            if digest != row["report_sha256"]:
                raise RuntimeError("monitoring report integrity check failed")
            return payload

    def list_reports(self, limit: int = 10, report_type: Optional[str] = None) -> list[dict[str, Any]]:
        with self._lock:
            if report_type is not None:
                cursor = self._connection.execute(
                    "SELECT report_json, report_sha256 FROM monitoring_reports WHERE report_type = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (report_type, limit),
                )
            else:
                cursor = self._connection.execute(
                    "SELECT report_json, report_sha256 FROM monitoring_reports ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (limit,),
                )
            reports = []
            for row in cursor.fetchall():
                digest = hashlib.sha256(row["report_json"].encode("utf-8")).hexdigest()
                if digest == row["report_sha256"]:
                    reports.append(json.loads(row["report_json"]))
            return reports

    def close(self) -> None:
        with self._lock:
            self._connection.close()


__all__ = ["MonitoringStore", "MonitoringStoreConfigurationError"]
