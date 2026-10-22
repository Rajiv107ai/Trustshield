"""Focused tests for the Phase 4 durable audit foundation."""

from __future__ import annotations

import json
import sqlite3

import pytest

from backend.contracts.audit import AUDIT_GENESIS_HASH, compute_audit_hash
from backend.services.audit_service import (
    AuditConfigurationError,
    AuditService,
)


def test_sqlite_append_read_and_hash_chain(tmp_path):
    service = AuditService(
        db_path=tmp_path / "audit.sqlite3",
        environment="development",
    )
    try:
        decision = service.append_decision(
            "REVIEW",
            transaction_id="txn_opaque_001",
            model_version="phase3",
            reason_codes=["HIGH_VELOCITY"],
            metadata={"risk_label": "medium", "score_bucket": "0.50-0.75"},
        )
        action = service.append_action(
            "queue_for_review",
            transaction_id="txn_opaque_001",
            actor_id="operator_opaque_001",
            outcome="queued",
            metadata={"queue": "manual_review"},
        )

        records = service.read_records()
        assert [record.record_id for record in records] == [decision.record_id, action.record_id]
        assert records[0].previous_hash == AUDIT_GENESIS_HASH
        assert records[1].previous_hash == records[0].record_hash
        assert records[0].record_hash == compute_audit_hash(records[0])
        assert records[1].record_hash == compute_audit_hash(records[1])
        assert service.verify_integrity().valid is True
        assert service.count() == 2
    finally:
        service.close()


def test_metadata_allowlist_rejects_secret_or_pii_fields(tmp_path):
    service = AuditService(db_path=tmp_path / "audit.sqlite3", environment="development")
    try:
        with pytest.raises(ValueError, match="forbidden|allowlisted"):
            service.append_action(
                "manual_review",
                subject_id="txn_opaque_002",
                metadata={"email": "person@example.invalid"},
            )

        with pytest.raises(ValueError, match="forbidden|allowlisted"):
            service.append_action(
                "manual_review",
                subject_id="txn_opaque_002",
                metadata={"request_body": "must-not-be-persisted"},
            )
        assert service.count() == 0
    finally:
        service.close()


def test_tampering_is_detected_by_hash_chain(tmp_path):
    db_path = tmp_path / "audit.sqlite3"
    service = AuditService(db_path=db_path, environment="development")
    try:
        service.append_decision(
            "ALLOW",
            transaction_id="txn_opaque_003",
            metadata={"risk_label": "low"},
        )
        service.append_action(
            "release_order",
            transaction_id="txn_opaque_003",
            metadata={"status": "completed"},
        )

        # Simulate an operator editing the SQLite file outside the service.
        # The public service has no update/delete operation, while verification
        # still has to detect file-level tampering.
        with sqlite3.connect(db_path) as connection:
            connection.execute(
                "UPDATE audit_records SET metadata_json = ? WHERE sequence = 1",
                (json.dumps({"risk_label": "critical"}),),
            )

        report = service.verify_integrity()
        assert report.valid is False
        assert report.first_invalid_sequence == 1
        assert report.error == "audit record hash mismatch"
    finally:
        service.close()


def test_production_requires_postgres_dsn_when_durability_is_required(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTSHIELD_ENV", "production")
    monkeypatch.delenv("TRUSTSHIELD_AUDIT_POSTGRES_DSN", raising=False)
    monkeypatch.delenv("TRUSTSHIELD_POSTGRES_DSN", raising=False)
    monkeypatch.delenv("AUDIT_POSTGRES_DSN", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    monkeypatch.delenv("TRUSTSHIELD_AUDIT_POSTGRES_URL", raising=False)
    monkeypatch.delenv("AUDIT_POSTGRES_URL", raising=False)
    monkeypatch.delenv("TRUSTSHIELD_AUDIT_DATABASE_URL", raising=False)
    monkeypatch.delenv("AUDIT_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TRUSTSHIELD_AUDIT_DURABILITY_REQUIRED", raising=False)
    monkeypatch.delenv("TRUSTSHIELD_REQUIRE_AUDIT_DURABILITY", raising=False)
    monkeypatch.delenv("AUDIT_DURABILITY_REQUIRED", raising=False)

    with pytest.raises(AuditConfigurationError, match="production requires.*PostgreSQL DSN"):
        AuditService(db_path=tmp_path / "must-not-use.sqlite3")


def test_production_dsn_selects_postgres_and_never_sqlite(tmp_path):
    service = AuditService(
        db_path=tmp_path / "must-not-use.sqlite3",
        environment="production",
        postgres_dsn="postgresql://audit_user:external-secret@example.invalid/audit",
    )
    try:
        assert service.backend == "postgresql"
        assert service.sqlite_path is None
        assert not (tmp_path / "must-not-use.sqlite3").exists()
    finally:
        service.close()
