"""Failure-injection coverage for release, auth, and action durability seams.

These tests deliberately break the inputs that a deployment can lose or
corrupt.  They exercise the existing model loader, readiness probe, auth
dependency, action contract, and durable outbox without changing application
routers or services.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import model_loader
from backend.adapters import FakePaymentAdapter
from backend.contracts.actions import ActionRequest, ActionStatus, ActionType
from backend.main import app, store
from backend.release_manifest import get_release_manifest_status
from backend.routers.actions import (
    get_action_outbox,
    get_audit_service,
    reset_action_services_for_tests,
)
from backend.services.action_outbox import ActionOutbox
from backend.services.audit_service import AuditService


OPERATOR_KEY = "failure-injection-operator-key-32chars"
ADMIN_KEY = "failure-injection-admin-key-32chars"


def _action_request_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "action": ActionType.PAYMENT_HOLD,
        "case_id": "case-failure-001",
        "reason": "failure injection test",
        "idempotency_key": "failure-action-key-001",
        "target": {"target_type": "payment", "target_id": "payment-001"},
        "target_confirmation": True,
        "policy_version": "failure-policy-v1",
    }
    payload.update(overrides)
    return payload


def test_missing_required_artifact_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A missing serving artifact must stop loading instead of serving partial models."""

    monkeypatch.setattr(model_loader, "MODELS_DIR", str(tmp_path))
    isolated_store = model_loader.ModelStore()

    with pytest.raises(FileNotFoundError, match="Required model artifacts"):
        isolated_store.load()

    assert isolated_store.is_loaded is False
    assert isolated_store.combined_graph_model is None


def test_corrupt_required_artifact_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Unreadable joblib bytes must not result in a partially loaded store."""

    for artifact_name in model_loader._REQUIRED:
        artifact = tmp_path / artifact_name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"this is not a joblib artifact")

    monkeypatch.setattr(model_loader, "MODELS_DIR", str(tmp_path))
    isolated_store = model_loader.ModelStore()

    with pytest.raises(Exception):
        isolated_store.load()

    assert isolated_store.is_loaded is False
    assert isolated_store.combined_graph_model is None


def test_corrupt_release_manifest_is_reported_without_success(tmp_path: Path) -> None:
    """Malformed release metadata is visible as invalid, never as a valid release."""

    manifest_path = tmp_path / "release-manifest.json"
    manifest_path.write_text("{not-json", encoding="utf-8")

    status = get_release_manifest_status(root_dir=tmp_path, manifest_path=manifest_path)

    assert status.valid is False
    assert status.errors


def test_production_readiness_fails_closed_when_pit_history_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A required point-in-time store makes readiness fail closed."""

    monkeypatch.setattr(store, "_loaded", True)
    monkeypatch.setattr(store, "combined_graph_model", object())
    monkeypatch.setattr(store, "rings_df", pd.DataFrame())
    monkeypatch.setattr(store, "transaction_history", None)
    monkeypatch.setattr(store, "load", lambda: None)
    monkeypatch.setenv("TRUSTSHIELD_ENV", "production")
    monkeypatch.setenv("TRUSTSHIELD_REQUIRE_PIT_HISTORY", "true")
    monkeypatch.setenv("TRUSTSHIELD_REQUIRE_RELEASE_MANIFEST", "false")
    monkeypatch.setenv("TRUSTSHIELD_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setenv("TRUSTSHIELD_API_KEY", OPERATOR_KEY)
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["pit_history_ready"] is False
    assert payload["details"]["pit_history_required"] is True


def test_expired_or_unrecognised_authentication_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An expired-looking bearer credential must fail closed like any revoked key."""

    monkeypatch.setenv("TRUSTSHIELD_AUTH_ENABLED", "true")
    monkeypatch.delenv("TRUSTSHIELD_DEV_INSECURE_BYPASS", raising=False)
    monkeypatch.setenv("TRUSTSHIELD_API_KEY", OPERATOR_KEY)
    monkeypatch.setenv("TRUSTSHIELD_ADMIN_KEY", ADMIN_KEY)
    expired_token = "eyJhbGciOiJub25lIn0.eyJleHAiOjF9.signature"

    with TestClient(app) as client:
        response = client.get(
            "/fraud-rings",
            headers={"Authorization": f"Bearer {expired_token}"},
        )

    assert response.status_code == 401
    assert "invalid authentication credentials" in response.json()["detail"].lower()


def test_invalid_action_contract_is_rejected_before_enqueue() -> None:
    """Strict action validation rejects extra fields and non-boolean confirmation."""

    with pytest.raises(ValidationError):
        ActionRequest(**_action_request_payload(unexpected_field="must-not-pass"))

    with pytest.raises(ValidationError):
        ActionRequest(**_action_request_payload(target_confirmation="true"))


@pytest.fixture
def action_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[tuple[TestClient, ActionOutbox]]:
    """Isolate the HTTP action boundary with real auth and durable SQLite."""

    monkeypatch.setenv("TRUSTSHIELD_ENV", "development")
    monkeypatch.setenv("TRUSTSHIELD_AUTH_ENABLED", "true")
    monkeypatch.delenv("TRUSTSHIELD_DEV_INSECURE_BYPASS", raising=False)
    monkeypatch.setenv("TRUSTSHIELD_API_KEY", OPERATOR_KEY)
    monkeypatch.setenv("TRUSTSHIELD_ADMIN_KEY", ADMIN_KEY)

    reset_action_services_for_tests()
    outbox = ActionOutbox(db_path=tmp_path / "actions.sqlite3", environment="development")
    audit = AuditService(db_path=tmp_path / "audit.sqlite3", environment="development")
    app.dependency_overrides[get_action_outbox] = lambda: outbox
    app.dependency_overrides[get_audit_service] = lambda: audit
    try:
        with TestClient(app) as client:
            yield client, outbox
    finally:
        app.dependency_overrides.pop(get_action_outbox, None)
        app.dependency_overrides.pop(get_audit_service, None)
        outbox.close()
        audit.close()
        reset_action_services_for_tests()


def test_duplicate_action_is_idempotent_and_changed_payload_conflicts(action_api) -> None:
    """Retrying one command is safe; reusing its key for another command is not."""

    client, outbox = action_api
    payload = {
        "case_id": "case-duplicate",
        "reason": "duplicate action test",
        "idempotency_key": "duplicate-action-key",
        "target_confirmation": True,
        "policy_version": "policy-v1",
    }
    headers = {"X-API-Key": OPERATOR_KEY}

    first = client.post("/actions/transactions/order-duplicate/hold", json=payload, headers=headers)
    duplicate = client.post("/actions/transactions/order-duplicate/hold", json=payload, headers=headers)
    conflict_payload = {**payload, "reason": "different command under same key"}
    conflict = client.post(
        "/actions/transactions/order-duplicate/hold",
        json=conflict_payload,
        headers=headers,
    )

    assert first.status_code == duplicate.status_code == 202
    assert duplicate.json()["command_id"] == first.json()["command_id"]
    assert duplicate.json()["status"] == ActionStatus.PENDING.value
    assert conflict.status_code == 409
    assert outbox._adapters["payment"].calls == []


def test_pending_action_survives_worker_restart_and_can_finish(tmp_path: Path) -> None:
    """A timeout remains pending, survives reopen, and is retryable by a worker."""

    db_path = tmp_path / "restartable-actions.sqlite3"
    request = _action_request_payload(idempotency_key="restart-action-key")
    with ActionOutbox(db_path=db_path, payment_adapter=FakePaymentAdapter(timeout=True)) as first_worker:
        command = first_worker.create_command(request)
        pending = first_worker.process_command(command.command_id)
        assert pending.status is ActionStatus.PENDING
        assert pending.attempts == 1
        assert pending.provider_receipt is None

    with ActionOutbox(db_path=db_path) as restarted_worker:
        restored = restarted_worker.get_command(command.command_id)
        assert restored is not None
        assert restored.status is ActionStatus.PENDING
        completed = restarted_worker.process_command(command.command_id)

    assert completed.status is ActionStatus.SUCCEEDED
    assert completed.attempts == 2
    assert completed.provider_receipt is not None
