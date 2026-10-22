"""API integration tests for the operator action command boundary."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.contracts.actions import ActionStatus, ActionType
from backend.main import app
from backend.routers.actions import (
    get_action_outbox,
    get_audit_service,
    process_action_for_development,
    reset_action_services_for_tests,
)
from backend.services.action_outbox import ActionOutbox
from backend.services.audit_service import AuditService


OPERATOR_KEY = "test-operator-api-key-32chars!!"
ADMIN_KEY = "test-admin-secret-key-32chars!!"


@pytest.fixture
def action_api(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[tuple[TestClient, ActionOutbox]]:
    """Use isolated durable stores and real authentication for each test."""

    monkeypatch.setenv("TRUSTSHIELD_AUTH_ENABLED", "true")
    monkeypatch.delenv("TRUSTSHIELD_DEV_INSECURE_BYPASS", raising=False)
    monkeypatch.setenv("TRUSTSHIELD_API_KEY", OPERATOR_KEY)
    monkeypatch.setenv("TRUSTSHIELD_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setenv("TRUSTSHIELD_ENV", "development")

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


def _payload(key: str = "action-key-001", **overrides):
    value = {
        "case_id": "case_001",
        "reason": "confirmed policy review",
        "idempotency_key": key,
        "target_confirmation": True,
        "policy_version": "policy-v1",
    }
    value.update(overrides)
    return value


def _operator_headers() -> dict[str, str]:
    return {"X-API-Key": OPERATOR_KEY}


def test_actions_require_operator_or_admin_authentication(action_api):
    client, _outbox = action_api
    body = _payload()

    assert client.post("/actions/transactions/order-001/hold", json=body).status_code == 401
    operator = client.post(
        "/actions/transactions/order-001/hold",
        json=body,
        headers=_operator_headers(),
    )
    assert operator.status_code == 202

    admin = client.post(
        "/actions/listings/listing-001/quarantine",
        json=_payload("admin-key-001"),
        headers={"X-API-Key": ADMIN_KEY},
    )
    assert admin.status_code == 202


def test_action_contract_requires_case_reason_idempotency_confirmation_and_policy(action_api):
    client, _outbox = action_api
    required = ("case_id", "reason", "idempotency_key", "target_confirmation", "policy_version")

    for field in required:
        body = _payload()
        body.pop(field)
        response = client.post(
            "/actions/transactions/order-002/approve",
            json=body,
            headers=_operator_headers(),
        )
        assert response.status_code == 422, field

    false_confirmation = client.post(
        "/actions/transactions/order-002/approve",
        json=_payload(target_confirmation=False),
        headers=_operator_headers(),
    )
    assert false_confirmation.status_code == 422


def test_action_is_pending_and_get_lookup_has_no_provider_receipt_before_worker(action_api):
    client, outbox = action_api
    audit = app.dependency_overrides[get_audit_service]()
    response = client.post(
        "/actions/transactions/order-003/hold",
        json=_payload(),
        headers=_operator_headers(),
    )

    assert response.status_code == 202
    receipt = response.json()
    assert receipt["status"] == ActionStatus.PENDING.value
    assert receipt["provider_receipt"] is None
    assert outbox._adapters["payment"].calls == []

    lookup = client.get(f"/actions/{receipt['command_id']}", headers=_operator_headers())
    assert lookup.status_code == 200
    assert lookup.json()["status"] == ActionStatus.PENDING.value
    assert lookup.json()["provider_receipt"] is None

    completed = process_action_for_development(receipt["command_id"], outbox=outbox, audit=audit)
    assert completed.status is ActionStatus.SUCCEEDED
    assert completed.provider_receipt is not None
    assert len(outbox._adapters["payment"].calls) == 1

    after_worker = client.get(f"/actions/{receipt['command_id']}", headers=_operator_headers())
    assert after_worker.status_code == 200
    assert after_worker.json()["status"] == ActionStatus.SUCCEEDED.value
    assert after_worker.json()["provider_receipt"]["receipt_id"] == completed.provider_receipt.receipt_id


def test_duplicate_idempotency_returns_same_pending_command_and_conflict_on_changed_request(action_api):
    client, outbox = action_api
    first = client.post(
        "/actions/transactions/order-004/hold",
        json=_payload("duplicate-key"),
        headers=_operator_headers(),
    )
    duplicate = client.post(
        "/actions/transactions/order-004/hold",
        json=_payload("duplicate-key"),
        headers=_operator_headers(),
    )

    assert first.status_code == duplicate.status_code == 202
    assert duplicate.json()["command_id"] == first.json()["command_id"]
    assert duplicate.json()["status"] == ActionStatus.PENDING.value
    assert outbox._adapters["payment"].calls == []

    conflict = client.post(
        "/actions/transactions/order-004/block",
        json=_payload("duplicate-key"),
        headers=_operator_headers(),
    )
    assert conflict.status_code == 409


@pytest.mark.parametrize(
    ("path", "expected_action", "adapter_domain"),
    [
        ("/actions/transactions/order-005/approve", ActionType.PAYMENT_RELEASE, "payment"),
        ("/actions/transactions/order-006/block", ActionType.PAYMENT_REFUND, "payment"),
        ("/actions/listings/listing-002/quarantine", ActionType.MARKETPLACE_SUSPEND_LISTING, "marketplace"),
        ("/actions/rings/ring-002/freeze", ActionType.GRAPH_FLAG_ENTITY, "graph"),
    ],
)
def test_routes_bind_allowlisted_action_and_provider_domain(action_api, path, expected_action, adapter_domain):
    client, outbox = action_api
    response = client.post(path, json=_payload(path.rsplit("/", 1)[-1]), headers=_operator_headers())
    assert response.status_code == 202
    command = outbox.get_command(response.json()["command_id"])
    assert command is not None
    assert command.action is expected_action
    assert outbox._adapter_for(command.model_copy(update={"action": expected_action})) is outbox._adapters[adapter_domain]


def test_audit_records_enqueue_event(action_api):
    client, _outbox = action_api
    response = client.post(
        "/actions/rings/ring-003/freeze",
        json=_payload("audit-key"),
        headers=_operator_headers(),
    )
    assert response.status_code == 202

    # The dependency is the real AuditService in this fixture.  Read through
    # the service's durable API rather than depending on implementation tables.
    audit = app.dependency_overrides[get_audit_service]()
    records = audit.read_records(event_type="action")
    assert len(records) == 1
    assert records[0].event_name == "action.command.enqueued"
    assert records[0].outcome == ActionStatus.PENDING.value
