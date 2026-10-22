"""Focused tests for the Phase 4 action integration foundation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.adapters import (
    FakeCaseManagementAdapter,
    FakeGraphAdapter,
    FakeIdentityAdapter,
    FakeMarketplaceAdapter,
    FakePaymentAdapter,
)
from backend.contracts.actions import ActionRequest, ActionStatus, ActionType
from backend.services.action_outbox import (
    ActionOutbox,
    ActionOutboxConfigurationError,
    IdempotencyConflictError,
)


def request_payload(**overrides):
    payload = {
        "action": ActionType.PAYMENT_HOLD,
        "case_id": "case_001",
        "reason": "confirmed policy review",
        "idempotency_key": "action-key-001",
        "target": {"target_type": "payment", "target_id": "payment_001"},
        "target_confirmation": True,
        "policy_version": "phase4-policy-v1",
    }
    payload.update(overrides)
    return payload


def test_action_request_requires_confirmation_policy_and_allowlisted_action():
    for field in ("case_id", "reason", "idempotency_key", "target", "target_confirmation", "policy_version"):
        payload = request_payload()
        del payload[field]
        with pytest.raises(ValidationError):
            ActionRequest(**payload)

    with pytest.raises(ValidationError):
        ActionRequest(**request_payload(target_confirmation=False))
    with pytest.raises(ValidationError):
        ActionRequest(**request_payload(target_confirmation=1))
    with pytest.raises(ValidationError):
        ActionRequest(**request_payload(action="provider_delete_everything"))
    with pytest.raises(ValidationError):
        ActionRequest(**request_payload(unexpected="must be rejected"))


def test_duplicate_idempotency_returns_same_command_and_conflicts_on_changed_payload(tmp_path):
    adapter = FakePaymentAdapter()
    with ActionOutbox(db_path=tmp_path / "outbox.sqlite3", payment_adapter=adapter) as outbox:
        first = outbox.create_command(request_payload())
        duplicate = outbox.create_command(request_payload())

        assert duplicate.command_id == first.command_id
        assert duplicate.status is ActionStatus.PENDING
        assert adapter.calls == []

        with pytest.raises(IdempotencyConflictError):
            outbox.create_command(request_payload(reason="different authorized reason"))


def test_timeout_leaves_command_pending_for_worker_retry(tmp_path):
    adapter = FakePaymentAdapter(timeout=True)
    with ActionOutbox(db_path=tmp_path / "outbox.sqlite3", payment_adapter=adapter) as outbox:
        command = outbox.create_command(request_payload())
        result = outbox.process_command(command.command_id)

        assert result.status is ActionStatus.PENDING
        assert result.attempts == 1
        assert result.provider_receipt is None
        assert result.error


def test_provider_rejection_is_persisted_as_failed(tmp_path):
    adapter = FakePaymentAdapter(fail=True)
    with ActionOutbox(db_path=tmp_path / "outbox.sqlite3", payment_adapter=adapter) as outbox:
        command = outbox.create_command(request_payload())
        result = outbox.process_command(command.command_id)

        assert result.status is ActionStatus.FAILED
        assert result.attempts == 1
        assert result.provider_receipt is None
        assert "rejected" in result.error


def test_provider_receipt_is_persisted_and_read_after_reopen(tmp_path):
    db_path = Path(tmp_path) / "outbox.sqlite3"
    with ActionOutbox(db_path=db_path) as outbox:
        command = outbox.create_command(request_payload())
        completed = outbox.process_command(command.command_id)
        assert completed.status is ActionStatus.SUCCEEDED
        receipt_id = completed.provider_receipt.receipt_id

    with ActionOutbox(db_path=db_path) as reopened:
        stored = reopened.get_command(command.command_id)
        assert stored is not None
        assert stored.status is ActionStatus.SUCCEEDED
        assert stored.provider_receipt is not None
        assert stored.provider_receipt.receipt_id == receipt_id


def test_production_requires_postgresql_dsn_and_never_creates_sqlite(tmp_path):
    db_path = Path(tmp_path) / "must-not-use.sqlite3"
    with pytest.raises(ActionOutboxConfigurationError, match="production requires.*PostgreSQL DSN"):
        ActionOutbox(db_path=db_path, environment="production")
    assert not db_path.exists()


def test_production_postgresql_configuration_does_not_select_sqlite(tmp_path):
    outbox = ActionOutbox(
        db_path=Path(tmp_path) / "must-not-use.sqlite3",
        environment="production",
        postgres_dsn="postgresql://user:password@example.invalid/trustshield",
        payment_adapter=FakePaymentAdapter(),
        marketplace_adapter=FakeMarketplaceAdapter(),
        identity_adapter=FakeIdentityAdapter(),
        graph_adapter=FakeGraphAdapter(),
        case_management_adapter=FakeCaseManagementAdapter(),
    )
    try:
        assert outbox.backend == "postgresql"
        assert outbox.sqlite_path is None
        assert not (Path(tmp_path) / "must-not-use.sqlite3").exists()
    finally:
        outbox.close()


def test_production_rejects_implicit_fake_provider_adapters():
    with pytest.raises(ActionOutboxConfigurationError, match="explicit provider adapters"):
        ActionOutbox(
            environment="production",
            postgres_dsn="postgresql://user:password@example.invalid/trustshield",
        )


def test_action_worker_daemon_processes_pending_batch(tmp_path):
    from scripts.run_action_worker import ActionWorkerDaemon

    db_path = Path(tmp_path) / "worker_outbox.sqlite3"
    with ActionOutbox(db_path=db_path) as outbox:
        cmd1 = outbox.create_command(request_payload(idempotency_key="key-001"))
        cmd2 = outbox.create_command(request_payload(idempotency_key="key-002"))
        cmd3 = outbox.create_command(request_payload(idempotency_key="key-003"))

        worker = ActionWorkerDaemon(outbox=outbox, batch_size=2, poll_interval=0.05)
        # First iteration processes batch of 2
        worker.run(once=True)

        assert outbox.get_command(cmd1.command_id).status is ActionStatus.SUCCEEDED
        assert outbox.get_command(cmd2.command_id).status is ActionStatus.SUCCEEDED
        assert outbox.get_command(cmd3.command_id).status is ActionStatus.PENDING

        # Second iteration processes remaining 1
        worker.run(once=True)
        assert outbox.get_command(cmd3.command_id).status is ActionStatus.SUCCEEDED


def test_action_worker_daemon_max_iterations_and_stop(tmp_path):
    from scripts.run_action_worker import ActionWorkerDaemon

    db_path = Path(tmp_path) / "worker_outbox_iter.sqlite3"
    with ActionOutbox(db_path=db_path) as outbox:
        worker = ActionWorkerDaemon(outbox=outbox, batch_size=5, poll_interval=0.01, max_iterations=2)
        exit_code = worker.run(once=False)
        assert exit_code == 0

