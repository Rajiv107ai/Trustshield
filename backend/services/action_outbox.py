"""Durable command outbox for Phase 4 operational actions.

Routes should call :meth:`ActionOutbox.create_command` only.  That method
validates and durably records a pending command; it does not call a provider.
The worker-facing :meth:`ActionOutbox.process_command` and
:meth:`ActionOutbox.process_pending` methods perform provider calls after the
command has been committed, then persist a succeeded/failed receipt.  A
provider timeout intentionally leaves the command pending so a later worker
attempt can retry it.

Development uses the standard-library SQLite driver.  Production is
fail-closed: a PostgreSQL DSN is required and there is no in-memory or SQLite
fallback.  PostgreSQL support is DB-API based and loaded lazily so local
development does not need to install a PostgreSQL driver.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol
from uuid import uuid4

from backend.adapters import (
    ActionAdapterTimeout,
    ActionProviderError,
    DeterministicFakeCaseManagementAdapter,
    DeterministicFakeGraphAdapter,
    DeterministicFakeIdentityAdapter,
    DeterministicFakeMarketplaceAdapter,
    DeterministicFakePaymentAdapter,
)
from backend.contracts.actions import (
    ActionReceipt,
    ActionRequest,
    ActionStatus,
    ActionType,
    ProviderReceipt,
)


ACTION_ENVIRONMENT_VARS = ("TRUSTSHIELD_ENV", "ENVIRONMENT", "ENV")
ACTION_SQLITE_PATH_ENV = "TRUSTSHIELD_ACTION_OUTBOX_SQLITE_PATH"
ACTION_SQLITE_PATH_ENV_VARS = (
    "TRUSTSHIELD_ACTION_OUTBOX_SQLITE_PATH",
    "TRUSTSHIELD_ACTION_OUTBOX_PATH",
    "TRUSTSHIELD_OUTBOX_SQLITE_PATH",
    "ACTION_OUTBOX_SQLITE_PATH",
)
ACTION_POSTGRES_DSN_ENV_VARS = (
    "TRUSTSHIELD_ACTION_POSTGRES_DSN",
    "TRUSTSHIELD_ACTION_POSTGRES_URL",
    "TRUSTSHIELD_ACTION_DSN",
    "TRUSTSHIELD_ACTION_DATABASE_DSN",
    "TRUSTSHIELD_ACTION_DATABASE_URL",
    "TRUSTSHIELD_POSTGRES_DSN",
    "TRUSTSHIELD_POSTGRES_URL",
    "TRUSTSHIELD_DATABASE_URL",
    "POSTGRES_DSN",
    "POSTGRES_URL",
    "DATABASE_DSN",
    "DATABASE_URL",
)
DEFAULT_SQLITE_PATH = ".runtime/action_outbox.sqlite3"


class ActionOutboxConfigurationError(RuntimeError):
    """Raised when action durability configuration is unsafe or incomplete."""


class ActionOutboxBackendUnavailable(ActionOutboxConfigurationError):
    """Raised when the selected durable database cannot be used."""


class IdempotencyConflictError(ValueError):
    """The same idempotency key was reused for a different command."""


class ActionCommandNotFoundError(KeyError):
    """A worker/status operation referred to an unknown command."""


ActionOutboxError = ActionOutboxConfigurationError
ActionOutboxIdempotencyConflict = IdempotencyConflictError


@dataclass(frozen=True)
class ActionOutboxConfiguration:
    """Resolved storage settings; DSNs are hidden from repr output."""

    environment: str
    backend: str
    sqlite_path: Optional[str]
    postgres_dsn: Optional[str] = field(default=None, repr=False)


class _OutboxStore(Protocol):
    def create_command(self, request: ActionRequest, fingerprint: str, command_id: str, now: str) -> ActionReceipt:
        ...

    def get_command(self, command_id: str) -> Optional[ActionReceipt]:
        ...

    def get_by_idempotency_key(self, idempotency_key: str) -> Optional[tuple[ActionReceipt, str]]:
        ...

    def increment_attempts(self, command_id: str, now: str) -> ActionReceipt:
        ...

    def finish_command(
        self,
        command_id: str,
        *,
        status: ActionStatus,
        provider_receipt: Optional[ProviderReceipt],
        error: Optional[str],
        now: str,
    ) -> ActionReceipt:
        ...

    def pending_command_ids(self, limit: int) -> list[str]:
        ...

    def close(self) -> None:
        ...


def _first_environment_value(names: Iterable[str]) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _is_production(environment: Optional[str]) -> bool:
    value = environment if environment is not None else _first_environment_value(ACTION_ENVIRONMENT_VARS)
    return (value or "development").strip().lower() in {"production", "prod"}


is_production_environment = _is_production


def _is_postgres_dsn(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.startswith("postgresql://") or lowered.startswith("postgres://")


def _configured_postgres_dsn(explicit_dsn: Optional[str]) -> Optional[str]:
    if explicit_dsn is not None and explicit_dsn.strip():
        dsn = explicit_dsn.strip()
        if not _is_postgres_dsn(dsn):
            raise ActionOutboxConfigurationError("action outbox database URL must be a PostgreSQL DSN")
        return dsn

    for name in ACTION_POSTGRES_DSN_ENV_VARS:
        value = os.getenv(name)
        if value is None or not value.strip():
            continue
        dsn = value.strip()
        # DATABASE_URL is common in local tooling.  It only selects this
        # backend when it is actually a PostgreSQL URL.
        if name == "DATABASE_URL" and not _is_postgres_dsn(dsn):
            continue
        if not _is_postgres_dsn(dsn):
            raise ActionOutboxConfigurationError("action outbox database URL must be a PostgreSQL DSN")
        return dsn
    return None


def load_action_outbox_configuration(
    *,
    sqlite_path: str | Path | None = None,
    db_path: str | Path | None = None,
    database_path: str | Path | None = None,
    postgres_dsn: Optional[str] = None,
    database_url: Optional[str] = None,
    database_dsn: Optional[str] = None,
    environment: Optional[str] = None,
) -> ActionOutboxConfiguration:
    """Resolve storage and reject production without a PostgreSQL DSN.

    ``sqlite_path``, ``db_path``, and ``database_path`` are compatibility
    spellings but must agree when more than one is supplied.  An explicit
    PostgreSQL DSN selects the PostgreSQL backend in every environment.
    """

    paths = [path for path in (sqlite_path, db_path, database_path) if path is not None]
    if paths and any(str(path) != str(paths[0]) for path in paths[1:]):
        raise ValueError("sqlite_path, db_path, and database_path cannot disagree")
    resolved_environment = (
        environment.strip().lower()
        if environment is not None and environment.strip()
        else (_first_environment_value(ACTION_ENVIRONMENT_VARS) or "development").strip().lower()
    )
    dsn_values = [
        value
        for value in (postgres_dsn, database_url, database_dsn)
        if value is not None and value.strip()
    ]
    if dsn_values and any(value.strip() != dsn_values[0].strip() for value in dsn_values[1:]):
        raise ValueError("PostgreSQL DSN aliases cannot disagree")
    dsn = _configured_postgres_dsn(dsn_values[0] if dsn_values else None)

    if _is_production(resolved_environment) and not dsn:
        raise ActionOutboxConfigurationError(
            "CRITICAL ACTION OUTBOX DURABILITY: production requires an externally configured PostgreSQL DSN"
        )

    if dsn:
        return ActionOutboxConfiguration(
            environment=resolved_environment,
            backend="postgresql",
            sqlite_path=None,
            postgres_dsn=dsn,
        )

    resolved_path = str(
        paths[0]
        if paths
        else (_first_environment_value(ACTION_SQLITE_PATH_ENV_VARS) or DEFAULT_SQLITE_PATH)
    )
    return ActionOutboxConfiguration(
        environment=resolved_environment,
        backend="sqlite",
        sqlite_path=resolved_path,
    )


validate_action_outbox_configuration = load_action_outbox_configuration
load_action_outbox_config = load_action_outbox_configuration
validate_production_action_outbox_configuration = load_action_outbox_configuration
ActionOutboxConfig = ActionOutboxConfiguration


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ActionOutboxBackendUnavailable("stored action timestamp is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _request_json(request: ActionRequest) -> str:
    return _json_dump(request.model_dump(mode="json", by_alias=True))


def _request_fingerprint(request: ActionRequest) -> str:
    return hashlib.sha256(_request_json(request).encode("utf-8")).hexdigest()


def _safe_error(exc: BaseException, *, fallback: str) -> str:
    """Keep provider diagnostics bounded and free of obvious credentials."""

    message = str(exc).strip()
    lowered = message.lower()
    if not message or "bearer " in lowered or "api_key" in lowered or "access_token" in lowered:
        return fallback
    message = " ".join(message.split())
    return message[:2000] or fallback


def _row_to_receipt(row: Mapping[str, Any]) -> ActionReceipt:
    request_payload = json.loads(row["request_json"])
    request = ActionRequest.model_validate(request_payload)
    provider_payload = row["provider_receipt_json"]
    provider_receipt = None
    if provider_payload:
        provider_receipt = ProviderReceipt.model_validate(json.loads(provider_payload))
    return ActionReceipt(
        command_id=str(row["command_id"]),
        action=request.action,
        case_id=request.case_id,
        reason=request.reason,
        idempotency_key=request.idempotency_key,
        target=request.target,
        target_confirmation=request.target_confirmation,
        policy_version=request.policy_version,
        requested_by=request.requested_by,
        status=ActionStatus(str(row["status"])),
        created_at=_parse_timestamp(str(row["created_at"])),
        updated_at=_parse_timestamp(str(row["updated_at"])),
        attempts=int(row["attempts"]),
        provider_receipt=provider_receipt,
        error=row["error"],
    )


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS action_outbox (
    command_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    request_json TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'succeeded', 'failed')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    provider_receipt_json TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS action_outbox_status_created_idx
    ON action_outbox(status, created_at, command_id);
"""


class _SQLiteOutboxStore:
    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            try:
                Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ActionOutboxBackendUnavailable("could not prepare SQLite action outbox directory") from exc
        try:
            self.connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            if path != ":memory:":
                self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = FULL")
            self.connection.executescript(_SQLITE_SCHEMA)
            self.connection.commit()
        except sqlite3.Error as exc:
            raise ActionOutboxBackendUnavailable("could not initialize SQLite action outbox") from exc
        self.lock = threading.RLock()

    def _select_one(self, query: str, parameters: tuple[Any, ...]) -> Optional[sqlite3.Row]:
        return self.connection.execute(query, parameters).fetchone()

    def create_command(self, request: ActionRequest, fingerprint: str, command_id: str, now: str) -> ActionReceipt:
        with self.lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                existing = self._select_one(
                    "SELECT * FROM action_outbox WHERE idempotency_key = ?",
                    (request.idempotency_key,),
                )
                if existing is not None:
                    self.connection.commit()
                    if existing["request_fingerprint"] != fingerprint:
                        raise IdempotencyConflictError(
                            "idempotency_key is already associated with a different action request"
                        )
                    return _row_to_receipt(existing)
                self.connection.execute(
                    """
                    INSERT INTO action_outbox (
                        command_id, idempotency_key, request_json, request_fingerprint,
                        status, created_at, updated_at, attempts, provider_receipt_json, error
                    ) VALUES (?, ?, ?, ?, 'pending', ?, ?, 0, NULL, NULL)
                    """,
                    (command_id, request.idempotency_key, _request_json(request), fingerprint, now, now),
                )
                self.connection.commit()
                row = self._select_one("SELECT * FROM action_outbox WHERE command_id = ?", (command_id,))
                if row is None:  # pragma: no cover - defensive database invariant
                    raise ActionOutboxBackendUnavailable("SQLite action command disappeared after insert")
                return _row_to_receipt(row)
            except IdempotencyConflictError:
                self.connection.rollback()
                raise
            except sqlite3.IntegrityError:
                self.connection.rollback()
                existing = self._select_one(
                    "SELECT * FROM action_outbox WHERE idempotency_key = ?",
                    (request.idempotency_key,),
                )
                if existing is None:
                    raise ActionOutboxBackendUnavailable("SQLite action command uniqueness failure")
                if existing["request_fingerprint"] != fingerprint:
                    raise IdempotencyConflictError(
                        "idempotency_key is already associated with a different action request"
                    )
                return _row_to_receipt(existing)
            except sqlite3.Error as exc:
                self.connection.rollback()
                raise ActionOutboxBackendUnavailable("could not create SQLite action command") from exc

    def get_command(self, command_id: str) -> Optional[ActionReceipt]:
        with self.lock:
            row = self._select_one("SELECT * FROM action_outbox WHERE command_id = ?", (command_id,))
            return _row_to_receipt(row) if row is not None else None

    def get_by_idempotency_key(self, idempotency_key: str) -> Optional[tuple[ActionReceipt, str]]:
        with self.lock:
            row = self._select_one(
                "SELECT * FROM action_outbox WHERE idempotency_key = ?",
                (idempotency_key,),
            )
            return (_row_to_receipt(row), str(row["request_fingerprint"])) if row is not None else None

    def increment_attempts(self, command_id: str, now: str) -> ActionReceipt:
        with self.lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                self.connection.execute(
                    "UPDATE action_outbox SET attempts = attempts + 1, updated_at = ? WHERE command_id = ?",
                    (now, command_id),
                )
                self.connection.commit()
            except sqlite3.Error as exc:
                self.connection.rollback()
                raise ActionOutboxBackendUnavailable("could not claim SQLite action command") from exc
            receipt = self.get_command(command_id)
            if receipt is None:
                raise ActionCommandNotFoundError(command_id)
            return receipt

    def finish_command(
        self,
        command_id: str,
        *,
        status: ActionStatus,
        provider_receipt: Optional[ProviderReceipt],
        error: Optional[str],
        now: str,
    ) -> ActionReceipt:
        provider_json = _json_dump(provider_receipt.model_dump(mode="json")) if provider_receipt else None
        with self.lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                self.connection.execute(
                    """
                    UPDATE action_outbox
                    SET status = ?, updated_at = ?, provider_receipt_json = ?, error = ?
                    WHERE command_id = ?
                    """,
                    (status.value, now, provider_json, error, command_id),
                )
                self.connection.commit()
            except sqlite3.Error as exc:
                self.connection.rollback()
                raise ActionOutboxBackendUnavailable("could not persist SQLite action result") from exc
            receipt = self.get_command(command_id)
            if receipt is None:
                raise ActionCommandNotFoundError(command_id)
            return receipt

    def pending_command_ids(self, limit: int) -> list[str]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT command_id FROM action_outbox WHERE status = 'pending' ORDER BY created_at, command_id LIMIT ?",
                (limit,),
            ).fetchall()
            return [str(row["command_id"]) for row in rows]

    def close(self) -> None:
        with self.lock:
            self.connection.close()


_POSTGRES_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS action_outbox (
        command_id TEXT PRIMARY KEY,
        idempotency_key TEXT NOT NULL UNIQUE,
        request_json TEXT NOT NULL,
        request_fingerprint TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'succeeded', 'failed')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        provider_receipt_json TEXT,
        error TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS action_outbox_status_created_idx ON action_outbox(status, created_at, command_id)",
)


class _PostgresOutboxStore:
    """Small optional DB-API PostgreSQL store used only with an explicit DSN."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.connection: Any = None
        self.lock = threading.RLock()

    def _connect(self) -> Any:
        if self.connection is not None:
            return self.connection
        driver: Any
        try:
            import psycopg as driver  # type: ignore[import-not-found]
        except ImportError:
            try:
                import psycopg2 as driver  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ActionOutboxBackendUnavailable(
                    "PostgreSQL action outbox requires psycopg or psycopg2"
                ) from exc
        try:
            self.connection = driver.connect(self.dsn)
            self.connection.autocommit = False
            cursor = self.connection.cursor()
            try:
                for statement in _POSTGRES_SCHEMA_STATEMENTS:
                    cursor.execute(statement)
                self.connection.commit()
            finally:
                cursor.close()
        except Exception as exc:
            if self.connection is not None:
                try:
                    self.connection.rollback()
                except Exception:
                    pass
            self.connection = None
            raise ActionOutboxBackendUnavailable("could not initialize PostgreSQL action outbox") from exc
        return self.connection

    @staticmethod
    def _rows(cursor: Any) -> list[dict[str, Any]]:
        names = [item[0] for item in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def _one(self, query: str, parameters: tuple[Any, ...]) -> Optional[dict[str, Any]]:
        connection = self._connect()
        cursor = connection.cursor()
        try:
            cursor.execute(query, parameters)
            row = cursor.fetchone()
            if row is None:
                return None
            names = [item[0] for item in cursor.description]
            return dict(zip(names, row))
        finally:
            cursor.close()

    def create_command(self, request: ActionRequest, fingerprint: str, command_id: str, now: str) -> ActionReceipt:
        with self.lock:
            connection = self._connect()
            cursor = connection.cursor()
            try:
                cursor.execute(
                    "SELECT * FROM action_outbox WHERE idempotency_key = %s FOR UPDATE",
                    (request.idempotency_key,),
                )
                row_values = self._rows(cursor)
                if row_values:
                    connection.commit()
                    row = row_values[0]
                    if row["request_fingerprint"] != fingerprint:
                        raise IdempotencyConflictError(
                            "idempotency_key is already associated with a different action request"
                        )
                    return _row_to_receipt(row)
                cursor.execute(
                    """
                    INSERT INTO action_outbox (
                        command_id, idempotency_key, request_json, request_fingerprint,
                        status, created_at, updated_at, attempts, provider_receipt_json, error
                    ) VALUES (%s, %s, %s, %s, 'pending', %s, %s, 0, NULL, NULL)
                    """,
                    (command_id, request.idempotency_key, _request_json(request), fingerprint, now, now),
                )
                connection.commit()
            except IdempotencyConflictError:
                connection.rollback()
                raise
            except Exception as exc:
                connection.rollback()
                raise ActionOutboxBackendUnavailable("could not create PostgreSQL action command") from exc
            finally:
                cursor.close()
            row = self._one("SELECT * FROM action_outbox WHERE command_id = %s", (command_id,))
            if row is None:
                raise ActionOutboxBackendUnavailable("PostgreSQL action command disappeared after insert")
            return _row_to_receipt(row)

    def get_command(self, command_id: str) -> Optional[ActionReceipt]:
        with self.lock:
            row = self._one("SELECT * FROM action_outbox WHERE command_id = %s", (command_id,))
            return _row_to_receipt(row) if row is not None else None

    def get_by_idempotency_key(self, idempotency_key: str) -> Optional[tuple[ActionReceipt, str]]:
        with self.lock:
            row = self._one("SELECT * FROM action_outbox WHERE idempotency_key = %s", (idempotency_key,))
            return (_row_to_receipt(row), str(row["request_fingerprint"])) if row is not None else None

    def increment_attempts(self, command_id: str, now: str) -> ActionReceipt:
        with self.lock:
            connection = self._connect()
            cursor = connection.cursor()
            try:
                cursor.execute(
                    "UPDATE action_outbox SET attempts = attempts + 1, updated_at = %s WHERE command_id = %s",
                    (now, command_id),
                )
                connection.commit()
            except Exception as exc:
                connection.rollback()
                raise ActionOutboxBackendUnavailable("could not claim PostgreSQL action command") from exc
            finally:
                cursor.close()
            receipt = self.get_command(command_id)
            if receipt is None:
                raise ActionCommandNotFoundError(command_id)
            return receipt

    def finish_command(
        self,
        command_id: str,
        *,
        status: ActionStatus,
        provider_receipt: Optional[ProviderReceipt],
        error: Optional[str],
        now: str,
    ) -> ActionReceipt:
        provider_json = _json_dump(provider_receipt.model_dump(mode="json")) if provider_receipt else None
        with self.lock:
            connection = self._connect()
            cursor = connection.cursor()
            try:
                cursor.execute(
                    """
                    UPDATE action_outbox
                    SET status = %s, updated_at = %s, provider_receipt_json = %s, error = %s
                    WHERE command_id = %s
                    """,
                    (status.value, now, provider_json, error, command_id),
                )
                connection.commit()
            except Exception as exc:
                connection.rollback()
                raise ActionOutboxBackendUnavailable("could not persist PostgreSQL action result") from exc
            finally:
                cursor.close()
            receipt = self.get_command(command_id)
            if receipt is None:
                raise ActionCommandNotFoundError(command_id)
            return receipt

    def pending_command_ids(self, limit: int) -> list[str]:
        with self.lock:
            connection = self._connect()
            cursor = connection.cursor()
            try:
                cursor.execute(
                    "SELECT command_id FROM action_outbox WHERE status = 'pending' ORDER BY created_at, command_id LIMIT %s",
                    (limit,),
                )
                return [str(row[0]) for row in cursor.fetchall()]
            finally:
                cursor.close()

    def close(self) -> None:
        with self.lock:
            if self.connection is not None:
                self.connection.close()
                self.connection = None


# Action names can be extended only here, alongside the allowlisted enum.
_ACTION_DOMAIN: dict[str, str] = {
    ActionType.PAYMENT_HOLD.value: "payment",
    ActionType.PAYMENT_RELEASE.value: "payment",
    ActionType.PAYMENT_REFUND.value: "payment",
    ActionType.MARKETPLACE_HOLD_ORDER.value: "marketplace",
    ActionType.MARKETPLACE_SUSPEND_LISTING.value: "marketplace",
    ActionType.MARKETPLACE_RESTORE_LISTING.value: "marketplace",
    ActionType.MARKETPLACE_SUSPEND.value: "marketplace",
    ActionType.MARKETPLACE_REINSTATE.value: "marketplace",
    ActionType.IDENTITY_LOCK_ACCOUNT.value: "identity",
    ActionType.IDENTITY_REQUIRE_VERIFICATION.value: "identity",
    ActionType.IDENTITY_UNLOCK_ACCOUNT.value: "identity",
    ActionType.IDENTITY_LOCK.value: "identity",
    ActionType.GRAPH_FLAG_ENTITY.value: "graph",
    ActionType.GRAPH_RECALCULATE_RISK.value: "graph",
    ActionType.GRAPH_ADD_CASE_EDGE.value: "graph",
    ActionType.GRAPH_FLAG.value: "graph",
    ActionType.CASE_CREATE.value: "case_management",
    ActionType.CASE_UPDATE.value: "case_management",
    ActionType.CASE_CLOSE.value: "case_management",
    ActionType.CASE_ADD_NOTE.value: "case_management",
}


class ActionOutbox:
    """Durable action command creation and worker processing service."""

    def __init__(
        self,
        sqlite_path: str | Path | None = None,
        *,
        db_path: str | Path | None = None,
        database_path: str | Path | None = None,
        postgres_dsn: Optional[str] = None,
        database_url: Optional[str] = None,
        database_dsn: Optional[str] = None,
        environment: Optional[str] = None,
        adapters: Optional[Mapping[str | ActionType, Any]] = None,
        payment_adapter: Any = None,
        marketplace_adapter: Any = None,
        identity_adapter: Any = None,
        graph_adapter: Any = None,
        case_management_adapter: Any = None,
        case_adapter: Any = None,
    ) -> None:
        self.configuration = load_action_outbox_configuration(
            sqlite_path=sqlite_path,
            db_path=db_path,
            database_path=database_path,
            postgres_dsn=postgres_dsn,
            database_url=database_url,
            database_dsn=database_dsn,
            environment=environment,
        )
        if self.configuration.backend == "sqlite":
            assert self.configuration.sqlite_path is not None
            self._store: _OutboxStore = _SQLiteOutboxStore(self.configuration.sqlite_path)
        elif self.configuration.backend == "postgresql":
            assert self.configuration.postgres_dsn is not None
            self._store = _PostgresOutboxStore(self.configuration.postgres_dsn)
        else:  # pragma: no cover - configuration currently has no disabled mode
            raise ActionOutboxConfigurationError("action outbox has no durable backend")
        self._closed = False
        if self.configuration.environment in {"production", "prod"}:
            configured_domains = {
                _ACTION_DOMAIN.get(key.value if isinstance(key, ActionType) else str(key), key.value if isinstance(key, ActionType) else str(key))
                for key in (adapters or {})
            }
            supplied_domains = {
                name
                for name, adapter in {
                    "payment": payment_adapter,
                    "marketplace": marketplace_adapter,
                    "identity": identity_adapter,
                    "graph": graph_adapter,
                    "case_management": case_management_adapter or case_adapter,
                }.items()
                if adapter is not None
            }
            missing_adapters = sorted({"payment", "marketplace", "identity", "graph", "case_management"} - supplied_domains - configured_domains)
            if missing_adapters:
                raise ActionOutboxConfigurationError(
                    "production action outbox requires explicit provider adapters: "
                    + ", ".join(missing_adapters)
                )
        self._adapters = self._build_adapters(
            adapters,
            payment_adapter=payment_adapter,
            marketplace_adapter=marketplace_adapter,
            identity_adapter=identity_adapter,
            graph_adapter=graph_adapter,
            case_management_adapter=case_management_adapter or case_adapter,
        )

    @property
    def durability_required(self) -> bool:
        """Whether this outbox is governed as a required durable store."""

        return self.configuration.environment in {"production", "prod"}

    @staticmethod
    def _build_adapters(
        adapters: Optional[Mapping[str | ActionType, Any]],
        *,
        payment_adapter: Any,
        marketplace_adapter: Any,
        identity_adapter: Any,
        graph_adapter: Any,
        case_management_adapter: Any,
    ) -> dict[str, Any]:
        registry: dict[str, Any] = {
            "payment": payment_adapter or DeterministicFakePaymentAdapter(),
            "marketplace": marketplace_adapter or DeterministicFakeMarketplaceAdapter(),
            "identity": identity_adapter or DeterministicFakeIdentityAdapter(),
            "graph": graph_adapter or DeterministicFakeGraphAdapter(),
            "case_management": case_management_adapter or DeterministicFakeCaseManagementAdapter(),
        }
        if adapters:
            for raw_key, adapter in adapters.items():
                key = raw_key.value if isinstance(raw_key, ActionType) else str(raw_key)
                domain = _ACTION_DOMAIN.get(key, key)
                if domain not in registry:
                    raise ValueError(f"unsupported action adapter key: {key}")
                registry[domain] = adapter
        return registry

    @property
    def backend(self) -> str:
        return self.configuration.backend

    @property
    def sqlite_path(self) -> Optional[str]:
        return self.configuration.sqlite_path

    @property
    def postgres_dsn(self) -> Optional[str]:
        return self.configuration.postgres_dsn

    def _ensure_open(self) -> None:
        if self._closed:
            raise ActionOutboxBackendUnavailable("action outbox is closed")

    @staticmethod
    def _coerce_request(request: ActionRequest | Mapping[str, Any]) -> ActionRequest:
        return request if isinstance(request, ActionRequest) else ActionRequest.model_validate(request)

    def create_command(self, request: ActionRequest | Mapping[str, Any]) -> ActionReceipt:
        """Validate and durably enqueue a command without calling a provider."""

        self._ensure_open()
        validated = self._coerce_request(request)
        return self._store.create_command(
            validated,
            _request_fingerprint(validated),
            uuid4().hex,
            _timestamp(_utc_now()),
        )

    # ``enqueue``/``create`` are route-integration-friendly spellings.
    enqueue = create_command
    create = create_command
    enqueue_action = create_command
    create_action = create_command
    submit = create_command

    def get_command(self, command_id: str) -> Optional[ActionReceipt]:
        self._ensure_open()
        return self._store.get_command(command_id)

    get = get_command
    get_receipt = get_command
    read_receipt = get_command
    read = get_command
    get_action = get_command
    status = get_command
    get_status = get_command

    def _adapter_for(self, request: ActionRequest) -> Any:
        domain = _ACTION_DOMAIN.get(request.action.value)
        if domain is None:
            raise ActionProviderError(f"no adapter is allowlisted for action {request.action.value}")
        adapter = self._adapters.get(domain)
        if adapter is None:
            raise ActionProviderError(f"no adapter is configured for action domain {domain}")
        return adapter

    @staticmethod
    def _invoke_adapter(adapter: Any, request: ActionRequest) -> ProviderReceipt:
        execute = getattr(adapter, "execute", None)
        if not callable(execute):
            execute = getattr(adapter, "perform", None)
        if not callable(execute):
            raise ActionProviderError("configured action adapter has no execute method")
        result = execute(request)
        if isinstance(result, ProviderReceipt):
            provider_receipt = result
        elif isinstance(result, Mapping):
            provider_receipt = ProviderReceipt.model_validate(result)
        else:
            raise ActionProviderError("action adapter returned an invalid provider receipt")
        if provider_receipt.action is not request.action:
            raise ActionProviderError("provider receipt action does not match the command")
        if provider_receipt.idempotency_key != request.idempotency_key:
            raise ActionProviderError("provider receipt idempotency key does not match the command")
        if provider_receipt.target != request.target:
            raise ActionProviderError("provider receipt target does not match the command")
        return provider_receipt

    def process_command(self, command_id: str | ActionReceipt) -> ActionReceipt:
        """Worker operation: call a provider outside command creation/route code."""

        self._ensure_open()
        if isinstance(command_id, ActionReceipt):
            command_id = command_id.command_id
        current = self._store.get_command(command_id)
        if current is None:
            raise ActionCommandNotFoundError(command_id)
        if current.status is not ActionStatus.PENDING:
            return current

        attempted = self._store.increment_attempts(command_id, _timestamp(_utc_now()))
        request = ActionRequest(
            action=attempted.action,
            case_id=attempted.case_id,
            reason=attempted.reason,
            idempotency_key=attempted.idempotency_key,
            target=attempted.target,
            target_confirmation=attempted.target_confirmation,
            policy_version=attempted.policy_version,
            requested_by=attempted.requested_by,
        )
        try:
            provider_receipt = self._invoke_adapter(self._adapter_for(request), request)
        except (ActionAdapterTimeout, TimeoutError) as exc:
            # A timeout gives no reliable provider outcome.  Keep pending so
            # a subsequent worker can safely retry with the same idempotency key.
            return self._store.finish_command(
                command_id,
                status=ActionStatus.PENDING,
                provider_receipt=None,
                error=_safe_error(exc, fallback="provider action timed out"),
                now=_timestamp(_utc_now()),
            )
        except Exception as exc:
            return self._store.finish_command(
                command_id,
                status=ActionStatus.FAILED,
                provider_receipt=None,
                error=_safe_error(exc, fallback="provider action failed"),
                now=_timestamp(_utc_now()),
            )
        return self._store.finish_command(
            command_id,
            status=ActionStatus.SUCCEEDED,
            provider_receipt=provider_receipt,
            error=None,
            now=_timestamp(_utc_now()),
        )

    process_one = process_command
    process_action = process_command
    worker_process_command = process_command

    def process_pending(self, *, limit: int = 10) -> list[ActionReceipt]:
        """Worker operation that processes a bounded batch of pending commands."""

        self._ensure_open()
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        command_ids = self._store.pending_command_ids(limit)
        return [self.process_command(command_id) for command_id in command_ids]

    worker_process = process_pending
    process_pending_commands = process_pending
    run_worker = process_pending
    run_once = process_pending

    def process(self, command_id: Optional[str] = None, *, limit: int = 10) -> ActionReceipt | list[ActionReceipt]:
        """Compatibility worker entry point for one command or a batch."""

        return self.process_command(command_id) if command_id is not None else self.process_pending(limit=limit)

    def close(self) -> None:
        if not self._closed:
            self._store.close()
            self._closed = True

    def __enter__(self) -> "ActionOutbox":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


# A semantic alias helps callers that use ``Service`` for all backend stores.
ActionOutboxService = ActionOutbox


__all__ = [
    "ACTION_ENVIRONMENT_VARS",
    "ACTION_POSTGRES_DSN_ENV_VARS",
    "ACTION_SQLITE_PATH_ENV",
    "ACTION_SQLITE_PATH_ENV_VARS",
    "ActionCommandNotFoundError",
    "ActionOutbox",
    "ActionOutboxBackendUnavailable",
    "ActionOutboxConfig",
    "ActionOutboxConfiguration",
    "ActionOutboxConfigurationError",
    "ActionOutboxError",
    "ActionOutboxIdempotencyConflict",
    "ActionOutboxService",
    "DEFAULT_SQLITE_PATH",
    "IdempotencyConflictError",
    "load_action_outbox_configuration",
    "load_action_outbox_config",
    "is_production_environment",
    "validate_action_outbox_configuration",
    "validate_production_action_outbox_configuration",
]
