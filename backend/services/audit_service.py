"""Durable append-only audit storage for TrustShield.

The development backend is intentionally small: it uses Python's standard
library ``sqlite3`` module, no ORM, and one append-only table.  Production
configuration is explicit.  A production process may select PostgreSQL only
through an externally configured PostgreSQL DSN; if audit durability is
required and that DSN is absent, configuration fails closed before a SQLite
database can be selected.

The optional PostgreSQL path uses a DB-API driver (``psycopg`` or
``psycopg2``) when one is supplied by the deployment.  Neither driver is a
runtime dependency of the development store.  This keeps local tests and
development lightweight while preventing accidental SQLite persistence in a
production process.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional

from backend.contracts.audit import (
    AUDIT_GENESIS_HASH,
    AUDIT_SCHEMA_VERSION,
    AuditRecordV1,
    audit_hash_material,
    compute_audit_hash,
    validate_audit_metadata,
)


AUDIT_ENVIRONMENT_VARS = ("TRUSTSHIELD_ENV", "ENVIRONMENT", "ENV")
AUDIT_SQLITE_PATH_ENV = "TRUSTSHIELD_AUDIT_SQLITE_PATH"
AUDIT_POSTGRES_DSN_ENV_VARS = (
    "TRUSTSHIELD_AUDIT_POSTGRES_DSN",
    "TRUSTSHIELD_POSTGRES_DSN",
    "AUDIT_POSTGRES_DSN",
    "POSTGRES_DSN",
    "TRUSTSHIELD_AUDIT_POSTGRES_URL",
    "AUDIT_POSTGRES_URL",
    "TRUSTSHIELD_AUDIT_DATABASE_URL",
    "AUDIT_DATABASE_URL",
    "DATABASE_URL",
)
AUDIT_DURABILITY_ENV_VARS = (
    "TRUSTSHIELD_AUDIT_DURABILITY_REQUIRED",
    "TRUSTSHIELD_REQUIRE_AUDIT_DURABILITY",
    "TRUSTSHIELD_REQUIRE_DURABLE_AUDIT",
    "TRUSTSHIELD_AUDIT_REQUIRED",
    "AUDIT_DURABILITY_REQUIRED",
)
DEFAULT_SQLITE_PATH = ".runtime/audit.sqlite3"


class AuditConfigurationError(RuntimeError):
    """Raised when audit persistence cannot be configured safely."""


class AuditBackendUnavailable(AuditConfigurationError):
    """Raised when a selected durable backend cannot be reached or loaded."""


class AuditIntegrityError(RuntimeError):
    """Raised when a write is attempted against a tampered audit chain."""


class AuditUnavailableError(RuntimeError):
    """Raised when an optional/disabled audit service receives a write."""


@dataclass(frozen=True)
class AuditConfiguration:
    """Resolved audit backend settings with secrets omitted from repr output."""

    environment: str
    backend: Literal["sqlite", "postgresql", "disabled"]
    sqlite_path: Optional[str]
    durability_required: bool
    postgres_dsn: Optional[str] = field(default=None, repr=False)


@dataclass(frozen=True)
class IntegrityReport:
    """Result of a complete hash-chain verification pass.

    ``__bool__`` makes the report convenient in guards while retaining a
    stable error and sequence number for diagnostics and tests.
    """

    valid: bool
    checked_records: int
    last_sequence: Optional[int] = None
    first_invalid_sequence: Optional[int] = None
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.valid

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bool):
            return self.valid == other
        if isinstance(other, IntegrityReport):
            return self.as_dict() == other.as_dict()
        return NotImplemented

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access for lightweight integration callers."""

        return getattr(self, key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checked_records": self.checked_records,
            "last_sequence": self.last_sequence,
            "first_invalid_sequence": self.first_invalid_sequence,
            "error": self.error,
        }


def _first_environment_value(names: Iterable[str]) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _truthy(value: Optional[str]) -> Optional[bool]:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise AuditConfigurationError("Audit durability flag must be a boolean value")


def is_production_environment(environment: Optional[str] = None) -> bool:
    """Return whether the selected or process environment is production."""

    value = environment if environment is not None else _first_environment_value(AUDIT_ENVIRONMENT_VARS)
    return (value or "development").strip().lower() in {"production", "prod"}


def _is_postgres_dsn(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.startswith("postgresql://") or lowered.startswith("postgres://")


def _configured_postgres_dsn(explicit_dsn: Optional[str]) -> Optional[str]:
    """Resolve an audit-specific DSN without hijacking a non-Postgres URL."""

    if explicit_dsn is not None and explicit_dsn.strip():
        dsn = explicit_dsn.strip()
        if not _is_postgres_dsn(dsn):
            raise AuditConfigurationError("Configured audit database URL must be a PostgreSQL DSN")
        return dsn

    for name in AUDIT_POSTGRES_DSN_ENV_VARS:
        value = os.getenv(name)
        if value is None or not value.strip():
            continue
        dsn = value.strip()
        # DATABASE_URL is commonly used by unrelated local tooling.  Only
        # treat it as an audit setting when it is actually a PostgreSQL URL.
        if name == "DATABASE_URL" and not _is_postgres_dsn(dsn):
            continue
        if not _is_postgres_dsn(dsn):
            raise AuditConfigurationError("Configured audit database URL must be a PostgreSQL DSN")
        return dsn
    return None


def load_audit_configuration(
    *,
    sqlite_path: str | Path | None = None,
    postgres_dsn: Optional[str] = None,
    environment: Optional[str] = None,
    durability_required: Optional[bool] = None,
    require_durability: Optional[bool] = None,
    require_audit_durability: Optional[bool] = None,
) -> AuditConfiguration:
    """Resolve audit storage and enforce the production fail-closed policy.

    Development/test environments default to a local SQLite file.  Production
    defaults to requiring durable audit storage.  An explicit production
    ``durability_required=false`` setting is allowed for a deliberately
    degraded deployment, but that deployment is represented as ``disabled``;
    it never silently falls back to SQLite.
    """

    aliases = [
        flag
        for flag in (durability_required, require_durability, require_audit_durability)
        if flag is not None
    ]
    if aliases and any(flag != aliases[0] for flag in aliases[1:]):
        raise ValueError("durability flags cannot disagree")
    if durability_required is None:
        durability_required = aliases[0] if aliases else None

    resolved_environment = (
        environment.strip().lower()
        if environment is not None and environment.strip()
        else (_first_environment_value(AUDIT_ENVIRONMENT_VARS) or "development").strip().lower()
    )
    production = resolved_environment in {"production", "prod"}

    if durability_required is None:
        raw_required = _first_environment_value(AUDIT_DURABILITY_ENV_VARS)
        parsed_required = _truthy(raw_required)
        durability_required = production if parsed_required is None else parsed_required

    dsn = _configured_postgres_dsn(postgres_dsn)

    if production:
        if durability_required and not dsn:
            raise AuditConfigurationError(
                "CRITICAL AUDIT DURABILITY: production requires an externally configured PostgreSQL DSN"
            )
        if dsn:
            return AuditConfiguration(
                environment=resolved_environment,
                backend="postgresql",
                sqlite_path=None,
                durability_required=bool(durability_required),
                postgres_dsn=dsn,
            )
        return AuditConfiguration(
            environment=resolved_environment,
            backend="disabled",
            sqlite_path=None,
            durability_required=False,
            postgres_dsn=None,
        )

    if dsn:
        return AuditConfiguration(
            environment=resolved_environment,
            backend="postgresql",
            sqlite_path=None,
            durability_required=bool(durability_required),
            postgres_dsn=dsn,
        )

    resolved_sqlite_path = str(
        sqlite_path
        if sqlite_path is not None
        else (os.getenv(AUDIT_SQLITE_PATH_ENV, "").strip() or DEFAULT_SQLITE_PATH)
    )
    return AuditConfiguration(
        environment=resolved_environment,
        backend="sqlite",
        sqlite_path=resolved_sqlite_path,
        durability_required=bool(durability_required),
    )


def validate_audit_configuration(**kwargs: Any) -> AuditConfiguration:
    """Validate audit settings and return the resolved configuration."""

    return load_audit_configuration(**kwargs)


validate_production_audit_configuration = validate_audit_configuration
AuditConfig = AuditConfiguration


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_records (
    sequence INTEGER PRIMARY KEY,
    schema_version TEXT NOT NULL,
    record_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL CHECK (event_type IN ('decision', 'action')),
    event_name TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    actor_id TEXT,
    subject_id TEXT,
    decision TEXT,
    action TEXT,
    outcome TEXT,
    model_version TEXT,
    reason_codes_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL UNIQUE
)

;
CREATE TABLE IF NOT EXISTS audit_chain_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    last_sequence INTEGER NOT NULL,
    last_hash TEXT NOT NULL
)
"""

_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_records (
    sequence BIGINT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    record_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL CHECK (event_type IN ('decision', 'action')),
    event_name TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    actor_id TEXT,
    subject_id TEXT,
    decision TEXT,
    action TEXT,
    outcome TEXT,
    model_version TEXT,
    reason_codes_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL UNIQUE
)

;
CREATE TABLE IF NOT EXISTS audit_chain_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    last_sequence BIGINT NOT NULL,
    last_hash TEXT NOT NULL
)
"""

_SELECT_COLUMNS = (
    "sequence, schema_version, record_id, event_type, event_name, occurred_at, "
    "actor_id, subject_id, decision, action, outcome, model_version, "
    "reason_codes_json, metadata_json, previous_hash, record_hash"
)


def _normalise_timestamp(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("occurred_at must be a valid datetime") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("occurred_at must be timezone-aware")
    return value.astimezone(timezone.utc)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _row_to_values(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a database row into contract-shaped Python values."""

    try:
        reason_codes = json.loads(row["reason_codes_json"])
        metadata = json.loads(row["metadata_json"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("audit JSON column is malformed") from exc
    if not isinstance(reason_codes, list) or not isinstance(metadata, dict):
        raise ValueError("audit JSON column has an invalid shape")
    return {
        "schema_version": row["schema_version"],
        "sequence": int(row["sequence"]),
        "record_id": row["record_id"],
        "event_type": row["event_type"],
        "event_name": row["event_name"],
        "occurred_at": row["occurred_at"],
        "actor_id": row["actor_id"],
        "subject_id": row["subject_id"],
        "decision": row["decision"],
        "action": row["action"],
        "outcome": row["outcome"],
        "model_version": row["model_version"],
        "reason_codes": reason_codes,
        "metadata": metadata,
        "previous_hash": row["previous_hash"],
        "record_hash": row["record_hash"],
    }


def _verify_rows(rows: Iterable[Mapping[str, Any]]) -> IntegrityReport:
    expected_sequence = 1
    expected_previous_hash = AUDIT_GENESIS_HASH
    checked = 0
    last_sequence: Optional[int] = None
    for row in rows:
        checked += 1
        try:
            values = _row_to_values(row)
            sequence = values["sequence"]
            last_sequence = sequence
            if sequence != expected_sequence:
                return IntegrityReport(
                    valid=False,
                    checked_records=checked,
                    last_sequence=last_sequence,
                    first_invalid_sequence=sequence,
                    error="audit sequence is not contiguous",
                )
            if values["previous_hash"] != expected_previous_hash:
                return IntegrityReport(
                    valid=False,
                    checked_records=checked,
                    last_sequence=last_sequence,
                    first_invalid_sequence=sequence,
                    error="audit previous hash does not match the chain",
                )
            expected_hash = compute_audit_hash(audit_hash_material(values))
            if values["record_hash"] != expected_hash:
                return IntegrityReport(
                    valid=False,
                    checked_records=checked,
                    last_sequence=last_sequence,
                    first_invalid_sequence=sequence,
                    error="audit record hash mismatch",
                )
            # Contract validation also catches a malicious edit that rewrites
            # the stored hash but violates the safe record shape.  The error
            # intentionally omits field values.
            AuditRecordV1(**values)
            expected_previous_hash = values["record_hash"]
            expected_sequence += 1
        except Exception as exc:  # malformed persisted data is an integrity failure
            sequence = None
            try:
                sequence = int(row["sequence"])
            except Exception:
                pass
            return IntegrityReport(
                valid=False,
                checked_records=checked,
                last_sequence=last_sequence,
                first_invalid_sequence=sequence,
                error=f"audit record is malformed: {type(exc).__name__}",
            )
    return IntegrityReport(
        valid=True,
        checked_records=checked,
        last_sequence=last_sequence,
    )


def _verify_chain_anchor(
    report: IntegrityReport,
    rows: list[Mapping[str, Any]],
    state: Optional[Mapping[str, Any]],
) -> IntegrityReport:
    """Check the durable head anchor so suffix deletion is detectable."""

    if not report.valid:
        return report
    if not rows:
        if state is None:
            return report
        return IntegrityReport(
            valid=False,
            checked_records=0,
            error="audit chain state points to missing records",
        )
    if state is None:
        return IntegrityReport(
            valid=False,
            checked_records=report.checked_records,
            last_sequence=report.last_sequence,
            error="audit chain state is missing",
        )
    try:
        state_sequence = int(state["last_sequence"])
        state_hash = str(state["last_hash"])
    except (KeyError, TypeError, ValueError):
        return IntegrityReport(
            valid=False,
            checked_records=report.checked_records,
            last_sequence=report.last_sequence,
            error="audit chain state is malformed",
        )
    if state_sequence != report.last_sequence or state_hash != rows[-1]["record_hash"]:
        return IntegrityReport(
            valid=False,
            checked_records=report.checked_records,
            last_sequence=report.last_sequence,
            first_invalid_sequence=report.last_sequence,
            error="audit chain head anchor mismatch",
        )
    return report


class _PostgresBackend:
    """Small DB-API PostgreSQL adapter loaded only when a DSN is configured."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._connection: Any = None
        self._schema_ready = False

    def _connect(self) -> Any:
        if self._connection is not None:
            return self._connection
        driver: Any
        try:  # pragma: no cover - deployment-only optional dependency
            import psycopg as driver
        except ImportError:
            try:  # pragma: no cover - deployment-only optional dependency
                import psycopg2 as driver
            except ImportError as exc:  # pragma: no cover - tested through guard message only
                raise AuditBackendUnavailable(
                    "PostgreSQL audit storage requires the externally managed psycopg driver"
                ) from exc
        try:  # pragma: no cover - deployment-only optional dependency
            self._connection = driver.connect(self._dsn)
            self._connection.autocommit = False
        except Exception as exc:  # pragma: no cover - deployment-only optional dependency
            raise AuditBackendUnavailable("PostgreSQL audit storage could not be connected") from exc
        return self._connection

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        connection = self._connect()
        cursor = connection.cursor()
        try:  # pragma: no cover - deployment-only optional dependency
            for statement in _POSTGRES_SCHEMA.split(";"):
                if statement.strip():
                    cursor.execute(statement)
            connection.commit()
            self._schema_ready = True
        except Exception as exc:  # pragma: no cover - deployment-only optional dependency
            connection.rollback()
            raise AuditBackendUnavailable("PostgreSQL audit schema could not be initialized") from exc
        finally:
            cursor.close()

    @staticmethod
    def _rows(cursor: Any) -> list[dict[str, Any]]:  # pragma: no cover - deployment-only optional dependency
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def append(self, values: dict[str, Any]) -> AuditRecordV1:  # pragma: no cover - deployment-only optional dependency
        self._ensure_schema()
        connection = self._connect()
        cursor = connection.cursor()
        try:
            cursor.execute("LOCK TABLE audit_records IN EXCLUSIVE MODE")
            cursor.execute("SELECT sequence, record_hash FROM audit_records ORDER BY sequence DESC LIMIT 1")
            latest = cursor.fetchone()
            sequence = int(latest[0]) + 1 if latest else 1
            previous_hash = latest[1] if latest else AUDIT_GENESIS_HASH
            record_values = {
                **values,
                "schema_version": AUDIT_SCHEMA_VERSION,
                "sequence": sequence,
                "previous_hash": previous_hash,
                "record_id": values.get("record_id") or uuid.uuid4().hex,
            }
            record_values["record_hash"] = compute_audit_hash(record_values)
            record = AuditRecordV1(**record_values)
            cursor.execute(
                "INSERT INTO audit_records (" + _SELECT_COLUMNS + ") "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    record.sequence,
                    record.schema_version,
                    record.record_id,
                    record.event_type,
                    record.event_name,
                    record.occurred_at.isoformat(),
                    record.actor_id,
                    record.subject_id,
                    record.decision,
                    record.action,
                    record.outcome,
                    record.model_version,
                    _json_dumps(record.reason_codes),
                    _json_dumps(record.metadata),
                    record.previous_hash,
                    record.record_hash,
                ),
            )
            cursor.execute(
                "INSERT INTO audit_chain_state (singleton, last_sequence, last_hash) VALUES (1, %s, %s) "
                "ON CONFLICT (singleton) DO UPDATE SET last_sequence = EXCLUDED.last_sequence, last_hash = EXCLUDED.last_hash",
                (record.sequence, record.record_hash),
            )
            connection.commit()
            return record
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()

    def read_records(
        self,
        *,
        limit: int,
        event_type: Optional[str] = None,
        subject_id: Optional[str] = None,
        after_sequence: Optional[int] = None,
        before_sequence: Optional[int] = None,
    ) -> list[AuditRecordV1]:  # pragma: no cover - deployment-only optional dependency
        self._ensure_schema()
        connection = self._connect()
        cursor = connection.cursor()
        clauses: list[str] = []
        params: list[Any] = []
        if event_type is not None:
            clauses.append("event_type = %s")
            params.append(event_type)
        if subject_id is not None:
            clauses.append("subject_id = %s")
            params.append(subject_id)
        if after_sequence is not None:
            clauses.append("sequence > %s")
            params.append(after_sequence)
        if before_sequence is not None:
            clauses.append("sequence < %s")
            params.append(before_sequence)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cursor.execute(
            "SELECT " + _SELECT_COLUMNS + " FROM audit_records" + where + " ORDER BY sequence ASC LIMIT %s",
            [*params, limit],
        )
        try:
            rows = self._rows(cursor)
        finally:
            cursor.close()
            connection.rollback()
        return [AuditRecordV1(**_row_to_values(row)) for row in rows]

    def verify_integrity(self) -> IntegrityReport:  # pragma: no cover - deployment-only optional dependency
        self._ensure_schema()
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute("SELECT " + _SELECT_COLUMNS + " FROM audit_records ORDER BY sequence ASC")
        try:
            rows = self._rows(cursor)
        finally:
            cursor.close()
            connection.rollback()
        state_cursor = connection.cursor()
        state_cursor.execute("SELECT last_sequence, last_hash FROM audit_chain_state WHERE singleton = 1")
        state_row = state_cursor.fetchone()
        state_cursor.close()
        connection.rollback()
        state = None if state_row is None else {
            "last_sequence": state_row[0],
            "last_hash": state_row[1],
        }
        return _verify_chain_anchor(_verify_rows(rows), rows, state)

    def get_record(self, record_id: str) -> Optional[AuditRecordV1]:  # pragma: no cover - deployment-only optional dependency
        self._ensure_schema()
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT " + _SELECT_COLUMNS + " FROM audit_records WHERE record_id = %s",
            (record_id,),
        )
        try:
            rows = self._rows(cursor)
        finally:
            cursor.close()
            connection.rollback()
        return AuditRecordV1(**_row_to_values(rows[0])) if rows else None

    def count(self) -> int:  # pragma: no cover - deployment-only optional dependency
        self._ensure_schema()
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM audit_records")
        try:
            count = int(cursor.fetchone()[0])
        finally:
            cursor.close()
            connection.rollback()
        return count

    def close(self) -> None:  # pragma: no cover - deployment-only optional dependency
        if self._connection is not None:
            self._connection.close()
            self._connection = None


class AuditService:
    """Append-only audit service with SQLite development persistence."""

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
        durability_required: Optional[bool] = None,
        require_durability: Optional[bool] = None,
        audit_durability_required: Optional[bool] = None,
        require_audit_durability: Optional[bool] = None,
    ):
        paths = [path for path in (sqlite_path, db_path, database_path) if path is not None]
        if paths and any(str(path) != str(paths[0]) for path in paths[1:]):
            raise ValueError("sqlite_path, db_path, and database_path cannot disagree")
        required_flags = [
            flag
            for flag in (
                durability_required,
                require_durability,
                audit_durability_required,
                require_audit_durability,
            )
            if flag is not None
        ]
        if required_flags and any(flag != required_flags[0] for flag in required_flags[1:]):
            raise ValueError("durability flags cannot disagree")
        resolved_path = paths[0] if paths else None
        resolved_required = required_flags[0] if required_flags else None
        dsn_values = [
            dsn
            for dsn in (postgres_dsn, database_url, database_dsn)
            if dsn is not None
        ]
        if dsn_values and any(dsn != dsn_values[0] for dsn in dsn_values[1:]):
            raise ValueError("PostgreSQL DSN aliases cannot disagree")
        resolved_dsn = dsn_values[0] if dsn_values else None
        self.configuration = load_audit_configuration(
            sqlite_path=resolved_path,
            postgres_dsn=resolved_dsn,
            environment=environment,
            durability_required=resolved_required,
        )
        self._lock = threading.RLock()
        self._connection: Optional[sqlite3.Connection] = None
        self._postgres: Optional[_PostgresBackend] = None
        self._closed = False

        if self.configuration.backend == "sqlite":
            assert self.configuration.sqlite_path is not None
            if self.configuration.sqlite_path != ":memory:":
                Path(self.configuration.sqlite_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                self.configuration.sqlite_path,
                timeout=30,
                isolation_level=None,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            if self.configuration.sqlite_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.executescript(_SQLITE_SCHEMA)
            self._initialize_sqlite_chain_state()
        elif self.configuration.backend == "postgresql":
            assert self.configuration.postgres_dsn is not None
            self._postgres = _PostgresBackend(self.configuration.postgres_dsn)

    @classmethod
    def from_environment(cls, **kwargs: Any) -> "AuditService":
        """Construct a service from environment settings and test overrides."""

        return cls(**kwargs)

    def _initialize_sqlite_chain_state(self) -> None:
        """Create a head anchor for a pre-existing SQLite audit table."""

        assert self._connection is not None
        state = self._connection.execute(
            "SELECT singleton FROM audit_chain_state WHERE singleton = 1"
        ).fetchone()
        if state is not None:
            return
        latest = self._connection.execute(
            "SELECT sequence, record_hash FROM audit_records ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if latest is not None:
            self._connection.execute(
                "INSERT INTO audit_chain_state (singleton, last_sequence, last_hash) VALUES (1, ?, ?)",
                (latest["sequence"], latest["record_hash"]),
            )

    @property
    def backend(self) -> str:
        return self.configuration.backend

    @property
    def durability_required(self) -> bool:
        return self.configuration.durability_required

    @property
    def sqlite_path(self) -> Optional[str]:
        return self.configuration.sqlite_path

    def _ensure_available(self) -> None:
        if self._closed:
            raise AuditUnavailableError("Audit service is closed")
        if self.configuration.backend == "disabled":
            raise AuditUnavailableError("Audit persistence is disabled by configuration")

    @staticmethod
    def _merge_metadata(
        metadata: Mapping[str, Any] | None,
        fields: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if metadata is not None and fields is not None:
            raise ValueError("metadata and fields are aliases; provide only one")
        return dict(metadata if metadata is not None else (fields or {}))

    @staticmethod
    def _subject_id(
        subject_id: Optional[str],
        transaction_id: Optional[str],
        entity_id: Optional[str],
    ) -> Optional[str]:
        values = [value for value in (subject_id, transaction_id, entity_id) if value is not None]
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError("subject_id, transaction_id, and entity_id cannot disagree")
        return values[0] if values else None

    def _append_values(self, values: dict[str, Any]) -> AuditRecordV1:
        self._ensure_available()
        if self.configuration.backend == "postgresql":
            assert self._postgres is not None
            with self._lock:
                # The PostgreSQL adapter performs its own transaction and
                # chain check once the externally managed driver is loaded.
                report = self._postgres.verify_integrity()
                if not report.valid:
                    raise AuditIntegrityError(report.error or "audit chain integrity check failed")
                return self._postgres.append(values)

        assert self._connection is not None
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    "SELECT " + _SELECT_COLUMNS + " FROM audit_records ORDER BY sequence ASC"
                ).fetchall()
                state = connection.execute(
                    "SELECT last_sequence, last_hash FROM audit_chain_state WHERE singleton = 1"
                ).fetchone()
                report = _verify_chain_anchor(
                    _verify_rows(rows),
                    [dict(row) for row in rows],
                    dict(state) if state is not None else None,
                )
                if not report.valid:
                    raise AuditIntegrityError(report.error or "audit chain integrity check failed")
                latest = connection.execute(
                    "SELECT sequence, record_hash FROM audit_records ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                sequence = int(latest["sequence"]) + 1 if latest else 1
                previous_hash = latest["record_hash"] if latest else AUDIT_GENESIS_HASH
                record_values = {
                    **values,
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "sequence": sequence,
                    "previous_hash": previous_hash,
                    "record_id": values.get("record_id") or uuid.uuid4().hex,
                }
                record_values["record_hash"] = compute_audit_hash(record_values)
                record = AuditRecordV1(**record_values)
                connection.execute(
                    "INSERT INTO audit_records (" + _SELECT_COLUMNS + ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.sequence,
                        record.schema_version,
                        record.record_id,
                        record.event_type,
                        record.event_name,
                        record.occurred_at.isoformat(),
                        record.actor_id,
                        record.subject_id,
                        record.decision,
                        record.action,
                        record.outcome,
                        record.model_version,
                        _json_dumps(record.reason_codes),
                        _json_dumps(record.metadata),
                        record.previous_hash,
                        record.record_hash,
                    ),
                )
                connection.execute(
                    "INSERT INTO audit_chain_state (singleton, last_sequence, last_hash) VALUES (1, ?, ?) "
                    "ON CONFLICT(singleton) DO UPDATE SET last_sequence = excluded.last_sequence, last_hash = excluded.last_hash",
                    (record.sequence, record.record_hash),
                )
                connection.execute("COMMIT")
                return record
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def append_record(
        self,
        *,
        event_type: Literal["decision", "action"],
        event_name: str,
        occurred_at: datetime | str | None = None,
        actor_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        decision: Optional[str] = None,
        action: Optional[str] = None,
        outcome: Optional[str] = None,
        model_version: Optional[str] = None,
        reason_codes: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        fields: Mapping[str, Any] | None = None,
        record_id: Optional[str] = None,
    ) -> AuditRecordV1:
        """Append one validated event and return its stored hash-chain record."""

        if event_type not in {"decision", "action"}:
            raise ValueError("event_type must be 'decision' or 'action'")
        if event_type == "decision" and not decision:
            raise ValueError("decision records require a decision value")
        if event_type == "action" and not action:
            raise ValueError("action records require an action value")
        normalized_metadata = validate_audit_metadata(self._merge_metadata(metadata, fields))
        values = {
            "record_id": record_id,
            "event_type": event_type,
            "event_name": event_name,
            "occurred_at": _normalise_timestamp(occurred_at),
            "actor_id": actor_id,
            "subject_id": subject_id,
            "decision": decision,
            "action": action,
            "outcome": outcome,
            "model_version": model_version,
            "reason_codes": list(reason_codes or []),
            "metadata": normalized_metadata,
        }
        return self._append_values(values)

    def append_decision(
        self,
        decision: str,
        *,
        subject_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        event_name: str = "decision.recorded",
        occurred_at: datetime | str | None = None,
        actor_id: Optional[str] = None,
        model_version: Optional[str] = None,
        reason_codes: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        fields: Mapping[str, Any] | None = None,
        risk_score: Optional[float] = None,
        record_id: Optional[str] = None,
    ) -> AuditRecordV1:
        """Append a model/policy decision without accepting raw request data."""

        merged = self._merge_metadata(metadata, fields)
        if risk_score is not None:
            if "risk_score" in merged and merged["risk_score"] != risk_score:
                raise ValueError("risk_score metadata disagrees with risk_score")
            merged["risk_score"] = risk_score
        return self.append_record(
            event_type="decision",
            event_name=event_name,
            occurred_at=occurred_at,
            actor_id=actor_id,
            subject_id=self._subject_id(subject_id, transaction_id, entity_id),
            decision=decision,
            model_version=model_version,
            reason_codes=reason_codes,
            metadata=merged,
            record_id=record_id,
        )

    def append_action(
        self,
        action: str,
        *,
        subject_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        event_name: str = "action.recorded",
        occurred_at: datetime | str | None = None,
        actor_id: Optional[str] = None,
        outcome: Optional[str] = None,
        reason_codes: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        fields: Mapping[str, Any] | None = None,
        record_id: Optional[str] = None,
    ) -> AuditRecordV1:
        """Append an investigator/operator action with safe scalar metadata."""

        return self.append_record(
            event_type="action",
            event_name=event_name,
            occurred_at=occurred_at,
            actor_id=actor_id,
            subject_id=self._subject_id(subject_id, transaction_id, entity_id),
            action=action,
            outcome=outcome,
            reason_codes=reason_codes,
            metadata=self._merge_metadata(metadata, fields),
            record_id=record_id,
        )

    # Explicit aliases keep the service discoverable for callers that name the
    # operation as a record rather than an append.
    append_decision_record = append_decision
    append_action_record = append_action
    append_decision_event = append_decision
    append_action_event = append_action
    record_decision = append_decision
    record_action = append_action

    def append(self, record: AuditRecordV1 | Mapping[str, Any] | None = None, **kwargs: Any) -> AuditRecordV1:
        """Append a generic record, recomputing sequence and hashes locally."""

        if record is not None and kwargs:
            raise ValueError("provide either record or keyword fields, not both")
        if record is None:
            return self.append_record(**kwargs)
        if isinstance(record, AuditRecordV1):
            values = record.model_dump(mode="python")
        elif isinstance(record, Mapping):
            values = dict(record)
        else:
            raise TypeError("record must be an AuditRecordV1 or mapping")
        aliases = {
            "record_type": "event_type",
            "kind": "event_type",
            "id": "record_id",
            "timestamp": "occurred_at",
            "created_at": "occurred_at",
            "prev_hash": "previous_hash",
            "hash": "record_hash",
            "transaction_id": "subject_id",
            "entity_id": "subject_id",
        }
        for alias, canonical in aliases.items():
            if canonical not in values and alias in values:
                values[canonical] = values[alias]
            values.pop(alias, None)
        for storage_field in ("schema_version", "sequence", "previous_hash", "record_hash"):
            values.pop(storage_field, None)
        if "occurred_at" in values:
            values["occurred_at"] = _normalise_timestamp(values["occurred_at"])
        return self.append_record(**values)

    def read_records(
        self,
        *,
        limit: int = 100,
        event_type: Optional[Literal["decision", "action"]] = None,
        subject_id: Optional[str] = None,
        after_sequence: Optional[int] = None,
        before_sequence: Optional[int] = None,
    ) -> list[AuditRecordV1]:
        """Read records in ascending chain order."""

        if limit < 1:
            raise ValueError("limit must be at least 1")
        self._ensure_available()
        if self.configuration.backend == "postgresql":
            assert self._postgres is not None
            with self._lock:
                return self._postgres.read_records(
                    limit=limit,
                    event_type=event_type,
                    subject_id=subject_id,
                    after_sequence=after_sequence,
                    before_sequence=before_sequence,
                )
        assert self._connection is not None
        with self._lock:
            clauses: list[str] = []
            params: list[Any] = []
            if event_type is not None:
                clauses.append("event_type = ?")
                params.append(event_type)
            if subject_id is not None:
                clauses.append("subject_id = ?")
                params.append(subject_id)
            if after_sequence is not None:
                clauses.append("sequence > ?")
                params.append(after_sequence)
            if before_sequence is not None:
                clauses.append("sequence < ?")
                params.append(before_sequence)
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = self._connection.execute(
                "SELECT " + _SELECT_COLUMNS + " FROM audit_records" + where + " ORDER BY sequence ASC LIMIT ?",
                [*params, limit],
            ).fetchall()
            try:
                return [AuditRecordV1(**_row_to_values(dict(row))) for row in rows]
            except Exception as exc:
                raise AuditIntegrityError("audit record could not be decoded") from exc

    read = read_records
    list_records = read_records

    def get_record(self, record_id: str) -> Optional[AuditRecordV1]:
        """Return one record by generated/supplied record ID."""

        self._ensure_available()
        if self.configuration.backend == "postgresql":
            assert self._postgres is not None
            with self._lock:
                return self._postgres.get_record(record_id)
        assert self._connection is not None
        with self._lock:
            row = self._connection.execute(
                "SELECT " + _SELECT_COLUMNS + " FROM audit_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            return AuditRecordV1(**_row_to_values(dict(row))) if row else None

    def verify_integrity(self) -> IntegrityReport:
        """Verify sequence continuity, every link, and every record hash."""

        if self.configuration.backend == "disabled":
            return IntegrityReport(
                valid=False,
                checked_records=0,
                error="audit persistence is disabled by configuration",
            )
        self._ensure_available()
        if self.configuration.backend == "postgresql":
            assert self._postgres is not None
            with self._lock:
                return self._postgres.verify_integrity()
        assert self._connection is not None
        with self._lock:
            rows = self._connection.execute(
                "SELECT " + _SELECT_COLUMNS + " FROM audit_records ORDER BY sequence ASC"
            ).fetchall()
            state = self._connection.execute(
                "SELECT last_sequence, last_hash FROM audit_chain_state WHERE singleton = 1"
            ).fetchone()
            row_values = [dict(row) for row in rows]
            state_values = dict(state) if state is not None else None
            return _verify_chain_anchor(_verify_rows(row_values), row_values, state_values)

    verify_hash_chain = verify_integrity

    def is_integrity_valid(self) -> bool:
        return bool(self.verify_integrity())

    def verify_integrity_status(self) -> bool:
        """Boolean convenience form for health/readiness checks."""

        return self.is_integrity_valid()

    def count(self) -> int:
        self._ensure_available()
        if self.configuration.backend == "postgresql":
            assert self._postgres is not None
            with self._lock:
                return self._postgres.count()
        assert self._connection is not None
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM audit_records").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if self._postgres is not None:
                self._postgres.close()
            self._closed = True

    def __enter__(self) -> "AuditService":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


AuditLogService = AuditService
AuditStore = AuditService
SQLiteAuditStore = AuditService
ProductionAuditConfigurationError = AuditConfigurationError


__all__ = [
    "AUDIT_DURABILITY_ENV_VARS",
    "AUDIT_ENVIRONMENT_VARS",
    "AUDIT_POSTGRES_DSN_ENV_VARS",
    "AUDIT_SQLITE_PATH_ENV",
    "AuditBackendUnavailable",
    "AuditConfig",
    "AuditConfiguration",
    "AuditConfigurationError",
    "AuditIntegrityError",
    "AuditLogService",
    "AuditStore",
    "AuditService",
    "AuditUnavailableError",
    "DEFAULT_SQLITE_PATH",
    "IntegrityReport",
    "SQLiteAuditStore",
    "is_production_environment",
    "load_audit_configuration",
    "ProductionAuditConfigurationError",
    "validate_audit_configuration",
    "validate_production_audit_configuration",
]
