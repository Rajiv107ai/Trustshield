"""Versioned, privacy-preserving audit contracts.

The audit boundary is deliberately smaller than an application request or a
model response.  Audit records contain decision/action facts and a narrow set
of operational metadata; callers cannot pass an arbitrary request payload to
the audit store.  Identifiers in this module are expected to be opaque or
tokenized upstream.  Raw names, contact details, credentials, and secrets are
rejected by the contract rather than being redacted after persistence.

``AuditRecordV1`` is storage-neutral.  The service layer adds the sequence and
hash-chain values before writing it to SQLite or PostgreSQL.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, ClassVar, Literal, Mapping, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


AUDIT_SCHEMA_VERSION = "audit.v1"
AUDIT_EVENT_TYPES = ("decision", "action")
AUDIT_GENESIS_HASH = "0" * 64

# The metadata map is intentionally finite.  Values are scalar operational
# facts only; identifiers must already be opaque/tokenized.  Adding a field to
# this set is a reviewable privacy decision instead of an accidental logging
# change at a call site.
AUDIT_METADATA_ALLOWLIST = frozenset(
    {
        "action_result",
        "action_type",
        "case_id",
        "confidence",
        "confidence_bucket",
        "correlation_id",
        "decision_source",
        "entity_id",
        "entity_type",
        "environment",
        "feature_source",
        "feature_schema_hash",
        "artifact_manifest_hash",
        "conformal_set",
        "fraud_probability",
        "http_status",
        "is_simulation",
        "latency_ms",
        "listing_id",
        "model_name",
        "model_stage",
        "order_id",
        "outcome",
        "policy_decision",
        "policy_version",
        "queue",
        "request_id",
        "resource_id",
        "resource_type",
        "risk_label",
        "risk_probability",
        "risk_score",
        "review_reason",
        "score_bucket",
        "seller_id",
        "source",
        "status",
        "trust_score",
        "trace_id",
        "transaction_id",
        "buyer_id",
        "version",
    }
)

# These names are kept separately so callers get a clear privacy error when
# they try a likely secret/PII field rather than a generic unknown-field error.
AUDIT_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "address",
        "api_key",
        "authorization",
        "bank_account",
        "card_number",
        "cookie",
        "cvv",
        "email",
        "first_name",
        "full_name",
        "last_name",
        "password",
        "phone",
        "raw_payload",
        "refresh_token",
        "secret",
        "shipping_address",
        "street_address",
        "token",
        "access_token",
    }
)

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_CODE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_AuditScalar = Union[str, int, float, bool, None]


def _utc_datetime(value: datetime | str, *, field_name: str) -> datetime:
    """Normalize a timestamp to UTC and reject naive timestamps."""

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a valid datetime") from exc
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a valid datetime")
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _safe_text(value: str, *, field_name: str, max_length: int = 256) -> str:
    """Validate a non-secret, non-control-character string field."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > max_length or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} contains unsupported characters")
    lowered = value.lower()
    # These checks cover accidental credential-shaped values without trying to
    # claim that arbitrary opaque identifiers can be cryptographically proven
    # safe by a schema.
    if "@" in value or "-----begin " in lowered or "bearer " in lowered:
        raise ValueError(f"{field_name} must be opaque and secret-free")
    if any(marker in lowered for marker in ("api_key", "access_token", "refresh_token", "password=")):
        raise ValueError(f"{field_name} must be opaque and secret-free")
    return value


def validate_audit_metadata(value: Mapping[str, Any] | None) -> dict[str, _AuditScalar]:
    """Validate and copy the allowlisted scalar metadata map.

    The returned dictionary is detached from the caller and therefore safe to
    serialize after the call.  Nested objects are rejected because they would
    make it too easy to smuggle a request or response payload into the audit
    record.
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping")

    result: dict[str, _AuditScalar] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError("metadata keys must be strings")
        key = raw_key.strip()
        if key in AUDIT_FORBIDDEN_FIELD_NAMES:
            raise ValueError(f"metadata field '{key}' is forbidden")
        if key not in AUDIT_METADATA_ALLOWLIST:
            raise ValueError(f"metadata field '{key}' is not allowlisted")
        if isinstance(raw_value, bool):
            result[key] = raw_value
        elif isinstance(raw_value, int) and not isinstance(raw_value, bool):
            result[key] = raw_value
        elif isinstance(raw_value, float):
            if not math.isfinite(raw_value):
                raise ValueError(f"metadata field '{key}' must be finite")
            result[key] = raw_value
        elif isinstance(raw_value, str):
            result[key] = _safe_text(raw_value, field_name=f"metadata.{key}")
        elif raw_value is None:
            result[key] = None
        else:
            raise ValueError(f"metadata field '{key}' must be a scalar")
    return result


class AuditRecordV1(BaseModel):
    """A single append-only decision or action audit record."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    schema_version: Literal[AUDIT_SCHEMA_VERSION] = Field(
        default=AUDIT_SCHEMA_VERSION,
        description="Version of the durable audit record contract",
    )
    sequence: int = Field(ge=1, description="Monotonic store sequence number")
    record_id: str = Field(
        min_length=1,
        max_length=128,
        validation_alias=AliasChoices("record_id", "id"),
    )
    event_type: Literal["decision", "action"] = Field(
        validation_alias=AliasChoices("event_type", "record_type", "kind")
    )
    event_name: str = Field(min_length=1, max_length=128)
    occurred_at: datetime = Field(
        validation_alias=AliasChoices("occurred_at", "timestamp", "created_at")
    )
    actor_id: Optional[str] = Field(default=None, max_length=256)
    subject_id: Optional[str] = Field(default=None, max_length=256)
    decision: Optional[str] = Field(default=None, max_length=64)
    action: Optional[str] = Field(default=None, max_length=128)
    outcome: Optional[str] = Field(default=None, max_length=128)
    model_version: Optional[str] = Field(default=None, max_length=128)
    reason_codes: list[str] = Field(default_factory=list, max_length=32)
    metadata: dict[str, _AuditScalar] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata", "fields"),
    )
    previous_hash: str = Field(
        min_length=64,
        max_length=64,
        validation_alias=AliasChoices("previous_hash", "prev_hash"),
    )
    record_hash: str = Field(
        min_length=64,
        max_length=64,
        validation_alias=AliasChoices("record_hash", "hash"),
    )

    FORBIDDEN_FIELDS: ClassVar[frozenset[str]] = AUDIT_FORBIDDEN_FIELD_NAMES
    ALLOWLISTED_METADATA_FIELDS: ClassVar[frozenset[str]] = AUDIT_METADATA_ALLOWLIST

    @field_validator("occurred_at")
    @classmethod
    def occurred_at_must_be_utc(cls, value: datetime) -> datetime:
        return _utc_datetime(value, field_name="occurred_at")

    @field_validator("record_id", "event_name", "actor_id", "subject_id", "decision", "action", "outcome", "model_version")
    @classmethod
    def text_fields_must_be_safe(cls, value: Optional[str], info) -> Optional[str]:
        if value is None:
            return None
        return _safe_text(value, field_name=info.field_name)

    @field_validator("reason_codes")
    @classmethod
    def reason_codes_must_be_codes(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for code in value:
            if not isinstance(code, str) or not _CODE.fullmatch(code.strip()):
                raise ValueError("reason_codes must contain short machine-readable codes")
            normalized.append(code.strip())
        return normalized

    @field_validator("metadata")
    @classmethod
    def metadata_must_be_allowlisted(cls, value: Mapping[str, Any]) -> dict[str, _AuditScalar]:
        return validate_audit_metadata(value)

    @field_validator("previous_hash", "record_hash")
    @classmethod
    def hashes_must_be_sha256(cls, value: str) -> str:
        if not _HEX_64.fullmatch(value):
            raise ValueError("hashes must be lowercase SHA-256 hex values")
        return value

    @model_validator(mode="after")
    def event_payload_matches_type(self) -> "AuditRecordV1":
        if self.event_type == "decision" and not self.decision:
            raise ValueError("decision records require a decision value")
        if self.event_type == "action" and not self.action:
            raise ValueError("action records require an action value")
        return self

    @property
    def event_kind(self) -> str:
        """Compatibility-friendly name for the decision/action discriminator."""

        return self.event_type

    @property
    def record_type(self) -> str:
        """Alias used by storage adapters that call the discriminator a type."""

        return self.event_type

    @property
    def created_at(self) -> datetime:
        """Compatibility alias for the persisted event timestamp."""

        return self.occurred_at

    @property
    def prev_hash(self) -> str:
        """Compatibility alias for the preceding chain hash."""

        return self.previous_hash

    @property
    def hash(self) -> str:
        """Compatibility alias for the current chain hash."""

        return self.record_hash

    def __getitem__(self, key: str) -> Any:
        """Provide read-only mapping-style access for storage adapters."""

        return getattr(self, key)

    @property
    def transaction_id(self) -> Optional[str]:
        """Alias used by transaction-scoring callers."""

        return self.subject_id

    @property
    def entity_id(self) -> Optional[str]:
        """Alias used by generic action/investigation callers."""

        return self.subject_id

    def hash_material(self) -> dict[str, Any]:
        """Return exactly the fields covered by ``record_hash``."""

        return audit_hash_material(self)


AuditRecord = AuditRecordV1
DecisionAuditRecord = AuditRecordV1
ActionAuditRecord = AuditRecordV1
AuditEvent = AuditRecordV1
AuditEventV1 = AuditRecordV1
DecisionAuditEvent = AuditRecordV1
ActionAuditEvent = AuditRecordV1


def audit_hash_material(record: AuditRecordV1 | Mapping[str, Any]) -> dict[str, Any]:
    """Build deterministic hash input without including the hash itself."""

    if isinstance(record, AuditRecordV1):
        values: Mapping[str, Any] = record.model_dump(mode="python")
    else:
        values = record

    occurred_at = values.get("occurred_at")
    if isinstance(occurred_at, datetime):
        occurred_at = _utc_datetime(occurred_at, field_name="occurred_at").isoformat()
    elif isinstance(occurred_at, str):
        occurred_at = _utc_datetime(occurred_at, field_name="occurred_at").isoformat()

    reason_codes = values.get("reason_codes") or []
    metadata = values.get("metadata") or {}
    return {
        "schema_version": values.get("schema_version", AUDIT_SCHEMA_VERSION),
        "sequence": int(values["sequence"]),
        "record_id": values["record_id"],
        "event_type": values["event_type"],
        "event_name": values["event_name"],
        "occurred_at": occurred_at,
        "actor_id": values.get("actor_id"),
        "subject_id": values.get("subject_id"),
        "decision": values.get("decision"),
        "action": values.get("action"),
        "outcome": values.get("outcome"),
        "model_version": values.get("model_version"),
        "reason_codes": list(reason_codes),
        "metadata": dict(metadata),
        "previous_hash": values["previous_hash"],
    }


def canonical_audit_json(record: AuditRecordV1 | Mapping[str, Any]) -> str:
    """Serialize hash input deterministically for all supported backends."""

    return json.dumps(
        audit_hash_material(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compute_audit_hash(record: AuditRecordV1 | Mapping[str, Any]) -> str:
    """Compute the SHA-256 hash for an audit record's canonical contents."""

    return sha256(canonical_audit_json(record).encode("utf-8")).hexdigest()


# Short aliases make the chain primitive easy to discover without duplicating
# the implementation or introducing a second hash format.
compute_record_hash = compute_audit_hash


__all__ = [
    "AUDIT_EVENT_TYPES",
    "AUDIT_FORBIDDEN_FIELD_NAMES",
    "AUDIT_GENESIS_HASH",
    "AUDIT_METADATA_ALLOWLIST",
    "AUDIT_SCHEMA_VERSION",
    "ActionAuditEvent",
    "ActionAuditRecord",
    "AuditEvent",
    "AuditEventV1",
    "AuditRecord",
    "AuditRecordV1",
    "DecisionAuditEvent",
    "DecisionAuditRecord",
    "audit_hash_material",
    "canonical_audit_json",
    "compute_audit_hash",
    "compute_record_hash",
    "validate_audit_metadata",
]
