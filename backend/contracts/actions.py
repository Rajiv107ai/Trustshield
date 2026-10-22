"""Strict contracts for Phase 4 operational actions.

The scoring and investigation APIs intentionally do not execute provider-side
effects.  An action request is first persisted by the action outbox and is
then processed by a worker.  This module is the storage/provider-neutral
boundary for that flow.

Only actions in :class:`ActionType` are accepted.  In particular, callers
must explicitly confirm the target with ``target_confirmation=True``; a
target identifier by itself is not enough to enqueue an operational action.
The models are strict and forbid undeclared fields so a route cannot
accidentally smuggle a request payload or provider credentials into the
outbox.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Mapping, Optional, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)
from pydantic.types import StringConstraints


ACTION_SCHEMA_VERSION = "actions.v1"
ACTIONS_V1_SCHEMA_VERSION = ACTION_SCHEMA_VERSION


class ActionType(str, Enum):
    """Allowlisted provider actions.

    The values are stable wire values.  Aliases that are useful to route
    integrations are explicit enum members rather than an open-ended string,
    which keeps validation closed while allowing the domain adapters to grow
    in a reviewable way.
    """

    PAYMENT_HOLD = "payment_hold"
    PAYMENT_RELEASE = "payment_release"
    PAYMENT_REFUND = "payment_refund"

    MARKETPLACE_HOLD_ORDER = "marketplace_hold_order"
    MARKETPLACE_SUSPEND_LISTING = "marketplace_suspend_listing"
    MARKETPLACE_RESTORE_LISTING = "marketplace_restore_listing"
    # Short names are retained as separate, explicitly allowlisted actions for
    # callers that use the marketplace provider's native vocabulary.
    MARKETPLACE_SUSPEND = "marketplace_suspend"
    MARKETPLACE_REINSTATE = "marketplace_reinstate"

    IDENTITY_LOCK_ACCOUNT = "identity_lock_account"
    IDENTITY_REQUIRE_VERIFICATION = "identity_require_verification"
    IDENTITY_UNLOCK_ACCOUNT = "identity_unlock_account"
    IDENTITY_LOCK = "identity_lock"

    GRAPH_FLAG_ENTITY = "graph_flag_entity"
    GRAPH_RECALCULATE_RISK = "graph_recalculate_risk"
    GRAPH_ADD_CASE_EDGE = "graph_add_case_edge"
    GRAPH_FLAG = "graph_flag"

    CASE_CREATE = "case_create"
    CASE_UPDATE = "case_update"
    CASE_CLOSE = "case_close"
    CASE_ADD_NOTE = "case_add_note"


# Friendly names for code that prefers the word ``Action`` or ``Allowed`` in
# type annotations.  They remain the same closed enum, not a second contract.
AllowedAction = ActionType
Action = ActionType


def _safe_text(
    value: str,
    *,
    field_name: str,
    max_length: int,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > max_length:
        raise ValueError(f"{field_name} is too long")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} contains unsupported control characters")
    lowered = value.lower()
    if "-----begin " in lowered or "bearer " in lowered:
        raise ValueError(f"{field_name} must not contain credentials")
    if any(marker in lowered for marker in ("api_key", "access_token", "refresh_token", "password=")):
        raise ValueError(f"{field_name} must not contain credentials")
    if pattern is not None and not pattern.fullmatch(value):
        raise ValueError(f"{field_name} has an invalid format")
    return value


OpaqueIdentifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]

_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class _ActionModel(BaseModel):
    """Shared strict Pydantic configuration for action contracts."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ActionTarget(_ActionModel):
    """Opaque provider target selected by the person or policy."""

    target_type: OpaqueIdentifier = Field(
        validation_alias=AliasChoices("target_type", "type", "entity_type"),
        serialization_alias="target_type",
        description="Provider target domain, for example payment or account",
    )
    target_id: OpaqueIdentifier = Field(
        validation_alias=AliasChoices("target_id", "id", "entity_id"),
        serialization_alias="target_id",
        description="Opaque identifier of the provider-side target",
    )

    @field_validator("target_type", "target_id")
    @classmethod
    def target_values_must_be_safe(cls, value: str, info) -> str:
        return _safe_text(value, field_name=info.field_name, max_length=256)

    @property
    def type(self) -> str:
        """Compatibility alias used by provider adapters."""

        return self.target_type

    @property
    def id(self) -> str:
        """Compatibility alias used by provider adapters."""

        return self.target_id


class ActionRequest(_ActionModel):
    """A fully confirmed, idempotent command to execute one action."""

    schema_version: Literal[ACTION_SCHEMA_VERSION] = Field(
        default=ACTION_SCHEMA_VERSION,
        description="Version of the operational action contract",
    )
    action: ActionType = Field(
        validation_alias=AliasChoices("action", "action_type"),
        serialization_alias="action",
        description="Allowlisted operational action",
    )
    case_id: OpaqueIdentifier = Field(
        description="Opaque case identifier that authorizes and explains the action"
    )
    reason: str = Field(
        min_length=1,
        max_length=2000,
        description="Human-readable reason for the provider-side action",
    )
    idempotency_key: str = Field(
        min_length=1,
        max_length=256,
        description="Stable caller key used to deduplicate command creation",
    )
    target: ActionTarget = Field(
        validation_alias=AliasChoices("target", "target_entity"),
        serialization_alias="target",
        description="Opaque provider target",
    )
    target_confirmation: StrictBool = Field(
        validation_alias=AliasChoices("target_confirmation", "target_confirmed"),
        serialization_alias="target_confirmation",
        description="Must be explicitly true before a side effect is queued",
        json_schema_extra={"const": True},
    )
    policy_version: str = Field(
        min_length=1,
        max_length=128,
        description="Policy version under which the action was authorized",
    )
    requested_by: Optional[OpaqueIdentifier] = Field(
        default=None,
        description="Optional opaque actor identifier",
    )

    @field_validator("case_id", "requested_by")
    @classmethod
    def identifiers_must_be_safe(cls, value: Optional[str], info) -> Optional[str]:
        if value is None:
            return None
        return _safe_text(value, field_name=info.field_name, max_length=256)

    @field_validator("reason")
    @classmethod
    def reason_must_be_safe(cls, value: str) -> str:
        return _safe_text(value, field_name="reason", max_length=2000)

    @field_validator("idempotency_key")
    @classmethod
    def idempotency_key_must_be_safe(cls, value: str) -> str:
        return _safe_text(
            value,
            field_name="idempotency_key",
            max_length=256,
            pattern=_KEY_PATTERN,
        )

    @field_validator("policy_version")
    @classmethod
    def policy_version_must_be_safe(cls, value: str) -> str:
        return _safe_text(
            value,
            field_name="policy_version",
            max_length=128,
            pattern=_VERSION_PATTERN,
        )

    @model_validator(mode="after")
    def target_must_be_confirmed(self) -> "ActionRequest":
        if self.target_confirmation is not True:
            raise ValueError("target_confirmation must be explicitly true")
        return self

    @property
    def action_name(self) -> str:
        """Return the stable string action value for adapters and storage."""

        return self.action.value

    @property
    def target_confirmed(self) -> bool:
        """Compatibility alias for integrations using past-tense wording."""

        return self.target_confirmation


class ActionStatus(str, Enum):
    """Durable lifecycle states for an outbox command."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ProviderReceipt(_ActionModel):
    """Provider acknowledgement persisted with the outbox command."""

    receipt_id: str = Field(
        min_length=1,
        max_length=256,
        validation_alias=AliasChoices("receipt_id", "provider_receipt_id"),
        serialization_alias="receipt_id",
    )
    provider: str = Field(min_length=1, max_length=128)
    action: ActionType = Field(
        validation_alias=AliasChoices("action", "action_type"),
        serialization_alias="action",
    )
    idempotency_key: str = Field(min_length=1, max_length=256)
    provider_reference: str = Field(min_length=1, max_length=256)
    target: ActionTarget
    accepted: StrictBool = True
    details: dict[str, Union[str, int, float, bool, None]] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("details", "metadata"),
        serialization_alias="details",
    )

    @field_validator("receipt_id", "provider", "idempotency_key", "provider_reference")
    @classmethod
    def receipt_text_must_be_safe(cls, value: str, info) -> str:
        return _safe_text(value, field_name=info.field_name, max_length=256)

    @field_validator("details")
    @classmethod
    def details_must_be_scalar(cls, value: Mapping[str, Any]) -> dict[str, Union[str, int, float, bool, None]]:
        result: dict[str, Union[str, int, float, bool, None]] = {}
        forbidden_keys = {
            "access_token",
            "api_key",
            "authorization",
            "card_number",
            "cvv",
            "password",
            "secret",
        }
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("provider receipt detail keys must be non-empty strings")
            if key.strip().lower() in forbidden_keys:
                raise ValueError(f"provider receipt detail field '{key}' is forbidden")
            if not isinstance(item, (str, int, float, bool)) and item is not None:
                raise ValueError("provider receipt details must be scalar")
            if isinstance(item, str):
                result[key.strip()] = _safe_text(item, field_name=f"details.{key}", max_length=512)
            elif isinstance(item, float) and not math.isfinite(item):
                raise ValueError("provider receipt details must contain finite numbers")
            else:
                result[key.strip()] = item
        return result

    @property
    def provider_receipt_id(self) -> str:
        """Compatibility alias for integrations using a verbose field name."""

        return self.receipt_id

    def __getitem__(self, key: str) -> Any:
        """Provide read-only mapping-style access to receipt fields."""

        return getattr(self, key)


class ActionReceipt(_ActionModel):
    """Durable command receipt returned by enqueue/status/worker methods."""

    schema_version: Literal[ACTION_SCHEMA_VERSION] = Field(default=ACTION_SCHEMA_VERSION)
    command_id: str = Field(min_length=1, max_length=128)
    action: ActionType
    case_id: OpaqueIdentifier
    reason: str = Field(min_length=1, max_length=2000)
    idempotency_key: str = Field(min_length=1, max_length=256)
    target: ActionTarget
    target_confirmation: StrictBool
    policy_version: str = Field(min_length=1, max_length=128)
    status: ActionStatus
    created_at: datetime
    updated_at: datetime
    attempts: int = Field(default=0, ge=0)
    provider_receipt: Optional[ProviderReceipt] = Field(
        default=None,
        validation_alias=AliasChoices("provider_receipt", "receipt"),
        serialization_alias="provider_receipt",
    )
    error: Optional[str] = Field(default=None, max_length=2000)
    requested_by: Optional[OpaqueIdentifier] = None

    @field_validator("command_id", "case_id", "reason", "idempotency_key", "policy_version", "error")
    @classmethod
    def receipt_text_must_be_safe(cls, value: Optional[str], info) -> Optional[str]:
        if value is None:
            return None
        return _safe_text(value, field_name=info.field_name, max_length=2000)

    @field_validator("created_at", "updated_at")
    @classmethod
    def timestamps_must_be_utc(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def receipt_state_is_consistent(self) -> "ActionReceipt":
        if self.target_confirmation is not True:
            raise ValueError("target_confirmation must be explicitly true")
        if self.status is ActionStatus.SUCCEEDED and self.provider_receipt is None:
            raise ValueError("succeeded action receipts require a provider receipt")
        if self.status is ActionStatus.FAILED and not self.error:
            raise ValueError("failed action receipts require an error")
        if self.status is ActionStatus.SUCCEEDED and self.error is not None:
            raise ValueError("succeeded action receipts may not contain an error")
        return self

    @property
    def command_status(self) -> ActionStatus:
        """Compatibility alias for status-oriented callers."""

        return self.status

    @property
    def last_error(self) -> Optional[str]:
        """Compatibility alias for worker diagnostics."""

        return self.error

    @property
    def receipt(self) -> Optional[ProviderReceipt]:
        """Compatibility alias for callers that use ``receipt``."""

        return self.provider_receipt

    def __getitem__(self, key: str) -> Any:
        """Provide read-only mapping-style access to receipt fields."""

        return getattr(self, key)


class ActionStatusResponse(_ActionModel):
    """Small status projection suitable for a future status route."""

    command_id: str = Field(min_length=1, max_length=128)
    status: ActionStatus
    attempts: int = Field(default=0, ge=0)
    provider_receipt: Optional[ProviderReceipt] = Field(
        default=None,
        validation_alias=AliasChoices("provider_receipt", "receipt"),
        serialization_alias="provider_receipt",
    )
    error: Optional[str] = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def status_fields_are_consistent(self) -> "ActionStatusResponse":
        if self.status is ActionStatus.SUCCEEDED and self.provider_receipt is None:
            raise ValueError("succeeded status requires a provider receipt")
        if self.status is ActionStatus.FAILED and not self.error:
            raise ValueError("failed status requires an error")
        return self


# Versioned and semantic aliases keep future route integration from having to
# duplicate contracts while making the likely import names discoverable.
ActionRequestV1 = ActionRequest
ActionCommandRequest = ActionRequest
ActionCommandReceipt = ActionReceipt
ActionReceiptV1 = ActionReceipt
ActionName = ActionType
AllowlistedAction = ActionType
ActionStatusV1 = ActionStatus
ActionCommandStatus = ActionStatus
ActionStatusModel = ActionStatusResponse
ActionStatusResponseV1 = ActionStatusResponse
ProviderActionReceipt = ProviderReceipt
ProviderReceiptV1 = ProviderReceipt


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "ACTIONS_V1_SCHEMA_VERSION",
    "Action",
    "ActionCommandReceipt",
    "ActionCommandRequest",
    "ActionCommandStatus",
    "ActionName",
    "ActionReceipt",
    "ActionReceiptV1",
    "ActionRequest",
    "ActionRequestV1",
    "ActionStatus",
    "ActionStatusModel",
    "ActionStatusResponse",
    "ActionStatusResponseV1",
    "ActionStatusV1",
    "ActionTarget",
    "ActionType",
    "AllowlistedAction",
    "AllowedAction",
    "ProviderActionReceipt",
    "ProviderReceipt",
    "ProviderReceiptV1",
]
