"""Canonical normalized event contract, version ``events.v1``.

Only identifiers and normalized transaction facts belong in this contract.
Names, raw addresses, payment credentials, API keys, access tokens, and other
sensitive payloads are not accepted because the model forbids every undeclared
field.  ``device_id`` and ``address_id`` are explicitly limited to opaque or
tokenized identifiers.  Validation can check their shape, but it cannot prove
that a value was cryptographically hashed, tokenized irreversibly, or derived
without sensitive input.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, ClassVar, Literal, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from pydantic.types import StringConstraints


EVENTS_V1_SCHEMA_VERSION = "events.v1"

# Keep identifier validation deliberately structural.  An opaque identifier
# can be a UUID, a deterministic token, or an upstream key; its cryptographic
# provenance cannot be established by a Pydantic model.
Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


def _require_utc(
    value: datetime | str,
    *,
    field_name: str,
) -> datetime:
    """Return a UTC-normalized datetime, rejecting naive/non-UTC values."""

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a valid datetime") from exc
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a valid datetime")

    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(
            f"{field_name} must be timezone-aware and expressed in UTC"
        )
    if offset != timedelta(0):
        raise ValueError(f"{field_name} must use UTC (offset +00:00)")
    return value.astimezone(timezone.utc)


class _ContractModel(BaseModel):
    """Shared strict configuration for all versioned contracts."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class CanonicalEventV1(_ContractModel):
    """A normalized e-commerce event accepted by the Stage 2 data boundary.

    The optional device and address fields must already be opaque/tokenized
    identifiers.  This contract does not—and cannot—prove that they are
    cryptographic hashes or that tokenization was performed securely.
    """

    event_id: Identifier = Field(description="Stable event identifier")
    event_time: datetime = Field(
        description="UTC-aware time at which the event occurred"
    )
    order_id: Identifier = Field(description="Opaque order identifier")
    buyer_id: Identifier = Field(description="Opaque buyer identifier")
    seller_id: Identifier = Field(description="Opaque seller identifier")
    listing_id: Identifier = Field(description="Opaque listing identifier")
    amount: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        description="Non-negative normalized transaction amount",
    )
    currency: Identifier = Field(
        description="Currency code or normalized currency identifier"
    )
    source: Identifier = Field(
        description="Upstream source that produced the event"
    )
    schema_version: Literal[EVENTS_V1_SCHEMA_VERSION] = Field(
        description="Canonical event schema version"
    )
    device_id: Optional[Identifier] = Field(
        default=None,
        validation_alias=AliasChoices("device_id", "tokenized_device_id"),
        serialization_alias="device_id",
        description=(
            "Optional opaque/tokenized device identifier. Shape validation "
            "cannot "
            "prove cryptographic hashing."
        ),
    )
    address_id: Optional[Identifier] = Field(
        default=None,
        validation_alias=AliasChoices("address_id", "tokenized_address_id"),
        serialization_alias="address_id",
        description=(
            "Optional opaque/tokenized address identifier. Shape validation "
            "cannot "
            "prove cryptographic hashing."
        ),
    )

    # This list documents producer and reviewer expectations.
    # ``extra='forbid'``
    # is the enforcement mechanism, so newly invented sensitive keys are also
    # rejected rather than relying on this finite list.
    FORBIDDEN_SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "name",
            "first_name",
            "last_name",
            "full_name",
            "email",
            "phone",
            "address",
            "street_address",
            "billing_address",
            "shipping_address",
            "payment",
            "payment_credentials",
            "payment_method",
            "card_number",
            "cvv",
            "cvc",
            "bank_account",
            "api_key",
            "access_token",
            "refresh_token",
            "token",
            "password",
            "secret",
        }
    )

    @field_validator("event_time")
    @classmethod
    def event_time_must_be_utc(cls, value: datetime) -> datetime:
        return _require_utc(value, field_name="event_time")

    @property
    def tokenized_device_id(self) -> Optional[str]:
        """Compatibility alias for the opaque device identifier field."""

        return self.device_id

    @property
    def tokenized_address_id(self) -> Optional[str]:
        """Compatibility alias for the opaque address identifier field."""

        return self.address_id


# Friendly aliases keep the version in the concrete model while allowing
# callers to use the domain term that best matches their ingestion code.
CanonicalEvent = CanonicalEventV1
CanonicalNormalizedEvent = CanonicalEventV1
CanonicalNormalizedEventV1 = CanonicalEventV1
NormalizedCanonicalEvent = CanonicalEventV1
NormalizedCanonicalEventV1 = CanonicalEventV1
NormalizedEventV1 = CanonicalEventV1
NormalizedEvent = CanonicalEventV1
EventV1 = CanonicalEventV1
Event = CanonicalEventV1


__all__ = [
    "EVENTS_V1_SCHEMA_VERSION",
    "CanonicalEvent",
    "CanonicalEventV1",
    "CanonicalNormalizedEvent",
    "CanonicalNormalizedEventV1",
    "Event",
    "EventV1",
    "NormalizedCanonicalEvent",
    "NormalizedCanonicalEventV1",
    "NormalizedEvent",
    "NormalizedEventV1",
]
