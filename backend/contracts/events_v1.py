"""Compatibility import surface for the ``events.v1`` contract."""

from .events import (
    EVENTS_V1_SCHEMA_VERSION,
    CanonicalEvent,
    CanonicalEventV1,
    CanonicalNormalizedEvent,
    CanonicalNormalizedEventV1,
    Event,
    EventV1,
    NormalizedCanonicalEvent,
    NormalizedCanonicalEventV1,
    NormalizedEvent,
    NormalizedEventV1,
)

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
