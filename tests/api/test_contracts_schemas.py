"""Focused tests for the Stage 2 contracts package."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from backend.contracts.events import CanonicalEventV1
from backend.contracts.labels import DelayedLabelV1, DelayedOutcomeV1


UTC = timezone.utc
EVENT_TIME = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def event_payload(**overrides):
    payload = {
        "event_id": "evt_001",
        "event_time": EVENT_TIME,
        "order_id": "order_001",
        "buyer_id": "buyer_001",
        "seller_id": "seller_001",
        "listing_id": "listing_001",
        "amount": 42.50,
        "currency": "USD",
        "source": "checkout",
        "schema_version": "events.v1",
    }
    payload.update(overrides)
    return payload


def label_payload(**overrides):
    payload = {
        "event_id": "evt_001",
        "event_time": EVENT_TIME,
        "label": "confirmed_fraud",
        "label_time": EVENT_TIME + timedelta(days=31),
        "label_source": "chargeback_review",
        "label_confidence": 0.98,
        "outcome_window_days": 30,
    }
    payload.update(overrides)
    return payload


def test_valid_event_accepts_opaque_ids_and_normalizes_utc():
    event = CanonicalEventV1(
        **event_payload(
            tokenized_device_id="device_tok_001",
            tokenized_address_id="address_tok_001",
        )
    )

    assert event.event_time.tzinfo == UTC
    assert event.device_id == "device_tok_001"
    assert event.address_id == "address_tok_001"
    assert event.model_dump()["schema_version"] == "events.v1"


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2025, 1, 1, 12, 0),
        datetime(2025, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=1))),
    ],
)
def test_event_rejects_naive_and_non_utc_timestamps(timestamp):
    with pytest.raises(ValidationError):
        CanonicalEventV1(**event_payload(event_time=timestamp))


@pytest.mark.parametrize(
    "field",
    [
        "event_id",
        "event_time",
        "order_id",
        "buyer_id",
        "seller_id",
        "listing_id",
        "amount",
        "currency",
        "source",
        "schema_version",
    ],
)
def test_event_requires_canonical_fields(field):
    payload = event_payload()
    del payload[field]

    with pytest.raises(ValidationError):
        CanonicalEventV1(**payload)


@pytest.mark.parametrize(
    "field",
    [
        "name",
        "address",
        "payment_credentials",
        "api_key",
        "token",
        "unexpected_extension",
    ],
)
def test_event_rejects_forbidden_sensitive_and_extra_fields(field):
    with pytest.raises(ValidationError):
        CanonicalEventV1(**event_payload(**{field: "must-not-cross-boundary"}))


def test_event_rejects_unknown_schema_version():
    with pytest.raises(ValidationError):
        CanonicalEventV1(**event_payload(schema_version="events.v2"))


def test_valid_delayed_label_and_maturity_helper():
    label = DelayedLabelV1(**label_payload())

    assert label.reference_time == EVENT_TIME
    assert label.is_mature() is True
    assert label.is_mature(EVENT_TIME + timedelta(days=29)) is False
    assert label.is_mature(as_of=EVENT_TIME + timedelta(days=30)) is True


def test_delayed_label_accepts_reference_time_alias():
    payload = label_payload()
    payload["reference_time"] = payload.pop("event_time")

    label = DelayedLabelV1(**payload)

    assert label.event_time == EVENT_TIME
    assert label.reference_time == EVENT_TIME


@pytest.mark.parametrize(
    "field",
    [
        "event_id",
        "event_time",
        "label",
        "label_time",
        "label_source",
        "label_confidence",
        "outcome_window_days",
    ],
)
def test_delayed_label_requires_outcome_fields(field):
    payload = label_payload()
    del payload[field]

    with pytest.raises(ValidationError):
        DelayedLabelV1(**payload)


@pytest.mark.parametrize(
    "field",
    [
        "event_time",
        "label_time",
    ],
)
def test_delayed_label_rejects_naive_and_non_utc_timestamps(field):
    for timestamp in (
        datetime(2025, 1, 1, 12, 0),
        datetime(2025, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=1))),
    ):
        payload = label_payload(**{field: timestamp})
        with pytest.raises(ValidationError):
            DelayedLabelV1(**payload)


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01, float("nan"), float("inf")],
)
def test_delayed_label_rejects_invalid_confidence(confidence):
    with pytest.raises(ValidationError):
        DelayedLabelV1(**label_payload(label_confidence=confidence))


def test_delayed_label_rejects_premature_label():
    with pytest.raises(ValidationError, match="label_time"):
        DelayedLabelV1(
            **label_payload(label_time=EVENT_TIME - timedelta(seconds=1))
        )


def test_delayed_outcome_has_its_own_versioned_contract():
    outcome = DelayedOutcomeV1(**label_payload())

    assert outcome.schema_version == "outcomes.v1"
    with pytest.raises(ValidationError):
        DelayedOutcomeV1(**label_payload(schema_version="outcomes.v2"))
