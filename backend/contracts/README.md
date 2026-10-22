# Stage 2 data contracts

This package contains the versioned Pydantic v2 boundaries for normalized
events and delayed outcomes. It is intentionally separate from
`backend/schemas.py`, which remains the serving API schema module.

## `events.v1`

`CanonicalEventV1` requires:

- `event_id` and a UTC-aware `event_time`;
- `order_id`, `buyer_id`, `seller_id`, and `listing_id`;
- `amount`, `currency`, and `source`; and
- the exact `schema_version` value `events.v1`.

`device_id` and `address_id` are optional opaque/tokenized identifiers. The
contract rejects undeclared fields, which prevents names, raw addresses,
payment credentials, API keys, access tokens, and similar sensitive values
from crossing this boundary. Structural validation of an opaque/tokenized ID
cannot prove that it is a cryptographic hash, that it was generated securely,
or that the original value was not sensitive.

## Delayed labels and outcomes

`DelayedLabelV1` and `DelayedOutcomeV1` carry the source `event_id` and
`event_time`, the `label`, `label_time`, `label_source`, confidence in the
range 0 through 1, and a non-negative `outcome_window_days`. Both timestamps must
be UTC-aware, and `label_time` cannot precede `event_time`.

`is_mature()` evaluates whether the outcome window has elapsed. By default it
uses `label_time`; replay and backfill code can pass a UTC-aware `as_of`
timestamp. `maturity_time` exposes the event time plus the declared window.
