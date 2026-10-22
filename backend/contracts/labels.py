"""Versioned delayed outcome and label contracts.

A delayed record points back to the original event through ``event_id`` and
``event_time``.  ``label_time`` is when the outcome was observed or recorded;
it may equal ``event_time`` but may never precede it.  A record is mature once
the requested outcome window has elapsed from ``event_time``.  The default
``is_mature()`` check uses ``label_time`` as the observation time; callers can
pass an explicit UTC ``as_of`` timestamp for backfills and replay.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .events import Identifier, _require_utc


LABELS_V1_SCHEMA_VERSION = "labels.v1"
OUTCOMES_V1_SCHEMA_VERSION = "outcomes.v1"


class _DelayedRecordBase(BaseModel):
    """Shared strict fields and temporal checks for delayed records."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    event_id: Identifier = Field(description="Identifier of the labeled event")
    event_time: datetime = Field(
        validation_alias=AliasChoices("event_time", "reference_time"),
        serialization_alias="event_time",
        description="UTC-aware time at which the source event occurred",
    )
    label: Identifier = Field(
        validation_alias=AliasChoices("label", "outcome"),
        serialization_alias="label",
        description="Outcome or ground-truth label",
    )
    label_time: datetime = Field(
        description=(
            "UTC-aware time at which the outcome label became available"
        )
    )
    label_source: Identifier = Field(
        description="System or process that produced the label"
    )
    label_confidence: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description=(
            "Confidence in the delayed outcome label, from 0 through 1"
        ),
    )
    outcome_window_days: int = Field(
        ge=0,
        description=(
            "Non-negative number of days after event_time in the "
            "outcome window"
        ),
    )

    @field_validator("event_time", "label_time")
    @classmethod
    def timestamps_must_be_utc(
        cls,
        value: datetime,
        info,
    ) -> datetime:
        return _require_utc(value, field_name=info.field_name)

    @model_validator(mode="after")
    def label_cannot_precede_event(self):
        if self.label_time < self.event_time:
            raise ValueError(
                "label_time must be greater than or equal to event_time"
            )
        return self

    @property
    def reference_time(self) -> datetime:
        """Alias for ``event_time`` used by label-ingestion pipelines."""

        return self.event_time

    @property
    def outcome(self) -> str:
        """Alias for ``label`` used by outcome-ingestion pipelines."""

        return self.label

    @property
    def maturity_time(self) -> datetime:
        """UTC time at which the declared outcome window has elapsed."""

        return self.event_time + timedelta(days=self.outcome_window_days)

    # A method is used rather than a stored boolean so maturity can be
    # evaluated
    # consistently during replay and backfill at any caller-selected timestamp.
    def is_mature(self, as_of: Optional[datetime | str] = None) -> bool:
        """Return whether the outcome window is complete at ``as_of``.

        When omitted, ``label_time`` is used.  ``as_of`` must be UTC-aware and
        UTC-valued just like the contract timestamps.
        """

        check_time = (
            self.label_time
            if as_of is None
            else _require_utc(as_of, field_name="as_of")
        )
        return check_time >= self.maturity_time


class DelayedLabelV1(_DelayedRecordBase):
    """A versioned delayed label record (``labels.v1``)."""

    schema_version: Literal[LABELS_V1_SCHEMA_VERSION] = Field(
        default=LABELS_V1_SCHEMA_VERSION,
        description="Delayed label schema version",
    )


class DelayedOutcomeV1(_DelayedRecordBase):
    """A versioned delayed outcome record (``outcomes.v1``)."""

    schema_version: Literal[OUTCOMES_V1_SCHEMA_VERSION] = Field(
        default=OUTCOMES_V1_SCHEMA_VERSION,
        description="Delayed outcome schema version",
    )


DelayedLabel = DelayedLabelV1
DelayedLabelRecordV1 = DelayedLabelV1
DelayedLabelRecord = DelayedLabelV1
LabelRecordV1 = DelayedLabelV1
OutcomeLabelV1 = DelayedLabelV1
OutcomeLabel = DelayedLabelV1
DelayedOutcome = DelayedOutcomeV1
DelayedOutcomeRecordV1 = DelayedOutcomeV1
DelayedOutcomeRecord = DelayedOutcomeV1
OutcomeRecordV1 = DelayedOutcomeV1


__all__ = [
    "LABELS_V1_SCHEMA_VERSION",
    "OUTCOMES_V1_SCHEMA_VERSION",
    "DelayedLabel",
    "DelayedLabelRecord",
    "DelayedLabelRecordV1",
    "DelayedLabelV1",
    "DelayedOutcome",
    "DelayedOutcomeRecord",
    "DelayedOutcomeRecordV1",
    "DelayedOutcomeV1",
    "LabelRecordV1",
    "OutcomeLabel",
    "OutcomeLabelV1",
    "OutcomeRecordV1",
]
