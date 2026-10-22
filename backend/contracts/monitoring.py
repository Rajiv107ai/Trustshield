"""Versioned contracts for aggregate model monitoring results.

Monitoring records are intentionally different from serving and event
contracts.  They contain a bounded monitoring window and aggregate metrics,
never transaction rows, customer identifiers, or model input payloads.  The
service modules in :mod:`backend.services` use these models as a final
boundary before returning a report to a caller or persisting one.

``monitoring.v1`` is storage-neutral.  A later version can add metrics without
changing the shape or meaning of the existing version.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Mapping, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


MONITORING_SCHEMA_VERSION = "monitoring.v1"
MONITORING_CONTRACT_VERSION = MONITORING_SCHEMA_VERSION

# These are the only subgroup dimensions that may cross the monitoring
# boundary.  They are deliberately finite and reviewable.  A caller cannot
# ask a report to group by an arbitrary request field or identifier.
APPROVED_SUBGROUP_FIELDS = frozenset(
    {
        "merchant_id",
        "merchant_age_band",
        "region",
        "account_age_band",
        "device_class",
        "listing_category",
        "payment_channel",
        "fraud_type",
    }
)
ALLOWED_SUBGROUP_FIELDS = APPROVED_SUBGROUP_FIELDS

SEVERITIES = ("OK", "WARNING", "CRITICAL")
ALERT_ACTIONS = ("none", "investigation", "shadow_evaluation", "promotion_review")

_SAFE_CODE = re.compile(r"^[A-Za-z0-9_.:/-]+$")
_FORBIDDEN_KEYS = frozenset(
    {
        "address",
        "api_key",
        "access_token",
        "bank_account",
        "card_number",
        "cookie",
        "cvv",
        "email",
        "event_id",
        "first_name",
        "full_name",
        "ip_address",
        "last_name",
        "name",
        "order_id",
        "password",
        "phone",
        "raw_payload",
        "refresh_token",
        "secret",
        "shipping_address",
        "street_address",
        "token",
        "transaction_id",
        "user_id",
    }
)


def _utc_datetime(value: datetime | str, *, field_name: str) -> datetime:
    """Normalize an aware timestamp to UTC and reject naive timestamps."""

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
    if offset != timedelta(0):
        raise ValueError(f"{field_name} must use UTC (offset +00:00)")
    return value.astimezone(timezone.utc)


def _safe_code(value: str, *, field_name: str, max_length: int = 128) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    value = value.strip()
    if not value or len(value) > max_length or not _SAFE_CODE.fullmatch(value):
        raise ValueError(f"{field_name} must be a short opaque code")
    lowered = value.lower()
    if "@" in value or "bearer " in lowered or "-----begin " in lowered:
        raise ValueError(f"{field_name} must not contain secret or personal data")
    return value


def _validate_aggregate(value: Any, *, path: str = "metrics", depth: int = 0) -> Any:
    """Validate a bounded aggregate payload without accepting row-level data."""

    # Reports contain a bounded but useful hierarchy (report -> metric family
    # -> feature -> quantile -> scalar).  The depth guard still prevents a
    # caller from smuggling a row-shaped object through an aggregate field.
    if depth > 8:
        raise ValueError(f"{path} is nested too deeply for an aggregate report")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, str):
        if len(value) > 256 or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError(f"{path} contains unsupported text")
        lowered = value.lower()
        if any(marker in lowered for marker in ("bearer ", "-----begin ", "api_key=", "password=")):
            raise ValueError(f"{path} contains a secret-shaped value")
        return value
    if isinstance(value, Mapping):
        if len(value) > 256:
            raise ValueError(f"{path} contains too many aggregate keys")
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise ValueError(f"{path} keys must be strings")
            key = raw_key.strip()
            if key.lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"{path}.{key} is not allowed in an aggregate report")
            if "@" in key or len(key) > 64 and key.lower() not in {"__missing__", "__other__"}:
                raise ValueError(f"{path}.{key} is not a privacy-safe aggregate key")
            if len(key) > 128:
                raise ValueError(f"{path} contains an overlong key")
            result[key] = _validate_aggregate(raw_value, path=f"{path}.{key}", depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            raise ValueError(f"{path} contains too many aggregate values")
        return [
            _validate_aggregate(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} must contain aggregate scalar, mapping, or list values")


class _MonitoringModel(BaseModel):
    """Shared strict configuration for monitoring contracts."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class MonitoringWindowV1(_MonitoringModel):
    """A UTC time interval associated with one aggregate monitoring record."""

    schema_version: Literal[MONITORING_SCHEMA_VERSION] = MONITORING_SCHEMA_VERSION
    window_id: str = Field(default="window", validation_alias=AliasChoices("window_id", "id"))
    record_id: Optional[str] = None
    window_start: datetime = Field(
        validation_alias=AliasChoices("window_start", "start", "start_time")
    )
    window_end: datetime = Field(validation_alias=AliasChoices("window_end", "end", "end_time"))
    baseline_window_id: Optional[str] = None
    baseline_window_start: Optional[datetime] = None
    baseline_window_end: Optional[datetime] = None
    population: str = "transaction"
    sample_count: int = Field(default=0, ge=0)
    reference_sample_count: int = Field(default=0, ge=0)
    current_sample_count: int = Field(default=0, ge=0)
    matured_label_count: int = Field(default=0, ge=0)
    recorded_at: Optional[datetime] = Field(
        default=None,
        validation_alias=AliasChoices("recorded_at", "created_at"),
    )

    @field_validator("window_id", "record_id", "baseline_window_id", "population")
    @classmethod
    def codes_must_be_safe(cls, value: Optional[str], info) -> Optional[str]:
        if value is None:
            return None
        return _safe_code(value, field_name=info.field_name)

    @field_validator(
        "window_start",
        "window_end",
        "baseline_window_start",
        "baseline_window_end",
        "recorded_at",
    )
    @classmethod
    def timestamps_must_be_utc(cls, value: Optional[datetime], info) -> Optional[datetime]:
        if value is None:
            return None
        return _utc_datetime(value, field_name=info.field_name)

    @model_validator(mode="after")
    def window_bounds_are_valid(self) -> "MonitoringWindowV1":
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be later than window_start")
        if (self.baseline_window_start is None) != (self.baseline_window_end is None):
            raise ValueError("baseline window requires both start and end")
        if self.baseline_window_start is not None and self.baseline_window_end <= self.baseline_window_start:
            raise ValueError("baseline_window_end must be later than baseline_window_start")
        return self

    @property
    def start(self) -> datetime:
        """Compatibility alias for consumers that call the boundary ``start``."""

        return self.window_start

    @property
    def end(self) -> datetime:
        """Compatibility alias for consumers that call the boundary ``end``."""

        return self.window_end


class MonitoringAlertV1(_MonitoringModel):
    """A threshold result that routes human/model-governance follow-up."""

    schema_version: Literal[MONITORING_SCHEMA_VERSION] = MONITORING_SCHEMA_VERSION
    metric_name: str = Field(min_length=1, max_length=128)
    metric_type: str = Field(default="monitoring", min_length=1, max_length=64)
    status: Literal["OK", "WARNING", "CRITICAL"] = "OK"
    observed_value: Optional[float] = Field(default=None, allow_inf_nan=False)
    warning_threshold: Optional[float] = Field(default=None, allow_inf_nan=False)
    critical_threshold: Optional[float] = Field(default=None, allow_inf_nan=False)
    direction: Literal["higher", "lower"] = "higher"
    recommended_action: Literal["none", "investigation", "shadow_evaluation", "promotion_review"] = "none"
    action: Optional[Literal["none", "investigation", "shadow_evaluation", "promotion_review"]] = None
    workflow_status: Optional[Literal["none", "investigation", "shadow_evaluation", "promotion_review"]] = None
    automatic_retraining: Literal[False] = False
    sample_count: int = Field(default=0, ge=0)
    reason: Optional[str] = Field(default=None, max_length=256)

    @field_validator("metric_name", "metric_type")
    @classmethod
    def metric_codes_must_be_safe(cls, value: str, info) -> str:
        return _safe_code(value, field_name=info.field_name)

    @field_validator("reason")
    @classmethod
    def reason_must_be_aggregate_text(cls, value: Optional[str]) -> Optional[str]:
        return _validate_aggregate(value, path="reason")

    @model_validator(mode="after")
    def workflow_aliases_match(self) -> "MonitoringAlertV1":
        if self.action is None:
            self.action = self.recommended_action
        if self.workflow_status is None:
            self.workflow_status = self.recommended_action
        if self.action != self.recommended_action or self.workflow_status != self.recommended_action:
            raise ValueError("alert action aliases must match recommended_action")
        return self

    @property
    def severity(self) -> str:
        """Compatibility alias for callers that use ``severity``."""

        return self.status


class MonitoringMetricV1(_MonitoringModel):
    """One named aggregate metric and its threshold context."""

    schema_version: Literal[MONITORING_SCHEMA_VERSION] = MONITORING_SCHEMA_VERSION
    metric_name: str = Field(min_length=1, max_length=128)
    metric_type: str = Field(default="monitoring", min_length=1, max_length=64)
    feature: Optional[str] = Field(default=None, max_length=128)
    value: Optional[float] = Field(default=None, allow_inf_nan=False)
    reference_value: Optional[float] = Field(default=None, allow_inf_nan=False)
    current_value: Optional[float] = Field(default=None, allow_inf_nan=False)
    delta: Optional[float] = Field(default=None, allow_inf_nan=False)
    warning_threshold: Optional[float] = Field(default=None, allow_inf_nan=False)
    critical_threshold: Optional[float] = Field(default=None, allow_inf_nan=False)
    direction: Literal["higher", "lower"] = "higher"
    status: Literal["OK", "WARNING", "CRITICAL"] = "OK"
    reference_sample_count: int = Field(default=0, ge=0)
    current_sample_count: int = Field(default=0, ge=0)
    missing_count: int = Field(default=0, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metric_name", "metric_type")
    @classmethod
    def metric_codes_must_be_safe(cls, value: str, info) -> str:
        return _safe_code(value, field_name=info.field_name)

    @field_validator("feature")
    @classmethod
    def feature_name_must_be_safe(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _safe_code(value, field_name="feature")

    @field_validator("details")
    @classmethod
    def details_must_be_aggregate(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        return _validate_aggregate(value, path="details")


class MonitoringSubgroupV1(_MonitoringModel):
    """Aggregate metrics for an approved subgroup dimension/value."""

    schema_version: Literal[MONITORING_SCHEMA_VERSION] = MONITORING_SCHEMA_VERSION
    field: str
    value: str = Field(min_length=1, max_length=128)
    sample_count: int = Field(ge=0)
    metrics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("field")
    @classmethod
    def subgroup_field_must_be_approved(cls, value: str) -> str:
        if value not in APPROVED_SUBGROUP_FIELDS:
            raise ValueError(f"subgroup field is not approved: {value}")
        return value

    @field_validator("metrics")
    @classmethod
    def subgroup_metrics_must_be_aggregate(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        return _validate_aggregate(value, path="metrics")


class _MonitoringReportV1(_MonitoringModel):
    """Shared report fields for drift and calibration monitoring."""

    schema_version: Literal[MONITORING_SCHEMA_VERSION] = MONITORING_SCHEMA_VERSION
    monitoring_window: MonitoringWindowV1
    status: Literal["OK", "WARNING", "CRITICAL"] = "OK"
    metrics: dict[str, Any] = Field(default_factory=dict)
    alerts: list[MonitoringAlertV1] = Field(default_factory=list, max_length=256)
    recommended_action: Literal["none", "investigation", "shadow_evaluation", "promotion_review"] = "none"
    workflow_status: Optional[Literal["none", "investigation", "shadow_evaluation", "promotion_review"]] = None
    automatic_retraining: Literal[False] = False
    aggregate_only: Literal[True] = True

    @field_validator("metrics")
    @classmethod
    def report_metrics_must_be_aggregate(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        return _validate_aggregate(value, path="metrics")

    @model_validator(mode="after")
    def report_status_matches_alerts(self) -> "_MonitoringReportV1":
        ranks = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
        if self.alerts and ranks[self.status] < max(ranks[alert.status] for alert in self.alerts):
            raise ValueError("report status cannot be less severe than an alert")
        if self.workflow_status is None:
            self.workflow_status = self.recommended_action
        elif self.workflow_status != self.recommended_action:
            raise ValueError("workflow_status must match recommended_action")
        return self


class DriftReportV1(_MonitoringReportV1):
    """Versioned aggregate feature/score/decision drift report."""

    report_type: Literal["drift"] = "drift"


class CalibrationReportV1(_MonitoringReportV1):
    """Versioned matured-label calibration and operating-point report."""

    report_type: Literal["calibration"] = "calibration"
    matured_label_count: int = Field(default=0, ge=0)
    immature_label_count: int = Field(default=0, ge=0)
    subgroup_slices: list[MonitoringSubgroupV1] = Field(default_factory=list, max_length=256)
    # ``subgroups`` is a compatibility spelling used by the independent
    # evaluation report; both names carry the same aggregate slice records.
    subgroups: list[MonitoringSubgroupV1] = Field(default_factory=list, max_length=256)

    @model_validator(mode="after")
    def subgroup_aliases_match(self) -> "CalibrationReportV1":
        if self.subgroup_slices and not self.subgroups:
            self.subgroups = list(self.subgroup_slices)
        elif self.subgroups and not self.subgroup_slices:
            self.subgroup_slices = list(self.subgroups)
        elif self.subgroup_slices != self.subgroups:
            raise ValueError("subgroup_slices and subgroups must match when both are supplied")
        return self


# Friendly aliases preserve the concrete version in the class name while
# keeping the API discoverable for callers that use the domain terms.
MonitoringWindow = MonitoringWindowV1
MonitoringWindowRecord = MonitoringWindowV1
MonitoringAlert = MonitoringAlertV1
MonitoringMetric = MonitoringMetricV1
MonitoringSubgroup = MonitoringSubgroupV1
MonitoringRecord = MonitoringWindowV1
DriftReport = DriftReportV1
CalibrationReport = CalibrationReportV1
# Explicit record/report names are useful to stores that keep the contract
# version in the class name rather than only in ``schema_version``.
MonitoringWindowRecordV1 = MonitoringWindowV1
MonitoringRecordV1 = MonitoringWindowV1
DriftMonitoringReportV1 = DriftReportV1
CalibrationMonitoringReportV1 = CalibrationReportV1


def classify_threshold(
    value: Optional[float],
    warning_threshold: Optional[float],
    critical_threshold: Optional[float],
    *,
    direction: Literal["higher", "lower"] = "higher",
) -> Literal["OK", "WARNING", "CRITICAL"]:
    """Classify a metric, with threshold equality belonging to the alert band."""

    if value is None or not math.isfinite(float(value)):
        return "OK"
    if warning_threshold is None and critical_threshold is None:
        return "OK"
    observed = float(value)
    if direction == "higher":
        if critical_threshold is not None and observed >= critical_threshold:
            return "CRITICAL"
        if warning_threshold is not None and observed >= warning_threshold:
            return "WARNING"
    else:
        if critical_threshold is not None and observed <= critical_threshold:
            return "CRITICAL"
        if warning_threshold is not None and observed <= warning_threshold:
            return "WARNING"
    return "OK"


__all__ = [
    "ALERT_ACTIONS",
    "ALLOWED_SUBGROUP_FIELDS",
    "APPROVED_SUBGROUP_FIELDS",
    "CalibrationReport",
    "CalibrationReportV1",
    "CalibrationMonitoringReportV1",
    "DriftReport",
    "DriftReportV1",
    "DriftMonitoringReportV1",
    "MONITORING_CONTRACT_VERSION",
    "MONITORING_SCHEMA_VERSION",
    "MonitoringAlert",
    "MonitoringAlertV1",
    "MonitoringMetric",
    "MonitoringMetricV1",
    "MonitoringSubgroup",
    "MonitoringSubgroupV1",
    "MonitoringWindow",
    "MonitoringWindowRecord",
    "MonitoringWindowV1",
    "MonitoringWindowRecordV1",
    "MonitoringRecord",
    "MonitoringRecordV1",
    "SEVERITIES",
    "classify_threshold",
]
