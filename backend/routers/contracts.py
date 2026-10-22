"""Private contract-validation boundary for normalized events and labels.

These routes validate payload shape for an upstream ingestion service. They do
not persist or echo raw event/customer data; only schema metadata and a safe
validation result leave the service.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError

from backend.auth import Role, require_role
from backend.contracts.events import CanonicalEventV1
from backend.contracts.labels import DelayedLabelV1


class ContractValidationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    schema_version: str
    mature: bool | None = None
    validation_errors: list[str] = []


router = APIRouter(prefix="/contracts", tags=["contracts"])


def _error_messages(exc: ValidationError) -> list[str]:
    return [str(error.get("msg", "invalid contract"))[:256] for error in exc.errors()]


@router.post("/events/validate", response_model=ContractValidationResponse, dependencies=[Depends(require_role(Role.OPERATOR))])
def validate_event(payload: dict[str, Any]) -> ContractValidationResponse:
    try:
        event = CanonicalEventV1.model_validate(payload)
    except ValidationError as exc:
        return ContractValidationResponse(
            valid=False,
            schema_version=str(payload.get("schema_version", "unknown")),
            validation_errors=_error_messages(exc),
        )
    return ContractValidationResponse(valid=True, schema_version=event.schema_version)


@router.post("/labels/validate", response_model=ContractValidationResponse, dependencies=[Depends(require_role(Role.OPERATOR))])
def validate_label(payload: dict[str, Any]) -> ContractValidationResponse:
    try:
        label = DelayedLabelV1.model_validate(payload)
    except ValidationError as exc:
        return ContractValidationResponse(
            valid=False,
            schema_version=str(payload.get("schema_version", "unknown")),
            validation_errors=_error_messages(exc),
        )
    return ContractValidationResponse(
        valid=True,
        schema_version=label.schema_version,
        mature=label.is_mature(),
    )


__all__ = ["ContractValidationResponse", "router", "validate_event", "validate_label"]
