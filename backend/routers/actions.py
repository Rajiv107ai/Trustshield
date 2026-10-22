"""Operator action command routes.

The action endpoints are deliberately split from the worker implementation.
An HTTP request validates and durably enqueues an :class:`ActionRequest`; it
never calls a provider adapter while the request is being handled.  A worker
can later process the pending command and persist a provider receipt.  The
small development-only worker helper at the bottom of this module is useful
for deterministic tests without exposing an HTTP endpoint that could be
mistaken for an immediate provider-side success.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping
from typing import Any, Callable, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import ValidationError

from backend.auth import AuthUser, Role, require_role
from backend.contracts.actions import ActionReceipt, ActionRequest, ActionType
from backend.services.action_outbox import (
    ActionCommandNotFoundError,
    ActionOutbox,
    ActionOutboxConfigurationError,
    IdempotencyConflictError,
)

try:
    from backend.services.audit_service import AuditService
except ImportError:  # pragma: no cover - only relevant to deliberately slim deployments
    AuditService = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/actions", tags=["actions"])
_UNPROCESSABLE_STATUS = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


# Route vocabulary is intentionally closed.  Each public route maps to one
# allowlisted action and one provider domain in ActionOutbox._ACTION_DOMAIN.
_TRANSACTION_ACTIONS: dict[str, ActionType] = {
    "approve": ActionType.PAYMENT_RELEASE,
    "hold": ActionType.PAYMENT_HOLD,
    "block": ActionType.PAYMENT_REFUND,
}
_LISTING_ACTION = ActionType.MARKETPLACE_SUSPEND_LISTING
_RING_ACTION = ActionType.GRAPH_FLAG_ENTITY


_service_lock = threading.RLock()
_action_outbox: Optional[ActionOutbox] = None
_audit_service: Any = None
_audit_service_initialized = False


def get_action_outbox() -> ActionOutbox:
    """Return the process action outbox, creating it from environment config.

    Construction is lazy so importing the FastAPI application does not create
    a database file or require production credentials.  In production the
    existing ActionOutbox configuration fails closed when no PostgreSQL DSN
    is configured; the route converts that configuration failure to a 503.
    """

    global _action_outbox
    with _service_lock:
        if _action_outbox is None:
            try:
                _action_outbox = ActionOutbox()
            except ActionOutboxConfigurationError as exc:
                logger.error("Action outbox is unavailable: %s", type(exc).__name__)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Action command durability is unavailable",
                ) from exc
        return _action_outbox


def get_audit_service() -> Any:
    """Return the optional audit service configured for this process.

    The import is optional for a slim deployment, but a production process
    that requires durable audit is never allowed to proceed as if auditing
    had succeeded when the service cannot be constructed.
    """

    global _audit_service, _audit_service_initialized
    with _service_lock:
        if _audit_service_initialized:
            return _audit_service
        _audit_service_initialized = True
        if AuditService is None:
            return None
        try:
            _audit_service = AuditService()
        except Exception as exc:
            # Keep the configuration error available to the enqueue path.  A
            # production request will fail closed; development can continue
            # only when audit is explicitly optional.
            logger.warning("Action audit service is unavailable: %s", type(exc).__name__)
            _audit_service = exc
        return _audit_service


def reset_action_services_for_tests() -> None:
    """Close and clear lazily-created services for isolated test fixtures."""

    global _action_outbox, _audit_service, _audit_service_initialized
    with _service_lock:
        if _action_outbox is not None:
            _action_outbox.close()
        if _audit_service is not None and hasattr(_audit_service, "close"):
            _audit_service.close()
        _action_outbox = None
        _audit_service = None
        _audit_service_initialized = False


def _is_production_environment() -> bool:
    value = os.getenv("TRUSTSHIELD_ENV", os.getenv("ENV", "development"))
    return value.strip().lower() in {"production", "prod"}


def _audit_required(audit: Any) -> bool:
    """Whether an audit failure must fail the action request closed."""

    if _is_production_environment():
        return True
    if audit is not None and not isinstance(audit, Exception):
        try:
            if bool(getattr(audit, "durability_required", False)):
                return True
        except Exception:
            return True
    for name in (
        "TRUSTSHIELD_AUDIT_DURABILITY_REQUIRED",
        "TRUSTSHIELD_REQUIRE_AUDIT_DURABILITY",
        "TRUSTSHIELD_REQUIRE_DURABLE_AUDIT",
        "TRUSTSHIELD_AUDIT_REQUIRED",
        "AUDIT_DURABILITY_REQUIRED",
    ):
        if os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _validation_detail(exc: ValidationError) -> list[dict[str, Any]]:
    """Return JSON-safe Pydantic errors for an HTTP 422 response."""

    return [
        {
            "loc": list(error.get("loc", ())),
            "msg": str(error.get("msg", "invalid action request")),
            "type": str(error.get("type", "value_error")),
        }
        for error in exc.errors()
    ]


def _route_request(
    payload: Mapping[str, Any],
    *,
    action: ActionType,
    target_type: str,
    target_id: str,
    user: AuthUser,
) -> ActionRequest:
    """Build the strict ActionRequest while binding path targets/actions.

    The public route body intentionally accepts the ActionRequest fields that
    callers supply (case, reason, idempotency, confirmation, and policy).
    ``action`` and ``target`` are route-bound values, but the complete
    ActionRequest contract is still used for validation, including strict
    booleans, allowlisted actions, target confirmation, and extra-field
    rejection.  Full contract-shaped bodies are accepted as a compatibility
    convenience when their action and target match the URL.
    """

    data = dict(payload)

    # If a caller supplied the action, validate that it agrees with the route
    # before replacing the route-bound value.  Keep invalid values visible to
    # ActionRequest so they receive the normal 422 contract error.
    supplied_action_key: Optional[str] = None
    if "action" in data:
        supplied_action_key = "action"
    elif "action_type" in data:
        supplied_action_key = "action_type"
    if supplied_action_key is not None:
        supplied_action = data[supplied_action_key]
        try:
            normalized_action = ActionType(supplied_action)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=_UNPROCESSABLE_STATUS,
                detail="action is not an allowlisted action",
            )
        if normalized_action is not None and normalized_action is not action:
            raise HTTPException(
                status_code=_UNPROCESSABLE_STATUS,
                detail="action does not match the requested route",
            )
        if supplied_action_key != "action":
            data.pop(supplied_action_key, None)
    data["action"] = action.value

    # Bind the target identifier from the URL.  A supplied target is accepted
    # only when it confirms the exact path identifier; this prevents a client
    # from authorizing one resource while enqueueing another.  Target type is
    # opaque by contract, so preserve a caller-supplied type when present.
    supplied_target_key: Optional[str] = None
    if "target" in data:
        supplied_target_key = "target"
    elif "target_entity" in data:
        supplied_target_key = "target_entity"
    if supplied_target_key is None:
        data["target"] = {"target_type": target_type, "target_id": target_id}
    else:
        raw_target = data[supplied_target_key]
        if isinstance(raw_target, Mapping):
            target_id_value = raw_target.get(
                "target_id",
                raw_target.get("id", raw_target.get("entity_id")),
            )
            if target_id_value != target_id:
                raise HTTPException(
                    status_code=_UNPROCESSABLE_STATUS,
                    detail="target.target_id must match the requested route resource",
                )
            if supplied_target_key != "target":
                data["target"] = dict(raw_target)
                data.pop(supplied_target_key, None)
        # Non-mapping values are left intact for ActionRequest to reject.

    # The authenticated actor is authoritative for the durable command and
    # audit record.  Do not allow a request body to impersonate another actor.
    data["requested_by"] = user.username

    try:
        return ActionRequest.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=_UNPROCESSABLE_STATUS,
            detail=_validation_detail(exc),
        ) from exc


def _append_enqueue_audit(audit: Any, request: ActionRequest, receipt: ActionReceipt, user: AuthUser) -> None:
    """Append a privacy-safe enqueue event, failing closed when required."""

    if audit is None or isinstance(audit, Exception):
        if _audit_required(audit):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Durable action audit is unavailable; action was not acknowledged",
            )
        return

    try:
        audit.append_action(
            request.action.value,
            subject_id=receipt.target.target_id,
            event_name="action.command.enqueued",
            actor_id=user.username,
            outcome=receipt.status.value,
            reason_codes=["operator_action_requested"],
            metadata={
                "action_type": request.action.value,
                "case_id": request.case_id,
                "policy_version": request.policy_version,
                "resource_id": receipt.target.target_id,
                "resource_type": receipt.target.target_type,
                "status": receipt.status.value,
                "outcome": receipt.status.value,
            },
        )
    except Exception as exc:
        if _audit_required(audit):
            logger.error("Durable action audit append failed: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Durable action audit is unavailable; action was not acknowledged",
            ) from exc
        logger.warning("Optional action audit append failed: %s", type(exc).__name__)


def _append_worker_audit(receipt: ActionReceipt, *, audit: Any = None) -> None:
    """Best-effort/required audit for a worker outcome."""

    if audit is None:
        audit = get_audit_service()
    if audit is None or isinstance(audit, Exception):
        if _audit_required(audit):
            raise RuntimeError("durable action audit is unavailable")
        return
    try:
        audit.append_action(
            receipt.action.value,
            subject_id=receipt.target.target_id,
            event_name="action.command.processed",
            actor_id=receipt.requested_by,
            outcome=receipt.status.value,
            reason_codes=["provider_action_processed"],
            metadata={
                "action_type": receipt.action.value,
                "case_id": receipt.case_id,
                "policy_version": receipt.policy_version,
                "resource_id": receipt.target.target_id,
                "resource_type": receipt.target.target_type,
                "status": receipt.status.value,
                "outcome": receipt.status.value,
            },
        )
    except Exception as exc:
        if _audit_required(audit):
            raise RuntimeError("durable action audit append failed") from exc
        logger.warning("Optional action worker audit append failed: %s", type(exc).__name__)


def _enqueue(
    payload: dict[str, Any],
    *,
    action: ActionType,
    target_type: str,
    target_id: str,
    user: AuthUser,
    outbox: ActionOutbox,
    audit: Any,
) -> ActionReceipt:
    request = _route_request(
        payload,
        action=action,
        target_type=target_type,
        target_id=target_id,
        user=user,
    )
    try:
        # create_command is intentionally the only operation used here.  The
        # provider adapter is called solely by the worker helper below.
        receipt = outbox.create_command(request)
    except IdempotencyConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="idempotency_key is already associated with a different action request",
        ) from exc
    except ActionOutboxConfigurationError as exc:
        logger.error("Action outbox is unavailable: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Action command durability is unavailable",
        ) from exc
    _append_enqueue_audit(audit, request, receipt, user)
    return receipt


def _transaction_route(action: ActionType) -> Callable[..., ActionReceipt]:
    """Build a transaction route whose path parameter is named ``order_id``."""

    def handler(
        order_id: str,
        payload: dict[str, Any] = Body(...),
        user: AuthUser = Depends(require_role(Role.OPERATOR)),
        outbox: ActionOutbox = Depends(get_action_outbox),
        audit: Any = Depends(get_audit_service),
    ) -> ActionReceipt:
        return _enqueue(
            payload,
            action=action,
            target_type="order",
            target_id=order_id,
            user=user,
            outbox=outbox,
            audit=audit,
        )

    return handler


def _listing_route() -> Callable[..., ActionReceipt]:
    """Build the listing quarantine route with its named path parameter."""

    def handler(
        listing_id: str,
        payload: dict[str, Any] = Body(...),
        user: AuthUser = Depends(require_role(Role.OPERATOR)),
        outbox: ActionOutbox = Depends(get_action_outbox),
        audit: Any = Depends(get_audit_service),
    ) -> ActionReceipt:
        return _enqueue(
            payload,
            action=_LISTING_ACTION,
            target_type="listing",
            target_id=listing_id,
            user=user,
            outbox=outbox,
            audit=audit,
        )

    return handler


def _ring_route() -> Callable[..., ActionReceipt]:
    """Build the ring freeze route with its named path parameter."""

    def handler(
        ring_id: str,
        payload: dict[str, Any] = Body(...),
        user: AuthUser = Depends(require_role(Role.OPERATOR)),
        outbox: ActionOutbox = Depends(get_action_outbox),
        audit: Any = Depends(get_audit_service),
    ) -> ActionReceipt:
        return _enqueue(
            payload,
            action=_RING_ACTION,
            target_type="ring",
            target_id=ring_id,
            user=user,
            outbox=outbox,
            audit=audit,
        )

    return handler


# Keep separate decorators/functions for generated OpenAPI and stable route
# names.  The handlers all use the same strict enqueue implementation.
for _verb, _action in _TRANSACTION_ACTIONS.items():
    router.add_api_route(
        f"/transactions/{{order_id}}/{_verb}",
        _transaction_route(_action),
        methods=["POST"],
        response_model=ActionReceipt,
        status_code=status.HTTP_202_ACCEPTED,
        name=f"{_verb}_transaction_action",
        operation_id=f"actions_transactions_{_verb}",
    )


router.add_api_route(
    "/listings/{listing_id}/quarantine",
    _listing_route(),
    methods=["POST"],
    response_model=ActionReceipt,
    status_code=status.HTTP_202_ACCEPTED,
    name="quarantine_listing_action",
    operation_id="actions_listings_quarantine",
)

router.add_api_route(
    "/rings/{ring_id}/freeze",
    _ring_route(),
    methods=["POST"],
    response_model=ActionReceipt,
    status_code=status.HTTP_202_ACCEPTED,
    name="freeze_ring_action",
    operation_id="actions_rings_freeze",
)


@router.get("/{action_id}", response_model=ActionReceipt, name="get_action_receipt")
def get_action(
    action_id: str,
    _user: AuthUser = Depends(require_role(Role.OPERATOR)),
    outbox: ActionOutbox = Depends(get_action_outbox),
) -> ActionReceipt:
    """Return pending, succeeded, or failed state for one action command."""

    try:
        receipt = outbox.get_command(action_id)
    except ActionOutboxConfigurationError as exc:
        logger.error("Action outbox is unavailable: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Action command durability is unavailable",
        ) from exc
    if receipt is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Action not found")
    return receipt


def process_action_for_development(
    action_id: str,
    *,
    outbox: Optional[ActionOutbox] = None,
    audit: Any = None,
) -> ActionReceipt:
    """Deterministically process one command outside HTTP in dev/tests only.

    This helper is intentionally not registered as an HTTP route.  It invokes
    the existing outbox worker, so the fake adapter cannot produce a provider
    receipt until the caller explicitly asks the worker to process the pending
    command.  Production invocation is rejected rather than becoming an
    accidental provider-side execution shortcut.
    """

    service = outbox or get_action_outbox()
    configured_environment = str(
        getattr(getattr(service, "configuration", None), "environment", "")
    ).strip().lower()
    if _is_production_environment() or configured_environment in {"production", "prod"}:
        raise RuntimeError("the deterministic action worker helper is disabled in production")
    try:
        receipt = service.process_command(action_id)
    except ActionCommandNotFoundError as exc:
        raise KeyError(action_id) from exc
    _append_worker_audit(receipt, audit=audit)
    return receipt


# Explicit aliases make the intended non-HTTP test/worker seam easy to find.
process_action_for_tests = process_action_for_development
run_action_worker_for_tests = process_action_for_development


__all__ = [
    "get_action",
    "get_action_outbox",
    "get_audit_service",
    "process_action_for_development",
    "process_action_for_tests",
    "reset_action_services_for_tests",
    "router",
    "run_action_worker_for_tests",
]
