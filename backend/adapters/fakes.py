"""Deterministic local adapters for Phase 4 action integration tests.

The fakes never perform a network call.  A receipt reference is derived from
the validated request, so submitting the same command produces the same
provider reference.  They can be configured to time out or fail to exercise
the outbox worker's retry and terminal-failure paths.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Optional

from backend.contracts.actions import ActionRequest, ActionType, ProviderReceipt

from .protocols import ActionAdapterTimeout, ActionProviderError


def _request_digest(request: ActionRequest) -> str:
    payload = request.model_dump(mode="json", by_alias=True)
    # The schema version and all confirmation fields are intentionally in the
    # digest.  A changed authorization context must not reuse a fake receipt.
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DeterministicFakeAdapter:
    """Shared implementation for deterministic provider fakes."""

    provider_name = "fake"

    def __init__(
        self,
        *,
        timeout: bool = False,
        fail: bool = False,
        should_timeout: Optional[bool] = None,
        should_fail: Optional[bool] = None,
        timeout_actions: Iterable[str | ActionType] = (),
        failure_actions: Iterable[str | ActionType] = (),
    ) -> None:
        self.timeout = timeout if should_timeout is None else should_timeout
        self.fail = fail if should_fail is None else should_fail
        self.timeout_actions = {self._action_name(item) for item in timeout_actions}
        self.failure_actions = {self._action_name(item) for item in failure_actions}
        self.calls: list[ActionRequest] = []

    @staticmethod
    def _action_name(action: str | ActionType) -> str:
        return action.value if isinstance(action, ActionType) else str(action)

    def execute(self, request: ActionRequest) -> ProviderReceipt:
        self.calls.append(request)
        action_name = request.action.value
        if self.timeout or action_name in self.timeout_actions:
            raise ActionAdapterTimeout(f"{self.provider_name} provider timed out")
        if self.fail or action_name in self.failure_actions:
            raise ActionProviderError(f"{self.provider_name} provider rejected the action")

        digest = _request_digest(request)
        short_digest = digest[:20]
        return ProviderReceipt(
            receipt_id=f"{self.provider_name}-receipt-{short_digest}",
            provider=self.provider_name,
            action=request.action,
            idempotency_key=request.idempotency_key,
            provider_reference=f"{self.provider_name}-reference-{short_digest}",
            target=request.target,
            accepted=True,
            details={
                "adapter": "deterministic_fake",
                "case_id": request.case_id,
                "policy_version": request.policy_version,
            },
        )

    # ``perform`` is a harmless compatibility alias for provider clients that
    # use that verb while the outbox protocol remains ``execute``.
    def perform(self, request: ActionRequest) -> ProviderReceipt:
        return self.execute(request)


class DeterministicFakePaymentAdapter(DeterministicFakeAdapter):
    provider_name = "fake-payment"


class DeterministicFakeMarketplaceAdapter(DeterministicFakeAdapter):
    provider_name = "fake-marketplace"


class DeterministicFakeIdentityAdapter(DeterministicFakeAdapter):
    provider_name = "fake-identity"


class DeterministicFakeGraphAdapter(DeterministicFakeAdapter):
    provider_name = "fake-graph"


class DeterministicFakeCaseManagementAdapter(DeterministicFakeAdapter):
    provider_name = "fake-case-management"


# Concise names are useful in tests and are the names exposed by the package.
FakePaymentAdapter = DeterministicFakePaymentAdapter
FakeMarketplaceAdapter = DeterministicFakeMarketplaceAdapter
FakeIdentityAdapter = DeterministicFakeIdentityAdapter
FakeGraphAdapter = DeterministicFakeGraphAdapter
FakeCaseManagementAdapter = DeterministicFakeCaseManagementAdapter
FakeCaseAdapter = DeterministicFakeCaseManagementAdapter
DeterministicFakeCaseAdapter = DeterministicFakeCaseManagementAdapter


__all__ = [
    "DeterministicFakeAdapter",
    "DeterministicFakeCaseAdapter",
    "DeterministicFakeCaseManagementAdapter",
    "DeterministicFakeGraphAdapter",
    "DeterministicFakeIdentityAdapter",
    "DeterministicFakeMarketplaceAdapter",
    "DeterministicFakePaymentAdapter",
    "FakeCaseAdapter",
    "FakeCaseManagementAdapter",
    "FakeGraphAdapter",
    "FakeIdentityAdapter",
    "FakeMarketplaceAdapter",
    "FakePaymentAdapter",
]
