"""Provider adapter protocols for Phase 4 actions.

Adapters are deliberately synchronous and small.  The action outbox invokes
them from its worker method, never while a router is handling the initial
command creation request.  A real integration can implement one of these
protocols without importing the SQLite outbox implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from backend.contracts.actions import ActionRequest, ProviderReceipt


class ActionAdapterError(RuntimeError):
    """Base error raised by an action provider adapter."""


class ActionAdapterTimeout(ActionAdapterError, TimeoutError):
    """The provider call did not complete and may be retried safely."""


class ActionProviderError(ActionAdapterError):
    """A provider rejected or could not execute an action."""


@runtime_checkable
class ActionAdapter(Protocol):
    """Protocol shared by every provider action adapter."""

    provider_name: str

    def execute(self, request: ActionRequest) -> ProviderReceipt:
        """Execute one already-validated action and return its receipt."""


@runtime_checkable
class PaymentAdapter(ActionAdapter, Protocol):
    """Payment-provider action adapter."""


@runtime_checkable
class MarketplaceAdapter(ActionAdapter, Protocol):
    """Marketplace-provider action adapter."""


@runtime_checkable
class IdentityAdapter(ActionAdapter, Protocol):
    """Identity-provider action adapter."""


@runtime_checkable
class GraphAdapter(ActionAdapter, Protocol):
    """Graph intelligence action adapter."""


@runtime_checkable
class CaseManagementAdapter(ActionAdapter, Protocol):
    """Case-management-provider action adapter."""


# Explicit names make the integration boundary readable to route/worker code.
PaymentProviderAdapter = PaymentAdapter
PaymentActionAdapter = PaymentAdapter
MarketplaceProviderAdapter = MarketplaceAdapter
MarketplaceActionAdapter = MarketplaceAdapter
IdentityProviderAdapter = IdentityAdapter
IdentityActionAdapter = IdentityAdapter
GraphProviderAdapter = GraphAdapter
GraphActionAdapter = GraphAdapter
CaseProviderAdapter = CaseManagementAdapter
CaseAdapter = CaseManagementAdapter
CaseActionAdapter = CaseManagementAdapter
ProviderAdapter = ActionAdapter


# Short aliases preserve a convenient import surface for callers that call
# these failures ``AdapterTimeout``/``AdapterError``.
AdapterTimeout = ActionAdapterTimeout
AdapterError = ActionProviderError
ProviderTimeout = ActionAdapterTimeout
ProviderError = ActionProviderError


__all__ = [
    "ActionAdapter",
    "ActionAdapterError",
    "ActionAdapterTimeout",
    "ActionProviderError",
    "AdapterError",
    "AdapterTimeout",
    "CaseAdapter",
    "CaseManagementAdapter",
    "GraphAdapter",
    "GraphActionAdapter",
    "GraphProviderAdapter",
    "IdentityAdapter",
    "IdentityActionAdapter",
    "IdentityProviderAdapter",
    "MarketplaceAdapter",
    "MarketplaceActionAdapter",
    "MarketplaceProviderAdapter",
    "PaymentAdapter",
    "PaymentActionAdapter",
    "PaymentProviderAdapter",
    "ProviderAdapter",
    "ProviderError",
    "ProviderTimeout",
]
