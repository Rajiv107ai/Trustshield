"""Marketplace action adapter compatibility exports."""

from .fakes import DeterministicFakeMarketplaceAdapter, FakeMarketplaceAdapter
from .protocols import MarketplaceAdapter, MarketplaceProviderAdapter

DeterministicMarketplaceAdapter = DeterministicFakeMarketplaceAdapter

__all__ = [
    "DeterministicFakeMarketplaceAdapter",
    "DeterministicMarketplaceAdapter",
    "FakeMarketplaceAdapter",
    "MarketplaceAdapter",
    "MarketplaceProviderAdapter",
]
