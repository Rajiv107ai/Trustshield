"""Identity action adapter compatibility exports."""

from .fakes import DeterministicFakeIdentityAdapter, FakeIdentityAdapter
from .protocols import IdentityAdapter, IdentityProviderAdapter

DeterministicIdentityAdapter = DeterministicFakeIdentityAdapter

__all__ = [
    "DeterministicFakeIdentityAdapter",
    "DeterministicIdentityAdapter",
    "FakeIdentityAdapter",
    "IdentityAdapter",
    "IdentityProviderAdapter",
]
