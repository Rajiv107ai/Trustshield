"""Graph action adapter compatibility exports."""

from .fakes import DeterministicFakeGraphAdapter, FakeGraphAdapter
from .protocols import GraphAdapter, GraphProviderAdapter

DeterministicGraphAdapter = DeterministicFakeGraphAdapter

__all__ = [
    "DeterministicFakeGraphAdapter",
    "DeterministicGraphAdapter",
    "FakeGraphAdapter",
    "GraphAdapter",
    "GraphProviderAdapter",
]
