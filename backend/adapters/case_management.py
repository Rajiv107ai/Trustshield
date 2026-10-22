"""Case-management action adapter compatibility exports."""

from .fakes import (
    DeterministicFakeCaseAdapter,
    DeterministicFakeCaseManagementAdapter,
    FakeCaseManagementAdapter,
)
from .protocols import CaseAdapter, CaseManagementAdapter, CaseProviderAdapter

DeterministicCaseManagementAdapter = DeterministicFakeCaseManagementAdapter

__all__ = [
    "CaseAdapter",
    "CaseManagementAdapter",
    "CaseProviderAdapter",
    "DeterministicCaseManagementAdapter",
    "DeterministicFakeCaseAdapter",
    "DeterministicFakeCaseManagementAdapter",
    "FakeCaseManagementAdapter",
]
