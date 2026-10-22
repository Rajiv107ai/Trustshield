"""Payment action adapter compatibility exports."""

from .fakes import DeterministicFakePaymentAdapter, FakePaymentAdapter
from .protocols import PaymentAdapter, PaymentProviderAdapter

DeterministicPaymentAdapter = DeterministicFakePaymentAdapter

__all__ = [
    "DeterministicFakePaymentAdapter",
    "DeterministicPaymentAdapter",
    "FakePaymentAdapter",
    "PaymentAdapter",
    "PaymentProviderAdapter",
]
