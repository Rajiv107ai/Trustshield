"""Regression coverage for the canonical API input boundary (audit Fix 13)."""

import pytest
from pydantic import ValidationError

from backend.schemas import ListingScoreRequest, ReturnScoreRequest, TransactionScoreRequest


@pytest.mark.parametrize(
    "payload",
    [
        {"amount": -1.0},
        {"price_vs_base_price_ratio": -0.1},
        {"buyer_return_rate_before": 1.1},
        {"device_shared_buyer_count": -1.0},
    ],
)
def test_transaction_boundary_rejects_out_of_range_values(payload):
    with pytest.raises(ValidationError):
        TransactionScoreRequest(**payload)


def test_listing_boundary_rejects_negative_prices_and_age():
    with pytest.raises(ValidationError):
        ListingScoreRequest(price=-1.0)
    with pytest.raises(ValidationError):
        ListingScoreRequest(seller_age_days_at_listing=-1.0)


def test_return_boundary_rejects_invalid_rate_and_amount():
    with pytest.raises(ValidationError):
        ReturnScoreRequest(order_amount=-1.0)
    with pytest.raises(ValidationError):
        ReturnScoreRequest(seller_return_rate_before=1.1)
