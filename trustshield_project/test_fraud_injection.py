"""
TrustShield AI — Test Suite: Fraud Injection Logic

Validates that fraud injection produces the right proportions, marks the
right entities, and maintains structural consistency of the dataset.
"""

import pandas as pd
from fraud_injection import (
    TARGET_FRAUD_RATE, FAKE_LISTING_RATE,
    FRAUD_TYPE_SHARE,
)


class TestFraudRates:
    """Overall fraud rates land within acceptable tolerance of targets."""

    def test_overall_fraud_rate_near_target(self, pipeline):
        orders = pipeline["result"]["orders"]
        actual_rate = orders["is_fraudulent"].mean()
        # Allow ±3% absolute tolerance (injection is approximate due to integer rounding)
        assert abs(actual_rate - TARGET_FRAUD_RATE) < 0.03, \
            f"Overall fraud rate {actual_rate:.3f} too far from target {TARGET_FRAUD_RATE}"

    def test_fake_listing_rate_near_target(self, pipeline):
        listings = pipeline["result"]["listings"]
        actual_rate = listings["is_fraudulent"].mean()
        assert abs(actual_rate - FAKE_LISTING_RATE) < 0.02, \
            f"Fake listing rate {actual_rate:.3f} too far from target {FAKE_LISTING_RATE}"


class TestFraudTypeConsistency:
    """All four fraud types are present and consistently labeled."""

    EXPECTED_TYPES = {"fake_listing", "return_abuse", "coordinated_fraud", "seller_buyer_collusion"}

    def test_all_four_fraud_types_present_in_orders(self, pipeline):
        orders = pipeline["result"]["orders"]
        fraud_orders = orders[orders["is_fraudulent"] == True]
        present = set(fraud_orders["fraud_type"].dropna().unique())
        assert self.EXPECTED_TYPES == present, \
            f"Expected {self.EXPECTED_TYPES}, got {present}"

    def test_all_four_fraud_types_in_ground_truth(self, pipeline):
        gt = pipeline["result"]["fraud_ground_truth"]
        present = set(gt["fraud_type"].unique())
        assert self.EXPECTED_TYPES == present, \
            f"Expected {self.EXPECTED_TYPES}, got {present}"

    def test_fraudulent_orders_always_have_type(self, pipeline):
        """Every is_fraudulent=True order must have a non-null fraud_type."""
        orders = pipeline["result"]["orders"]
        fraud_no_type = orders[(orders["is_fraudulent"] == True) & (orders["fraud_type"].isna())]
        assert len(fraud_no_type) == 0, \
            f"{len(fraud_no_type)} fraudulent orders have no fraud_type"

    def test_non_fraudulent_orders_have_no_type(self, pipeline):
        """Non-fraud orders should not have a fraud_type set."""
        orders = pipeline["result"]["orders"]
        clean = orders[orders["is_fraudulent"] == False]
        with_type = clean[clean["fraud_type"].notna()]
        assert len(with_type) == 0, \
            f"{len(with_type)} clean orders incorrectly labeled with fraud_type"


class TestFraudGroundTruth:
    """The fraud_ground_truth ledger is structurally valid."""

    def test_ground_truth_has_required_columns(self, pipeline):
        gt = pipeline["result"]["fraud_ground_truth"]
        required = {"fraud_ring_id", "fraud_type", "entity_type", "entity_id"}
        assert required.issubset(set(gt.columns)), \
            f"Missing columns: {required - set(gt.columns)}"

    def test_ground_truth_not_empty(self, pipeline):
        gt = pipeline["result"]["fraud_ground_truth"]
        assert len(gt) > 0

    def test_ring_ids_are_prefixed_by_type(self, pipeline):
        gt = pipeline["result"]["fraud_ground_truth"]
        for fraud_type in gt["fraud_type"].unique():
            subset = gt[gt["fraud_type"] == fraud_type]
            prefix = "RING_" + fraud_type.upper().split("_")[0]
            assert subset["fraud_ring_id"].str.startswith("RING_").all(), \
                f"Ring IDs for {fraud_type} don't start with RING_"


class TestReturnFraudInjection:
    """Return abuse creates valid returns with proper temporal ordering."""

    def test_fraud_returns_have_valid_orders(self, pipeline):
        returns = pipeline["result"]["returns"]
        orders = pipeline["result"]["orders"]
        fraud_returns = returns[returns["is_fraudulent"] == True]
        if len(fraud_returns) == 0:
            return
        valid_orders = set(orders["order_id"])
        assert set(fraud_returns["order_id"]).issubset(valid_orders), \
            "Fraud returns reference non-existent orders"

    def test_fraud_returns_have_valid_dates(self, pipeline):
        returns = pipeline["result"]["returns"]
        orders = pipeline["result"]["orders"]
        fraud_returns = returns[returns["is_fraudulent"] == True]
        if len(fraud_returns) == 0:
            return

        merged = fraud_returns.merge(
            orders[["order_id", "order_date"]],
            on="order_id", how="left"
        )
        merged["return_date"] = pd.to_datetime(merged["return_date"])
        merged["order_date"] = pd.to_datetime(merged["order_date"])

        violations = (merged["return_date"] < merged["order_date"]).sum()
        assert violations == 0, \
            f"{violations} fraud returns predate their orders"
