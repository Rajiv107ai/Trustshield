"""
TrustShield AI — Test Suite: Temporal Safety & Data Leakage

These tests codify the leakage-audit checklists that were previously
print-based assertions scattered across baseline_model.py,
phase2_specialized_models.py, and graph_features.py.

If any of these fail, it means a code change has introduced data leakage
into the ML pipeline — exactly the kind of silent regression that is
hardest to catch in ML projects.
"""

import pandas as pd
import numpy as np
from baseline_model import TRAIN_END, VAL_END


# =========================================================================
# Phase 1C — Order-level feature leakage
# =========================================================================

class TestFeatureSetSanity:
    """No fraud-mechanism or ground-truth columns leak into the feature set."""

    BANNED_COLS = {
        "is_fraudulent", "fraud_type", "price_anomaly", "image_mismatch",
        "fraud_ring_id", "displayed_product_id",
    }

    def test_no_banned_columns_in_features(self, pipeline):
        used_banned = self.BANNED_COLS.intersection(pipeline["feature_cols"])
        assert not used_banned, f"Banned columns in feature set: {used_banned}"

    def test_fraud_ring_id_absent_from_df(self, pipeline):
        assert "fraud_ring_id" not in pipeline["df"].columns, \
            "fraud_ring_id leaked into the feature DataFrame"


class TestTemporalIntegrity:
    """Temporal-safe features never look into the future."""

    def test_buyer_returns_never_exceed_orders(self, pipeline):
        """A buyer cannot have returned more items than they ordered."""
        df = pipeline["df"]
        violations = (df["buyer_returns_before"] > df["buyer_orders_before"]).sum()
        assert violations == 0, f"{violations} rows where returns > orders"

    def test_no_negative_seller_age(self, pipeline):
        """seller_age_days < 0 would mean the order predates the seller's signup."""
        df = pipeline["df"]
        neg = (df["seller_age_days"] < 0).sum()
        assert neg == 0, f"{neg} orders with negative seller age"

    def test_no_negative_buyer_age(self, pipeline):
        """buyer_age_days < 0 would mean the order predates the buyer's signup."""
        df = pipeline["df"]
        neg = (df["buyer_age_days"] < 0).sum()
        assert neg == 0, f"{neg} orders with negative buyer age"

    def test_return_rate_bounded(self, pipeline):
        """buyer_return_rate_before should be in [0, 1] if returns ≤ orders."""
        df = pipeline["df"]
        rate = df["buyer_return_rate_before"]
        assert (rate >= 0).all(), "Negative return rates found"
        assert (rate <= 1.0001).all(), "Return rate exceeds 1.0"


class TestTemporalSplit:
    """Train/val/test splits are strictly ordered by date with no overlap."""

    def test_train_before_val(self, pipeline):
        train_max = pipeline["train"]["order_date"].max()
        val_min = pipeline["val"]["order_date"].min()
        assert train_max <= val_min, \
            f"Train max {train_max} overlaps with val min {val_min}"

    def test_val_before_test(self, pipeline):
        val_max = pipeline["val"]["order_date"].max()
        test_min = pipeline["test"]["order_date"].min()
        assert val_max <= test_min, \
            f"Val max {val_max} overlaps with test min {test_min}"

    def test_train_ends_before_cutoff(self, pipeline):
        train_max = pipeline["train"]["order_date"].max()
        assert train_max <= TRAIN_END

    def test_val_ends_before_cutoff(self, pipeline):
        val_max = pipeline["val"]["order_date"].max()
        assert val_max <= VAL_END

    def test_no_empty_splits(self, pipeline):
        assert len(pipeline["train"]) > 0, "Train set is empty"
        assert len(pipeline["val"]) > 0, "Val set is empty"
        assert len(pipeline["test"]) > 0, "Test set is empty"


# =========================================================================
# Fraud injection leakage guards
# =========================================================================

class TestFraudInjectionLeakage:
    """fraud_ring_id never touches feature tables (listings, orders, returns)."""

    def test_no_fraud_ring_id_in_listings(self, pipeline):
        assert "fraud_ring_id" not in pipeline["result"]["listings"].columns

    def test_no_fraud_ring_id_in_orders(self, pipeline):
        assert "fraud_ring_id" not in pipeline["result"]["orders"].columns

    def test_no_fraud_ring_id_in_returns(self, pipeline):
        assert "fraud_ring_id" not in pipeline["result"]["returns"].columns

    def test_fraud_ground_truth_has_ring_ids(self, pipeline):
        """The ground truth ledger SHOULD have fraud_ring_id."""
        gt = pipeline["result"]["fraud_ground_truth"]
        assert "fraud_ring_id" in gt.columns


# =========================================================================
# Price anomaly propagation (the critical bug we fixed)
# =========================================================================

class TestPriceAnomalyPropagation:
    """The discounted price from fake listings is reflected in orders."""

    def test_fake_listing_orders_have_discounted_prices(self, pipeline):
        listings = pipeline["result"]["listings"]
        orders = pipeline["result"]["orders"]

        # Get fake listings that have price anomalies
        anomaly_listings = listings[
            (listings["is_fraudulent"] == True) &
            (listings["price_anomaly"] == True)
        ]
        if len(anomaly_listings) == 0:
            return  # nothing to test

        # For each anomaly listing, the order amount should match the
        # listing's (discounted) price, not the original base_price
        for _, listing in anomaly_listings.iterrows():
            matching_orders = orders[orders["listing_id"] == listing["listing_id"]]
            if len(matching_orders) == 0:
                continue
            # All orders for this listing should have the discounted price
            assert (matching_orders["amount"] == listing["price"]).all(), \
                f"Order amounts don't match discounted listing price for {listing['listing_id']}"


# =========================================================================
# Return temporal safety
# =========================================================================

class TestReturnTemporalSafety:
    """Returns cannot predate their own orders."""

    def test_no_return_before_order(self, pipeline):
        returns = pipeline["result"]["returns"]
        orders = pipeline["result"]["orders"]

        merged = returns.merge(
            orders[["order_id", "order_date"]],
            on="order_id", how="left"
        )
        merged["return_date"] = pd.to_datetime(merged["return_date"])
        merged["order_date"] = pd.to_datetime(merged["order_date"])

        violations = (merged["return_date"] < merged["order_date"]).sum()
        assert violations == 0, f"{violations} returns predate their order"
