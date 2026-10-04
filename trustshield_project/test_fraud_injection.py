"""Tests validating synthetic fraud injection rates, labels, and entity links."""

import pandas as pd
from fraud_injection import TARGET_FRAUD_RATE, FAKE_LISTING_RATE, FRAUD_TYPE_SHARE


class TestFraudRates:
    def test_overall_fraud_rate_near_target(self, pipeline):
        actual_rate = pipeline["result"]["orders"]["is_fraudulent"].mean()
        assert abs(actual_rate - TARGET_FRAUD_RATE) < 0.03

    def test_fake_listing_rate_near_target(self, pipeline):
        actual_rate = pipeline["result"]["listings"]["is_fraudulent"].mean()
        assert abs(actual_rate - FAKE_LISTING_RATE) < 0.02


class TestFraudTypeConsistency:
    EXPECTED_TYPES = {"fake_listing", "return_abuse", "coordinated_fraud", "seller_buyer_collusion"}

    def test_all_four_fraud_types_present_in_orders(self, pipeline):
        fraud_orders = pipeline["result"]["orders"][pipeline["result"]["orders"]["is_fraudulent"] == True]
        assert self.EXPECTED_TYPES == set(fraud_orders["fraud_type"].dropna().unique())

    def test_all_four_fraud_types_in_ground_truth(self, pipeline):
        assert self.EXPECTED_TYPES == set(pipeline["result"]["fraud_ground_truth"]["fraud_type"].unique())

    def test_fraudulent_orders_always_have_type(self, pipeline):
        orders = pipeline["result"]["orders"]
        assert len(orders[(orders["is_fraudulent"] == True) & (orders["fraud_type"].isna())]) == 0

    def test_non_fraudulent_orders_have_no_type(self, pipeline):
        orders = pipeline["result"]["orders"]
        assert len(orders[(orders["is_fraudulent"] == False) & (orders["fraud_type"].notna())]) == 0


class TestFraudGroundTruth:
    def test_ground_truth_has_required_columns(self, pipeline):
        required = {"fraud_ring_id", "fraud_type", "entity_type", "entity_id"}
        assert required.issubset(set(pipeline["result"]["fraud_ground_truth"].columns))

    def test_ground_truth_not_empty(self, pipeline):
        assert len(pipeline["result"]["fraud_ground_truth"]) > 0

    def test_ring_ids_are_prefixed_by_type(self, pipeline):
        gt = pipeline["result"]["fraud_ground_truth"]
        for fraud_type in gt["fraud_type"].unique():
            subset = gt[gt["fraud_type"] == fraud_type]
            assert subset["fraud_ring_id"].str.startswith("RING_").all()


class TestReturnFraudInjection:
    def test_fraud_returns_have_valid_orders(self, pipeline):
        returns = pipeline["result"]["returns"]
        orders = pipeline["result"]["orders"]
        fraud_returns = returns[returns["is_fraudulent"] == True]
        if len(fraud_returns) > 0:
            assert set(fraud_returns["order_id"]).issubset(set(orders["order_id"]))

    def test_fraud_returns_have_valid_dates(self, pipeline):
        returns = pipeline["result"]["returns"]
        orders = pipeline["result"]["orders"]
        fraud_returns = returns[returns["is_fraudulent"] == True]
        if len(fraud_returns) > 0:
            merged = fraud_returns.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
            assert (pd.to_datetime(merged["return_date"]) >= pd.to_datetime(merged["order_date"])).all()
