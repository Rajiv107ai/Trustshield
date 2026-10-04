"""Tests for entity shapes, onboarding timelines, and structural invariants."""

import pandas as pd
from entity_generator import N_SELLERS, N_BUYERS, SIM_START, SIM_END


class TestEntityCounts:
    def test_seller_count(self, pipeline):
        assert len(pipeline["catalog"]["sellers"]) == N_SELLERS

    def test_buyer_count(self, pipeline):
        assert len(pipeline["txn"]["buyers"]) == N_BUYERS

    def test_no_duplicate_seller_ids(self, pipeline):
        assert pipeline["catalog"]["sellers"]["seller_id"].is_unique

    def test_no_duplicate_buyer_ids(self, pipeline):
        assert pipeline["txn"]["buyers"]["buyer_id"].is_unique

    def test_no_duplicate_order_ids(self, pipeline):
        assert pipeline["result"]["orders"]["order_id"].is_unique


class TestOnboardingTimeline:
    def test_seller_signup_in_range(self, pipeline):
        dates = pd.to_datetime(pipeline["catalog"]["sellers"]["signup_date"])
        assert (dates >= SIM_START).all() and (dates <= SIM_END).all()

    def test_buyer_signup_in_range(self, pipeline):
        dates = pd.to_datetime(pipeline["txn"]["buyers"]["signup_date"])
        assert (dates >= SIM_START).all() and (dates <= SIM_END).all()

    def test_listing_dates_in_range(self, pipeline):
        dates = pd.to_datetime(pipeline["result"]["listings"]["listing_date"])
        assert (dates >= SIM_START).all() and (dates <= SIM_END).all()


class TestSharingMechanics:
    def test_device_sharing_references_valid_buyers(self, pipeline):
        buyers = set(pipeline["txn"]["buyers"]["buyer_id"])
        sharing = pipeline["base"]["device_sharing_log"]
        if len(sharing) > 0:
            assert set(sharing["buyer_id"]).issubset(buyers)
            assert set(sharing["shared_with_buyer_id"]).issubset(buyers)

    def test_address_sharing_references_valid_buyers(self, pipeline):
        buyers = set(pipeline["txn"]["buyers"]["buyer_id"])
        sharing = pipeline["base"]["address_sharing_log"]
        if len(sharing) > 0:
            assert set(sharing["buyer_id"]).issubset(buyers)
            assert set(sharing["shared_with_buyer_id"]).issubset(buyers)

    def test_no_self_sharing_device(self, pipeline):
        sharing = pipeline["base"]["device_sharing_log"]
        if len(sharing) > 0:
            assert (sharing["buyer_id"] != sharing["shared_with_buyer_id"]).all()

    def test_no_self_sharing_address(self, pipeline):
        sharing = pipeline["base"]["address_sharing_log"]
        if len(sharing) > 0:
            assert (sharing["buyer_id"] != sharing["shared_with_buyer_id"]).all()


class TestOrderIntegrity:
    def test_orders_reference_valid_buyers(self, pipeline):
        assert set(pipeline["result"]["orders"]["buyer_id"]).issubset(set(pipeline["txn"]["buyers"]["buyer_id"]))

    def test_orders_reference_valid_sellers(self, pipeline):
        assert set(pipeline["result"]["orders"]["seller_id"]).issubset(set(pipeline["catalog"]["sellers"]["seller_id"]))

    def test_orders_reference_valid_listings(self, pipeline):
        assert set(pipeline["result"]["orders"]["listing_id"]).issubset(set(pipeline["result"]["listings"]["listing_id"]))

    def test_order_amounts_positive(self, pipeline):
        assert (pipeline["result"]["orders"]["amount"] > 0).all()
