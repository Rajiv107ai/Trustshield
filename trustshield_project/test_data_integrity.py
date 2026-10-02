"""
TrustShield AI — Test Suite: Data Integrity & Entity Generators

Validates entity shapes, onboarding timelines, sharing mechanics, and
basic structural invariants of the synthetic dataset.
"""

import pandas as pd
from entity_generator import N_SELLERS, N_BUYERS, SIM_START, SIM_END


class TestEntityCounts:
    """The generated entity tables have the right shape and no duplicates."""

    def test_seller_count(self, pipeline):
        assert len(pipeline["catalog"]["sellers"]) == N_SELLERS

    def test_buyer_count(self, pipeline):
        assert len(pipeline["txn"]["buyers"]) == N_BUYERS

    def test_no_duplicate_seller_ids(self, pipeline):
        sellers = pipeline["catalog"]["sellers"]
        assert sellers["seller_id"].is_unique

    def test_no_duplicate_buyer_ids(self, pipeline):
        buyers = pipeline["txn"]["buyers"]
        assert buyers["buyer_id"].is_unique

    def test_no_duplicate_order_ids(self, pipeline):
        orders = pipeline["result"]["orders"]
        assert orders["order_id"].is_unique


class TestOnboardingTimeline:
    """All entity dates fall within the simulation window."""

    def test_seller_signup_in_range(self, pipeline):
        sellers = pipeline["catalog"]["sellers"]
        dates = pd.to_datetime(sellers["signup_date"])
        assert (dates >= SIM_START).all()
        assert (dates <= SIM_END).all()

    def test_buyer_signup_in_range(self, pipeline):
        buyers = pipeline["txn"]["buyers"]
        dates = pd.to_datetime(buyers["signup_date"])
        assert (dates >= SIM_START).all()
        assert (dates <= SIM_END).all()

    def test_listing_dates_in_range(self, pipeline):
        listings = pipeline["result"]["listings"]
        dates = pd.to_datetime(listings["listing_date"])
        assert (dates >= SIM_START).all()
        assert (dates <= SIM_END).all()


class TestSharingMechanics:
    """Device and address sharing logs are structurally valid."""

    def test_device_sharing_references_valid_buyers(self, pipeline):
        buyers = set(pipeline["txn"]["buyers"]["buyer_id"])
        sharing = pipeline["base"]["device_sharing_log"]
        if len(sharing) == 0:
            return
        assert set(sharing["buyer_id"]).issubset(buyers)
        assert set(sharing["shared_with_buyer_id"]).issubset(buyers)

    def test_address_sharing_references_valid_buyers(self, pipeline):
        buyers = set(pipeline["txn"]["buyers"]["buyer_id"])
        sharing = pipeline["base"]["address_sharing_log"]
        if len(sharing) == 0:
            return
        assert set(sharing["buyer_id"]).issubset(buyers)
        assert set(sharing["shared_with_buyer_id"]).issubset(buyers)

    def test_no_self_sharing_device(self, pipeline):
        sharing = pipeline["base"]["device_sharing_log"]
        if len(sharing) == 0:
            return
        assert (sharing["buyer_id"] != sharing["shared_with_buyer_id"]).all()

    def test_no_self_sharing_address(self, pipeline):
        sharing = pipeline["base"]["address_sharing_log"]
        if len(sharing) == 0:
            return
        assert (sharing["buyer_id"] != sharing["shared_with_buyer_id"]).all()



class TestOrderIntegrity:
    """Orders reference only entities that existed at order time."""

    def test_orders_reference_valid_buyers(self, pipeline):
        buyers = set(pipeline["txn"]["buyers"]["buyer_id"])
        orders = pipeline["result"]["orders"]
        assert set(orders["buyer_id"]).issubset(buyers)

    def test_orders_reference_valid_sellers(self, pipeline):
        sellers = set(pipeline["catalog"]["sellers"]["seller_id"])
        orders = pipeline["result"]["orders"]
        assert set(orders["seller_id"]).issubset(sellers)

    def test_orders_reference_valid_listings(self, pipeline):
        listings = set(pipeline["result"]["listings"]["listing_id"])
        orders = pipeline["result"]["orders"]
        assert set(orders["listing_id"]).issubset(listings)

    def test_order_amounts_positive(self, pipeline):
        orders = pipeline["result"]["orders"]
        assert (orders["amount"] > 0).all()
