"""Tests for entity shapes, onboarding timelines, and structural invariants."""

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
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

    def test_sharing_logs_timestamps_in_valid_range_after_signups(self, pipeline):
        """Regression test for Phase 1: every sharing relationship first_seen_date
        must be >= max(signup_a, signup_b), <= SIM_END, and strictly >= SIM_START.
        """
        buyers_df = pipeline["txn"]["buyers"].set_index("buyer_id")
        for log_key in ["address_sharing_log", "device_sharing_log"]:
            sharing = pipeline["base"][log_key]
            assert len(sharing) > 0, f"Expected non-empty {log_key}"
            first_seen = pd.to_datetime(sharing["first_seen_date"])
            signup_a = pd.to_datetime(buyers_df.loc[sharing["buyer_id"], "signup_date"].values)
            signup_b = pd.to_datetime(buyers_df.loc[sharing["shared_with_buyer_id"], "signup_date"].values)
            latest_signup = np.maximum(signup_a.values, signup_b.values)

            # Assert no 1970-era dates and strictly within simulation horizon
            assert (first_seen >= SIM_START).all(), f"Found dates before SIM_START in {log_key}"
            assert (first_seen <= SIM_END).all(), f"Found dates after SIM_END in {log_key}"
            assert (first_seen.values >= latest_signup).all(), f"Found first_seen earlier than account signups in {log_key}"

    def test_buyer_address_sharing_collision_consistency(self, pipeline):
        """Regression test for Phase 1: only logged sharing pairs share addresses in buyers table."""
        buyers_df = pipeline["txn"]["buyers"]
        addr_counts = buyers_df["address_id"].value_counts()
        shared_addresses = set(addr_counts[addr_counts > 1].index)
        logged_shared = set(pipeline["base"]["address_sharing_log"]["shared_address_id"])
        # All shared addresses in buyers table must come from the logged sharing relationships
        assert shared_addresses.issubset(logged_shared), "Found unlogged random address collisions among buyers"


class TestOrderIntegrity:
    def test_orders_reference_valid_buyers(self, pipeline):
        assert set(pipeline["result"]["orders"]["buyer_id"]).issubset(set(pipeline["txn"]["buyers"]["buyer_id"]))

    def test_orders_reference_valid_sellers(self, pipeline):
        assert set(pipeline["result"]["orders"]["seller_id"]).issubset(set(pipeline["catalog"]["sellers"]["seller_id"]))

    def test_orders_reference_valid_listings(self, pipeline):
        assert set(pipeline["result"]["orders"]["listing_id"]).issubset(set(pipeline["result"]["listings"]["listing_id"]))

    def test_order_amounts_positive(self, pipeline):
        assert (pipeline["result"]["orders"]["amount"] > 0).all()

    def test_order_and_return_dates_chronology(self, pipeline):
        """Regression test for Phase 1: orders occur after buyer/seller/listing onboarding and returns occur after orders."""
        orders = pipeline["result"]["orders"]
        returns = pipeline["result"]["returns"]
        buyers = pipeline["txn"]["buyers"].set_index("buyer_id")
        sellers = pipeline["catalog"]["sellers"].set_index("seller_id")
        listings = pipeline["result"]["listings"].set_index("listing_id")

        order_dates = pd.to_datetime(orders["order_date"])
        buyer_signups = pd.to_datetime(buyers.loc[orders["buyer_id"], "signup_date"].values)
        seller_signups = pd.to_datetime(sellers.loc[orders["seller_id"], "signup_date"].values)
        listing_dates = pd.to_datetime(listings.loc[orders["listing_id"], "listing_date"].values)

        assert (order_dates.values >= buyer_signups.values).all(), "Order placed before buyer signup"
        assert (order_dates.values >= seller_signups.values).all(), "Order placed before seller signup"
        assert (order_dates.values >= listing_dates.values).all(), "Order placed before listing date"
        assert (order_dates <= SIM_END).all()
        assert (order_dates >= SIM_START).all()

        if len(returns) > 0:
            orders_idx = orders.set_index("order_id")
            return_dates = pd.to_datetime(returns["return_date"])
            ref_order_dates = pd.to_datetime(orders_idx.loc[returns["order_id"], "order_date"].values)
            assert (return_dates.values >= ref_order_dates.values).all(), "Return occurred before order"
            assert (return_dates <= SIM_END).all()
            assert (return_dates >= SIM_START).all()



class TestPipelineDeterminism:
    """Issue #5: Regression tests ensuring repeated pipeline calls within the same
    process are fully deterministic when starting from a controlled seed.
    """

    def test_repeated_build_base_entities_deterministic(self):
        """Calling build_base_entities() twice consecutively in the same process
        must yield identical tables.
        """
        from entity_generator import build_base_entities

        b1 = build_base_entities()
        b2 = build_base_entities()

        for key in ["addresses", "devices", "sellers", "buyers", "device_mapping",
                    "address_sharing_log", "device_sharing_log"]:
            assert b1[key].equals(b2[key]), f"Mismatch in base entity table '{key}' on repeated call"

    def test_controlled_seed_reproducibility(self):
        """Interleaving runs with different seeds must not compromise repeatability."""
        from entity_generator import build_base_entities

        run_a1 = build_base_entities(seed=42)
        run_b = build_base_entities(seed=12345)
        run_a2 = build_base_entities(seed=42)

        assert not run_a1["addresses"].equals(run_b["addresses"]), "Different seeds should produce different data"
        assert run_a1["addresses"].equals(run_a2["addresses"]), "Same seed must produce identical data across runs"

    def test_repeated_pipeline_deterministic(self):
        """Running the full pipeline twice in the same process must yield identical key frames and hashes."""
        import hashlib
        from entity_generator import generate_full_pipeline

        p1 = generate_full_pipeline(seed=42)
        p2 = generate_full_pipeline(seed=42)

        # Compare key dataframes
        for frame_name in ["orders", "listings", "returns", "fraud_ground_truth"]:
            df1 = p1["result"][frame_name]
            df2 = p2["result"][frame_name]
            assert df1.equals(df2), f"Pipeline result '{frame_name}' differs across runs"

        # Compare stable hashes of orders table
        h1 = hashlib.sha256(np.asarray(hash_pandas_object(p1["result"]["orders"])).tobytes()).hexdigest()
        h2 = hashlib.sha256(np.asarray(hash_pandas_object(p2["result"]["orders"])).tobytes()).hexdigest()
        assert h1 == h2, f"Orders hash mismatch: {h1} != {h2}"

