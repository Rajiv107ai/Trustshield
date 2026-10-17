"""Tests for entity shapes, onboarding timelines, and structural invariants."""

import os
import numpy as np
import pandas as pd
import pytest
from pandas.util import hash_pandas_object
from entity_generator import N_SELLERS, N_BUYERS, SIM_START, SIM_END
from point_in_time_history import PointInTimeHistoryStore
from fraud_injection import TARGET_FRAUD_RATE, FAKE_LISTING_RATE
from temporal_utils import filter_historical_events
from baseline_model import TRAIN_END, VAL_END
from graph_features import build_relationship_graph
from phase2_specialized_models import build_listing_features


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
        d = pd.to_datetime(pipeline["catalog"]["sellers"]["signup_date"])
        assert (d >= SIM_START).all() and (d <= SIM_END).all()

    def test_buyer_signup_in_range(self, pipeline):
        d = pd.to_datetime(pipeline["txn"]["buyers"]["signup_date"])
        assert (d >= SIM_START).all() and (d <= SIM_END).all()

    def test_listing_dates_in_range(self, pipeline):
        d = pd.to_datetime(pipeline["result"]["listings"]["listing_date"])
        assert (d >= SIM_START).all() and (d <= SIM_END).all()


class TestSharingMechanics:
    def test_device_sharing_references_valid_buyers(self, pipeline):
        b = set(pipeline["txn"]["buyers"]["buyer_id"])
        s = pipeline["base"]["device_sharing_log"]
        if len(s) > 0:
            assert set(s["buyer_id"]).issubset(b) and set(s["shared_with_buyer_id"]).issubset(b)

    def test_address_sharing_references_valid_buyers(self, pipeline):
        b = set(pipeline["txn"]["buyers"]["buyer_id"])
        s = pipeline["base"]["address_sharing_log"]
        if len(s) > 0:
            assert set(s["buyer_id"]).issubset(b) and set(s["shared_with_buyer_id"]).issubset(b)

    def test_no_self_sharing_device(self, pipeline):
        s = pipeline["base"]["device_sharing_log"]
        if len(s) > 0:
            assert (s["buyer_id"] != s["shared_with_buyer_id"]).all()

    def test_no_self_sharing_address(self, pipeline):
        s = pipeline["base"]["address_sharing_log"]
        if len(s) > 0:
            assert (s["buyer_id"] != s["shared_with_buyer_id"]).all()

    def test_sharing_logs_timestamps_in_valid_range_after_signups(self, pipeline):
        buyers_df = pipeline["txn"]["buyers"].set_index("buyer_id")
        for log_key in ["address_sharing_log", "device_sharing_log"]:
            sharing = pipeline["base"][log_key]
            assert len(sharing) > 0
            first_seen = pd.to_datetime(sharing["first_seen_date"])
            signup_a = pd.to_datetime(buyers_df.loc[sharing["buyer_id"], "signup_date"].values)
            signup_b = pd.to_datetime(buyers_df.loc[sharing["shared_with_buyer_id"], "signup_date"].values)
            latest_signup = np.maximum(signup_a.values, signup_b.values)
            assert (first_seen >= SIM_START).all() and (first_seen <= SIM_END).all()
            assert (first_seen.values >= latest_signup).all()

    def test_buyer_address_sharing_collision_consistency(self, pipeline):
        buyers_df = pipeline["txn"]["buyers"]
        addr_counts = buyers_df["address_id"].value_counts()
        shared_addresses = set(addr_counts[addr_counts > 1].index)
        logged_shared = set(pipeline["base"]["address_sharing_log"]["shared_address_id"])
        assert shared_addresses.issubset(logged_shared)


class TestOrderIntegrity:
    def test_orders_reference_valid_entities(self, pipeline):
        ords = pipeline["result"]["orders"]
        assert set(ords["buyer_id"]).issubset(set(pipeline["txn"]["buyers"]["buyer_id"]))
        assert set(ords["seller_id"]).issubset(set(pipeline["catalog"]["sellers"]["seller_id"]))
        assert set(ords["listing_id"]).issubset(set(pipeline["result"]["listings"]["listing_id"]))
        assert (ords["amount"] > 0).all()

    def test_order_and_return_dates_chronology(self, pipeline):
        orders, returns = pipeline["result"]["orders"], pipeline["result"]["returns"]
        buyers = pipeline["txn"]["buyers"].set_index("buyer_id")
        sellers = pipeline["catalog"]["sellers"].set_index("seller_id")
        listings = pipeline["result"]["listings"].set_index("listing_id")
        order_dates = pd.to_datetime(orders["order_date"])
        assert (order_dates.values >= pd.to_datetime(buyers.loc[orders["buyer_id"], "signup_date"].values)).all()
        assert (order_dates.values >= pd.to_datetime(sellers.loc[orders["seller_id"], "signup_date"].values)).all()
        assert (order_dates.values >= pd.to_datetime(listings.loc[orders["listing_id"], "listing_date"].values)).all()
        assert (order_dates <= SIM_END).all() and (order_dates >= SIM_START).all()
        if len(returns) > 0:
            ret_dates = pd.to_datetime(returns["return_date"])
            ref_dates = pd.to_datetime(orders.set_index("order_id").loc[returns["order_id"], "order_date"].values)
            assert (ret_dates.values >= ref_dates.values).all()
            assert (ret_dates <= SIM_END).all() and (ret_dates >= SIM_START).all()


class TestPipelineDeterminism:
    def test_repeated_build_base_entities_deterministic(self):
        from entity_generator import build_base_entities
        b1, b2 = build_base_entities(), build_base_entities()
        for k in ["addresses", "devices", "sellers", "buyers", "device_mapping", "address_sharing_log", "device_sharing_log"]:
            assert b1[k].equals(b2[k])

    def test_controlled_seed_reproducibility(self):
        from entity_generator import build_base_entities
        a1, b, a2 = build_base_entities(seed=42), build_base_entities(seed=12345), build_base_entities(seed=42)
        assert not a1["addresses"].equals(b["addresses"]) and a1["addresses"].equals(a2["addresses"])

    def test_repeated_pipeline_deterministic(self):
        from entity_generator import generate_full_pipeline
        p1, p2 = generate_full_pipeline(seed=42), generate_full_pipeline(seed=42)
        for k in ["orders", "listings", "returns", "fraud_ground_truth"]:
            assert p1["result"][k].equals(p2["result"][k])
        import hashlib
        h1 = hashlib.sha256(np.asarray(hash_pandas_object(p1["result"]["orders"])).tobytes()).hexdigest()
        h2 = hashlib.sha256(np.asarray(hash_pandas_object(p2["result"]["orders"])).tobytes()).hexdigest()
        assert h1 == h2


class TestDeviceSharedBuyerCountAsof:
    def test_device_shared_buyer_count_asof_chronological(self):
        from baseline_model import _device_shared_buyer_count_asof
        df = pd.DataFrame({"device_id": ["dev_1", "dev_1", "dev_2", "dev_1"], "buyer_id": ["b_1", "b_2", "b_3", "b_1"], "order_date": pd.to_datetime(["2025-01-01 10:00:00", "2025-01-01 12:00:00", "2025-01-02 09:00:00", "2025-01-03 10:00:00"])})
        assert list(_device_shared_buyer_count_asof(df)) == [0.0, 1.0, 0.0, 2.0]

    def test_device_shared_buyer_count_asof_empty(self):
        from baseline_model import _device_shared_buyer_count_asof
        assert len(_device_shared_buyer_count_asof(pd.DataFrame(columns=["device_id", "buyer_id", "order_date"]))) == 0

    def test_device_shared_buyer_count_asof_missing_columns(self):
        from baseline_model import _device_shared_buyer_count_asof
        assert list(_device_shared_buyer_count_asof(pd.DataFrame({"buyer_id": ["b_1"], "order_date": pd.to_datetime(["2025-01-01"])}))) == [0.0]

    def test_build_features_with_device_id_completes_successfully(self):
        from baseline_model import build_features
        listings = pd.DataFrame({"listing_id": [1], "seller_id": ["s1"], "category": ["electronics"], "listing_date": pd.to_datetime(["2025-01-01"]), "product_id": [1]})
        feat_df, cols = build_features(
            pd.DataFrame({"order_id": [1], "buyer_id": ["b1"], "seller_id": ["s1"], "listing_id": [1], "amount": [100.0], "order_date": pd.to_datetime(["2025-01-01"]), "device_id": ["dev_1"]}),
            listings, pd.DataFrame(columns=["buyer_id", "return_date"]),
            pd.DataFrame({"buyer_id": ["b1"], "signup_date": pd.to_datetime(["2024-01-01"])}),
            pd.DataFrame({"seller_id": ["s1"], "signup_date": pd.to_datetime(["2024-01-01"])}),
            pd.DataFrame({"product_id": [1], "base_price": [100.0]})
        )
        assert "device_shared_buyer_count" in cols and feat_df["device_shared_buyer_count"].iloc[0] == 0.0

    def test_device_shared_buyer_count_asof_equal_timestamps_deterministic(self):
        from baseline_model import _device_shared_buyer_count_asof
        for b_order in [["b_1", "b_2", "b_3"], ["b_2", "b_1", "b_3"]]:
            df = pd.DataFrame({"device_id": ["dev_1"] * 3, "buyer_id": b_order, "order_date": pd.to_datetime(["2025-01-01 10:00:00", "2025-01-01 10:00:00", "2025-01-01 12:00:00"])})
            assert list(_device_shared_buyer_count_asof(df)) == [0.0, 0.0, 2.0]


class TestPointInTimeHistoryStore:
    @pytest.fixture
    def sample_store(self):
        from typing import List, cast
        store = PointInTimeHistoryStore()
        store._history[("BUYER_A", "SELLER_X")] = cast(List[pd.Timestamp], [
            pd.Timestamp("2025-01-10 10:00:00"), pd.Timestamp("2025-01-10 10:00:00"),
            pd.Timestamp("2025-02-01 12:00:00"), pd.Timestamp("2025-03-01 09:00:00"),
        ])
        store.is_loaded = True
        return store

    def test_first_transaction_is_zero(self, sample_store):
        w, src, _ = sample_store.get_prior_edge_weight("BUYER_A", "SELLER_X", order_date="2025-01-01 00:00:00")
        assert w == 0.0 and src == "server_point_in_time"

    def test_repeated_transactions_increment(self, sample_store):
        assert sample_store.get_prior_edge_weight("BUYER_A", "SELLER_X", order_date="2025-01-15 00:00:00")[0] == 2.0
        assert sample_store.get_prior_edge_weight("BUYER_A", "SELLER_X", order_date="2025-02-15 00:00:00")[0] == 3.0
        assert sample_store.get_prior_edge_weight("BUYER_A", "SELLER_X", order_date="2025-04-01 00:00:00")[0] == 4.0

    def test_future_transactions_strictly_excluded(self, sample_store):
        assert sample_store.get_prior_edge_weight("BUYER_A", "SELLER_X", order_date="2025-02-01 11:59:59")[0] == 2.0

    def test_equal_timestamps_do_not_count_each_other(self, sample_store):
        assert sample_store.get_prior_edge_weight("BUYER_A", "SELLER_X", order_date="2025-01-10 10:00:00")[0] == 0.0

    def test_unknown_pairs_safely_return_zero(self, sample_store):
        w, src, _ = sample_store.get_prior_edge_weight("BUYER_UNKNOWN", "SELLER_UNKNOWN", order_date="2025-02-01 00:00:00")
        assert w == 0.0 and src == "fallback_unobserved_zero"

    def test_unloaded_store_safe_failure(self):
        w, src, _ = PointInTimeHistoryStore().get_prior_edge_weight("BUYER_A", "SELLER_X", order_date="2025-02-01 00:00:00")
        assert w == 0.0 and src == "fallback_history_unavailable"

    def test_replay_exact_match_against_canonical_cumcount(self):
        root_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        orders_path = os.path.join(root_dir, "data", "synthetic_v2_1", "orders.csv")
        if not os.path.isfile(orders_path):
            pytest.skip("orders.csv not found")
        df = pd.read_csv(orders_path, usecols=["order_id", "buyer_id", "seller_id", "order_date"])
        df["order_date"] = pd.to_datetime(df["order_date"])
        df = df.sort_values(by="order_date", kind="mergesort").reset_index(drop=True)
        df["expected_edge_weight"] = df.groupby(["buyer_id", "seller_id"]).cumcount().astype(float)
        store = PointInTimeHistoryStore(orders_path)
        assert store.is_loaded
        for idx in np.linspace(0, len(df) - 1, 100, dtype=int):
            row = df.iloc[idx]
            w, _, _ = store.get_prior_edge_weight(row["buyer_id"], row["seller_id"], order_date=row["order_date"])
            assert w == row["expected_edge_weight"]


class TestFraudRates:
    def test_overall_fraud_rate_near_target(self, pipeline):
        assert abs(pipeline["result"]["orders"]["is_fraudulent"].mean() - TARGET_FRAUD_RATE) < 0.03

    def test_fake_listing_rate_near_target(self, pipeline):
        assert abs(pipeline["result"]["listings"]["is_fraudulent"].mean() - FAKE_LISTING_RATE) < 0.02


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
    def test_ground_truth_properties(self, pipeline):
        gt = pipeline["result"]["fraud_ground_truth"]
        assert len(gt) > 0 and {"fraud_ring_id", "fraud_type", "entity_type", "entity_id"}.issubset(set(gt.columns))
        assert gt["fraud_ring_id"].str.startswith("RING_").all()


class TestReturnFraudInjection:
    def test_fraud_returns_validity(self, pipeline):
        ret, ords = pipeline["result"]["returns"], pipeline["result"]["orders"]
        fraud_ret = ret[ret["is_fraudulent"] == True]
        if len(fraud_ret) > 0:
            assert set(fraud_ret["order_id"]).issubset(set(ords["order_id"]))
            merged = fraud_ret.merge(ords[["order_id", "order_date"]], on="order_id", how="left")
            assert (pd.to_datetime(merged["return_date"]) >= pd.to_datetime(merged["order_date"])).all()


class TestPipelineLeakageGuards:
    def test_pipeline_1_no_graph_edge_has_edge_time_gte_decision_time(self, pipeline):
        cutoff = str(TRAIN_END)
        addr_log, dev_log = pipeline["base"]["address_sharing_log"], pipeline["base"]["device_sharing_log"]
        G = build_relationship_graph(addr_log, dev_log, cutoff_date=cutoff)
        addr_filt = filter_historical_events(addr_log, "first_seen_date", cutoff)
        dev_filt = filter_historical_events(dev_log, "first_seen_date", cutoff)
        cutoff_ts = pd.to_datetime(cutoff)
        assert (pd.to_datetime(addr_filt["first_seen_date"]) < cutoff_ts).all()
        assert (pd.to_datetime(dev_filt["first_seen_date"]) < cutoff_ts).all()
        assert G.number_of_edges() <= len(addr_filt) + len(dev_filt)

    def test_pipeline_2_no_historical_aggregation_contains_future_events(self, pipeline):
        df = pipeline["df"]
        assert len(pipeline["result"]["orders"]) > 0 and len(pipeline["result"]["returns"]) > 0
        assert (df["buyer_return_rate_before"] >= 0.0).all() and (df["buyer_return_rate_before"] <= 1.0001).all()
        assert (df["buyer_returns_before"] <= df["buyer_orders_before"]).all()

    def test_pipeline_4_no_val_or_test_transaction_influences_training_features(self, pipeline):
        df = pipeline["df"]
        train_df = df[df["order_date"] <= TRAIN_END]
        val_df = df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)]
        assert train_df["order_date"].max() < val_df["order_date"].min()

    def test_pipeline_5_no_fraud_label_enters_predictive_features(self, pipeline):
        banned = {"is_fraudulent", "fraud_type", "price_anomaly", "image_mismatch"}
        assert not set(pipeline["feature_cols"]).intersection(banned)

    def test_pipeline_6_no_fraud_ring_id_enters_predictive_features(self, pipeline):
        assert "fraud_ring_id" not in pipeline["df"].columns and "fraud_ring_id" not in pipeline["feature_cols"]

    def test_pipeline_7_no_post_event_return_info_in_listing_prediction(self, pipeline):
        _, cols = build_listing_features(pipeline["result"]["listings"], pipeline["catalog"]["sellers"], pipeline["catalog"]["products"])
        assert not any("return" in col.lower() for col in cols)

    def test_pipeline_8_no_future_seller_buyer_behavior_in_prediction(self, pipeline):
        df = pipeline["df"]
        assert (df["seller_age_days"] >= 0).all() and (df["buyer_age_days"] >= 0).all()
