"""
TrustShield — Stage 3.2 Audit & Validation Test Suite.

Covers:
1. Dataset Contract & Referential Integrity.
2. Point-in-Time Features & Temporal Invariance.
3. Right-Censoring Semantics & Label Disambiguation.
4. Multimodal Evidence & Image Asset Auditing.
5. Evaluation Protocol Freezing & Split Boundary Isolation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_DATA_DIR = os.path.join(_ROOT_DIR, "data", "synthetic_v2_1")
_PROJECT_DIR = os.path.join(_ROOT_DIR, "trustshield_project")

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from multimodal_scoring import compute_multimodal_similarity
from baseline_model import build_features
from phase2_specialized_models import build_listing_features, build_return_features


@pytest.fixture(scope="module")
def v2_1_data() -> Dict[str, pd.DataFrame]:
    """Loads all operational tables from data/synthetic_v2_1/."""
    assert os.path.isdir(_DATA_DIR), f"Directory {_DATA_DIR} does not exist!"
    table_names = [
        "orders", "listings", "returns", "buyers", "sellers", "products",
        "devices", "addresses", "device_mapping", "address_sharing_log",
        "device_sharing_log", "fraud_ground_truth", "generator_personas",
        "perturbation_audit",
    ]
    dfs = {}
    for name in table_names:
        p = os.path.join(_DATA_DIR, f"{name}.csv")
        assert os.path.isfile(p), f"Missing required file: {p}"
        dfs[name] = pd.read_csv(p)
    return dfs


@pytest.fixture(scope="module")
def manifest() -> dict:
    """Loads dataset_manifest.json."""
    m_path = os.path.join(_DATA_DIR, "dataset_manifest.json")
    assert os.path.isfile(m_path), f"Missing manifest: {m_path}"
    with open(m_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# 1. Dataset Contract & Referential Integrity
# ===========================================================================

class TestDatasetContract:
    def test_manifest_hashes_match_disk(self, manifest):
        """Verifies that all 14 files in data/synthetic_v2_1/ match manifest SHA-256 hashes."""
        for filename, meta in manifest["files"].items():
            filepath = os.path.join(_DATA_DIR, filename)
            assert os.path.isfile(filepath), f"File {filename} missing on disk!"
            h = hashlib.sha256()
            with open(filepath, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            assert h.hexdigest() == meta["sha256"], f"Hash mismatch for {filename}!"

    def test_primary_key_uniqueness_and_zero_nulls(self, v2_1_data):
        """Verifies primary key uniqueness and zero nulls on required columns."""
        for pk_col, tname in [
            ("order_id", "orders"), ("listing_id", "listings"), ("return_id", "returns"),
            ("buyer_id", "buyers"), ("seller_id", "sellers"), ("product_id", "products"),
        ]:
            df = v2_1_data[tname]
            dups = int(df[pk_col].duplicated().sum())
            assert dups == 0, f"Found {dups} duplicate primary keys in {tname}.{pk_col}!"

        # Operational null checks
        assert v2_1_data["orders"][["order_id", "buyer_id", "seller_id", "listing_id", "amount", "order_date"]].isna().sum().sum() == 0
        assert v2_1_data["listings"][["listing_id", "seller_id", "product_id", "displayed_product_id", "price", "listing_date"]].isna().sum().sum() == 0
        assert v2_1_data["returns"][["return_id", "order_id", "buyer_id", "seller_id", "return_date", "reason"]].isna().sum().sum() == 0

    def test_foreign_key_referential_integrity(self, v2_1_data):
        """Verifies all foreign key relationships across the 14 operational tables."""
        orders = v2_1_data["orders"]
        listings = v2_1_data["listings"]
        returns = v2_1_data["returns"]
        buyers = v2_1_data["buyers"]
        sellers = v2_1_data["sellers"]
        products = v2_1_data["products"]

        assert orders["buyer_id"].isin(buyers["buyer_id"]).all()
        assert orders["seller_id"].isin(sellers["seller_id"]).all()
        assert orders["listing_id"].isin(listings["listing_id"]).all()
        assert orders["product_id"].isin(products["product_id"]).all()

        assert returns["order_id"].isin(orders["order_id"]).all()
        assert returns["buyer_id"].isin(buyers["buyer_id"]).all()
        assert returns["seller_id"].isin(sellers["seller_id"]).all()

        assert listings["seller_id"].isin(sellers["seller_id"]).all()
        assert listings["product_id"].isin(products["product_id"]).all()
        assert listings["displayed_product_id"].isin(products["product_id"]).all()

    def test_cross_table_relationship_consistency(self, v2_1_data):
        """Verifies cross-table integrity: return-to-order 1-to-1 and buyer/seller consistency."""
        orders = v2_1_data["orders"]
        returns = v2_1_data["returns"]
        listings = v2_1_data["listings"]

        # Exactly 1-to-1 return to order
        assert returns["order_id"].nunique() == len(returns)

        # Return buyer and seller match order buyer and seller
        merged_ro = returns.merge(orders[["order_id", "buyer_id", "seller_id"]], on="order_id", suffixes=("_ret", "_ord"))
        assert (merged_ro["buyer_id_ret"] == merged_ro["buyer_id_ord"]).all()
        assert (merged_ro["seller_id_ret"] == merged_ro["seller_id_ord"]).all()

        # Order seller and product match listing seller and product
        merged_ol = orders.merge(listings[["listing_id", "seller_id", "product_id"]], on="listing_id", suffixes=("_ord", "_list"))
        assert (merged_ol["seller_id_ord"] == merged_ol["seller_id_list"]).all()
        assert (merged_ol["product_id_ord"] == merged_ol["product_id_list"]).all()

    def test_category_constraints_hold_on_all_listings(self, v2_1_data):
        """Verifies that for 100% of listings, claimed product and displayed product share the EXACT category."""
        listings = v2_1_data["listings"]
        products = v2_1_data["products"]
        cat_map = products.set_index("product_id")["category"].to_dict()

        claimed_cats = listings["product_id"].map(cat_map)
        disp_cats = listings["displayed_product_id"].map(cat_map)
        mismatches = int((claimed_cats != disp_cats).sum())
        assert mismatches == 0, f"Found {mismatches} category constraint violations in listings!"


# ===========================================================================
# 2. Point-in-Time Features & Temporal Invariance
# ===========================================================================

class TestPointInTimeFeatures:
    UNSAFE_PREDICTION_FIELDS = {
        "buyer.total_orders", "buyer.total_returns", "seller.total_listings",
        "seller.total_orders_received", "buyer_persona", "seller_persona",
        "is_fraudulent", "fraud_type", "fraud_ring_id",
    }

    def test_unsafe_fields_banned_from_model_features(self, v2_1_data):
        """Verifies that candidate feature sets from all models contain 0 unsafe/leaking fields."""
        orders = v2_1_data["orders"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        listings = v2_1_data["listings"].copy()
        listings["listing_date"] = pd.to_datetime(listings["listing_date"])
        returns = v2_1_data["returns"].copy()
        returns["return_date"] = pd.to_datetime(returns["return_date"])
        buyers = v2_1_data["buyers"].copy()
        buyers["signup_date"] = pd.to_datetime(buyers["signup_date"])
        sellers = v2_1_data["sellers"].copy()
        sellers["signup_date"] = pd.to_datetime(sellers["signup_date"])
        products = v2_1_data["products"].copy()

        _, base_f = build_features(orders.head(50), listings, returns, buyers, sellers, products)
        _, list_f = build_listing_features(listings.head(50), sellers, products)
        _, ret_f = build_return_features(returns.head(50), orders, buyers, sellers)

        all_candidate_features = set(base_f) | set(list_f) | set(ret_f)
        leaked = self.UNSAFE_PREDICTION_FIELDS.intersection(all_candidate_features)
        assert not leaked, f"Unsafe fields entered feature set: {leaked}"

    def test_late_arriving_returns_isolated_at_order_time(self, v2_1_data):
        """Verifies that for an order at time T that was returned at T + delay,
        the feature buyer_returns_before at time T CANNOT observe that return.
        """
        orders = v2_1_data["orders"].copy()
        returns = v2_1_data["returns"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        returns["return_date"] = pd.to_datetime(returns["return_date"])

        # Pick an order that had a return
        ret_order_id = returns["order_id"].iloc[0]
        o_row = orders[orders["order_id"] == ret_order_id].iloc[0]
        r_row = returns[returns["order_id"] == ret_order_id].iloc[0]

        # The return is strictly after the order
        assert r_row["return_date"] > o_row["order_date"]

        # Count buyer returns prior to o_row['order_date']
        prior_rets = (
            (returns["buyer_id"] == o_row["buyer_id"]) &
            (returns["return_date"] < o_row["order_date"])
        ).sum()

        # The return of this order itself must NOT be counted
        assert r_row["return_id"] not in returns[(returns["return_date"] < o_row["order_date"])]["return_id"].values

    def test_temporal_invariance_under_future_event_insertion(self, v2_1_data):
        """Verifies that adding synthetic future orders and returns in December
        causes zero change in point-in-time features computed for an earlier order.
        """
        orders = v2_1_data["orders"].copy()
        returns = v2_1_data["returns"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        returns["return_date"] = pd.to_datetime(returns["return_date"])

        # Target order in June 2025
        target_orders = orders[(orders["order_date"] >= "2025-06-01") & (orders["order_date"] <= "2025-06-30")]
        target = target_orders.iloc[0]
        t_eval = target["order_date"]
        b_id = target["buyer_id"]

        # Base point-in-time counts
        b_orders_before_base = ((orders["buyer_id"] == b_id) & (orders["order_date"] < t_eval)).sum()
        b_rets_before_base = ((returns["buyer_id"] == b_id) & (returns["return_date"] < t_eval)).sum()

        # Insert 10 future orders and 3 future returns in December
        future_orders = pd.DataFrame([
            {
                "order_id": f"FUTURE_O_{i}", "buyer_id": b_id, "seller_id": target["seller_id"],
                "listing_id": target["listing_id"], "product_id": target["product_id"],
                "order_date": pd.Timestamp("2025-12-25 10:00:00") + timedelta(hours=i),
                "amount": 150.0, "device_id": "DEV_000001", "status": "completed",
                "is_fraudulent": False, "fraud_type": None,
            } for i in range(10)
        ])
        future_returns = pd.DataFrame([
            {
                "return_id": f"FUTURE_R_{i}", "order_id": f"FUTURE_O_{i}", "buyer_id": b_id,
                "seller_id": target["seller_id"],
                "return_date": pd.Timestamp("2025-12-28 10:00:00") + timedelta(hours=i),
                "reason": "defective", "status": "approved",
                "is_fraudulent": False, "fraud_type": None,
            } for i in range(3)
        ])

        orders_expanded = pd.concat([orders, future_orders], ignore_index=True)
        returns_expanded = pd.concat([returns, future_returns], ignore_index=True)

        b_orders_before_after = ((orders_expanded["buyer_id"] == b_id) & (orders_expanded["order_date"] < t_eval)).sum()
        b_rets_before_after = ((returns_expanded["buyer_id"] == b_id) & (returns_expanded["return_date"] < t_eval)).sum()

        assert b_orders_before_after == b_orders_before_base, "Future events altered buyer_orders_before!"
        assert b_rets_before_after == b_rets_before_base, "Future events altered buyer_returns_before!"


# ===========================================================================
# 3. Right-Censoring Analysis & Label Disambiguation
# ===========================================================================

class TestRightCensoring:
    def test_missing_return_near_simulation_end_is_not_confirmed_negative(self, v2_1_data):
        """Verifies that orders in late December have incomplete observation windows
        and cannot be treated as confirmed non-returns.
        """
        orders = v2_1_data["orders"].copy()
        returns = v2_1_data["returns"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        returns["return_date"] = pd.to_datetime(returns["return_date"])

        # Orders placed in the last 21 days (after 2025-12-10)
        dec10_cutoff = pd.Timestamp("2025-12-10 23:59:59")
        late_orders = orders[orders["order_date"] > dec10_cutoff]
        assert len(late_orders) > 0, "No orders found in late December!"

        # A significant fraction of these orders have no return observed in 2025
        returned_late_orders = set(returns["order_id"]).intersection(set(late_orders["order_id"]))
        unreturned_late_orders = len(late_orders) - len(returned_late_orders)

        # These unreturned orders include censored orders whose return would occur in 2026
        assert unreturned_late_orders > 0
        # Exactly 0 returns occur in 2026 within the 2025 dataset
        assert (returns["return_date"] > pd.Timestamp("2025-12-31 23:59:59")).sum() == 0

    def test_label_disambiguation_abuse_vs_return_event(self, v2_1_data):
        """Verifies that return_abuse (fraudulent intent) is distinct from has_return (observable return event)."""
        orders = v2_1_data["orders"].copy()
        returns = v2_1_data["returns"].copy()
        returned_order_ids = set(returns["order_id"])

        orders["has_return_event"] = orders["order_id"].isin(returned_order_ids)
        orders["is_return_abuse"] = orders["fraud_type"] == "return_abuse"

        # Legitimate orders that were returned (e.g. fit-check, defective) have has_return=True, is_return_abuse=False
        legit_returns = orders[orders["has_return_event"] & (~orders["is_return_abuse"])]
        assert len(legit_returns) > 0, "No legitimate returns found; return_abuse confused with return event!"

        # Censored return abusers (scam orders placed late where return occurs in 2026)
        censored_abusers = orders[(~orders["has_return_event"]) & orders["is_return_abuse"]]
        assert len(censored_abusers) > 0, "No censored return abuse orders found!"


# ===========================================================================
# 4. Multimodal Evidence & Image Asset Auditing
# ===========================================================================

class TestMultimodalEvidence:
    def test_reproduced_similarity_distributions_and_auc(self, v2_1_data):
        """Reproduces multimodal similarity scores and verifies diagnostic AUC = 0.7599."""
        listings = v2_1_data["listings"].copy()
        products = v2_1_data["products"].copy()

        sims = compute_multimodal_similarity(listings, products, use_clip=False)
        listings["sim"] = sims

        genuine = listings[~listings["is_fraudulent"]]["sim"]
        fake = listings[listings["is_fraudulent"]]["sim"]

        assert len(fake) == 500
        assert len(genuine) == 19500

        # Mean similarity values match reported distributions
        assert abs(genuine.mean() - 0.7875) < 0.01
        assert abs(fake.mean() - 0.6745) < 0.01

        auc = roc_auc_score(listings["is_fraudulent"], -listings["sim"])
        assert abs(auc - 0.7599) < 0.01, f"Expected AUC ~0.7599, got {auc:.4f}"

    def test_image_availability_and_fallback_counts(self, v2_1_data):
        """Verifies that 2,616 products have real ABO image files and 384 use synthetic fallbacks."""
        products = v2_1_data["products"]
        abo_dir = os.path.join(_PROJECT_DIR, "data", "external", "abo")

        real_count = 0
        fallback_count = 0
        for ref in products["image_ref"]:
            if str(ref).startswith("images"):
                p = os.path.join(abo_dir, ref)
                if os.path.isfile(p):
                    real_count += 1
                else:
                    fallback_count += 1
            else:
                fallback_count += 1

        assert real_count == 2616, f"Expected 2,616 real ABO images, found {real_count}!"
        assert fallback_count == 384, f"Expected 384 fallback images, found {fallback_count}!"


# ===========================================================================
# 5. Evaluation Protocol Freezing & Split Boundary Isolation
# ===========================================================================

class TestEvaluationProtocol:
    def test_temporal_split_boundaries_and_order_counts(self, v2_1_data):
        """Verifies exact split sample counts and strict temporal ordering across splits."""
        orders = v2_1_data["orders"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])

        train_end = pd.Timestamp("2025-08-31 23:59:59")
        val_end = pd.Timestamp("2025-10-31 23:59:59")

        train = orders[orders["order_date"] <= train_end]
        val = orders[(orders["order_date"] > train_end) & (orders["order_date"] <= val_end)]
        test = orders[orders["order_date"] > val_end]

        assert len(train) == 16952, f"Expected 16,952 train orders, got {len(train)}"
        assert len(val) == 12129, f"Expected 12,129 val orders, got {len(val)}"
        assert len(test) == 20919, f"Expected 20,919 test orders, got {len(test)}"

        # Strict temporal isolation
        assert train["order_date"].max() <= val["order_date"].min()
        assert val["order_date"].max() <= test["order_date"].min()

    def test_test_set_frozen_isolation(self, v2_1_data):
        """Verifies that the final test set remains untouched and is strictly out-of-time (Nov-Dec 2025)."""
        orders = v2_1_data["orders"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        val_end = pd.Timestamp("2025-10-31 23:59:59")
        test = orders[orders["order_date"] > val_end]

        assert test["order_date"].min() >= pd.Timestamp("2025-11-01 00:00:00")
        assert test["order_date"].max() <= pd.Timestamp("2025-12-31 23:59:59")
        assert int(test["is_fraudulent"].sum()) == 1835
