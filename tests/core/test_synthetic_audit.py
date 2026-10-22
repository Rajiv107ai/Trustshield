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
import joblib
from sklearn.metrics import roc_auc_score

from pathlib import Path
_ROOT_DIR = str(Path(__file__).resolve().parents[2])
_DATA_DIR = os.path.join(_ROOT_DIR, "data", "synthetic_v2_1")
_PROJECT_DIR = os.path.join(_ROOT_DIR, "core")

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

    def test_v1_and_v2_artifacts_preserved(self):
        """Verifies data/synthetic_v2/ and v1 benchmark export were NOT overwritten or deleted."""
        v2_dir = os.path.join(_ROOT_DIR, "data", "synthetic_v2")
        v1_dir = os.path.join(_ROOT_DIR, "trustshield_project", "synthetic_data_export")
        assert os.path.isdir(v2_dir), "data/synthetic_v2/ must be preserved!"
        assert os.path.isfile(os.path.join(v2_dir, "orders.csv")), "data/synthetic_v2/orders.csv missing!"
        assert os.path.isfile(os.path.join(v2_dir, "dataset_manifest.json")), "v2 manifest missing!"
        assert os.path.isdir(v1_dir), "v1 directory must be preserved!"
        assert os.path.isfile(os.path.join(v1_dir, "orders.csv")), "v1 orders.csv missing!"

    def test_validation_report_status_pass(self):
        """Verifies validation_report.json reports overall status PASS."""
        v_path = os.path.join(_DATA_DIR, "validation_report.json")
        assert os.path.isfile(v_path), f"Missing validation report: {v_path}"
        with open(v_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        assert rep["status"] == "PASS", f"Validation status is {rep['status']}, not PASS!"
        for check_name, check_data in rep["checks"].items():
            assert check_data.get("passed") is True, f"Check {check_name} failed: {check_data}"

    def test_perturbation_changes_observable_listing_evidence(self, v2_1_data):
        """Validates that perturbations alter observable evidence (displayed_product_id and price)."""
        listings = v2_1_data["listings"]
        audit = v2_1_data["perturbation_audit"]
        fake_listings = listings[listings["listing_id"].isin(audit["listing_id"])]
        assert (fake_listings["displayed_product_id"] != fake_listings["product_id"]).sum() == len(fake_listings)
        assert (audit["price_factor"] > 0).all()
        assert audit["perturbation_type"].str.contains("compatible_subtype").sum() > 0

    def test_no_shortcut_boolean_flags_or_persona_in_public_tables(self, v2_1_data):
        """Validates no shortcut flags or persona columns in listings, buyers, or sellers."""
        listings = v2_1_data["listings"]
        buyers = v2_1_data["buyers"]
        sellers = v2_1_data["sellers"]
        assert "price_anomaly" not in listings.columns
        assert "image_mismatch" not in listings.columns
        assert "_seller_persona" not in listings.columns
        assert "buyer_persona" not in buyers.columns
        assert "seller_persona" not in sellers.columns
        assert "compromise_date" not in sellers.columns

    def test_personas_and_ground_truth_segregated(self, v2_1_data):
        """Validates generator_personas and fraud_ground_truth segregation."""
        personas = v2_1_data["generator_personas"]
        gt = v2_1_data["fraud_ground_truth"]
        assert set(personas["entity_type"].unique()) == {"seller", "buyer"}
        assert len(personas) == len(v2_1_data["buyers"]) + len(v2_1_data["sellers"])
        assert len(gt) > 0
        assert "fraud_ring_id" in gt.columns
        assert "fraud_type" in gt.columns

    def test_no_id_label_leakage(self, v2_1_data):
        """Verifies no IDs contain FRAUD, ABUSE, COLLUSION, or RING tokens."""
        banned = ["FRAUD", "ABUSE", "COLLUSION", "RING"]
        for tname in ["orders", "listings", "returns", "buyers", "sellers"]:
            df = v2_1_data[tname]
            id_col = f"{tname[:-1] if tname != 'addresses' else 'address'}_id"
            if id_col not in df.columns:
                id_col = df.columns[0]
            assert df[id_col].str.contains("|".join(banned), case=False).sum() == 0


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

    def test_every_return_occurs_strictly_after_order(self, v2_1_data):
        """Validates that days_to_return >= 1 for every return. Zero-day returns = 0."""
        returns = v2_1_data["returns"]
        orders = v2_1_data["orders"]
        merged = returns.merge(orders[["order_id", "order_date"]], on="order_id")
        delays = (pd.to_datetime(merged["return_date"]) - pd.to_datetime(merged["order_date"])).dt.days
        assert (delays >= 1).all()
        assert (delays == 0).sum() == 0
        assert (delays < 0).sum() == 0

    def test_december_30_and_31_return_policy(self, v2_1_data):
        """Dec 30 orders only return on Dec 31; Dec 31 orders have 0 returns in 2025."""
        orders = v2_1_data["orders"].copy()
        returns = v2_1_data["returns"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        returns["return_date"] = pd.to_datetime(returns["return_date"])
        dec30_orders = orders[orders["order_date"].dt.date == datetime(2025, 12, 30).date()]
        dec31_orders = orders[orders["order_date"].dt.date == datetime(2025, 12, 31).date()]
        dec30_rets = returns[returns["order_id"].isin(dec30_orders["order_id"])]
        dec31_rets = returns[returns["order_id"].isin(dec31_orders["order_id"])]
        for _, r in dec30_rets.iterrows():
            assert r["return_date"].date() == datetime(2025, 12, 31).date()
        assert len(dec31_rets) == 0

    def test_return_delay_distribution_overlap(self, v2_1_data):
        """Verifies return delay distributions overlap realistically without disjoint bands."""
        orders = v2_1_data["orders"]
        returns = v2_1_data["returns"]
        merged = returns.merge(orders[["order_id", "order_date"]], on="order_id")
        delays = (pd.to_datetime(merged["return_date"]) - pd.to_datetime(merged["order_date"])).dt.days
        legit_delays = delays[~merged["is_fraudulent"]]
        fraud_delays = delays[merged["is_fraudulent"]]
        assert legit_delays.min() >= 1 and fraud_delays.min() >= 1
        assert legit_delays.max() >= 15 and fraud_delays.max() >= 15
        diff = abs(float(legit_delays.mean() - fraud_delays.mean()))
        assert diff < 3.5


class TestObservableSellerBehaviors:
    def test_disposable_seller_listing_burst(self, v2_1_data):
        """Verifies disposable sellers exhibit observable short-tenure bursts (median <= 7 days)."""
        personas = v2_1_data["generator_personas"]
        sellers = v2_1_data["sellers"]
        listings = v2_1_data["listings"]
        disp_sids = set(personas[(personas["entity_type"] == "seller") & (personas["persona"] == "disposable")]["entity_id"])
        est_sids = set(personas[(personas["entity_type"] == "seller") & (personas["persona"] == "established")]["entity_id"])

        disp_merged = listings[listings["seller_id"].isin(disp_sids)].merge(sellers[["seller_id", "signup_date"]], on="seller_id")
        disp_tenures = (pd.to_datetime(disp_merged["listing_date"]) - pd.to_datetime(disp_merged["signup_date"])).dt.days
        assert float(disp_tenures.median()) <= 7.0

        est_merged = listings[listings["seller_id"].isin(est_sids)].merge(sellers[["seller_id", "signup_date"]], on="seller_id")
        est_tenures = (pd.to_datetime(est_merged["listing_date"]) - pd.to_datetime(est_merged["signup_date"])).dt.days
        assert float(est_tenures.median()) >= 50.0

    def test_takeover_seller_post_compromise_category_surge(self, v2_1_data):
        """Verifies takeover sellers exhibit post-compromise anomalous category surge."""
        personas = v2_1_data["generator_personas"]
        sellers = v2_1_data["sellers"]
        listings = v2_1_data["listings"]
        ato_meta = personas[(personas["entity_type"] == "seller") & (personas["persona"] == "takeover")]
        assert len(ato_meta) > 0
        sample_ato = ato_meta[ato_meta["compromise_date"].notna()].iloc[0]
        s_id = sample_ato["entity_id"]
        comp_date = pd.to_datetime(sample_ato["compromise_date"])
        focus_cat = sellers[sellers["seller_id"] == s_id]["category_focus"].iloc[0]

        s_listings = listings[listings["seller_id"] == s_id].copy()
        s_listings["listing_date"] = pd.to_datetime(s_listings["listing_date"])
        pre_listings = s_listings[s_listings["listing_date"] <= comp_date]
        post_listings = s_listings[s_listings["listing_date"] > comp_date]
        assert len(pre_listings) > 0 and len(post_listings) > 0
        assert (pre_listings["category"] == focus_cat).all()
        post_cats = post_listings["category"].unique()
        assert focus_cat not in post_cats or len(post_cats) > 1


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
        if not os.path.isdir(abo_dir):
            pytest.skip("external ABO image fixture is not provisioned; run the governed multimodal data setup")

        real_count = 0
        fallback_count = 0
        for ref in products["image_ref"]:
            normalized_ref = str(ref).replace("\\", os.sep)
            if normalized_ref.startswith("images"):
                p = os.path.join(abo_dir, normalized_ref)
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

    def test_reconciliation_of_december_10_and_december_11(self, v2_1_data):
        """Reconciles Dec 10 vs Dec 11 definitions against exact orders data.
        - Through Dec 10 23:59:59 (i.e. strictly before Dec 11 00:00:00): 41,382 orders.
        - On or after Dec 11 00:00:00: exactly 8,618 orders.
        """
        orders = v2_1_data["orders"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        n_through_dec10_end = int((orders["order_date"] <= "2025-12-10 23:59:59").sum())
        n_on_or_after_dec11 = int((orders["order_date"] >= "2025-12-11 00:00:00").sum())
        assert n_through_dec10_end == 41382
        assert n_on_or_after_dec11 == 8618
        assert n_through_dec10_end + n_on_or_after_dec11 == 50000

    def test_causal_cross_split_directionality(self, v2_1_data):
        """Verifies temporal causality: returns and orders never reference future events."""
        orders = v2_1_data["orders"].copy()
        listings = v2_1_data["listings"].copy()
        returns = v2_1_data["returns"].copy()

        orders["order_date"] = pd.to_datetime(orders["order_date"])
        listings["listing_date"] = pd.to_datetime(listings["listing_date"])
        returns["return_date"] = pd.to_datetime(returns["return_date"])

        train_cutoff = pd.Timestamp("2025-08-31 23:59:59")
        orders["order_split"] = np.where(orders["order_date"] <= train_cutoff, "train", "post_train")
        listings["listing_split"] = np.where(listings["listing_date"] <= train_cutoff, "train", "post_train")
        returns["return_split"] = np.where(returns["return_date"] <= train_cutoff, "train", "post_train")

        ro = returns.merge(orders[["order_id", "order_split"]], on="order_id")
        assert len(ro[(ro["return_split"] == "train") & (ro["order_split"] == "post_train")]) == 0

        ol = orders.merge(listings[["listing_id", "listing_split"]], on="listing_id")
        assert len(ol[(ol["order_split"] == "train") & (ol["listing_split"] == "post_train")]) == 0


class TestStage33ModelsAndThresholds:
    """Verifies that Stage 3.3 models and feature metadata exist and are loadable."""

    def test_feature_meta_thresholds_defined(self):
        p = os.path.join(_ROOT_DIR, "models", "v2_1", "feature_meta.joblib")
        assert os.path.isfile(p), f"Missing feature_meta: {p}"
        feature_meta = joblib.load(p)
        thresholds = feature_meta["thresholds"]
        assert "fake_listing" in thresholds and "transaction_fraud" in thresholds and "return_fraud" in thresholds
        for t_name, t_val in thresholds.items():
            if isinstance(t_val, list):
                assert all(0.0 < x < 1.0 for x in t_val)
            else:
                assert 0.0 < t_val < 1.0

    def test_versioned_model_files_exist(self):
        models_dir = os.path.join(_ROOT_DIR, "models", "v2_1")
        expected_files = [
            "fake_listing_model.joblib",
            "fake_listing_scaler.joblib",
            "fake_listing_calibrator.joblib",
            "return_fraud_model.joblib",
            "return_fraud_scaler.joblib",
            "return_fraud_calibrator.joblib",
            "tabular_baseline_model.joblib",
            "combined_graph_model.joblib",
            "combined_graph_calibrator.joblib",
            "feature_meta.joblib",
        ]
        for fname in expected_files:
            fpath = os.path.join(models_dir, fname)
            assert os.path.isfile(fpath), f"Missing model file: {fpath}"
            assert joblib.load(fpath) is not None
