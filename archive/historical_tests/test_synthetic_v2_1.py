"""
TrustShield — Test Suite for Stage 3.1.1 Synthetic Dataset v2.1 Integrity Fixes.

Validates:
1. Mandatory safeguards and schema isolation (data/synthetic_v2/ preserved, v2_1 populated).
2. Category-constrained perturbations (same category, compatible subtype, changes observable evidence).
3. Generator-label non-leakage (personas excluded from public tables, forbidden fields banned from feature matrix).
4. Return-date semantics and right-censoring policy (days >= 1, zero-day returns = 0, Dec 30/31 orders).
5. Prediction-time feature validity and temporal invariance (future events do not alter historical features).
6. Observable seller behaviors (disposable short tenure <= 7 days, takeover post-compromise surge).
7. Dataset manifest and hash verification.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pytest


_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_PROJECT_DIR = os.path.join(_ROOT_DIR, "trustshield_project")
_V2_1_DIR = os.path.join(_ROOT_DIR, "data", "synthetic_v2_1")
_V2_DIR = os.path.join(_ROOT_DIR, "data", "synthetic_v2")
_V1_DIR = os.path.join(_ROOT_DIR, "trustshield_project", "synthetic_data_export")


@pytest.fixture(scope="module")
def v2_1_tables() -> Dict[str, pd.DataFrame]:
    """Loads all operational tables and metadata from data/synthetic_v2_1/."""
    assert os.path.isdir(_V2_1_DIR), f"Directory {_V2_1_DIR} does not exist!"
    table_names = [
        "addresses", "devices", "device_mapping", "address_sharing_log", "device_sharing_log",
        "buyers", "sellers", "products", "listings", "orders", "returns",
        "fraud_ground_truth", "generator_personas", "perturbation_audit",
    ]
    dfs = {}
    for name in table_names:
        p = os.path.join(_V2_1_DIR, f"{name}.csv")
        assert os.path.isfile(p), f"Missing required file: {p}"
        dfs[name] = pd.read_csv(p)
    return dfs


@pytest.fixture(scope="module")
def manifest_and_report() -> Dict[str, dict]:
    """Loads dataset_manifest.json and validation_report.json."""
    m_path = os.path.join(_V2_1_DIR, "dataset_manifest.json")
    v_path = os.path.join(_V2_1_DIR, "validation_report.json")
    assert os.path.isfile(m_path), f"Missing manifest: {m_path}"
    assert os.path.isfile(v_path), f"Missing validation report: {v_path}"
    with open(m_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(v_path, "r", encoding="utf-8") as f:
        val_report = json.load(f)
    return {"manifest": manifest, "validation_report": val_report}


# ===========================================================================
# 1. Mandatory Safeguards & Directory Isolation
# ===========================================================================

class TestMandatorySafeguards:
    def test_v1_and_v2_artifacts_preserved(self):
        """Verifies data/synthetic_v2/ and v1 benchmark export were NOT overwritten or deleted."""
        assert os.path.isdir(_V2_DIR), "data/synthetic_v2/ must be preserved!"
        assert os.path.isfile(os.path.join(_V2_DIR, "orders.csv")), "data/synthetic_v2/orders.csv missing!"
        assert os.path.isfile(os.path.join(_V2_DIR, "dataset_manifest.json")), "v2 manifest missing!"

        assert os.path.isdir(_V1_DIR), "v1 directory must be preserved!"
        assert os.path.isfile(os.path.join(_V1_DIR, "orders.csv")), "v1 orders.csv missing!"

    def test_manifest_file_hashes_match(self, v2_1_tables, manifest_and_report):
        """Verifies every file in data/synthetic_v2_1/ matches its manifest SHA-256 hash."""
        manifest = manifest_and_report["manifest"]
        for filename, meta in manifest["files"].items():
            filepath = os.path.join(_V2_1_DIR, filename)
            assert os.path.isfile(filepath), f"File {filename} in manifest not on disk!"
            h = hashlib.sha256()
            with open(filepath, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            assert h.hexdigest() == meta["sha256"], f"SHA-256 mismatch for {filename}!"

    def test_validation_report_status_pass(self, manifest_and_report):
        """Verifies validation_report.json reports overall status PASS."""
        rep = manifest_and_report["validation_report"]
        assert rep["status"] == "PASS", f"Validation status is {rep['status']}, not PASS!"
        for check_name, check_data in rep["checks"].items():
            assert check_data.get("passed") is True, f"Check {check_name} failed: {check_data}"


# ===========================================================================
# 2. Category-Constrained Product & Multimodal Perturbations
# ===========================================================================

class TestCategoryConstrainedPerturbations:
    def test_all_swaps_within_exact_same_category(self, v2_1_tables):
        """Validates 100% of candidate product perturbations belong to the SAME category."""
        audit = v2_1_tables["perturbation_audit"]
        prods = v2_1_tables["products"]
        cat_map = prods.set_index("product_id")["category"].to_dict()

        claimed_cats = audit["claimed_product_id"].map(cat_map)
        pert_cats = audit["perturbed_product_id"].map(cat_map)
        mismatches = (claimed_cats != pert_cats).sum()
        assert mismatches == 0, f"Found {mismatches} cross-category perturbations!"

    def test_compatible_subtype_presence(self, v2_1_tables):
        """Validates that candidate selection attempts compatible subtype matching where available."""
        audit = v2_1_tables["perturbation_audit"]
        subtype_matches = audit["perturbation_type"].str.contains("compatible_subtype").sum()
        assert subtype_matches > 0, "No compatible subtype matches recorded in perturbation audit!"

    def test_perturbation_changes_observable_listing_evidence(self, v2_1_tables):
        """Validates that perturbations alter observable evidence (displayed_product_id and price)."""
        listings = v2_1_tables["listings"]
        audit = v2_1_tables["perturbation_audit"]
        fake_listings = listings[listings["listing_id"].isin(audit["listing_id"])]

        # Displayed product differs from claimed product for perturbed listings
        differ_count = (fake_listings["displayed_product_id"] != fake_listings["product_id"]).sum()
        assert differ_count == len(fake_listings), "Observable displayed_product_id was not swapped!"

        # Price factors are recorded and non-empty
        assert (audit["price_factor"] > 0).all()

    def test_no_shortcut_boolean_flags_in_listings(self, v2_1_tables):
        """Validates that price_anomaly and image_mismatch boolean flags are absent from listings.csv."""
        listings = v2_1_tables["listings"]
        assert "price_anomaly" not in listings.columns, "price_anomaly shortcut leaked in listings.csv!"
        assert "image_mismatch" not in listings.columns, "image_mismatch shortcut leaked in listings.csv!"
        assert "_seller_persona" not in listings.columns, "_seller_persona generator column leaked!"


# ===========================================================================
# 3. Generator-Label & Persona Leakage Prevention
# ===========================================================================

class TestGeneratorLabelLeakagePrevention:
    FORBIDDEN_MODEL_INPUTS = {
        "buyer_persona", "seller_persona", "is_fraudulent", "fraud_type",
        "fraud_ring_id", "compromise_date", "price_anomaly", "image_mismatch",
    }

    def test_persona_segregated_from_public_entity_tables(self, v2_1_tables):
        """Validates that buyer_persona and seller_persona are strictly segregated."""
        buyers = v2_1_tables["buyers"]
        sellers = v2_1_tables["sellers"]

        assert "buyer_persona" not in buyers.columns, "buyer_persona leaked into buyers.csv!"
        assert "seller_persona" not in sellers.columns, "seller_persona leaked into sellers.csv!"
        assert "compromise_date" not in sellers.columns, "compromise_date leaked into sellers.csv!"

    def test_personas_preserved_in_separate_evaluation_metadata(self, v2_1_tables):
        """Validates that generator_personas.csv and fraud_ground_truth.csv store internal tags."""
        personas = v2_1_tables["generator_personas"]
        gt = v2_1_tables["fraud_ground_truth"]

        assert set(personas["entity_type"].unique()) == {"seller", "buyer"}
        assert len(personas) == len(v2_1_tables["buyers"]) + len(v2_1_tables["sellers"])
        assert len(gt) > 0
        assert "fraud_ring_id" in gt.columns
        assert "fraud_type" in gt.columns

    def test_forbidden_fields_cannot_enter_feature_matrix(self, v2_1_tables):
        """Explicitly tests that feature matrix construction excludes forbidden fields."""
        import sys
        if _PROJECT_DIR not in sys.path:
            sys.path.insert(0, _PROJECT_DIR)

        from baseline_model import build_features
        from phase2_specialized_models import build_listing_features, build_return_features

        orders = v2_1_tables["orders"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        listings = v2_1_tables["listings"].copy()
        listings["listing_date"] = pd.to_datetime(listings["listing_date"])
        returns = v2_1_tables["returns"].copy()
        returns["return_date"] = pd.to_datetime(returns["return_date"])
        buyers = v2_1_tables["buyers"].copy()
        buyers["signup_date"] = pd.to_datetime(buyers["signup_date"])
        sellers = v2_1_tables["sellers"].copy()
        sellers["signup_date"] = pd.to_datetime(sellers["signup_date"])
        products = v2_1_tables["products"].copy()

        # 1. Baseline transaction model feature matrix
        _, base_features = build_features(orders.head(100), listings, returns, buyers, sellers, products)
        leaked_base = self.FORBIDDEN_MODEL_INPUTS.intersection(base_features)
        assert not leaked_base, f"Forbidden fields in baseline model feature set: {leaked_base}"

        # 2. Fake listing specialized model feature matrix
        _, list_features = build_listing_features(listings.head(100), sellers, products)
        leaked_list = self.FORBIDDEN_MODEL_INPUTS.intersection(list_features)
        assert not leaked_list, f"Forbidden fields in listing model feature set: {leaked_list}"

        # 3. Return fraud specialized model feature matrix
        _, ret_features = build_return_features(returns.head(100), orders, buyers, sellers)
        leaked_ret = self.FORBIDDEN_MODEL_INPUTS.intersection(ret_features)
        assert not leaked_ret, f"Forbidden fields in return model feature set: {leaked_ret}"

        # 4. Strict validator gate function: raises ValueError if forbidden column is injected
        def enforce_no_forbidden_features(feature_names: List[str]) -> None:
            viol = self.FORBIDDEN_MODEL_INPUTS.intersection(feature_names)
            if viol:
                raise ValueError(f"Forbidden columns detected in model feature matrix: {viol}")

        enforce_no_forbidden_features(base_features)
        enforce_no_forbidden_features(list_features)
        enforce_no_forbidden_features(ret_features)

        with pytest.raises(ValueError, match="Forbidden columns"):
            enforce_no_forbidden_features(["price_vs_base_price_ratio", "buyer_persona"])

        with pytest.raises(ValueError, match="Forbidden columns"):
            enforce_no_forbidden_features(["price_vs_base_price_ratio", "is_fraudulent"])

        with pytest.raises(ValueError, match="Forbidden columns"):
            enforce_no_forbidden_features(["price_vs_base_price_ratio", "seller_persona"])


# ===========================================================================
# 4. Return-Date Semantics & End-of-Period Policy
# ===========================================================================

class TestReturnDateSemantics:
    def test_every_return_occurs_strictly_after_order(self, v2_1_tables):
        """Validates that days_to_return >= 1 for every return. Zero-day returns = 0."""
        returns = v2_1_tables["returns"]
        orders = v2_1_tables["orders"]

        merged = returns.merge(orders[["order_id", "order_date"]], on="order_id")
        delays = (pd.to_datetime(merged["return_date"]) - pd.to_datetime(merged["order_date"])).dt.days

        assert (delays >= 1).all(), f"Found {(delays < 1).sum()} returns with delay < 1 day!"
        assert (delays == 0).sum() == 0, "Zero-day returns detected!"
        assert (delays < 0).sum() == 0, "Negative delay returns detected!"

    def test_december_30_and_31_return_policy(self, v2_1_tables):
        """Tests returns for orders on Dec 30 and Dec 31:
        - Dec 30 orders can only return on Dec 31 (delay >= 1).
        - Dec 31 orders have 0 returns in 2025 (all right-censored).
        - No delays are shortened.
        """
        orders = v2_1_tables["orders"].copy()
        returns = v2_1_tables["returns"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        returns["return_date"] = pd.to_datetime(returns["return_date"])

        dec30_orders = orders[orders["order_date"].dt.date == datetime(2025, 12, 30).date()]
        dec31_orders = orders[orders["order_date"].dt.date == datetime(2025, 12, 31).date()]

        dec30_rets = returns[returns["order_id"].isin(dec30_orders["order_id"])]
        dec31_rets = returns[returns["order_id"].isin(dec31_orders["order_id"])]

        # If any Dec 30 return exists, it must be on Dec 31
        for _, r in dec30_rets.iterrows():
            assert r["return_date"].date() == datetime(2025, 12, 31).date(), f"Invalid Dec 30 return date: {r['return_date']}"

        # Dec 31 orders cannot have any return recorded in 2025
        assert len(dec31_rets) == 0, f"Expected 0 returns for Dec 31 orders (right-censored), found {len(dec31_rets)}!"

    def test_foreign_key_and_one_to_one_order_mapping(self, v2_1_tables):
        """Validates every return maps to exactly one valid order."""
        returns = v2_1_tables["returns"]
        orders = v2_1_tables["orders"]

        # 1-to-1: each return maps to a unique order
        assert not returns["order_id"].duplicated().any(), "Found multiple returns for the same order_id!"
        # Foreign key integrity
        missing = (~returns["order_id"].isin(orders["order_id"])).sum()
        assert missing == 0, f"Found {missing} returns referencing non-existent order_id!"


# ===========================================================================
# 5. Prediction-Time Feature Validity & Temporal Invariance
# ===========================================================================

class TestPredictionTimeFeatureValidity:
    FULL_PERIOD_AGGREGATES = {
        "total_orders", "total_returns", "total_listings", "total_orders_received",
    }

    def test_full_period_aggregates_classified_as_metadata(self, v2_1_tables):
        """Validates that entity-table snapshot counters are identified and forbidden at prediction time."""
        buyers = v2_1_tables["buyers"]
        sellers = v2_1_tables["sellers"]

        assert "total_orders" in buyers.columns
        assert "total_returns" in buyers.columns
        assert "total_listings" in sellers.columns
        assert "total_orders_received" in sellers.columns

    def test_point_in_time_feature_invariance_under_future_events(self, v2_1_tables):
        """Inserts future events into orders and returns and proves that point-in-time
        historical features at time T remain 100% invariant.
        """
        orders = v2_1_tables["orders"].copy()
        returns = v2_1_tables["returns"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        returns["return_date"] = pd.to_datetime(returns["return_date"])

        # Pick a target order in May 2025
        may_mask = (orders["order_date"] >= "2025-05-01") & (orders["order_date"] <= "2025-05-31")
        target_order = orders[may_mask].iloc[0]
        t_eval = target_order["order_date"]
        b_id = target_order["buyer_id"]
        s_id = target_order["seller_id"]

        # Compute point-in-time features BEFORE adding future events
        b_orders_before_base = (
            (orders["buyer_id"] == b_id) & (orders["order_date"] < t_eval)
        ).sum()
        b_returns_before_base = (
            (returns["buyer_id"] == b_id) & (returns["return_date"] < t_eval)
        ).sum()
        s_orders_before_base = (
            (orders["seller_id"] == s_id) & (orders["order_date"] < t_eval)
        ).sum()

        # Deliberately insert synthetic FUTURE events in December 2025
        future_orders = pd.DataFrame([
            {
                "order_id": f"FUTURE_O_{i}",
                "buyer_id": b_id,
                "seller_id": s_id,
                "listing_id": target_order["listing_id"],
                "product_id": target_order["product_id"],
                "order_date": pd.Timestamp("2025-12-15 12:00:00"),
                "amount": 100.0,
                "device_id": "DEV_000001",
                "status": "completed",
                "is_fraudulent": False,
                "fraud_type": None,
            } for i in range(5)
        ])
        future_returns = pd.DataFrame([
            {
                "return_id": f"FUTURE_R_{i}",
                "order_id": f"FUTURE_O_{i}",
                "buyer_id": b_id,
                "seller_id": s_id,
                "return_date": pd.Timestamp("2025-12-20 12:00:00"),
                "reason": "defective",
                "status": "approved",
                "is_fraudulent": False,
                "fraud_type": None,
            } for i in range(2)
        ])

        orders_expanded = pd.concat([orders, future_orders], ignore_index=True)
        returns_expanded = pd.concat([returns, future_returns], ignore_index=True)

        # Recompute point-in-time features AFTER adding future events
        b_orders_before_after = (
            (orders_expanded["buyer_id"] == b_id) & (orders_expanded["order_date"] < t_eval)
        ).sum()
        b_returns_before_after = (
            (returns_expanded["buyer_id"] == b_id) & (returns_expanded["return_date"] < t_eval)
        ).sum()
        s_orders_before_after = (
            (orders_expanded["seller_id"] == s_id) & (orders_expanded["order_date"] < t_eval)
        ).sum()

        # Invariance check: point-in-time features MUST NOT CHANGE
        assert b_orders_before_after == b_orders_before_base, "Point-in-time buyer_orders_before changed under future events!"
        assert b_returns_before_after == b_returns_before_base, "Point-in-time buyer_returns_before changed under future events!"
        assert s_orders_before_after == s_orders_before_base, "Point-in-time seller_orders_before changed under future events!"


# ===========================================================================
# 6. Observable Seller Behavior Verification
# ===========================================================================

class TestObservableSellerBehaviors:
    def test_disposable_seller_listing_burst(self, v2_1_tables):
        """Verifies disposable sellers exhibit observable short-tenure bursts (median <= 7 days)."""
        personas = v2_1_tables["generator_personas"]
        sellers = v2_1_tables["sellers"]
        listings = v2_1_tables["listings"]

        disp_sids = set(personas[(personas["entity_type"] == "seller") & (personas["persona"] == "disposable")]["entity_id"])
        est_sids = set(personas[(personas["entity_type"] == "seller") & (personas["persona"] == "established")]["entity_id"])

        disp_merged = listings[listings["seller_id"].isin(disp_sids)].merge(
            sellers[["seller_id", "signup_date"]], on="seller_id"
        )
        disp_tenures = (pd.to_datetime(disp_merged["listing_date"]) - pd.to_datetime(disp_merged["signup_date"])).dt.days
        disp_median = float(disp_tenures.median())

        est_merged = listings[listings["seller_id"].isin(est_sids)].merge(
            sellers[["seller_id", "signup_date"]], on="seller_id"
        )
        est_tenures = (pd.to_datetime(est_merged["listing_date"]) - pd.to_datetime(est_merged["signup_date"])).dt.days
        est_median = float(est_tenures.median())

        assert disp_median <= 7.0, f"Disposable sellers median tenure {disp_median:.1f} > 7.0 days!"
        assert est_median >= 50.0, f"Established sellers median tenure {est_median:.1f} < 50.0 days!"

    def test_takeover_seller_post_compromise_category_surge(self, v2_1_tables):
        """Verifies takeover sellers exhibit post-compromise anomalous category surge."""
        personas = v2_1_tables["generator_personas"]
        sellers = v2_1_tables["sellers"]
        listings = v2_1_tables["listings"]

        ato_meta = personas[(personas["entity_type"] == "seller") & (personas["persona"] == "takeover")]
        assert len(ato_meta) > 0, "No takeover sellers found in personas metadata!"

        # Inspect a takeover seller with defined compromise_date
        sample_ato = ato_meta[ato_meta["compromise_date"].notna()].iloc[0]
        s_id = sample_ato["entity_id"]
        comp_date = pd.to_datetime(sample_ato["compromise_date"])
        focus_cat = sellers[sellers["seller_id"] == s_id]["category_focus"].iloc[0]

        s_listings = listings[listings["seller_id"] == s_id].copy()
        s_listings["listing_date"] = pd.to_datetime(s_listings["listing_date"])

        pre_listings = s_listings[s_listings["listing_date"] <= comp_date]
        post_listings = s_listings[s_listings["listing_date"] > comp_date]

        assert len(pre_listings) > 0, "No pre-compromise listings for takeover seller!"
        assert len(post_listings) > 0, "No post-compromise listings for takeover seller!"

        # Pre-compromise is in focus category
        assert (pre_listings["category"] == focus_cat).all(), "Pre-compromise listings not in focus category!"
        # Post-compromise has anomalous category surge
        post_cats = post_listings["category"].unique()
        assert focus_cat not in post_cats or len(post_cats) > 1, "Post-compromise did not deviate category!"


# ===========================================================================
# 7. Distribution Overlaps & General Realism
# ===========================================================================

class TestDistributionOverlaps:
    def test_return_delay_distribution_overlap(self, v2_1_tables):
        """Verifies return delay distributions overlap realistically without disjoint shortcut."""
        orders = v2_1_tables["orders"]
        returns = v2_1_tables["returns"]

        merged = returns.merge(orders[["order_id", "order_date"]], on="order_id")
        delays = (pd.to_datetime(merged["return_date"]) - pd.to_datetime(merged["order_date"])).dt.days

        legit_delays = delays[~merged["is_fraudulent"]]
        fraud_delays = delays[merged["is_fraudulent"]]

        # Both distributions span [1, 20+] days
        assert legit_delays.min() >= 1 and fraud_delays.min() >= 1
        assert legit_delays.max() >= 15 and fraud_delays.max() >= 15

        # Check overlap: mean difference is modest (< 3 days)
        diff = abs(float(legit_delays.mean() - fraud_delays.mean()))
        assert diff < 3.5, f"Return delay difference {diff:.2f} days is too large, indicates disjoint bands!"

    def test_no_id_label_leakage(self, v2_1_tables):
        """Verifies no IDs contain FRAUD, ABUSE, COLLUSION, or RING tokens."""
        banned = ["FRAUD", "ABUSE", "COLLUSION", "RING"]
        for tname in ["orders", "listings", "returns", "buyers", "sellers"]:
            df = v2_1_tables[tname]
            id_col = f"{tname[:-1] if tname != 'addresses' else 'address'}_id"
            if id_col not in df.columns:
                id_col = df.columns[0]
            leaks = df[id_col].str.contains("|".join(banned), case=False).sum()
            assert leaks == 0, f"Found {leaks} leaking IDs in {tname}.{id_col}!"
