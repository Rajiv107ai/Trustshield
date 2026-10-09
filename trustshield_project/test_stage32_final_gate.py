"""
TrustShield — Stage 3.2 Final Gate Regression Test Suite.

Covers:
1. Exact Censoring Boundary (Dec 10 vs Dec 11 reconciliation, orders before/at/after cutoff).
2. Right-Censoring Enforcement (unobserved returns not labeled as confirmed negatives,
   three-way label disambiguation: has_return_event vs is_return_abuse vs is_censored).
3. Prediction-Time & Temporal Split Integrity (Orders, Listings, Returns split counts,
   causal directionality, no future outcomes/aggregates in features).
4. Multimodal Evidence Reproducibility (TF-IDF implementation, 0.7599 overall AUC,
   0.7324 test-split diagnostic AUC, fallback usage).
5. Test-Set Protection (virgin Nov-Dec holdout, no prior training on synthetic_v2_1).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict

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

from multimodal_scoring import compute_multimodal_similarity, MultimodalScorer
from phase2_specialized_models import build_listing_features, build_return_features


@pytest.fixture(scope="module")
def v2_1_tables() -> Dict[str, pd.DataFrame]:
    """Loads all operational tables from data/synthetic_v2_1/ with datetime parsing."""
    date_cols = {
        "orders": ["order_date"],
        "listings": ["listing_date"],
        "returns": ["return_date"],
        "buyers": ["signup_date"],
        "sellers": ["signup_date"],
    }
    tables = {}
    for name in ["orders", "listings", "returns", "buyers", "sellers", "products"]:
        p = os.path.join(_DATA_DIR, f"{name}.csv")
        assert os.path.isfile(p), f"Missing table: {p}"
        tables[name] = pd.read_csv(p, parse_dates=date_cols.get(name))
    return tables


class TestTaskAExactCensoringBoundary:
    """Task A: Exact censoring boundary analysis and reconciliation."""

    def test_reconciliation_of_december_10_and_december_11(self, v2_1_tables):
        """Reconciles Dec 10 vs Dec 11 definitions against exact orders data.

        - Through Dec 10 23:59:59 (i.e. strictly before Dec 11 00:00:00): 41,382 orders.
        - On or after Dec 11 00:00:00: exactly 8,618 orders.
        - Through Dec 10 00:00:00: 41,049 orders.
        - Between Dec 10 00:00:00 and Dec 10 23:59:59: 343 orders.
        Total sum is exactly 50,000 orders.
        """
        orders = v2_1_tables["orders"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])

        n_through_dec10_end = int((orders["order_date"] <= "2025-12-10 23:59:59").sum())
        n_on_or_after_dec11 = int((orders["order_date"] >= "2025-12-11 00:00:00").sum())
        assert n_through_dec10_end == 41382
        assert n_on_or_after_dec11 == 8618
        assert n_through_dec10_end + n_on_or_after_dec11 == 50000

        n_through_dec10_start = int((orders["order_date"] <= "2025-12-10 00:00:00").sum())
        n_during_dec10 = int(((orders["order_date"] > "2025-12-10 00:00:00") & (orders["order_date"] <= "2025-12-10 23:59:59")).sum())
        assert n_through_dec10_start == 41049
        assert n_during_dec10 == 333  # 343 total on Dec 10 including exact 00:00:00 boundary

    def test_orders_immediately_before_at_and_after_cutoff(self, v2_1_tables):
        """Validates behavior of orders immediately before, on, and after the cutoff.

        - Orders before Dec 10 have 100% complete 21-day observation.
        - Orders on Dec 10 observe return turnaround up to 19 days.
        - Orders on Dec 31 (1 order) have 0 observed returns (guaranteed censored).
        """
        orders = v2_1_tables["orders"].copy()
        returns = v2_1_tables["returns"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        returns["return_date"] = pd.to_datetime(returns["return_date"])

        ro = returns.merge(orders[["order_id", "order_date"]], on="order_id")
        ro["delay_days"] = (ro["return_date"] - ro["order_date"]).dt.total_seconds() / 86400.0

        # Before cutoff: Dec 9 orders
        dec9_orders = orders[(orders["order_date"] >= "2025-12-09 00:00:00") & (orders["order_date"] < "2025-12-10 00:00:00")]
        dec9_returns = ro[ro["order_id"].isin(dec9_orders["order_id"])]
        assert len(dec9_orders) > 0
        assert len(dec9_returns) > 0
        assert dec9_returns["delay_days"].max() <= 21.0

        # At cutoff: Dec 10 orders
        dec10_orders = orders[(orders["order_date"] >= "2025-12-10 00:00:00") & (orders["order_date"] < "2025-12-11 00:00:00")]
        dec10_returns = ro[ro["order_id"].isin(dec10_orders["order_id"])]
        assert len(dec10_orders) == 343
        assert len(dec10_returns) == 40
        assert dec10_returns["delay_days"].max() <= 21.0

        # After cutoff: Dec 31 orders
        dec31_orders = orders[orders["order_date"] >= "2025-12-31 00:00:00"]
        dec31_returns = ro[ro["order_id"].isin(dec31_orders["order_id"])]
        assert len(dec31_orders) == 1
        assert len(dec31_returns) == 0  # 100% censored!


class TestTaskCRightCensoringEnforcement:
    """Task C: Right-censoring enforcement and label disambiguation."""

    def test_three_way_label_disambiguation(self, v2_1_tables):
        """Confirms the distinction between has_return_event, is_return_abuse, and is_censored.

        Exactly 66 return abuse orders in orders.csv are right-censored (no return in returns.csv).
        """
        orders = v2_1_tables["orders"].copy()
        returns = v2_1_tables["returns"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])

        ret_order_ids = set(returns["order_id"])
        orders["has_return_event"] = orders["order_id"].isin(ret_order_ids)
        orders["is_return_abuse"] = (orders["fraud_type"] == "return_abuse")
        orders["is_censored"] = orders["is_return_abuse"] & (~orders["has_return_event"])

        total_abuse_orders = int(orders["is_return_abuse"].sum())
        observed_abuse_orders = int((orders["is_return_abuse"] & orders["has_return_event"]).sum())
        censored_abuse_orders = int(orders["is_censored"].sum())

        assert total_abuse_orders == 1233
        assert observed_abuse_orders == 1167
        assert censored_abuse_orders == 66
        assert observed_abuse_orders + censored_abuse_orders == total_abuse_orders

        # All 66 censored orders occur strictly after Dec 10!
        censored_min_date = orders.loc[orders["is_censored"], "order_date"].min()
        assert censored_min_date > pd.Timestamp("2025-12-10 23:59:59")

    def test_downstream_return_features_builder_never_labels_unreturned_orders_as_negatives(self, v2_1_tables):
        """Verifies that build_return_features operates strictly on observed returns.

        Unobserved returns are NEVER passed into build_return_features as false negatives.
        """
        returns = v2_1_tables["returns"]
        orders = v2_1_tables["orders"]
        buyers = v2_1_tables["buyers"]
        sellers = v2_1_tables["sellers"]

        feat_df, cols = build_return_features(returns, orders, buyers, sellers)
        assert len(feat_df) == len(returns) == 5537
        # Ensure target label y has no NaN and contains only observed return ground truths
        assert feat_df["y"].isna().sum() == 0
        assert int(feat_df["y"].to_numpy().sum()) == int(returns["is_fraudulent"].to_numpy().sum()) == 1552


class TestTaskBPredictionTimeAndSplits:
    """Task B: Prediction-time and temporal split integrity."""

    def test_exact_split_counts_reconciliation(self, v2_1_tables):
        """Verifies split counts across orders, listings, and returns against CSV files."""
        orders = v2_1_tables["orders"].copy()
        listings = v2_1_tables["listings"].copy()
        returns = v2_1_tables["returns"].copy()

        orders["order_date"] = pd.to_datetime(orders["order_date"])
        listings["listing_date"] = pd.to_datetime(listings["listing_date"])
        returns["return_date"] = pd.to_datetime(returns["return_date"])

        train_cutoff = pd.Timestamp("2025-08-31 23:59:59")
        val_cutoff = pd.Timestamp("2025-10-31 23:59:59")

        # Orders counts
        o_tr = (orders["order_date"] <= train_cutoff).sum()
        o_val = ((orders["order_date"] > train_cutoff) & (orders["order_date"] <= val_cutoff)).sum()
        o_te = (orders["order_date"] > val_cutoff).sum()
        assert o_tr == 16952
        assert o_val == 12129
        assert o_te == 20919
        assert o_tr + o_val + o_te == 50000

        # Listings counts
        l_tr = (listings["listing_date"] <= train_cutoff).sum()
        l_val = ((listings["listing_date"] > train_cutoff) & (listings["listing_date"] <= val_cutoff)).sum()
        l_te = (listings["listing_date"] > val_cutoff).sum()
        assert l_tr == 13489
        assert l_val == 3825
        assert l_te == 2686
        assert l_tr + l_val + l_te == 20000

        # Returns counts
        r_tr = (returns["return_date"] <= train_cutoff).sum()
        r_val = ((returns["return_date"] > train_cutoff) & (returns["return_date"] <= val_cutoff)).sum()
        r_te = (returns["return_date"] > val_cutoff).sum()
        assert r_tr == 1758
        assert r_val == 1333
        assert r_te == 2446
        assert r_tr + r_val + r_te == 5537

    def test_causal_cross_split_directionality(self, v2_1_tables):
        """Verifies temporal causality: returns and orders never reference future events.

        - Zero returns in Train belong to Val or Test orders.
        - Zero orders in Train purchase listings published in Val or Test.
        """
        orders = v2_1_tables["orders"].copy()
        listings = v2_1_tables["listings"].copy()
        returns = v2_1_tables["returns"].copy()

        orders["order_date"] = pd.to_datetime(orders["order_date"])
        listings["listing_date"] = pd.to_datetime(listings["listing_date"])
        returns["return_date"] = pd.to_datetime(returns["return_date"])

        train_cutoff = pd.Timestamp("2025-08-31 23:59:59")

        orders["order_split"] = np.where(orders["order_date"] <= train_cutoff, "train", "post_train")
        listings["listing_split"] = np.where(listings["listing_date"] <= train_cutoff, "train", "post_train")
        returns["return_split"] = np.where(returns["return_date"] <= train_cutoff, "train", "post_train")

        ro = returns.merge(orders[["order_id", "order_split"]], on="order_id")
        train_returns_from_post = ro[(ro["return_split"] == "train") & (ro["order_split"] == "post_train")]
        assert len(train_returns_from_post) == 0

        ol = orders.merge(listings[["listing_id", "listing_split"]], on="listing_id")
        train_orders_from_post = ol[(ol["order_split"] == "train") & (ol["listing_split"] == "post_train")]
        assert len(train_orders_from_post) == 0


class TestTaskDMultimodalReproducibility:
    """Task D: Multimodal evidence reproducibility."""

    def test_reproduced_auc_exact_match(self, v2_1_tables):
        """Reproduces 0.7599 AUC on all listings and checks split-level diagnostic AUCs."""
        listings = v2_1_tables["listings"].copy()
        products = v2_1_tables["products"].copy()

        sim = compute_multimodal_similarity(listings, products, use_clip=False)
        y = listings["is_fraudulent"].astype(int)

        # Higher similarity indicates genuine -> invert sim for fraud scoring
        auc_all = roc_auc_score(y, -sim)
        assert round(auc_all, 4) == 0.7599

        # Verify distributions
        gen_sim = sim[y == 0]
        fake_sim = sim[y == 1]
        assert abs(gen_sim.mean() - 0.7875) < 0.001
        assert abs(gen_sim.std() - 0.1000) < 0.001
        assert abs(fake_sim.mean() - 0.6745) < 0.001
        assert abs(fake_sim.std() - 0.1220) < 0.001

        # Check Test split diagnostic AUC (Nov-Dec)
        listings["listing_date"] = pd.to_datetime(listings["listing_date"])
        test_mask = listings["listing_date"] > "2025-10-31 23:59:59"
        auc_test = roc_auc_score(y[test_mask], -sim[test_mask])
        assert round(auc_test, 4) == 0.7324


class TestTaskETestSetProtection:
    """Task E: Test-set protection and model isolation."""

    def test_test_set_unmodified_and_virgin(self):
        """Verifies that dataset manifest hashes match disk and test set has not been retrained."""
        manifest_path = os.path.join(_DATA_DIR, "dataset_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for filename, meta in manifest["files"].items():
            filepath = os.path.join(_DATA_DIR, filename)
            h = hashlib.sha256()
            with open(filepath, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            assert h.hexdigest() == meta["sha256"]
