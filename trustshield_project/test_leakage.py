"""Tests verifying absence of data leakage and temporal consistency."""

from typing import Any

import pandas as pd
from baseline_model import TRAIN_END, VAL_END


class TestFeatureSetSanity:
    BANNED_COLS = {
        "is_fraudulent", "fraud_type", "price_anomaly", "image_mismatch",
        "fraud_ring_id", "displayed_product_id", "share_type", "fraud_linked",
        "graph_fraud_degree_ratio", "community_id",
        "trust_score", "risk_score", "overall_fraud_probability",
    }

    def test_no_banned_columns_in_features(self, pipeline):
        used_banned = self.BANNED_COLS.intersection(pipeline["feature_cols"])
        assert not used_banned, f"Banned columns in feature set: {used_banned}"

    def test_fraud_ring_id_absent_from_df(self, pipeline):
        assert "fraud_ring_id" not in pipeline["df"].columns

    def test_features_strictly_invariant_to_label_permutation(self, pipeline):
        """Permuting is_fraudulent, fraud_type, or fraud_ring_id must have ZERO impact
        on tabular or graph features.
        
        If a feature depended on labels (e.g. a hypothetical graph_fraud_degree_ratio),
        permuting the label vector would alter the feature values.
        """
        import numpy as np
        from baseline_model import build_features
        from graph_features import (
            attach_relationship_snapshot_features,
            build_monthly_snapshots,
            attach_snapshot_features,
            add_edge_weight_before,
            SIM_START,
        )

        res = pipeline["result"]
        base = pipeline["base"]
        txn = pipeline["txn"]
        cat = pipeline["catalog"]

        def _extract(ord_in, list_in):
            d, tc = build_features(ord_in, list_in, res["returns"], txn["buyers"], cat["sellers"], cat["products"])
            d = attach_relationship_snapshot_features(d, base["address_sharing_log"], base["device_sharing_log"])
            snaps, months = build_monthly_snapshots(ord_in.assign(order_date=pd.to_datetime(ord_in["order_date"])), SIM_START)
            d = attach_snapshot_features(d, snaps, months)
            return add_edge_weight_before(d), tc

        df1, tab_cols = _extract(res["orders"], res["listings"])

        # Permuted copies with scrambled labels
        rng = np.random.RandomState(42)
        orders_perm = res["orders"].copy()
        orders_perm["is_fraudulent"] = rng.permutation(orders_perm["is_fraudulent"].values)
        if "fraud_type" in orders_perm.columns:
            orders_perm["fraud_type"] = rng.permutation(orders_perm["fraud_type"].values)
        listings_perm = res["listings"].copy()
        listings_perm["is_fraudulent"] = rng.permutation(listings_perm["is_fraudulent"].values)

        df2, _ = _extract(orders_perm, listings_perm)

        eval_cols = tab_cols + [
            "share_degree", "share_component_size", "buyer_seller_degree", "buyer_pagerank",
            "seller_buyer_degree", "seller_pagerank", "seller_buyer_concentration_hhi", "buyer_seller_edge_weight_before"
        ]
        for col in eval_cols:
            np.testing.assert_array_equal(df1[col].to_numpy(), df2[col].to_numpy(), err_msg=f"Feature '{col}' changed on label permutation!")


class TestTemporalIntegrity:
    def test_buyer_returns_never_exceed_orders(self, pipeline):
        df = pipeline["df"]
        violations = (df["buyer_returns_before"] > df["buyer_orders_before"]).sum()
        assert violations == 0

    def test_no_negative_seller_age(self, pipeline):
        assert (pipeline["df"]["seller_age_days"] < 0).sum() == 0

    def test_no_negative_buyer_age(self, pipeline):
        assert (pipeline["df"]["buyer_age_days"] < 0).sum() == 0

    def test_return_rate_bounded(self, pipeline):
        rate = pipeline["df"]["buyer_return_rate_before"]
        assert (rate >= 0).all() and (rate <= 1.0001).all()


class TestTemporalSplit:
    def test_train_before_val(self, pipeline):
        assert pipeline["train"]["order_date"].max() <= pipeline["val"]["order_date"].min()

    def test_val_before_test(self, pipeline):
        assert pipeline["val"]["order_date"].max() <= pipeline["test"]["order_date"].min()

    def test_train_ends_before_cutoff(self, pipeline):
        assert pipeline["train"]["order_date"].max() <= TRAIN_END

    def test_val_ends_before_cutoff(self, pipeline):
        assert pipeline["val"]["order_date"].max() <= VAL_END

    def test_no_empty_splits(self, pipeline):
        assert len(pipeline["train"]) > 0 and len(pipeline["val"]) > 0 and len(pipeline["test"]) > 0


class TestFraudInjectionLeakage:
    def test_no_fraud_ring_id_in_listings(self, pipeline):
        assert "fraud_ring_id" not in pipeline["result"]["listings"].columns

    def test_no_fraud_ring_id_in_orders(self, pipeline):
        assert "fraud_ring_id" not in pipeline["result"]["orders"].columns

    def test_no_fraud_ring_id_in_returns(self, pipeline):
        assert "fraud_ring_id" not in pipeline["result"]["returns"].columns

    def test_fraud_ground_truth_has_ring_ids(self, pipeline):
        assert "fraud_ring_id" in pipeline["result"]["fraud_ground_truth"].columns


class TestPriceAnomalyPropagation:
    def test_fake_listing_orders_have_discounted_prices(self, pipeline):
        listings = pipeline["result"]["listings"]
        orders = pipeline["result"]["orders"]

        anomaly_listings = listings[
            (listings["is_fraudulent"] == True) & (listings["price_anomaly"] == True)
        ]
        for _, listing in anomaly_listings.iterrows():
            matching_orders = orders[orders["listing_id"] == listing["listing_id"]]
            if not matching_orders.empty:
                assert (matching_orders["amount"] == listing["price"]).all()


class TestReturnTemporalSafety:
    def test_no_return_before_order(self, pipeline):
        returns = pipeline["result"]["returns"]
        orders = pipeline["result"]["orders"]

        merged = returns.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
        violations = (pd.to_datetime(merged["return_date"]) < pd.to_datetime(merged["order_date"])).sum()
        assert violations == 0


# ---------------------------------------------------------------------------
# Issue #3 — Same-timestamp prior-count regression tests
# ---------------------------------------------------------------------------

class TestSameTimestampOrderLeakage:
    def test_same_timestamp_both_get_zero_prior_count(self):
        from baseline_model import _cumulative_count_asof
        ts = pd.Timestamp("2025-06-15 10:00:00")
        orders = pd.DataFrame({"buyer_id": ["B001", "B001"], "order_date": [ts, ts], "order_id": ["O1", "O2"]})
        result = _cumulative_count_asof(orders, group_col="buyer_id", date_col="order_date")
        assert result.iloc[0] == 0 and result.iloc[1] == 0

    def test_earlier_order_counted_for_later(self):
        from baseline_model import _cumulative_count_asof
        orders = pd.DataFrame({"buyer_id": ["B001", "B001"], "order_date": [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-15")], "order_id": ["O1", "O2"]})
        result = _cumulative_count_asof(orders, group_col="buyer_id", date_col="order_date")
        assert result.iloc[0] == 0 and result.iloc[1] == 1


class TestSameTimestampEdgeWeightLeakage:
    def test_same_timestamp_edge_weight_both_zero(self):
        from graph_features import add_edge_weight_before
        ts = pd.Timestamp("2025-07-01 12:00:00")
        df = pd.DataFrame({"buyer_id": ["B001", "B001"], "seller_id": ["S001", "S001"], "order_date": [ts, ts], "order_id": ["O1", "O2"]})
        result = add_edge_weight_before(df)
        assert result["buyer_seller_edge_weight_before"].iloc[0] == 0 and result["buyer_seller_edge_weight_before"].iloc[1] == 0

    def test_prior_order_counted_for_later_order(self):
        from graph_features import add_edge_weight_before
        df = pd.DataFrame({"buyer_id": ["B001", "B001"], "seller_id": ["S001", "S001"], "order_date": [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-15")], "order_id": ["O1", "O2"]})
        result = add_edge_weight_before(df).sort_values("order_date").reset_index(drop=True)
        assert result.loc[0, "buyer_seller_edge_weight_before"] == 0 and result.loc[1, "buyer_seller_edge_weight_before"] == 1


# ---------------------------------------------------------------------------
# Issue #4 — Graph features temporal safety tests
def _make_sharing_logs(first_seen_date):
    import pandas as pd
    d = {"buyer_id": ["B001"], "shared_with_buyer_id": ["B002"], "shared_address_id": ["ADDR_001"], "share_type": ["fraud_linked"]}
    if first_seen_date is not None:
        d["first_seen_date"] = [first_seen_date]
    dev_cols = ["buyer_id", "shared_with_buyer_id", "shared_device_id", "share_type"]
    if first_seen_date is not None:
        dev_cols.append("first_seen_date")
    return pd.DataFrame(d), pd.DataFrame(columns=dev_cols)


class TestGraphTemporalSafety:
    """Prove that a sharing relationship created in the future cannot affect
    an earlier order's share_degree or share_component_size.
    """

    def test_future_sharing_does_not_affect_past_order(self):
        """An order scored before a sharing relationship first_seen_date must see
        share_degree=0, not the degree from the future sharing edge.
        """
        import pandas as pd
        from graph_features import build_relationship_graph, compute_relationship_features

        early_order_date = pd.Timestamp("2025-03-01")
        future_sharing_date = pd.Timestamp("2025-06-01")

        addr_log, dev_log = _make_sharing_logs(future_sharing_date)
        G_early = build_relationship_graph(addr_log, dev_log, cutoff_date=early_order_date)
        b001_early = compute_relationship_features(G_early, ["B001", "B002"]).query("buyer_id == 'B001'").iloc[0]
        assert b001_early["share_degree"] == 0
        assert b001_early["share_component_size"] == 1

    def test_sharing_visible_after_first_seen_date(self):
        import pandas as pd
        from graph_features import build_relationship_graph, compute_relationship_features
        addr_log, dev_log = _make_sharing_logs(pd.Timestamp("2025-03-01"))
        G_later = build_relationship_graph(addr_log, dev_log, cutoff_date=pd.Timestamp("2025-06-01"))
        b001_later = compute_relationship_features(G_later, ["B001", "B002"]).query("buyer_id == 'B001'").iloc[0]
        assert b001_later["share_degree"] == 1
        assert b001_later["share_component_size"] == 2

    def test_no_first_seen_date_column_falls_back_to_all_relationships(self):
        import pandas as pd
        from graph_features import build_relationship_graph, compute_relationship_features
        addr_log, dev_log = _make_sharing_logs(None)
        G = build_relationship_graph(addr_log, dev_log, cutoff_date=None)
        b001 = compute_relationship_features(G, ["B001", "B002"]).query("buyer_id == 'B001'").iloc[0]
        assert b001["share_degree"] == 1

class TestRunPhase3GraphTemporalSafety:
    def test_future_sharing_invisible_to_training_period_with_cutoff(self):
        import pandas as pd
        from baseline_model import TRAIN_END
        from graph_features import build_relationship_graph, compute_relationship_features
        addr_log, dev_log = _make_sharing_logs(TRAIN_END + pd.Timedelta(days=60))
        G_train = build_relationship_graph(addr_log, dev_log, cutoff_date=TRAIN_END)
        b001_train = compute_relationship_features(G_train, ["B001", "B002"]).query("buyer_id == 'B001'").iloc[0]
        assert b001_train["share_degree"] == 0

    def test_future_sharing_visible_after_val_end_cutoff(self):
        import pandas as pd
        from baseline_model import TRAIN_END, VAL_END
        from graph_features import build_relationship_graph, compute_relationship_features
        addr_log, dev_log = _make_sharing_logs(TRAIN_END + pd.Timedelta(days=30))
        G_val = build_relationship_graph(addr_log, dev_log, cutoff_date=VAL_END)
        b001_val = compute_relationship_features(G_val, ["B001", "B002"]).query("buyer_id == 'B001'").iloc[0]
        assert b001_val["share_degree"] == 1

class TestFeedbackLoopGuard:
    def test_model_output_columns_are_banned(self):
        import pandas as pd
        import pytest
        from baseline_model import leakage_audit
        df = pd.DataFrame({"buyer_returns_before": [0], "buyer_orders_before": [1], "seller_age_days": [30], "buyer_age_days": [30]})
        for col in ["trust_score", "risk_score", "overall_fraud_probability", "predicted_fraud", "decision", "model_reason_code", "model_probability"]:
            with pytest.raises(AssertionError):
                leakage_audit(df, [col])

    def test_legitimate_features_are_not_banned(self):
        """Legitimate feature columns must pass the feedback-loop guard."""
        import pandas as pd
        from baseline_model import leakage_audit

        df = pd.DataFrame({
            "buyer_returns_before": [0],
            "buyer_orders_before": [1],
            "seller_age_days": [30],
            "buyer_age_days": [30],
            "amount": [99.0],
        })
        # Should not raise
        leakage_audit(df, ["amount", "buyer_returns_before", "buyer_orders_before"])


# ---------------------------------------------------------------------------
# Check 2 — Point-In-Time Graph & Event Leakage Prevention Tests
# ---------------------------------------------------------------------------

class TestPointInTimeGraphAndReturnLeakagePrevention:
    """Verifies that future events strictly cannot alter feature representations of earlier transactions."""

    def test_future_orders_do_not_affect_earlier_snapshot_features(self):
        import numpy as np
        from graph_features import build_monthly_snapshots, attach_snapshot_features
        m1_dates = pd.date_range("2025-01-05", periods=5, freq="D")
        m2_dates = pd.date_range("2025-02-05", periods=5, freq="D")
        base_orders = pd.DataFrame({
            "order_id": [f"ORD_M1_{i}" for i in range(5)] + [f"ORD_M2_{i}" for i in range(5)],
            "buyer_id": ["B1", "B2", "B1", "B3", "B2", "B1", "B2", "B3", "B1", "B2"],
            "seller_id": ["S1", "S1", "S2", "S2", "S1", "S1", "S2", "S1", "S2", "S1"],
            "order_date": list(m1_dates) + list(m2_dates),
            "amount": [100.0] * 10,
        })
        sim_start = pd.Timestamp("2025-01-01")
        snapshots_v1, months_v1 = build_monthly_snapshots(base_orders, sim_start)
        df_v1 = attach_snapshot_features(base_orders, snapshots_v1, months_v1)

        m3_orders = pd.DataFrame({
            "order_id": [f"ORD_M3_{i}" for i in range(10)],
            "buyer_id": ["B1", "B2"] * 5,
            "seller_id": ["S1"] * 10,
            "order_date": list(pd.date_range("2025-03-05", periods=10, freq="D")),
            "amount": [500.0] * 10,
        })
        all_orders = pd.concat([base_orders, m3_orders], ignore_index=True)
        snapshots_v2, months_v2 = build_monthly_snapshots(all_orders, sim_start)
        df_v2 = attach_snapshot_features(all_orders, snapshots_v2, months_v2)

        check_cols = ["buyer_seller_degree", "buyer_pagerank", "seller_buyer_degree", "seller_pagerank", "seller_buyer_concentration_hhi"]
        for col in check_cols:
            np.testing.assert_allclose(np.array(df_v1.loc[:9, col], dtype=np.float64), np.array(df_v2.loc[:9, col], dtype=np.float64), rtol=1e-5, atol=1e-5)

    def test_future_returns_do_not_leak_into_earlier_order_features(self):
        from baseline_model import build_features
        b_df = pd.DataFrame({"buyer_id": ["B_TEST"], "signup_date": [pd.Timestamp("2025-01-01")]})
        s_df = pd.DataFrame({"seller_id": ["S_TEST"], "signup_date": [pd.Timestamp("2025-01-01")]})
        p_df = pd.DataFrame({"product_id": ["P_TEST"], "base_price": [50.0]})
        l_df = pd.DataFrame({"listing_id": ["L_TEST"], "seller_id": ["S_TEST"], "product_id": ["P_TEST"], "category": ["electronics"], "listing_date": [pd.Timestamp("2025-01-02")], "price": [50.0]})
        o_df = pd.DataFrame({"order_id": ["ORD_1", "ORD_2"], "buyer_id": ["B_TEST", "B_TEST"], "seller_id": ["S_TEST", "S_TEST"], "listing_id": ["L_TEST", "L_TEST"], "order_date": [pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-20")], "amount": [50.0, 50.0], "device_id": ["DEV_TEST", "DEV_TEST"]})
        r_df = pd.DataFrame({"return_id": ["RET_1"], "order_id": ["ORD_1"], "buyer_id": ["B_TEST"], "return_date": [pd.Timestamp("2025-01-15")]})
        feat_df, _ = build_features(o_df, l_df, r_df, b_df, s_df, p_df)
        assert feat_df[feat_df["order_id"] == "ORD_1"].iloc[0]["buyer_returns_before"] == 0
        assert feat_df[feat_df["order_id"] == "ORD_2"].iloc[0]["buyer_returns_before"] == 1

    def test_future_pair_orders_do_not_leak_into_earlier_edge_weight(self):
        from graph_features import add_edge_weight_before
        df = pd.DataFrame({"order_id": ["ORD_1", "ORD_2", "ORD_3"], "buyer_id": ["B1", "B1", "B1"], "seller_id": ["S1", "S1", "S1"], "order_date": [pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-15"), pd.Timestamp("2025-03-01")], "amount": [10.0, 10.0, 10.0]})
        res = add_edge_weight_before(df)
        assert res.sort_values("order_date")["buyer_seller_edge_weight_before"].tolist() == [0, 1, 2]


class TestShuffledLabelCanary:
    def test_shuffled_train_labels_yield_chance_test_auc(self):
        import numpy as np
        from xgboost import XGBClassifier
        from sklearn.metrics import roc_auc_score
        from baseline_model import build_features, TRAIN_END, VAL_END
        p = "trustshield_project/synthetic_data_export"
        orders = pd.read_csv(f"{p}/orders.csv").assign(order_date=lambda d: pd.to_datetime(d["order_date"]))
        listings = pd.read_csv(f"{p}/listings.csv").assign(listing_date=lambda d: pd.to_datetime(d["listing_date"]))
        returns = pd.read_csv(f"{p}/returns.csv").assign(return_date=lambda d: pd.to_datetime(d["return_date"]))
        buyers = pd.read_csv(f"{p}/buyers.csv").assign(signup_date=lambda d: pd.to_datetime(d["signup_date"]))
        sellers = pd.read_csv(f"{p}/sellers.csv").assign(signup_date=lambda d: pd.to_datetime(d["signup_date"]))
        products = pd.read_csv(f"{p}/products.csv")

        df, feat_cols = build_features(orders, listings, returns, buyers, sellers, products)
        train = pd.DataFrame(df[df["order_date"] <= TRAIN_END])
        test = pd.DataFrame(df[df["order_date"] > VAL_END])
        y_test = np.asarray(test["is_fraudulent"].astype(int))
        X_train, X_test = pd.DataFrame(train[feat_cols]).fillna(0), pd.DataFrame(test[feat_cols]).fillna(0)
        y_train_orig = np.asarray(train["is_fraudulent"])

        shuffled_aucs = []
        for seed in range(10):
            rng = np.random.RandomState(seed)
            clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="logloss", verbosity=0)
            clf.fit(X_train, rng.permutation(y_train_orig))
            shuffled_aucs.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

        mean_auc = float(np.mean(shuffled_aucs))
        assert 0.47 <= mean_auc <= 0.53, f"Canary failed: mean={mean_auc:.4f}"


class TestPointInTimeMonthlyGraphIsolation:
    def test_month_m_orders_use_strictly_prior_month_bipartite_graph(self):
        from graph_features import build_monthly_snapshots
        orders = pd.DataFrame({
            "order_id": [f"O_{i}" for i in range(6)],
            "buyer_id": ["B1", "B2", "B1", "B3", "B2", "B1"],
            "seller_id": ["S1", "S1", "S2", "S2", "S1", "S2"],
            "order_date": [pd.Timestamp(f"2025-0{m}-10") for m in [1, 1, 2, 2, 3, 3]],
            "amount": [100.0] * 6,
        })
        snapshots, _ = build_monthly_snapshots(orders, pd.Timestamp("2025-01-01"))
        assert len(snapshots) >= 2
        buyer_feat_m1, _ = snapshots[1]
        assert set(buyer_feat_m1["buyer_id"]).issubset({"B1", "B2"})
        assert "B3" not in buyer_feat_m1["buyer_id"].values

    def test_sharing_features_derived_strictly_prior_to_order_date(self):
        from graph_features import attach_relationship_snapshot_features
        orders = pd.DataFrame({"order_id": ["O1", "O2"], "buyer_id": ["B1", "B1"], "order_date": [pd.Timestamp("2025-03-01"), pd.Timestamp("2025-05-01")]})
        addr_log = pd.DataFrame({"buyer_id": ["B1"], "shared_with_buyer_id": ["B2"], "shared_address_id": ["A1"], "share_type": ["device_cluster"], "first_seen_date": [pd.Timestamp("2025-04-01")]})
        dev_log = pd.DataFrame(columns=["buyer_id", "shared_with_buyer_id", "shared_device_id", "share_type", "first_seen_date"])
        res = attach_relationship_snapshot_features(orders, addr_log, dev_log, sim_start=pd.Timestamp("2025-01-01"))
        o1 = res[res["order_id"] == "O1"].iloc[0]
        o2 = res[res["order_id"] == "O2"].iloc[0]
        assert o1["share_degree"] == 0 and o1["share_component_size"] == 1
        assert o2["share_degree"] == 1 and o2["share_component_size"] == 2

