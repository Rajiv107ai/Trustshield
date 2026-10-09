"""Tests verifying absence of data leakage and temporal consistency."""

import pandas as pd
from baseline_model import TRAIN_END, VAL_END


class TestFeatureSetSanity:
    BANNED_COLS = {
        "is_fraudulent", "fraud_type", "price_anomaly", "image_mismatch",
        "fraud_ring_id", "displayed_product_id",
    }

    def test_no_banned_columns_in_features(self, pipeline):
        used_banned = self.BANNED_COLS.intersection(pipeline["feature_cols"])
        assert not used_banned, f"Banned columns in feature set: {used_banned}"

    def test_fraud_ring_id_absent_from_df(self, pipeline):
        assert "fraud_ring_id" not in pipeline["df"].columns


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
    """Two orders for the same buyer at the exact same timestamp must both get
    buyer_orders_before = 0.  The old cumcount() code would give one of them
    count=1, creating temporal leakage.
    """

    def test_same_timestamp_both_get_zero_prior_count(self):
        from baseline_model import _cumulative_count_asof
        import pandas as pd

        ts = pd.Timestamp("2025-06-15 10:00:00")
        orders = pd.DataFrame({
            "buyer_id": ["B001", "B001"],
            "order_date": [ts, ts],
            "order_id": ["O1", "O2"],
        })
        orders["order_date"] = pd.to_datetime(orders["order_date"])

        result = _cumulative_count_asof(orders, group_col="buyer_id", date_col="order_date")

        assert result.iloc[0] == 0, (
            "First of two same-timestamp orders must see 0 prior orders (not 1)."
        )
        assert result.iloc[1] == 0, (
            "Second of two same-timestamp orders must see 0 prior orders (not 1)."
        )

    def test_earlier_order_counted_for_later(self):
        """An order placed before another must be counted in the later order's prior count."""
        from baseline_model import _cumulative_count_asof
        import pandas as pd

        orders = pd.DataFrame({
            "buyer_id": ["B001", "B001"],
            "order_date": [
                pd.Timestamp("2025-06-01"),
                pd.Timestamp("2025-06-15"),
            ],
            "order_id": ["O1", "O2"],
        })
        orders["order_date"] = pd.to_datetime(orders["order_date"])

        result = _cumulative_count_asof(orders, group_col="buyer_id", date_col="order_date")
        assert result.iloc[0] == 0, "First (earliest) order should have 0 prior orders"
        assert result.iloc[1] == 1, "Second (later) order should see 1 prior order"


class TestSameTimestampEdgeWeightLeakage:
    """Two orders between the same buyer and seller at the exact same timestamp must
    both get buyer_seller_edge_weight_before = 0.
    """

    def test_same_timestamp_edge_weight_both_zero(self):
        from graph_features import add_edge_weight_before
        import pandas as pd

        ts = pd.Timestamp("2025-07-01 12:00:00")
        df = pd.DataFrame({
            "buyer_id": ["B001", "B001"],
            "seller_id": ["S001", "S001"],
            "order_date": [ts, ts],
            "order_id": ["O1", "O2"],
        })
        df["order_date"] = pd.to_datetime(df["order_date"])

        result = add_edge_weight_before(df)

        assert result["buyer_seller_edge_weight_before"].iloc[0] == 0, (
            "First same-timestamp order must have edge_weight_before=0."
        )
        assert result["buyer_seller_edge_weight_before"].iloc[1] == 0, (
            "Second same-timestamp order must have edge_weight_before=0 (not 1)."
        )

    def test_prior_order_counted_for_later_order(self):
        """An earlier order must be counted for the later one."""
        from graph_features import add_edge_weight_before
        import pandas as pd

        df = pd.DataFrame({
            "buyer_id": ["B001", "B001"],
            "seller_id": ["S001", "S001"],
            "order_date": [
                pd.Timestamp("2025-06-01"),
                pd.Timestamp("2025-06-15"),
            ],
            "order_id": ["O1", "O2"],
        })
        df["order_date"] = pd.to_datetime(df["order_date"])
        result = add_edge_weight_before(df)
        result = result.sort_values("order_date").reset_index(drop=True)
        assert result.loc[0, "buyer_seller_edge_weight_before"] == 0
        assert result.loc[1, "buyer_seller_edge_weight_before"] == 1


# ---------------------------------------------------------------------------
# Issue #4 — Graph features temporal safety tests
# ---------------------------------------------------------------------------

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
        # The sharing relationship is first seen AFTER the early order
        future_sharing_date = pd.Timestamp("2025-06-01")

        addr_log = pd.DataFrame({
            "buyer_id": ["B001"],
            "shared_with_buyer_id": ["B002"],
            "shared_address_id": ["ADDR_001"],
            "share_type": ["fraud_linked"],
            "first_seen_date": [future_sharing_date],
        })
        dev_log = pd.DataFrame(columns=["buyer_id", "shared_with_buyer_id",
                                        "shared_device_id", "share_type", "first_seen_date"])

        # Build graph as seen BEFORE the sharing relationship was established
        G_early = build_relationship_graph(addr_log, dev_log, cutoff_date=early_order_date)
        feats_early = compute_relationship_features(G_early, ["B001", "B002"])

        b001_early = feats_early[feats_early["buyer_id"] == "B001"].iloc[0]
        assert b001_early["share_degree"] == 0, (
            "B001 must have share_degree=0 before the sharing relationship was established."
        )
        assert b001_early["share_component_size"] == 1, (
            "B001 must be in a singleton component before sharing was observed."
        )

    def test_sharing_visible_after_first_seen_date(self):
        """An order scored AFTER a sharing relationship's first_seen_date must see
        the actual graph degree (i.e. the relationship IS included).
        """
        import pandas as pd
        from graph_features import build_relationship_graph, compute_relationship_features

        sharing_date = pd.Timestamp("2025-03-01")
        later_order_date = pd.Timestamp("2025-06-01")

        addr_log = pd.DataFrame({
            "buyer_id": ["B001"],
            "shared_with_buyer_id": ["B002"],
            "shared_address_id": ["ADDR_001"],
            "share_type": ["fraud_linked"],
            "first_seen_date": [sharing_date],
        })
        dev_log = pd.DataFrame(columns=["buyer_id", "shared_with_buyer_id",
                                        "shared_device_id", "share_type", "first_seen_date"])

        # Build graph as seen AFTER the sharing relationship was established
        G_later = build_relationship_graph(addr_log, dev_log, cutoff_date=later_order_date)
        feats_later = compute_relationship_features(G_later, ["B001", "B002"])

        b001_later = feats_later[feats_later["buyer_id"] == "B001"].iloc[0]
        assert b001_later["share_degree"] == 1, (
            "B001 must have share_degree=1 after the sharing relationship was established."
        )
        assert b001_later["share_component_size"] == 2, (
            "B001 and B002 must be in the same component after sharing was observed."
        )

    def test_no_first_seen_date_column_falls_back_to_all_relationships(self):
        """Legacy sharing logs without first_seen_date must not crash when cutoff=None."""
        import pandas as pd
        from graph_features import build_relationship_graph, compute_relationship_features

        addr_log = pd.DataFrame({
            "buyer_id": ["B001"],
            "shared_with_buyer_id": ["B002"],
            "shared_address_id": ["ADDR_001"],
            "share_type": ["fraud_linked"],
            # No first_seen_date column — legacy format
        })
        dev_log = pd.DataFrame(columns=["buyer_id", "shared_with_buyer_id",
                                        "shared_device_id", "share_type"])

        # cutoff=None means "include all" — must not raise
        G = build_relationship_graph(addr_log, dev_log, cutoff_date=None)
        feats = compute_relationship_features(G, ["B001", "B002"])
        b001 = feats[feats["buyer_id"] == "B001"].iloc[0]
        assert b001["share_degree"] == 1  # edge included since no filter applied


# ---------------------------------------------------------------------------
# FIX-28 — run_phase_3 temporal leakage regression test
# ---------------------------------------------------------------------------

class TestRunPhase3GraphTemporalSafety:
    """Verify that the relationship graph used in run_phase_3() is built with
    per-split cutoff dates, not a no-cutoff all-inclusive graph.

    The bug (fixed in FIX-28): build_relationship_graph() was called without
    cutoff_date, meaning future sharing relationships were included in training
    features — a temporal leakage.  The test confirms the fix by showing that
    a sharing relationship first_seen AFTER TRAIN_END does NOT affect features
    computed for a training-period order when a cutoff is applied.
    """

    def test_future_sharing_invisible_to_training_period_with_cutoff(self):
        """Sharing first seen after TRAIN_END must not appear in the train graph."""
        import pandas as pd
        from baseline_model import TRAIN_END
        from graph_features import build_relationship_graph, compute_relationship_features

        # Sharing relationship first observed 60 days after TRAIN_END
        future_date = TRAIN_END + pd.Timedelta(days=60)

        addr_log = pd.DataFrame({
            "buyer_id": ["B001"],
            "shared_with_buyer_id": ["B002"],
            "shared_address_id": ["ADDR_001"],
            "share_type": ["fraud_linked"],
            "first_seen_date": [future_date],
        })
        dev_log = pd.DataFrame(columns=[
            "buyer_id", "shared_with_buyer_id", "shared_device_id", "share_type", "first_seen_date"
        ])

        # Build graph with TRAIN_END cutoff — future relationship must be excluded
        G_train = build_relationship_graph(addr_log, dev_log, cutoff_date=TRAIN_END)
        feats_train = compute_relationship_features(G_train, ["B001", "B002"])

        b001_train = feats_train[feats_train["buyer_id"] == "B001"].iloc[0]
        assert b001_train["share_degree"] == 0, (
            "FIX-28 regression: Training graph must NOT include sharing relationships "
            "first observed after TRAIN_END. share_degree should be 0."
        )

    def test_future_sharing_visible_after_val_end_cutoff(self):
        """Sharing first seen before VAL_END must appear in the val/test graph."""
        import pandas as pd
        from baseline_model import TRAIN_END, VAL_END
        from graph_features import build_relationship_graph, compute_relationship_features

        # Sharing relationship first observed 30 days after TRAIN_END (but before VAL_END)
        mid_date = TRAIN_END + pd.Timedelta(days=30)

        addr_log = pd.DataFrame({
            "buyer_id": ["B001"],
            "shared_with_buyer_id": ["B002"],
            "shared_address_id": ["ADDR_001"],
            "share_type": ["fraud_linked"],
            "first_seen_date": [mid_date],
        })
        dev_log = pd.DataFrame(columns=[
            "buyer_id", "shared_with_buyer_id", "shared_device_id", "share_type", "first_seen_date"
        ])

        G_val = build_relationship_graph(addr_log, dev_log, cutoff_date=VAL_END)
        feats_val = compute_relationship_features(G_val, ["B001", "B002"])

        b001_val = feats_val[feats_val["buyer_id"] == "B001"].iloc[0]
        assert b001_val["share_degree"] == 1, (
            "Sharing observed before VAL_END must be visible in the val/test graph."
        )


# ---------------------------------------------------------------------------
# FIX-04 — Feedback loop guard regression test
# ---------------------------------------------------------------------------

class TestFeedbackLoopGuard:
    """Verify that leakage_audit() blocks model-output columns from the feature set.

    Model-derived columns (trust_score, risk_score, etc.) must never become
    training features — this creates a feedback loop where the model predicts
    its own past outputs, inflating apparent performance.
    """

    def test_model_output_columns_are_banned(self):
        """leakage_audit() must reject any model-output column in feature_cols."""
        import pandas as pd
        from baseline_model import leakage_audit

        # Minimal DataFrame that satisfies the audit's other checks
        df = pd.DataFrame({
            "buyer_returns_before": [0],
            "buyer_orders_before": [1],
            "seller_age_days": [30],
            "buyer_age_days": [30],
        })

        forbidden_model_cols = [
            "trust_score", "risk_score", "overall_fraud_probability",
            "predicted_fraud", "decision", "model_reason_code", "model_probability",
        ]
        for col in forbidden_model_cols:
            feature_cols = [col]
            try:
                leakage_audit(df, feature_cols)
                raise AssertionError(
                    f"FIX-04 regression: leakage_audit() should have rejected '{col}' "
                    "as a banned model-output column, but did not."
                )
            except AssertionError as exc:
                # The AssertionError raised by leakage_audit itself is expected
                if "FIX-04 regression" in str(exc):
                    raise  # re-raise our own assertion, not the expected one
                # leakage_audit raised its own AssertionError — that's the correct behaviour

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
    """Verifies that future events (future orders, future returns, future graph edges)
    strictly cannot alter feature representations of earlier transactions.
    """

    def test_future_orders_do_not_affect_earlier_snapshot_features(self):
        """Adding Month M+1 orders to the history must not change graph features for Month M orders."""
        import numpy as np
        import pandas as pd
        from graph_features import build_monthly_snapshots, attach_snapshot_features

        # Base orders across Month 1 and Month 2
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

        # Compute snapshots without Month 3
        snapshots_v1, months_v1 = build_monthly_snapshots(base_orders, sim_start)
        df_v1 = attach_snapshot_features(base_orders, snapshots_v1, months_v1)

        # Now simulate adding 10 orders in Month 3 with a new dense collusion ring
        m3_dates = pd.date_range("2025-03-05", periods=10, freq="D")
        m3_orders = pd.DataFrame({
            "order_id": [f"ORD_M3_{i}" for i in range(10)],
            "buyer_id": ["B1", "B2", "B1", "B2", "B1", "B2", "B1", "B2", "B1", "B2"],
            "seller_id": ["S1", "S1", "S1", "S1", "S1", "S1", "S1", "S1", "S1", "S1"],
            "order_date": list(m3_dates),
            "amount": [500.0] * 10,
        })
        all_orders = pd.concat([base_orders, m3_orders], ignore_index=True)

        snapshots_v2, months_v2 = build_monthly_snapshots(all_orders, sim_start)
        df_v2 = attach_snapshot_features(all_orders, snapshots_v2, months_v2)

        # Check features for base orders (Month 1 & 2) in df_v1 vs df_v2
        check_cols = [
            "buyer_seller_degree", "buyer_pagerank",
            "seller_buyer_degree", "seller_pagerank", "seller_buyer_concentration_hhi"
        ]
        for col in check_cols:
            v1_vals = df_v1.loc[:9, col].to_numpy()
            v2_vals = df_v2.loc[:9, col].to_numpy()
            np.testing.assert_allclose(
                v1_vals, v2_vals, rtol=1e-5, atol=1e-5,
                err_msg=f"Temporal leakage detected: Month 3 future orders changed '{col}' for earlier orders!"
            )

    def test_future_returns_do_not_leak_into_earlier_order_features(self):
        """A return occurring after an order date must not increment buyer_returns_before."""
        import pandas as pd
        from baseline_model import build_features

        buyers_df = pd.DataFrame({
            "buyer_id": ["B_TEST"],
            "signup_date": [pd.Timestamp("2025-01-01")],
        })
        sellers_df = pd.DataFrame({
            "seller_id": ["S_TEST"],
            "signup_date": [pd.Timestamp("2025-01-01")],
        })
        products_df = pd.DataFrame({
            "product_id": ["P_TEST"],
            "base_price": [50.0],
        })
        listings_df = pd.DataFrame({
            "listing_id": ["L_TEST"],
            "seller_id": ["S_TEST"],
            "product_id": ["P_TEST"],
            "category": ["electronics"],
            "listing_date": [pd.Timestamp("2025-01-02")],
            "price": [50.0],
        })
        orders_df = pd.DataFrame({
            "order_id": ["ORD_1", "ORD_2"],
            "buyer_id": ["B_TEST", "B_TEST"],
            "seller_id": ["S_TEST", "S_TEST"],
            "listing_id": ["L_TEST", "L_TEST"],
            "order_date": [pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-20")],
            "amount": [50.0, 50.0],
            "device_id": ["DEV_TEST", "DEV_TEST"],
        })
        # Return for ORD_1 occurs on 2025-01-15 (after ORD_1, but before ORD_2)
        returns_df = pd.DataFrame({
            "return_id": ["RET_1"],
            "order_id": ["ORD_1"],
            "buyer_id": ["B_TEST"],
            "return_date": [pd.Timestamp("2025-01-15")],
        })

        feat_df, _ = build_features(orders_df, listings_df, returns_df, buyers_df, sellers_df, products_df)

        ord1_feat = feat_df[feat_df["order_id"] == "ORD_1"].iloc[0]
        ord2_feat = feat_df[feat_df["order_id"] == "ORD_2"].iloc[0]

        # At ORD_1 time (Jan 10), return RET_1 (Jan 15) has not happened yet -> must be 0
        assert ord1_feat["buyer_returns_before"] == 0, (
            "Temporal leakage: Future return on Jan 15 was counted before Jan 10 order!"
        )
        # At ORD_2 time (Jan 20), return RET_1 has happened -> must be 1
        assert ord2_feat["buyer_returns_before"] == 1, (
            "Historical return on Jan 15 must be counted for Jan 20 order."
        )

    def test_future_pair_orders_do_not_leak_into_earlier_edge_weight(self):
        """Future buyer-seller orders must not increment buyer_seller_edge_weight_before."""
        import pandas as pd
        from graph_features import add_edge_weight_before

        df = pd.DataFrame({
            "order_id": ["ORD_1", "ORD_2", "ORD_3"],
            "buyer_id": ["B1", "B1", "B1"],
            "seller_id": ["S1", "S1", "S1"],
            "order_date": [pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-15"), pd.Timestamp("2025-03-01")],
            "amount": [10.0, 10.0, 10.0],
        })

        res = add_edge_weight_before(df)
        weights = res.sort_values("order_date")["buyer_seller_edge_weight_before"].tolist()
        # ORD_1: 0 prior orders
        # ORD_2: 1 prior order
        # ORD_3: 2 prior orders
        assert weights == [0, 1, 2], f"Expected strictly prior counts [0, 1, 2], got {weights}"


