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
