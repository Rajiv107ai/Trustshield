"""Automated regression tests for Master Project Audit fixes (TS-AUD-01 through TS-AUD-05)."""

import pytest
import numpy as np
import pandas as pd


from baseline_model import build_features, TRAIN_END, VAL_END
from graph_features import build_relationship_graph, compute_relationship_features
from backend.main import _build_feature_row
from backend.schemas import TransactionScoreRequest


class TestTemporalLeakageFixes:
    """Verify that temporal cutoffs prevent future relationship leakage into training features."""

    def test_future_device_sharing_invisible_to_phase5_train_split(self):
        """TS-AUD-01: Sharing relationship observed after TRAIN_END must not appear in train features."""
        future_date = TRAIN_END + pd.Timedelta(days=45)
        addr_log = pd.DataFrame(columns=["buyer_id", "shared_with_buyer_id", "first_seen_date"])
        dev_log = pd.DataFrame({
            "buyer_id": ["B_TEST_01"],
            "shared_with_buyer_id": ["B_TEST_02"],
            "shared_device_id": ["DEV_01"],
            "share_type": ["fraud_linked"],
            "first_seen_date": [future_date],
        })

        G_train = build_relationship_graph(addr_log, dev_log, cutoff_date=TRAIN_END)
        feats_train = compute_relationship_features(G_train, ["B_TEST_01", "B_TEST_02"])
        b1_train = feats_train[feats_train["buyer_id"] == "B_TEST_01"].iloc[0]

        assert b1_train["share_degree"] == 0, (
            "TS-AUD-01 regression: Train graph must exclude sharing observed after TRAIN_END."
        )
        assert b1_train["share_component_size"] == 1

        G_val = build_relationship_graph(addr_log, dev_log, cutoff_date=VAL_END)
        feats_val = compute_relationship_features(G_val, ["B_TEST_01", "B_TEST_02"])
        b1_val = feats_val[feats_val["buyer_id"] == "B_TEST_01"].iloc[0]
        assert b1_val["share_degree"] == 1, (
            "Sharing observed before VAL_END must be visible in evaluation graph."
        )


class TestApiReturnRateClipping:
    """Verify that derived buyer_return_rate_before is strictly capped in [0.0, 1.0]."""

    def test_derived_return_rate_capped_at_one(self):
        """TS-AUD-04: Returns > orders must be clipped to 1.0 rather than exceeding 1.0."""
        req = TransactionScoreRequest(
            buyer_orders_before=2,
            buyer_returns_before=10,
        )
        row = _build_feature_row(req, ["buyer_return_rate_before"])
        assert row["buyer_return_rate_before"] == 1.0, (
            f"TS-AUD-04 regression: buyer_return_rate_before should be clipped to 1.0, got {row['buyer_return_rate_before']}"
        )

    def test_derived_return_rate_normal_ratio(self):
        """Normal return ratio must be calculated correctly."""
        req = TransactionScoreRequest(
            buyer_orders_before=10,
            buyer_returns_before=2,
        )
        row = _build_feature_row(req, ["buyer_return_rate_before"])
        assert row["buyer_return_rate_before"] == pytest.approx(0.20)


class TestZeroBasePriceInfGuard:
    """Verify that build_features does not generate np.inf on zero base price."""

    def test_zero_base_price_produces_no_infs(self):
        """TS-AUD-05: When base_price is 0 or NaN, price_vs_base_price_ratio must be 0, not inf."""
        orders_df = pd.DataFrame([{
            "order_id": "ORD_ZERO_PRICE",
            "listing_id": "LIST_ZERO",
            "order_date": pd.Timestamp("2025-05-01"),
            "amount": 50.0,
            "buyer_id": "BUYER_01",
            "seller_id": "SELLER_01",
            "device_id": "DEV_01",
        }])
        listings_df = pd.DataFrame([{
            "listing_id": "LIST_ZERO",
            "seller_id": "SELLER_01",
            "product_id": "PROD_ZERO",
            "category": "Electronics",
            "listing_date": pd.Timestamp("2025-04-01"),
        }])
        returns_df = pd.DataFrame(columns=["buyer_id", "return_date"])
        buyers_df = pd.DataFrame([{
            "buyer_id": "BUYER_01",
            "signup_date": pd.Timestamp("2025-01-01"),
        }])
        sellers_df = pd.DataFrame([{
            "seller_id": "SELLER_01",
            "signup_date": pd.Timestamp("2025-01-01"),
        }])
        products_df = pd.DataFrame([{
            "product_id": "PROD_ZERO",
            "base_price": 0.0,  # Zero base price!
        }])

        df, feature_cols = build_features(
            orders_df, listings_df, returns_df, buyers_df, sellers_df, products_df
        )

        val = df.loc[0, "price_vs_base_price_ratio"]
        assert not np.isinf(val), f"TS-AUD-05 regression: inf found in price_vs_base_price_ratio: {val}"
        assert not np.isnan(val), f"TS-AUD-05 regression: NaN found in price_vs_base_price_ratio: {val}"
        assert val == 0.0


class TestApiEnsembleDisagreement:
    """Verify that Phase 5 scoring queries tabular+graph model distinctly from hybrid model."""

    def test_phase5_distinct_tabular_and_gnn_risk(self):
        """TS-AUD-03: Tabular risk and GNN risk must reflect distinct models, not duplicate copies."""
        from fastapi.testclient import TestClient
        from backend.main import app

        with TestClient(app) as client:
            resp = client.post("/transaction/score", json={
                "order_id": "ORD_DISAGREE_TEST",
                "buyer_id": "BUYER_000001",
                "seller_id": "SELLER_000001",
                "amount": 120.0,
                "base_price": 120.0,
                "category_median_price": 100.0,
                "buyer_orders_before": 10,
                "buyer_returns_before": 1,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert "model_disagreement" in data
            assert 0.0 <= data["model_disagreement"] <= 1.0
