"""Unit tests for graph topological features, specialized models, and GNN components."""

import numpy as np
import pandas as pd
import networkx as nx
import torch
import pytest

from graph_features import (
    build_relationship_graph,
    compute_relationship_features,
    build_monthly_snapshots,
    attach_snapshot_features,
    add_edge_weight_before,
    detect_fraud_rings,
)
from phase2_specialized_models import (
    build_listing_features,
    build_return_features,
)
from gnn_model import (
    build_node_index,
    build_node_features,
    build_edge_index,
    GraphSAGEEncoder,
    EdgeClassifier,
)
from phase5_hybrid_model import attach_gnn_embeddings
from multimodal_scoring import compute_multimodal_similarity, explain_listing_risk


class TestGraphFeatures:
    def test_build_relationship_graph(self, pipeline):
        base = pipeline["base"]
        G = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"])
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_compute_relationship_features(self, pipeline):
        base = pipeline["base"]
        G = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"])
        buyers = list(pipeline["txn"]["buyers"]["buyer_id"].unique()[:20])
        feat_df = compute_relationship_features(G, buyers)
        assert len(feat_df) == 20
        assert set(feat_df.columns) == {"buyer_id", "share_degree", "share_component_size"}
        assert (feat_df["share_degree"] >= 0).all()
        assert (feat_df["share_component_size"] >= 1).all()

    def test_monthly_snapshots_and_attachment(self, pipeline):
        orders = pipeline["result"]["orders"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        snapshots, months = build_monthly_snapshots(orders, orders["order_date"].min())
        assert len(snapshots) > 0
        assert len(months) > 0

        attached = attach_snapshot_features(orders.head(50), snapshots, months)
        assert len(attached) == 50
        expected_cols = [
            "buyer_seller_degree", "buyer_pagerank",
            "seller_buyer_degree", "seller_pagerank", "seller_buyer_concentration_hhi"
        ]
        for col in expected_cols:
            assert col in attached.columns
            assert not bool(attached[col].isna().to_numpy().any())

    def test_add_edge_weight_before(self, pipeline):
        orders = pipeline["result"]["orders"].head(50).copy()
        res = add_edge_weight_before(orders)
        assert "buyer_seller_edge_weight_before" in res.columns
        assert (res["buyer_seller_edge_weight_before"] >= 0).all()

    def test_detect_fraud_rings(self, pipeline):
        base = pipeline["base"]
        G = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"])
        orders = pipeline["df"].assign(fraud_score=0.45)
        rings = detect_fraud_rings(G, orders, score_col="fraud_score", min_ring_size=2)
        assert isinstance(rings, pd.DataFrame)
        if len(rings) > 0:
            assert "ring_id" in rings.columns
            assert "size" in rings.columns
            assert (rings["size"] >= 2).all()
            assert (rings["avg_risk_score"] >= 0.0).all()

    def test_detect_fraud_rings_leakage_assertion(self, pipeline):
        base = pipeline["base"]
        G = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"])
        orders = pipeline["df"].copy()
        with pytest.raises(AssertionError, match="Leakage"):
            detect_fraud_rings(G, orders, score_col="is_fraudulent")


class TestSpecializedModels:
    def test_build_listing_features(self, pipeline):
        listings = pipeline["result"]["listings"]
        sellers = pipeline["catalog"]["sellers"]
        products = pipeline["catalog"]["products"]
        df, cols = build_listing_features(listings, sellers, products)
        assert len(df) == len(listings)
        assert set(cols).issubset(set(df.columns))
        assert not bool(df[cols].isna().to_numpy().any())

    def test_build_return_features(self, pipeline):
        returns = pipeline["result"]["returns"]
        orders = pipeline["result"]["orders"]
        buyers = pipeline["txn"]["buyers"]
        sellers = pipeline["catalog"]["sellers"]
        df, cols = build_return_features(returns, orders, buyers, sellers)
        assert len(df) == len(returns)
        assert set(cols).issubset(set(df.columns))
        assert not bool(df[cols].isna().to_numpy().any())


class TestGNNComponents:
    def test_build_node_index(self):
        buyer_ids = ["B_001", "B_002", "B_003"]
        seller_ids = ["S_001", "S_002"]
        b_idx, s_idx, n_b, n_tot = build_node_index(buyer_ids, seller_ids)
        assert n_b == 3
        assert n_tot == 6
        assert set(b_idx.values()) == {1, 2, 3}
        assert set(s_idx.values()) == {4, 5}

    def test_graphsage_encoder_forward(self):
        encoder = GraphSAGEEncoder(in_dim=7, hidden_dim=16, out_dim=8)
        x = torch.randn(10, 7)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = encoder(x, edge_index)
        assert out.shape == (10, 8)

    def test_edge_classifier_forward(self):
        classifier = EdgeClassifier(emb_dim=8, edge_feat_dim=2, hidden_dim=16)
        b_emb = torch.randn(5, 8)
        s_emb = torch.randn(5, 8)
        edge_feats = torch.randn(5, 2)
        logits = classifier(b_emb, s_emb, edge_feats)
        assert logits.shape == (5,)

    def test_attach_gnn_embeddings_cold_start(self):
        df = pd.DataFrame({
            "order_id": ["O_1", "O_2"],
            "buyer_id": ["B_KNOWN", "B_UNKNOWN"],
            "seller_id": ["S_KNOWN", "S_UNKNOWN"],
        })
        buyer_embs = {"B_KNOWN": np.ones(16, dtype=np.float32)}
        seller_embs = {"S_KNOWN": np.ones(16, dtype=np.float32) * 2}

        df_aug, cols = attach_gnn_embeddings(df, buyer_embs, seller_embs, emb_dim=16)
        assert len(cols) == 32
        assert np.allclose(df_aug.loc[0, "gnn_buyer_emb_0"], 1.0)
        assert np.allclose(df_aug.loc[0, "gnn_seller_emb_0"], 2.0)
        # Cold start fallback to 0.0
        assert np.allclose(df_aug.loc[1, "gnn_buyer_emb_0"], 0.0)
        assert np.allclose(df_aug.loc[1, "gnn_seller_emb_0"], 0.0)


class TestMultimodalPhase4:
    def test_multimodal_similarity_bounds(self, pipeline):
        listings = pipeline["result"]["listings"].head(100)
        products = pipeline["catalog"]["products"]
        sims = compute_multimodal_similarity(listings, products)
        assert len(sims) == len(listings)
        assert (sims >= 0.05).all() and (sims <= 1.0).all()
        assert not bool(sims.isna().to_numpy().any())

    def test_multimodal_separation(self, pipeline):
        listings = pipeline["result"]["listings"]
        products = pipeline["catalog"]["products"]
        sims = compute_multimodal_similarity(listings, products)
        normal_mask = ~listings["is_fraudulent"].fillna(False)
        fraud_mask = listings["is_fraudulent"].fillna(False)

        if fraud_mask.any():
            avg_normal = sims[normal_mask].mean()
            avg_fraud = sims[fraud_mask].mean()
            assert avg_normal > avg_fraud

    def test_investigator_explanation_high_risk(self):
        features = {
            "multimodal_similarity_score": 0.22,
            "price_vs_base_price_ratio": 0.35,
            "seller_age_days_at_listing": 5,
            "seller_listings_before": 1,
        }
        res = explain_listing_risk(features, risk_score=0.88)
        assert res["risk_level"] == "HIGH"
        assert "Suspend" in res["recommended_action"]
        assert len(res["key_evidence"]) >= 2
        assert any("mismatch" in ev.lower() for ev in res["key_evidence"])

    def test_investigator_explanation_low_risk(self):
        features = {
            "multimodal_similarity_score": 0.89,
            "price_vs_base_price_ratio": 0.98,
            "seller_age_days_at_listing": 180,
            "seller_listings_before": 25,
        }
        res = explain_listing_risk(features, risk_score=0.12)
        assert res["risk_level"] == "LOW"
        assert "approval" in res["recommended_action"].lower()

