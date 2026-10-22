"""Comprehensive automated test suite for TrustShield Phase 2 Architecture:
- Part 1: Advanced Trust Engine & Conformal Uncertainty (calibrated risk, conformal sets)
- Part 2: GenAI Forensic Investigation Agent & Policy Rules (evidence grounding, hallucination guard)
- Part 3: Graph Topological Features, Fraud Ring Detection & Specialized XGBoost Models
- Part 4: Multimodal Scoring & Listing Intelligence
"""

import numpy as np
import pandas as pd
import pytest
torch = pytest.importorskip("torch", reason="optional Phase 2 CLIP/GNN test dependency is not installed")
import networkx as nx


from trustshield_project.advanced_trust_engine import (
    AdvancedTrustEngine,
    ConformalPredictor,
    OperationalDecision,
)
from trustshield_project.policy_rules import ForensicRAGIndex
from trustshield_project.dossier_generator import (
    GenAIInvestigationAgent,
    RuleBasedInvestigationAgent,
    verify_dossier_grounding,
)
from graph_features import (
    build_relationship_graph,
    compute_relationship_features,
    build_monthly_snapshots,
    attach_snapshot_features,
    add_edge_weight_before,
    detect_fraud_rings,
)
from core.specialized_fraud_detectors import (
    build_listing_features,
    build_return_features,
)

from multimodal_scoring import compute_multimodal_similarity, explain_listing_risk


# ===========================================================================
# Part 5 & 6: Advanced Trust Engine & Conformal Uncertainty Tests
# ===========================================================================

class TestAdvancedTrustEngineAndUncertainty:
    def test_conformal_predictor_coverage(self):
        conf = ConformalPredictor(alpha=0.10)
        conf.calibrate(np.array([0.05, 0.10, 0.15, 0.85, 0.90, 0.95]), np.array([0, 0, 0, 1, 1, 1]))
        assert 0 in conf.predict_set(0.02) and 1 in conf.predict_set(0.98)

    def test_calibrated_probability_pass_through(self):
        # P0 Fix: High-risk calibrated probability must never be downgraded by heuristics
        engine = AdvancedTrustEngine()
        res_high = engine.score({"tabular_risk": 0.95, "graph_risk": 0.0}, model_probability=0.95)
        assert res_high.calibrated_risk == 0.95
        assert res_high.decision == OperationalDecision.BLOCK

    def test_advanced_trust_engine_scoring(self):
        res = AdvancedTrustEngine().score({"tabular_risk": 0.88, "graph_risk": 0.90})
        assert res.calibrated_risk > 0.80 and res.decision in (OperationalDecision.HOLD, OperationalDecision.BLOCK)
        assert res.predictive_uncertainty < 0.60 and res.detector_disagreement < 0.10

    def test_cost_matrix_and_reason_codes(self):
        engine = AdvancedTrustEngine()
        # Low risk with shared hardware signal surfaced as reason code, not corrupting probability
        res = engine.score({"tabular_risk": 0.10, "graph_risk": 0.85}, model_probability=0.10)
        assert res.calibrated_risk == 0.10
        assert any("GRAPH_RISK" in r for r in res.reason_codes)



class _MockTrustResult:
    calibrated_risk = 0.88
    decision = OperationalDecision.BLOCK
    confidence = 0.92
    detector_disagreement = 0.12
    conformal_prediction_set = {1}
    reason_codes = ["HIGH_RETURN_VELOCITY", "SUSPICIOUS_HARDWARE_COLLISION"]


class TestGenAIInvestigationAgent:
    _TXN = {"order_id": "ORD_12345", "amount": 250.0, "buyer_return_rate_before": 0.65, "cold_start": True, "price_vs_base_price_ratio": 0.45, "share_degree": 3, "share_component_size": 5}

    def test_rag_search(self):
        chunks = ForensicRAGIndex().search("hardware device collision and shared fingerprints", top_k=2)
        assert len(chunks) > 0 and "POL-01" in [c.chunk_id for c in chunks]

    def test_dossier_generation_and_hallucination_guard(self):
        dossier = GenAIInvestigationAgent().generate_dossier(self._TXN, _MockTrustResult())
        assert dossier.case_id == "CASE_ORD_12345" and dossier.fraud_risk_score == 0.88 and dossier.operational_decision == "BLOCK"
        assert dossier.grounding_verification_passed is True and len(dossier.structured_evidence) >= 3 and len(dossier.retrieved_policy_guidelines) > 0

    def test_hallucination_guard_detects_corruption(self):
        import dataclasses
        valid = RuleBasedInvestigationAgent().generate_dossier(self._TXN, _MockTrustResult())
        assert valid.grounding_verification_passed is True
        assert verify_dossier_grounding(dataclasses.replace(valid, fraud_risk_score=0.42), self._TXN, _MockTrustResult()) is False
        assert verify_dossier_grounding(dataclasses.replace(valid, operational_decision="ALLOW"), self._TXN, _MockTrustResult()) is False
        assert verify_dossier_grounding(dataclasses.replace(valid, structured_evidence=["Transaction Amount: INR 99999.00"]), self._TXN, _MockTrustResult()) is False





# ===========================================================================
# Part 12: Graph Topological Features, Specialized Models & GNN Encoders
# ===========================================================================

class TestGraphFeatures:
    def test_build_relationship_graph(self, pipeline):
        G = build_relationship_graph(pipeline["base"]["address_sharing_log"], pipeline["base"]["device_sharing_log"])
        assert isinstance(G, nx.Graph) and G.number_of_nodes() > 0 and G.number_of_edges() > 0

    def test_compute_relationship_features(self, pipeline):
        G = build_relationship_graph(pipeline["base"]["address_sharing_log"], pipeline["base"]["device_sharing_log"])
        buyers = list(pipeline["txn"]["buyers"]["buyer_id"].unique()[:20])
        feat_df = compute_relationship_features(G, buyers)
        assert len(feat_df) == 20 and set(feat_df.columns) == {"buyer_id", "share_degree", "share_component_size"}
        assert (feat_df["share_degree"] >= 0).all() and (feat_df["share_component_size"] >= 1).all()

    def test_monthly_snapshots_and_attachment(self, pipeline):
        orders = pipeline["result"]["orders"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        snapshots, months = build_monthly_snapshots(orders, orders["order_date"].min())
        attached = attach_snapshot_features(orders.head(50), snapshots, months)
        assert len(attached) == 50
        for col in ["buyer_seller_degree", "buyer_pagerank", "seller_buyer_degree", "seller_pagerank", "seller_buyer_concentration_hhi"]:
            assert col in attached.columns and not bool(attached[col].isna().to_numpy().any())

    def test_add_edge_weight_before(self, pipeline):
        res = add_edge_weight_before(pipeline["result"]["orders"].head(50).copy())
        assert "buyer_seller_edge_weight_before" in res.columns and (res["buyer_seller_edge_weight_before"] >= 0).all()

    def test_detect_fraud_rings(self, pipeline):
        G = build_relationship_graph(pipeline["base"]["address_sharing_log"], pipeline["base"]["device_sharing_log"])
        rings = detect_fraud_rings(G, pipeline["df"].assign(fraud_score=0.45), score_col="fraud_score", min_ring_size=2)
        assert isinstance(rings, pd.DataFrame)
        if len(rings) > 0:
            assert "ring_id" in rings.columns and (rings["size"] >= 2).all() and (rings["avg_risk_score"] >= 0.0).all()

    def test_detect_fraud_rings_leakage_assertion(self, pipeline):
        G = build_relationship_graph(pipeline["base"]["address_sharing_log"], pipeline["base"]["device_sharing_log"])
        with pytest.raises(AssertionError, match="Leakage"):
            detect_fraud_rings(G, pipeline["df"].copy(), score_col="is_fraudulent")


class TestSpecializedModels:
    def test_build_listing_features(self, pipeline):
        df, cols = build_listing_features(pipeline["result"]["listings"], pipeline["catalog"]["sellers"], pipeline["catalog"]["products"])
        assert len(df) == len(pipeline["result"]["listings"]) and set(cols).issubset(set(df.columns)) and not bool(df[cols].isna().to_numpy().any())

    def test_build_return_features(self, pipeline):
        df, cols = build_return_features(pipeline["result"]["returns"], pipeline["result"]["orders"], pipeline["txn"]["buyers"], pipeline["catalog"]["sellers"])
        assert len(df) == len(pipeline["result"]["returns"]) and set(cols).issubset(set(df.columns)) and not bool(df[cols].isna().to_numpy().any())


class TestMultimodalPhase4:
    def test_multimodal_similarity_bounds(self, pipeline):
        sims = compute_multimodal_similarity(pipeline["result"]["listings"].head(100), pipeline["catalog"]["products"])
        assert len(sims) == 100 and (sims >= 0.05).all() and (sims <= 1.0).all() and not bool(sims.isna().to_numpy().any())

    def test_multimodal_separation(self, pipeline):
        listings = pipeline["result"]["listings"]
        sims = compute_multimodal_similarity(listings, pipeline["catalog"]["products"])
        fraud_mask = listings["is_fraudulent"].fillna(False)
        if fraud_mask.any():
            assert sims[~fraud_mask].mean() > sims[fraud_mask].mean()

    def test_investigator_explanation_high_risk(self):
        res = explain_listing_risk({"multimodal_similarity_score": 0.22, "price_vs_base_price_ratio": 0.35, "seller_age_days_at_listing": 5, "seller_listings_before": 1}, risk_score=0.88)
        assert res["risk_level"] == "HIGH" and "Suspend" in res["recommended_action"] and len(res["key_evidence"]) >= 2 and any("mismatch" in ev.lower() for ev in res["key_evidence"])

    def test_investigator_explanation_low_risk(self):
        res = explain_listing_risk({"multimodal_similarity_score": 0.89, "price_vs_base_price_ratio": 0.98, "seller_age_days_at_listing": 180, "seller_listings_before": 25}, risk_score=0.12)
        assert res["risk_level"] == "LOW" and "approval" in res["recommended_action"].lower()


