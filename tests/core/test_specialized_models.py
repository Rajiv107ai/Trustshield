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
from core.calibration import (
    optimize_dollar_weighted_cost_threshold,
    compute_cost_weighted_sample_weights,
    CapacityBudgetCalibrator,
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
    detect_dense_fraud_clusters,
    compute_collusion_indicators,
)
from core.specialized_fraud_detectors import (
    build_listing_features,
    build_return_features,
)

from multimodal_scoring import (
    compute_multimodal_similarity,
    explain_listing_risk,
    compute_image_dhash,
    compute_image_hamming_distance,
    scan_off_platform_contact,
    detect_catalog_clone,
)


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

    def test_bayesian_noisy_or_fusion_elevation(self):
        engine = AdvancedTrustEngine(fusion_mode="bayesian_noisy_or")
        # Low tabular risk (0.20) but severe return abuse (0.95)
        res = engine.score({"tabular_risk": 0.20, "return_risk": 0.95})
        assert res.calibrated_risk > 0.70
        assert res.decision in (OperationalDecision.HOLD, OperationalDecision.BLOCK)
        assert res.aggregation_method_used == "bayesian_noisy_or_fusion"

    def test_bayesian_fusion_monotonicity(self):
        engine = AdvancedTrustEngine(fusion_mode="bayesian_noisy_or")
        # High tabular risk (0.95) must never be downgraded by lower specialized signals (0.20)
        res = engine.score({"tabular_risk": 0.95, "multimodal_risk": 0.20})
        assert res.calibrated_risk >= 0.95
        assert res.decision == OperationalDecision.BLOCK

    def test_conformal_bounded_interval(self):
        conf = ConformalPredictor(alpha=0.10)
        probs = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
        labels = np.array([0, 0, 0, 1, 1, 1])
        conf.calibrate(probs, labels)

        low, high = conf.predict_interval(0.90)
        assert 0.0 <= low <= high <= 1.0
        assert high >= 0.90

        engine = AdvancedTrustEngine(conformal_predictor=conf)
        res = engine.score({"tabular_risk": 0.88})
        assert hasattr(res, "risk_interval")
        assert res.risk_interval[0] <= res.calibrated_risk <= res.risk_interval[1]
        assert "lower" in res.risk_bounds and "upper" in res.risk_bounds

    def test_conformal_risk_control(self):
        conf = ConformalPredictor()
        probs = np.array([0.05, 0.12, 0.18, 0.70, 0.85, 0.92, 0.96])
        labels = np.array([0, 0, 0, 1, 1, 1, 1])
        tau = conf.calibrate_conformal_risk_control(probs, labels, alpha_risk=0.50)
        assert 0.01 <= tau <= 0.99

    def test_dollar_weighted_cost_threshold(self):
        y_true = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.05, 0.15, 0.30, 0.75, 0.90])
        amounts = np.array([50.0, 50.0, 50.0, 500.0, 1000.0])
        metrics = optimize_dollar_weighted_cost_threshold(y_true, scores, amounts, fp_cost=20.0)
        assert "threshold" in metrics
        assert metrics["fraud_value_saved"] > 0
        assert metrics["value_recall_pct"] >= 50.0

    def test_compute_cost_weighted_sample_weights(self):
        y = np.array([0, 0, 1, 1])
        amounts = np.array([50.0, 50.0, 40.0, 80.0])
        weights = compute_cost_weighted_sample_weights(y, amounts, base_fp_cost=20.0, max_weight_multiplier=5.0)
        assert len(weights) == 4
        assert weights[0] == 1.0 and weights[1] == 1.0
        assert weights[3] > weights[2]
        assert weights[3] <= 5.0

    def test_capacity_budget_calibrator(self):
        calibrator = CapacityBudgetCalibrator(target_budgets=(0.02, 0.05, 0.10))
        np.random.seed(42)
        scores = np.random.uniform(0.0, 1.0, 1000)
        labels = (scores > 0.7).astype(int)
        amounts = np.random.uniform(20.0, 200.0, 1000)
        calibrator.fit(scores, labels, amounts)

        t_02 = calibrator.get_threshold(0.02)
        t_05 = calibrator.get_threshold(0.05)
        t_10 = calibrator.get_threshold(0.10)
        assert t_02 >= t_05 >= t_10
        assert 0.02 in calibrator.budget_metrics
        assert calibrator.budget_metrics[0.02]["budget_pct"] == 2.0





class _MockTrustResult:
    calibrated_risk = 0.88
    risk_interval = (0.82, 0.94)
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
        assert dossier.risk_interval == (0.82, 0.94)
        assert any("Conformal" in ev for ev in dossier.structured_evidence)
        assert dossier.grounding_verification_passed is True and len(dossier.structured_evidence) >= 3 and len(dossier.retrieved_policy_guidelines) > 0

    def test_hallucination_guard_detects_corruption(self):
        import dataclasses
        valid = RuleBasedInvestigationAgent().generate_dossier(self._TXN, _MockTrustResult())
        assert valid.grounding_verification_passed is True
        assert verify_dossier_grounding(dataclasses.replace(valid, fraud_risk_score=0.42), self._TXN, _MockTrustResult()) is False
        assert verify_dossier_grounding(dataclasses.replace(valid, operational_decision="ALLOW"), self._TXN, _MockTrustResult()) is False
        assert verify_dossier_grounding(dataclasses.replace(valid, structured_evidence=["Transaction Amount: INR 99999.00"]), self._TXN, _MockTrustResult()) is False
        assert verify_dossier_grounding(dataclasses.replace(valid, risk_interval=(0.10, 0.20)), self._TXN, _MockTrustResult()) is False





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

    def test_compute_collusion_indicators_synthetic(self):
        G = nx.Graph()
        G.add_edge("B1", "B2")  # B1 and B2 share infrastructure (cluster 0)

        sample_orders = pd.DataFrame([
            {"order_id": "O1", "buyer_id": "B1", "seller_id": "S1", "order_date": "2025-01-01"},
            {"order_id": "O2", "buyer_id": "B2", "seller_id": "S1", "order_date": "2025-01-02"},
            {"order_id": "O3", "buyer_id": "B3", "seller_id": "S2", "order_date": "2025-01-03"},
            {"order_id": "O4", "buyer_id": "B4", "seller_id": "S2", "order_date": "2025-01-04"},
        ])
        res = compute_collusion_indicators(sample_orders, G)
        assert "seller_buyer_shared_infra_ratio" in res.columns
        # For S1 second order (O2), prior order O1 was from linked B1 => shared ratio is 1.0
        s1_o2 = res[res["order_id"] == "O2"].iloc[0]
        assert s1_o2["seller_buyer_shared_infra_ratio"] == 1.0
        # For S2 second order (O4), prior order O3 was from unlinked B3 => shared ratio is 0.0
        s2_o4 = res[res["order_id"] == "O4"].iloc[0]
        assert s2_o4["seller_buyer_shared_infra_ratio"] == 0.0

    def test_detect_dense_fraud_clusters_topology(self):
        G = nx.Graph()
        # Clique K4: nodes 1,2,3,4 (all connected)
        for u in [1, 2, 3, 4]:
            for v in [1, 2, 3, 4]:
                if u < v:
                    G.add_edge(u, v)
        # 4-node Chain: nodes 5,6,7,8 (linear chain, max degree 2 < 3)
        G.add_edge(5, 6)
        G.add_edge(6, 7)
        G.add_edge(7, 8)

        orders = pd.DataFrame([
            {"order_id": f"O{i}", "buyer_id": i, "fraud_score": 0.80}
            for i in range(1, 9)
        ])
        clusters = detect_dense_fraud_clusters(G, orders, score_col="fraud_score", min_ring_size=3)
        assert len(clusters) == 2
        # K4 should be identified as CLIQUE with density 1.0
        clique_row = clusters[clusters["size"] == 4].iloc[0]
        assert clique_row["topology_type"] == "CLIQUE"
        assert clique_row["density"] == 1.0
        assert clique_row["cluster_threat_score"] > 0.70
        # 4-node Chain should be TREE_OR_CHAIN with density < 1.0
        chain_row = clusters[clusters["size"] == 4].iloc[1]
        assert chain_row["topology_type"] == "TREE_OR_CHAIN"
        assert chain_row["density"] < 1.0



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

    def test_scan_off_platform_contact(self):
        # WhatsApp + UPI detection test
        scam_text = "Urgent: Contact on WhatsApp wa.me/9876543210 or pay to merchant@okhdfcbank for direct order"
        findings = scan_off_platform_contact(scam_text)
        assert findings["has_violation"] is True
        assert findings["threat_score"] >= 0.80
        rules = [v["rule"] for v in findings["violations"]]
        assert "whatsapp_invite" in rules or "upi_vpa" in rules

        # Benign listing test
        clean_text = "Genuine cotton t-shirt with crew neck and breathable fabric."
        clean_findings = scan_off_platform_contact(clean_text)
        assert clean_findings["has_violation"] is False
        assert clean_findings["threat_score"] == 0.0

    def test_image_dhash_and_hamming(self):
        from PIL import Image
        # Create two identical small test images
        img1 = Image.new("RGB", (64, 64), color="red")
        img2 = Image.new("RGB", (64, 64), color="red")
        img3 = Image.new("RGB", (64, 64), color="blue")

        h1 = compute_image_dhash(img1)
        h2 = compute_image_dhash(img2)
        assert h1 is not None and len(h1) == 16
        assert compute_image_hamming_distance(h1, h2) == 0

        # Different images should have valid distance
        h3 = compute_image_dhash(img3)
        assert compute_image_hamming_distance(h1, h3) >= 0

    def test_detect_catalog_clone(self):
        target_h = "0000ffff0000ffff"
        ref_hashes = {
            "listing_101": "0000ffff0000ffff",  # identical (dist=0)
            "listing_102": "ffff0000ffff0000",  # completely different (dist=64)
        }
        res = detect_catalog_clone(target_h, ref_hashes, threshold_distance=5)
        assert res["is_clone"] is True
        assert len(res["matched_listings"]) == 1
        assert res["matched_listings"][0]["listing_id"] == "listing_101"

    def test_investigator_explanation_forensics(self):
        res = explain_listing_risk(
            {
                "multimodal_similarity_score": 0.85,
                "price_vs_base_price_ratio": 1.0,
                "seller_age_days_at_listing": 100,
                "seller_listings_before": 10,
                "title": "Cheap phone wa.me/1234567890 pay outside",
            },
            risk_score=0.25,
        )
        assert res["risk_level"] == "HIGH"
        assert any("Off-platform" in ev for ev in res["key_evidence"])



