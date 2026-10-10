"""Comprehensive automated test suite for TrustShield Phase 2 Research Architecture:
- Part 1: Heterogeneous GNN (PyG HeteroData, explicit typed edges, temporal cutoff)
- Part 2: Temporal GNN (Continuous Time2Vec harmonic encoding, temporal guard assertion)
- Part 3 & 4: Multimodal CLIP & FAISS Vector Search (near-duplicates, cross-seller image reuse)
- Part 5 & 6: Advanced Trust Engine & Conformal Uncertainty (stacking meta-learner, conformal sets)
- Part 7: Advanced Fraud Ring Intelligence (community burstiness, HHI, cluster risk classification)
- Part 8: Neo4j Graph Investigator (Cypher query generation, in-memory traversal)
- Part 9 & 10: GenAI Forensic Investigation Agent & RAG (evidence grounding, hallucination guard)
- Part 11: MLOps Experiment Lineage Tracker (parameter logging, SHA256 artifact hashing)
"""

import numpy as np
import pandas as pd
import pytest
import torch
import networkx as nx

from trustshield_project.hetero_gnn import (
    build_hetero_graph,
    HeteroGNN,
    extract_hetero_embeddings,
    PYG_AVAILABLE,
)
from trustshield_project.temporal_gnn import (
    Time2Vec,
    TemporalGNN,
)

from trustshield_project.multimodal_clip_faiss import (
    MultimodalFAISSIndex,
    extract_multimodal_listing_features,
)
from trustshield_project.advanced_trust_engine import (
    AdvancedTrustEngine,
    ConformalPredictor,
    StackingRiskMetaLearner,
    OperationalDecision,
)
from trustshield_project.advanced_ring_intelligence import (
    discover_candidate_communities,
    analyze_candidate_community,
)
from trustshield_project.neo4j_investigator import Neo4jInvestigator
from trustshield_project.investigation_rag import ForensicRAGIndex
from trustshield_project.investigation_agent import (
    GenAIInvestigationAgent,
    RuleBasedInvestigationAgent,
    verify_dossier_grounding,
)
from trustshield_project.mlops_pipeline import ExperimentTracker


# ===========================================================================
# Part 1: Heterogeneous GNN Tests
# ===========================================================================

@pytest.mark.skipif(not PYG_AVAILABLE, reason="torch_geometric not available")
class TestHeteroGNN:
    @pytest.fixture
    def mock_data(self):
        buyers = pd.DataFrame({
            "buyer_id": ["B1", "B2", "B3"],
            "device_id": ["D1", "D1", "D2"],
            "address_id": ["A1", "A2", "A1"],
            "orders_before": [5, 2, 8],
            "returns_before": [1, 0, 2],
        })
        sellers = pd.DataFrame({
            "seller_id": ["S1", "S2"],
            "total_listings_before": [20, 5],
            "seller_age_days": [200, 30],
        })
        orders = pd.DataFrame({
            "order_id": ["O1", "O2", "O3"],
            "buyer_id": ["B1", "B2", "B3"],
            "seller_id": ["S1", "S1", "S2"],
            "order_date": ["2024-01-10", "2024-01-20", "2024-02-05"],
        })
        dev_log = pd.DataFrame({
            "seller_id": ["S1"],
            "shared_device_id": ["D1"],
            "first_seen_date": ["2024-01-05"],
        })
        addr_log = pd.DataFrame({
            "seller_id": ["S2"],
            "shared_address_id": ["A1"],
            "first_seen_date": ["2024-01-15"],
        })
        return buyers, sellers, orders, dev_log, addr_log

    def test_build_hetero_graph_structure(self, mock_data):
        buyers, sellers, orders, dev_log, addr_log = mock_data
        data, id_maps = build_hetero_graph(
            buyers, sellers, orders, dev_log, addr_log, cutoff_date="2024-02-01"
        )
        assert "buyer" in data.node_types
        assert "seller" in data.node_types
        assert "device" in data.node_types
        assert "address" in data.node_types
        assert len(id_maps["buyer"]) == 3
        assert len(id_maps["seller"]) == 2

    def test_hetero_gnn_forward_pass(self, mock_data):
        buyers, sellers, orders, dev_log, addr_log = mock_data
        data, id_maps = build_hetero_graph(buyers, sellers, orders, dev_log, addr_log)
        model = HeteroGNN(data.metadata(), hidden_channels=8, out_channels=8)
        b_embs, s_embs = extract_hetero_embeddings(model, data, id_maps)
        assert len(b_embs) == 3
        assert len(s_embs) == 2
        assert b_embs["B1"].shape == (8,)


# ===========================================================================
# Part 2: Temporal GNN Tests
# ===========================================================================

class TestTemporalGNN:
    def test_time2vec_encoding(self):
        t2v = Time2Vec(out_dim=8)
        delta_t = torch.tensor([[0.5], [1.0], [5.2]], dtype=torch.float)
        emb = t2v(delta_t)
        assert emb.shape == (3, 8)
        assert not torch.isnan(emb).any()

    def test_temporal_leakage_guard_raises_error(self):
        t2v = Time2Vec(out_dim=8)
        # Negative delta_t indicates future information!
        bad_delta_t = torch.tensor([[-1.5], [2.0]], dtype=torch.float)
        with pytest.raises(AssertionError, match="Temporal leakage detected"):
            t2v(bad_delta_t)

    def test_temporal_gnn_forward(self):
        model = TemporalGNN(in_features=4, hidden_features=8, out_features=8, time_dim=8)
        h = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        delta_t = torch.tensor([[0.1], [0.5], [1.2], [3.0]], dtype=torch.float)

        logits, embs = model(h, edge_index, delta_t)
        assert logits.shape == (5, 1)
        assert embs.shape == (5, 8)


# ===========================================================================
# Part 3 & 4: Multimodal CLIP + FAISS Tests
# ===========================================================================

class TestMultimodalCLIPAndFAISS:
    def test_faiss_index_and_near_duplicates(self):
        index = MultimodalFAISSIndex(dim=8)
        # Add 3 seller listings
        embs = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # L1 (Seller S1)
            [0.99, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],     # L2 (Seller S2 - Near duplicate!)
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # L3 (Seller S3 - Distinct)
        ], dtype=np.float32)

        index.add_listings(["L1", "L2", "L3"], ["S1", "S2", "S3"], embs)

        # Query L1 from Seller S1
        query_emb = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        feats = index.query_similarity_features(query_emb, ["S1"], k=2)

        assert len(feats) == 1
        f = feats[0]
        assert f["faiss_max_image_similarity"] >= 0.95
        assert f["faiss_near_duplicate_count"] >= 1
        assert f["faiss_cross_seller_image_reuse"] == 1.0  # Shared with Seller S2!

    def test_extract_multimodal_listing_features(self):
        df = pd.DataFrame({
            "listing_id": ["L1", "L2"],
            "seller_id": ["S1", "S2"],
        })
        img_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        txt_embs = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)

        out = extract_multimodal_listing_features(df, img_embs, txt_embs)
        assert "clip_image_text_similarity" in out.columns
        assert "clip_semantic_mismatch" in out.columns
        assert out["clip_image_text_similarity"].iloc[0] > 0.80


# ===========================================================================
# Part 5 & 6: Advanced Trust Engine & Conformal Uncertainty Tests
# ===========================================================================

class TestAdvancedTrustEngineAndUncertainty:
    def test_conformal_predictor_coverage(self):
        conf = ConformalPredictor(alpha=0.10)
        probs = np.array([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])
        labels = np.array([0, 0, 0, 1, 1, 1])
        conf.calibrate(probs, labels)

        # Low risk -> {0}
        set_low = conf.predict_set(0.02)
        assert 0 in set_low

        # High risk -> {1}
        set_high = conf.predict_set(0.98)
        assert 1 in set_high

    def test_stacking_meta_learner(self):
        meta = StackingRiskMetaLearner()
        X = pd.DataFrame({
            "tabular_risk": [0.05, 0.10, 0.15, 0.20, 0.25, 0.75, 0.80, 0.85, 0.90, 0.95],
            "graph_risk": [0.05, 0.08, 0.12, 0.18, 0.22, 0.70, 0.78, 0.82, 0.88, 0.92],
        })
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        meta.fit(X, y)

        r_low = meta.predict_risk({"tabular_risk": 0.1, "graph_risk": 0.1})
        r_high = meta.predict_risk({"tabular_risk": 0.9, "graph_risk": 0.9})
        assert r_low < 0.35
        assert r_high > 0.65
        assert r_low < r_high

    def test_advanced_trust_engine_scoring(self):
        engine = AdvancedTrustEngine()
        comp_scores = {
            "tabular_risk": 0.85,
            "graph_risk": 0.90,
            "hetero_gnn_risk": 0.88,
        }
        res = engine.score(comp_scores)
        assert res.calibrated_risk > 0.80
        assert res.decision in (OperationalDecision.HOLD, OperationalDecision.BLOCK)
        assert res.predictive_uncertainty < 0.60
        assert res.detector_disagreement < 0.10


# ===========================================================================
# Part 7: Advanced Fraud Ring Intelligence Tests
# ===========================================================================

class TestAdvancedRingIntelligence:
    def test_community_discovery_and_burstiness(self):
        # Create a small multi-graph
        G = nx.MultiGraph()
        G.add_edge("B1", "D1", relation="uses_device")
        G.add_edge("B2", "D1", relation="uses_device")
        G.add_edge("B3", "D1", relation="uses_device")

        communities = discover_candidate_communities(G, min_size=3)
        assert len(communities) >= 1
        comm = communities[0]

        orders = pd.DataFrame({
            "order_id": ["O1", "O2", "O3"],
            "buyer_id": ["B1", "B2", "B3"],
            "seller_id": ["S1", "S1", "S1"],
            "order_date": ["2024-01-01 10:00:00", "2024-01-01 10:05:00", "2024-01-01 10:10:00"],
        })
        returns = pd.DataFrame(columns=["order_id"])

        report = analyze_candidate_community(comm, orders, returns, G, cluster_id="RING_01")
        assert report.member_count == len(comm)
        assert report.hardware_sharing_density > 0.50
        assert report.merchant_concentration_hhi == 1.0  # All funneled to Seller S1
        assert report.cluster_classification in ("suspicious_cluster", "high_risk_candidate_ring")


# ===========================================================================
# Part 8: Neo4j Investigator Tests
# ===========================================================================

class TestNeo4jInvestigator:
    def test_cypher_generation(self):
        cypher = Neo4jInvestigator.cypher_shared_device_query("SELLER_42", days=30)
        assert "MATCH (s:Seller" in cypher
        assert "SELLER_42" in cypher

    def test_in_memory_simulation(self):
        G = nx.MultiGraph()
        G.add_node("S1")
        G.add_node("DEV_99")
        G.add_node("B1")
        G.add_edge("S1", "DEV_99", relation="uses_device")
        G.add_edge("DEV_99", "B1", relation="uses_device")

        inv = Neo4jInvestigator(G)
        results = inv.query_buyers_connected_via_device_in_memory("S1")
        assert len(results) == 1
        assert results[0]["buyer_id"] == "B1"
        assert results[0]["shared_device_id"] == "DEV_99"


# ===========================================================================
# Part 9 & 10: GenAI Investigation Agent & RAG Tests
# ===========================================================================

class TestGenAIInvestigationAgent:
    def test_rag_search(self):
        rag = ForensicRAGIndex()
        chunks = rag.search("hardware device collision and shared fingerprints", top_k=2)
        assert len(chunks) > 0
        assert "POL-01" in [c.chunk_id for c in chunks]

    def test_dossier_generation_and_hallucination_guard(self):
        agent = GenAIInvestigationAgent()
        txn = {
            "order_id": "ORD_12345",
            "amount": 250.0,
            "buyer_return_rate_before": 0.65,
            "cold_start": True,
            "price_vs_base_price_ratio": 0.45,
            "share_degree": 3,
            "share_component_size": 5,
        }

        # Mock Trust Engine Result
        class MockTrustResult:
            calibrated_risk = 0.88
            decision = OperationalDecision.BLOCK
            confidence = 0.92
            detector_disagreement = 0.12
            conformal_prediction_set = {1}
            reason_codes = ["HIGH_RETURN_VELOCITY", "SUSPICIOUS_HARDWARE_COLLISION"]

        dossier = agent.generate_dossier(txn, MockTrustResult())
        assert dossier.case_id == "CASE_ORD_12345"
        assert dossier.fraud_risk_score == 0.88  # Matches model exactly
        assert dossier.operational_decision == "BLOCK"
        assert dossier.grounding_verification_passed is True
        assert len(dossier.structured_evidence) >= 3
        assert len(dossier.retrieved_policy_guidelines) > 0

    def test_hallucination_guard_detects_corruption(self):
        import dataclasses
        agent = RuleBasedInvestigationAgent()
        txn = {
            "order_id": "ORD_12345",
            "amount": 250.0,
            "buyer_return_rate_before": 0.65,
            "cold_start": True,
            "price_vs_base_price_ratio": 0.45,
            "share_degree": 3,
            "share_component_size": 5,
        }

        class MockTrustResult:
            calibrated_risk = 0.88
            decision = OperationalDecision.BLOCK
            confidence = 0.92
            detector_disagreement = 0.12
            conformal_prediction_set = {1}
            reason_codes = ["HIGH_RETURN_VELOCITY"]

        valid_dossier = agent.generate_dossier(txn, MockTrustResult())
        assert valid_dossier.grounding_verification_passed is True

        # 1. Corrupted risk score
        corrupted_risk = dataclasses.replace(valid_dossier, fraud_risk_score=0.42)
        assert verify_dossier_grounding(corrupted_risk, txn, MockTrustResult()) is False

        # 2. Corrupted decision
        corrupted_decision = dataclasses.replace(valid_dossier, operational_decision="ALLOW")
        assert verify_dossier_grounding(corrupted_decision, txn, MockTrustResult()) is False

        # 3. Corrupted evidence amount
        corrupted_evidence = dataclasses.replace(valid_dossier, structured_evidence=["Transaction Amount: INR 99999.00"])
        assert verify_dossier_grounding(corrupted_evidence, txn, MockTrustResult()) is False


# ===========================================================================
# Part 11: MLOps Lineage Tracking Tests
# ===========================================================================

class TestMLOpsPipeline:
    def test_experiment_tracker(self, tmp_path):
        tracker = ExperimentTracker(registry_dir=tmp_path / "registry")
        test_model_file = tmp_path / "model.bin"
        test_model_file.write_bytes(b"HeteroGNN_weights_v1")

        run = tracker.log_run(
            experiment_name="HeteroGNN_vs_Baseline",
            model_name="HeteroGNN",
            model_version="2.0.0",
            parameters={"hidden_dim": 16, "lr": 0.01},
            metrics={"roc_auc": 0.845, "pr_auc": 0.590},
            artifacts={"model_weights": test_model_file},
        )

        assert run.model_name == "HeteroGNN"
        assert "model_weights" in run.artifact_hashes
        assert len(run.artifact_hashes["model_weights"]) == 64  # SHA256 length

        runs = tracker.list_runs()
        assert len(runs) == 1
        assert runs[0]["metrics"]["roc_auc"] == 0.845
