"""Comprehensive regression and contract tests for Phase 3.1 remediation.

Covers:
- Task 1: Serving-model selection (default phase3, explicit phase5-hybrid, invalid model 400, config 500, cold-start)
- Task 2: Separation of raw probability, calibrated probability, policy score, and decision
- Task 3: Conformal prediction truthfulness (authoritative server set, calibration status, q_hat)
- Task 4: Listing-model 0.85 contract preservation
"""

import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app, store

VALID_API_KEY = "dev-operator-api-key-32chars-long"
AUTH_HEADERS = {"X-API-Key": VALID_API_KEY}


@pytest.fixture(scope="module")
def client():
    store.load()
    with TestClient(app, headers=AUTH_HEADERS) as c:
        yield c


class TestServingModelSelection:
    """Task 1 tests: Default routing to Phase 3, explicit selection, error handling."""

    def test_default_serving_model_is_phase3(self, client):
        """When no model_version is passed, default serving model must be Phase 3."""
        # Ensure env var is unset or defaults to phase3
        orig_env = os.environ.get("TRUSTSHIELD_SERVING_MODEL")
        try:
            if "TRUSTSHIELD_SERVING_MODEL" in os.environ:
                del os.environ["TRUSTSHIELD_SERVING_MODEL"]

            resp = client.post("/transaction/score", json={
                "order_id": "ORD_DEFAULT_ROUTING",
                "amount": 100.0,
                "base_price": 100.0,
                "category_median_price": 95.0,
                "buyer_orders_before": 5,
                "buyer_returns_before": 1,
                "buyer_age_days": 60.0,
                "seller_age_days": 180.0,
                "seller_total_listings_before": 20,
            })
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["model_version"] == "phase3"
            assert "tabular+graph" in data["model_used"]
            assert data["scoring_mode"] == "production_model_phase3_combined"
        finally:
            if orig_env is not None:
                os.environ["TRUSTSHIELD_SERVING_MODEL"] = orig_env

    def test_explicit_phase5_selection_via_body(self, client):
        """Authenticated caller requesting phase5-hybrid in body must receive Phase 5 model scoring."""
        resp = client.post("/transaction/score", json={
            "order_id": "ORD_EXPLICIT_P5_BODY",
            "model_version": "phase5-hybrid",
            "amount": 120.0,
            "base_price": 120.0,
            "category_median_price": 100.0,
            "buyer_orders_before": 10,
            "buyer_returns_before": 1,
            "buyer_age_days": 90.0,
            "seller_age_days": 200.0,
            "seller_total_listings_before": 30,
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["model_version"] == "phase5-hybrid"
        assert "Hybrid GNN" in data["model_used"]
        assert data["scoring_mode"] == "production_model_phase5_hybrid"

    def test_explicit_phase5_selection_via_header(self, client):
        """Authenticated caller requesting phase5-hybrid via X-TrustShield-Model-Version must receive Phase 5."""
        headers = {**AUTH_HEADERS, "X-TrustShield-Model-Version": "phase5-hybrid"}
        resp = client.post("/transaction/score", json={
            "order_id": "ORD_EXPLICIT_P5_HEADER",
            "amount": 120.0,
            "base_price": 120.0,
            "category_median_price": 100.0,
            "buyer_orders_before": 10,
            "buyer_returns_before": 1,
            "buyer_age_days": 90.0,
            "seller_age_days": 200.0,
            "seller_total_listings_before": 30,
        }, headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["model_version"] == "phase5-hybrid"
        assert "Hybrid GNN" in data["model_used"]

    def test_invalid_model_version_in_body_returns_400(self, client):
        """Requesting an invalid model name in the body must return 400 Bad Request."""
        resp = client.post("/transaction/score", json={
            "order_id": "ORD_INVALID_MODEL",
            "model_version": "nonexistent_model_v9",
            "amount": 100.0,
        })
        assert resp.status_code == 400
        assert "Invalid model version requested" in resp.json()["detail"]

    def test_invalid_model_version_in_header_returns_400(self, client):
        """Requesting an invalid model name in the header must return 400 Bad Request."""
        headers = {**AUTH_HEADERS, "X-TrustShield-Model-Version": "fake_gnn"}
        resp = client.post("/transaction/score", json={
            "order_id": "ORD_INVALID_HEADER",
            "amount": 100.0,
        }, headers=headers)
        assert resp.status_code == 400
        assert "Invalid model version requested" in resp.json()["detail"]

    def test_invalid_server_env_configuration_returns_500(self, client):
        """If TRUSTSHIELD_SERVING_MODEL is set to an unsupported string, server returns 500."""
        orig_env = os.environ.get("TRUSTSHIELD_SERVING_MODEL")
        try:
            os.environ["TRUSTSHIELD_SERVING_MODEL"] = "unsupported_architecture"
            resp = client.post("/transaction/score", json={
                "order_id": "ORD_BAD_CONFIG",
                "amount": 100.0,
            })
            assert resp.status_code == 500
            assert "Invalid server configuration for TRUSTSHIELD_SERVING_MODEL" in resp.json()["detail"]
        finally:
            if orig_env is not None:
                os.environ["TRUSTSHIELD_SERVING_MODEL"] = orig_env
            else:
                os.environ.pop("TRUSTSHIELD_SERVING_MODEL", None)

    def test_cold_start_entity_behavior_phase3(self, client):
        """Cold start entity under Phase 3 must have cold_start=True and appropriate reason code."""
        resp = client.post("/transaction/score", json={
            "order_id": "ORD_COLD_P3",
            "model_version": "phase3",
            "amount": 50.0,
            "buyer_orders_before": 0,
            "buyer_age_days": 1.0,
            "seller_age_days": 2.0,
            "seller_total_listings_before": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["cold_start"] is True
        assert "COLD_START_INSUFFICIENT_HISTORY" in data["reason_codes"]
        assert data["evidence_availability"]["gnn_embeddings"] is False

    def test_cold_start_entity_behavior_phase5(self, client):
        """Cold start entity under Phase 5 must gracefully report embedding unavailability."""
        resp = client.post("/transaction/score", json={
            "order_id": "ORD_COLD_P5",
            "model_version": "phase5-hybrid",
            "buyer_id": "BUYER_TOTALLY_NEW_9999",
            "seller_id": "SELLER_TOTALLY_NEW_9999",
            "amount": 50.0,
            "buyer_orders_before": 0,
            "buyer_age_days": 1.0,
            "seller_age_days": 2.0,
            "seller_total_listings_before": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["cold_start"] is True
        assert data["infrastructure_sources"]["buyer_embedding"] in ["not_found", "unavailable"]
        assert data["infrastructure_sources"]["seller_embedding"] in ["not_found", "unavailable"]


class TestScoringSemanticsSeparation:
    """Task 2 tests: Clear separation of raw score, calibrated probability, policy score, decision."""

    def test_distinct_scoring_fields_present_and_valid(self, client):
        """Response must distinctly provide raw probability, calibrated probability, policy score, and decision."""
        resp = client.post("/transaction/score", json={
            "order_id": "ORD_SEMANTICS_01",
            "amount": 80.0,
            "base_price": 100.0,
            "category_median_price": 95.0,
            "buyer_orders_before": 15,
            "buyer_returns_before": 1,
            "buyer_age_days": 120.0,
            "seller_age_days": 300.0,
            "seller_total_listings_before": 50,
        })
        assert resp.status_code == 200
        data = resp.json()

        # 1. Raw uncalibrated score
        assert "raw_fraud_probability" in data
        assert 0.0 <= data["raw_fraud_probability"] <= 1.0

        # 2. Statistical calibrated probability
        assert "calibrated_fraud_probability" in data
        assert 0.0 <= data["calibrated_fraud_probability"] <= 1.0

        # 3. Composite operational policy score
        assert "policy_risk_score" in data
        assert 0.0 <= data["policy_risk_score"] <= 1.0

        # 4. Backward-compatible policy score aliases
        assert data["canonical_trust_engine_risk"] == data["policy_risk_score"]
        assert data["overall_fraud_probability"] == data["policy_risk_score"]

        # 5. Operational decision
        assert data["decision"] in ["ALLOW", "REVIEW", "HOLD", "BLOCK"]
        assert 0.0 <= data["trust_score"] <= 100.0

    def test_high_risk_transaction_routing(self, client):
        """Severe risk signals must trigger elevated policy score and non-ALLOW decision."""
        resp = client.post("/transaction/score", json={
            "order_id": "ORD_HIGH_RISK_POLICY",
            "amount": 1000.0,
            "base_price": 100.0,
            "category_median_price": 95.0,
            "buyer_orders_before": 1,
            "buyer_returns_before": 1,
            "share_degree": 8.0,
            "share_component_size": 15.0,
            "multimodal_similarity_score": 0.05,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["policy_risk_score"] > 0.40
        assert data["decision"] in ["REVIEW", "HOLD", "BLOCK"]
        assert data["risk_label"] in ["medium", "high"]


class TestConformalPredictionTruthfulness:
    """Task 3 tests: Authoritative server conformal set, calibration status, and quantile."""

    def test_conformal_fields_in_response(self, client):
        """Transaction score response must include authoritative conformal fields."""
        resp = client.post("/transaction/score", json={
            "order_id": "ORD_CONFORMAL_PROBE",
            "amount": 100.0,
        })
        assert resp.status_code == 200
        data = resp.json()

        assert "conformal_prediction_set" in data
        assert isinstance(data["conformal_prediction_set"], list)
        assert set(data["conformal_prediction_set"]).issubset({0, 1})
        assert len(data["conformal_prediction_set"]) >= 1

        assert "conformal_is_calibrated" in data
        assert isinstance(data["conformal_is_calibrated"], bool)

        assert "conformal_coverage_guarantee" in data
        assert data["conformal_coverage_guarantee"] == 0.95

        if data["conformal_is_calibrated"]:
            assert "conformal_q_hat" in data
            assert data["conformal_q_hat"] is not None
            assert 0.0 < data["conformal_q_hat"] < 1.0


class TestListingModelContractPreservation:
    """Task 4 tests: Retain 0.85 similarity default and 5-feature contract."""

    def test_listing_model_085_default_preserved(self, client):
        """Missing similarity score must default to 0.85 without regression."""
        resp = client.post("/listing/analyze", json={
            "listing_id": "LIST_CONTRACT_TEST",
            "price": 100.0,
            "base_price": 100.0,
            "category_median_price": 95.0,
            "seller_age_days_at_listing": 120,
            "seller_listings_before": 15,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["multimodal_similarity_score"] == 0.85
        assert 0.0 <= data["fake_listing_probability"] <= 1.0


class TestStreamProbabilitySemantics:
    """Phase 3.3 tests: SSE stream event probability semantics and field separation."""

    def test_stream_event_schema_separates_calibrated_and_policy_risk(self):
        """StreamTransactionEvent must support distinct calibrated_risk and policy_risk_score."""
        from backend.schemas import StreamTransactionEvent

        evt = StreamTransactionEvent(
            event_id="EVT_TEST_01",
            timestamp="12:00:00",
            order_id="ORD_TEST_01",
            buyer_id="BUYER_01",
            seller_id="SELLER_01",
            amount=150.0,
            risk_score=0.10,
            calibrated_risk=0.04,
            policy_risk_score=0.28,
            risk_level="medium",
            decision="REVIEW",
            trust_score=72.0,
        )
        assert evt.calibrated_risk == 0.04
        assert evt.policy_risk_score == 0.28
        assert evt.calibrated_risk != evt.policy_risk_score

    def test_stream_event_generation_maps_distinct_probabilities(self, client):
        """When scoring yields different calibrated prob vs policy risk, stream event preserves separation."""
        from backend.schemas import TransactionScoreRequest, StreamTransactionEvent
        from backend.main import score_transaction

        # Transaction with severe graph collusion elevating policy risk above calibrated tabular risk
        req = TransactionScoreRequest(
            order_id="ORD_STREAM_SEMANTICS",
            amount=800.0,
            base_price=100.0,
            category_median_price=95.0,
            buyer_orders_before=2,
            buyer_returns_before=1,
            share_degree=12.0,
            share_component_size=20.0,
        )
        res = score_transaction(req)
        assert res.calibrated_fraud_probability is not None
        assert res.policy_risk_score is not None
        # Calibrated tabular probability and composite policy risk differ substantially
        assert res.calibrated_fraud_probability != res.policy_risk_score

        # Map to StreamTransactionEvent according to stream_service logic
        cal_prob = (
            res.calibrated_fraud_probability
            if res.calibrated_fraud_probability is not None
            else res.overall_fraud_probability
        )
        policy_risk = (
            res.policy_risk_score
            if res.policy_risk_score is not None
            else res.overall_fraud_probability
        )

        event = StreamTransactionEvent(
            event_id="EVT_TEST_DISTINCT",
            timestamp="12:00:00",
            order_id=req.order_id,
            buyer_id="B1",
            seller_id="S1",
            amount=req.amount,
            risk_score=res.raw_fraud_probability or res.overall_fraud_probability,
            calibrated_risk=cal_prob,
            policy_risk_score=policy_risk,
            risk_level=res.risk_label,
            decision=res.decision,
            trust_score=res.trust_score,
            reason_codes=res.reason_codes,
        )
        assert event.calibrated_risk == res.calibrated_fraud_probability
        assert event.policy_risk_score == res.policy_risk_score
        assert event.calibrated_risk != event.policy_risk_score
        assert abs(event.calibrated_risk - 0.1671) < 0.05
        assert abs(event.policy_risk_score - 0.5835) < 0.05


class TestStageATrustEngineConsolidation:
    """Stage A Simplification: Canonical Trust Engine consolidation and legacy deprecation."""

    def test_fail_loud_when_canonical_trust_engine_missing(self, client):
        """When CanonicalTrustEngine is unavailable, scoring must fail loud with HTTP 503 rather than silently switching."""
        from unittest.mock import patch
        import backend.main
        with patch.object(backend.main, "_canonical_trust_engine", None):
            resp = client.post("/transaction/score", json={
                "order_id": "ORD_FAIL_LOUD_PROBE",
                "amount": 100.0,
            })
            assert resp.status_code == 503
            data = resp.json()
            assert "Canonical Trust Engine is unavailable" in data["detail"]
            assert "Legacy fallback is disabled" in data["detail"]

    def test_canonical_engine_threshold_boundaries(self):
        """Canonical engine decisions must respect boundaries: 0.25 (ALLOW/REVIEW), 0.55 (REVIEW/HOLD), 0.85 (HOLD/BLOCK)."""
        from trustshield_project.advanced_trust_engine import CanonicalTrustEngine, OperationalDecision
        engine = CanonicalTrustEngine()

        # 1. Below 0.25 -> ALLOW
        res_allow = engine.score({"tabular_risk": 0.2499})
        assert res_allow.decision == OperationalDecision.ALLOW

        # 2. At 0.25 -> REVIEW
        res_review_lower = engine.score({"tabular_risk": 0.2500})
        assert res_review_lower.decision == OperationalDecision.REVIEW

        # 3. Below 0.55 -> REVIEW
        res_review_upper = engine.score({"tabular_risk": 0.5499})
        assert res_review_upper.decision == OperationalDecision.REVIEW

        # 4. At 0.55 -> HOLD
        res_hold_lower = engine.score({"tabular_risk": 0.5500})
        assert res_hold_lower.decision == OperationalDecision.HOLD

        # 5. Below 0.85 -> HOLD
        res_hold_upper = engine.score({"tabular_risk": 0.8499})
        assert res_hold_upper.decision == OperationalDecision.HOLD

        # 6. At 0.85 -> BLOCK
        res_block = engine.score({"tabular_risk": 0.8500})
        assert res_block.decision == OperationalDecision.BLOCK

    def test_cold_start_confidence_penalty(self):
        """Cold-start entities must receive a 30% confidence penalty and cold-start reason code."""
        from trustshield_project.advanced_trust_engine import CanonicalTrustEngine
        engine = CanonicalTrustEngine()

        res_warm = engine.score({"tabular_risk": 0.20}, is_cold_start=False)
        res_cold = engine.score({"tabular_risk": 0.20}, is_cold_start=True)

        assert res_cold.confidence < res_warm.confidence
        assert abs(res_cold.confidence - round(res_warm.confidence * 0.70, 4)) < 1e-4
        assert "COLD_START_INSUFFICIENT_HISTORY" in res_cold.reason_codes

    def test_legacy_trust_engine_deprecation_notice(self):
        """Legacy trust_engine.py must contain the DEPRECATION NOTICE for backward-compatibility tracking."""
        import trustshield_project.trust_engine as legacy_mod
        assert "[DEPRECATED]" in legacy_mod.__doc__
        assert "DO NOT USE FOR PRODUCTION SERVING" in legacy_mod.__doc__
        assert "CanonicalTrustEngine" in legacy_mod.__doc__


class TestStageBInfrastructureDecoupling:
    """Stage B Simplification: Infrastructure Decoupling and Research Dependency Isolation."""

    def test_default_services_disabled_without_network_call(self):
        """By default, Redis and Neo4j must be disabled and never attempt socket connections."""
        from backend.services.redis_service import RedisService
        from backend.services.neo4j_service import Neo4jService

        redis = RedisService()
        assert redis.enabled is False
        assert redis.is_available is False
        assert redis.ping().status == "disabled"

        neo4j = Neo4jService()
        assert neo4j.enabled is False
        assert neo4j.is_available is False
        assert neo4j.ping().status == "disabled"

    def test_transaction_scoring_default_phase3_with_infrastructure_disabled(self, client):
        """Default Phase 3 scoring must succeed with HTTP 200 and transparent provenance when Redis/Neo4j are disabled."""
        resp = client.post("/transaction/score", json={
            "order_id": "ORD_STAGE_B_PROBE",
            "amount": 250.0,
            "base_price": 250.0,
            "category_median_price": 250.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["order_id"] == "ORD_STAGE_B_PROBE"
        assert data["model_version"] == "phase3"
        assert data["scoring_mode"] == "production_model_phase3_combined"
        assert data["infrastructure_sources"]["graph_topology"] == "disk_artifact"
        assert data["infrastructure_sources"]["embeddings"] == "not_used_in_phase3"

    def test_readiness_probe_with_disabled_services(self, client):
        """Readiness probe must distinguish disabled from unavailable and not fail when optional services are disabled."""
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["redis_ready"] is False
        assert data["neo4j_ready"] is False
        assert data["details"]["redis_status"] == "disabled"
        assert data["details"]["neo4j_status"] == "disabled"
        assert data["components"]["redis"] == "unavailable"
        assert data["components"]["neo4j"] == "unavailable"

    def test_neo4j_production_security_guard(self, monkeypatch):
        """In production, missing password only raises RuntimeError if Neo4j is enabled; disabled Neo4j boots safely."""
        from backend.services.neo4j_service import Neo4jService

        monkeypatch.setenv("TRUSTSHIELD_ENV", "production")
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        # 1. Disabled Neo4j in production -> Must succeed without error
        svc_disabled = Neo4jService(enabled=False, password=None)
        assert svc_disabled.enabled is False
        assert svc_disabled.is_available is False

        # 2. Enabled Neo4j in production without password -> Must fail closed
        with pytest.raises(RuntimeError, match="CRITICAL SECURITY: NEO4J_PASSWORD"):
            Neo4jService(enabled=True, password=None)

    def test_feature_flag_isolation_clip_and_phase5(self, monkeypatch):
        """ModelStore feature flags CLIP_ENABLED=false and PHASE5_ENABLED=false isolate optional research loaders."""
        from backend.model_loader import ModelStore

        monkeypatch.setenv("CLIP_ENABLED", "false")
        monkeypatch.setenv("PHASE5_ENABLED", "false")

        isolated_store = ModelStore()
        isolated_store.load()

        assert isolated_store.clip_loaded is False
        assert isolated_store.phase5_loaded is False
        assert isolated_store.combined_graph_model is not None
        assert isolated_store.is_loaded is True

    def test_clip_multimodal_scorer_instantiates_without_torch(self, monkeypatch):
        """CLIPMultimodalScorer gracefully initializes device='cpu' when torch is absent."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from trustshield_project.multimodal_scoring import CLIPMultimodalScorer
        scorer = CLIPMultimodalScorer(device=None)
        assert scorer.device == "cpu"

    def test_end_to_end_scoring_in_blocked_research_environment(self, monkeypatch):
        """Phase 3 transaction scoring must execute with 100% fidelity when research dependencies are blocked."""
        import sys

        class BlockResearchMetaPath:
            blocked = {"torch", "torch_geometric", "transformers"}
            def find_spec(self, fullname, path, target=None):
                top_name = fullname.split(".")[0]
                if top_name in self.blocked:
                    raise ModuleNotFoundError(f"No module named {fullname!r} (ISOLATION TEST)")
                return None

        blocker = BlockResearchMetaPath()
        monkeypatch.setattr(sys, "meta_path", [blocker] + sys.meta_path)
        for mod in list(sys.modules.keys()):
            if mod in ("torch", "torch_geometric", "transformers") or mod.startswith(("torch.", "torch_geometric.", "transformers.")):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        with pytest.raises(ModuleNotFoundError):
            __import__("torch")

        from backend.main import score_transaction
        from backend.schemas import TransactionScoreRequest

        req = TransactionScoreRequest(
            order_id="ORD_ISOLATION_UNIT_TEST",
            amount=175.50,
            base_price=175.0,
            category_median_price=175.0,
        )
        res = score_transaction(req)
        assert res.order_id == "ORD_ISOLATION_UNIT_TEST"
        assert res.model_version == "phase3"
        assert res.scoring_mode == "production_model_phase3_combined"
        assert res.infrastructure_sources["graph_topology"] == "disk_artifact"
