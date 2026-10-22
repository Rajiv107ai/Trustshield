"""Tests for Autonomous Multi-Agent Fraud Investigation Copilot.

Validates:
- Tool executions (Graph Specialist, Feature Velocity Specialist, Outbox Auditor, Policy Specialist)
- Cryptographic Dispute Brief synthesis, SHA-256 evidence hashing, and HMAC-SHA256 signatures
- Tamper-evidence detection (tampered digest or signature triggers verification failure)
- End-to-end API endpoints /investigation/copilot/analyze and /investigation/copilot/verify-brief
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.copilot_service import (
    AutonomousInvestigationCopilot,
    DisputeBriefTool,
    FeatureVelocityTool,
    GraphForensicsTool,
    OutboxAuditorTool,
    PolicyRAGTool,
    copilot_service,
)
from backend.schemas import (
    CopilotInvestigationRequest,
    VerifyBriefRequest,
)


def test_graph_forensics_tool():
    tool = GraphForensicsTool()
    data, narrative = tool.execute("buyer", "BUYER_00000")
    assert "entity_id" in data
    assert "topological_metrics" in data
    assert isinstance(data["connected_devices"], list)
    assert len(narrative) > 10


def test_feature_velocity_tool():
    tool = FeatureVelocityTool()
    data, narrative = tool.execute("BUYER_00000", as_of="2025-11-01 10:00:00")
    assert "sliding_velocities" in data
    assert "total_lifetime_orders" in data
    assert "return_abuse_rate" in data
    assert len(narrative) > 10


def test_outbox_auditor_tool():
    tool = OutboxAuditorTool()
    data, narrative = tool.execute("ORD_78901")
    assert "active_pending_commands" in data
    assert isinstance(data["active_pending_commands"], list)
    assert len(narrative) > 5


def test_policy_rag_tool():
    tool = PolicyRAGTool()
    policies, narrative = tool.execute("DEVICE_FARM")
    assert len(policies) == 1
    assert "Hardware Collision Standard" in policies[0]

    all_policies, _ = tool.execute("ALL")
    assert len(all_policies) >= 4


def test_cryptographic_dispute_brief_tamper_evidence():
    brief_tool = DisputeBriefTool()
    brief = brief_tool.execute(
        case_id="CASE_TEST_001",
        entity_id="ORD_TEST_99",
        hypothesis="Coordinated Synthetic Identity Collusion",
        graph_data={"total_shared_buyers_discovered": 4, "is_part_of_collusion_ring": True, "topological_metrics": {"share_degree": 3.0}},
        velocity_data={"total_lifetime_orders": 5, "return_abuse_rate": 0.2, "sliding_velocities": {"orders_1h": 2, "spend_24h": 500.0}},
        model_risk=0.88,
        model_decision="BLOCK",
    )

    assert brief.brief_id.startswith("BRIEF_")
    assert brief.sha256_evidence_digest
    assert brief.hmac_signature
    assert brief.chargeback_defense_viability == "STRONG"
    assert "TRUSTSHIELD FORENSIC DISPUTE & CHARGEBACK DEFENSE BRIEF" in brief.markdown_content

    # 1. Valid signature verification
    v_req = VerifyBriefRequest(
        brief_id=brief.brief_id,
        sha256_evidence_digest=brief.sha256_evidence_digest,
        hmac_signature=brief.hmac_signature,
    )
    v_res = copilot_service.verify_brief(v_req)
    assert v_res.is_valid is True
    assert v_res.signature_matches is True

    # 2. Tampered signature detection
    v_bad_sig = VerifyBriefRequest(
        brief_id=brief.brief_id,
        sha256_evidence_digest=brief.sha256_evidence_digest,
        hmac_signature="tampered_bad_signature_00000000000000000000",
    )
    v_bad_res = copilot_service.verify_brief(v_bad_sig)
    assert v_bad_res.is_valid is False
    assert v_bad_res.signature_matches is False

    # 3. Tampered evidence digest detection
    v_bad_digest = VerifyBriefRequest(
        brief_id=brief.brief_id,
        sha256_evidence_digest="deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        hmac_signature=brief.hmac_signature,
    )
    v_bad_dig_res = copilot_service.verify_brief(v_bad_digest)
    assert v_bad_dig_res.is_valid is False


def test_autonomous_copilot_investigation_e2e():
    copilot = AutonomousInvestigationCopilot()
    req = CopilotInvestigationRequest(
        entity_type="transaction",
        entity_id="ORD_78901",
        include_dispute_brief=True,
    )

    res = copilot.investigate(req)
    assert res.status == "COMPLETED"
    assert res.primary_fraud_hypothesis
    assert res.copilot_narrative
    assert res.counterfactual_defense
    assert res.recommended_action
    assert res.dispute_brief is not None
    assert len(res.tool_executions) == 5

    # Check all 5 specialized agents were invoked
    agents_called = {t.agent for t in res.tool_executions}
    assert "Network Topology Specialist" in agents_called
    assert "Feature Store Specialist" in agents_called
    assert "Action Outbox Auditor" in agents_called
    assert "Policy & Compliance Specialist" in agents_called
    assert "Legal & Dispute Synthesizer" in agents_called


def test_copilot_api_endpoints():
    headers = {"X-API-Key": "dev-operator-api-key-32chars-long"}
    with TestClient(app) as client:
        # 1. Analyze endpoint
        payload = {
            "entity_type": "transaction",
            "entity_id": "ORD_API_TEST",
            "include_dispute_brief": True,
        }
        res = client.post("/investigation/copilot/analyze", json=payload, headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["case_id"] == "CASE_COPILOT_ORD_API_TEST"
        assert "copilot_narrative" in data
        assert "dispute_brief" in data
        assert len(data["tool_executions"]) >= 4

        # 2. Verify brief endpoint
        brief = data["dispute_brief"]
        verify_payload = {
            "brief_id": brief["brief_id"],
            "sha256_evidence_digest": brief["sha256_evidence_digest"],
            "hmac_signature": brief["hmac_signature"],
        }
        v_res = client.post("/investigation/copilot/verify-brief", json=verify_payload)
        assert v_res.status_code == 200
        v_data = v_res.json()
        assert v_data["is_valid"] is True
        assert v_data["signature_matches"] is True
