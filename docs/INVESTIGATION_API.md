# Forensic Investigation Dossier API

## 1. Overview

The Forensic Investigation Dossier API (`POST /investigation/generate-dossier`) provides operations analysts and compliance officers with evidence-grounded investigation reports.

Instead of outputting an unconstrained text string, the API produces a strictly structured schema that prevents speculative claims and provides transparent chain of evidence.

---

## 2. API Endpoint Specification

- **Route:** `POST /investigation/generate-dossier`
- **Request Headers:** `Content-Type: application/json`

### Request Schema (`DossierRequest`)
```json
{
  "entity_type": "transaction",
  "entity_id": "ORD_78910",
  "transaction_data": {
    "amount": 890.0,
    "buyer_id": "BUYER_RING_MEMBER_04",
    "seller_id": "SELLER_91",
    "share_degree": 9
  },
  "include_graph_evidence": true
}
```

#### Supported Entity Types
- `transaction` / `order`
- `buyer`
- `seller`
- `ring`

---

## 3. Grounded Synthesis Pipeline

The dossier pipeline enforces strict separation of concerns:
```
Request (entity_type, entity_id)
    │
    ▼
1. Validate Entity & Input Bounds
    │
    ▼
2. Context Resolution (Redis 16D cache / disk catalog)
    │
    ▼
3. Model Scoring & Conformal Coverage (Unified Trust Engine)
    │
    ▼
4. Graph Traversal with Strict Cutoff (Neo4j / disk rings)
    │
    ▼
5. Policy Retrieval (Forensic RAG Index)
    │
    ▼
6. GenAI / Deterministic Synthesis Agent
    │
    ▼
7. Evidence Grounding Verification Guard
    │
    ▼
Structured Response (DossierResponse)
```

---

## 4. Strict Separation of Truth

Per Rule #12, the response explicitly separates four epistemological tiers:
1. **`observed_evidence` (Factual Ground Truth):**
   - Verified amounts, historical return rates, timestamped ledger facts, device collision counts.
2. **`model_inference` (Statistical ML Predictions):**
   - Calibrated risk score, classifier confidence, detector disagreement, conformal coverage set (`{0}`, `{1}`, or `{0, 1}`), triggered reason codes.
3. **`graph_findings` (Topology & Collusion Intelligence):**
   - Cluster size, shared entity subgraphs, parameterized 2-hop ego connections, temporal cutoff enforcement.
4. **`recommendation` (Human Action Protocol):**
   - Recommended operational decision (`BLOCK`, `HOLD`, `REVIEW`, `ALLOW`), protocol severity level, and specific evidence required for clearance.

---

## 5. Evidence Grounding & Verification Rules

1. **Grounded Invariants:** The investigation agent cannot invent entity IDs, chargeback amounts, or nonexistent connections.
2. **Deterministic Fallback:** When an external LLM is not configured, TrustShield uses its native deterministic rule-grounded synthesis agent (`GenAIInvestigationAgent` in `trustshield_project/investigation_agent.py`).
3. **Audit Provenance:** Every response includes explicit metadata tracking:
   ```json
   "provenance": {
     "model_version": "phase5-hybrid",
     "graph_source": "neo4j | disk_artifact",
     "engine": "TrustShield Grounded Forensic Agent"
   }
   ```
4. **Automated Verification Guard:** Compares the final narrative against observed inputs; if unverified claims are detected, `grounding_verification_passed` reports false with remediation notes.
