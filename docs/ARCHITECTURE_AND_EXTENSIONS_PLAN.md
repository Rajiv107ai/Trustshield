# TrustShield AI — Production Architecture & Extensions Plan

## Executive Summary
This document formalizes the production-grade extension of the TrustShield platform, integrating:
1. **Live Neo4j + Redis Infrastructure Layer** with zero-crash offline fallbacks.
2. **Forensic Dossier API** (`POST /investigation/generate-dossier`) grounded in actual model inferences and RAG policies.
3. **Real-Time SSE Transaction Stream** (`GET /stream/transactions`) with client lifecycle management and Next.js live streaming.
4. **Prometheus Metrics & Grafana Observability** (`GET /metrics`, Docker service mesh orchestration, and latency histograms).
5. **Strict Temporal Isolation** (`event_time < decision_time`) and verifiable fallback observability.

---

## 1. System Architecture Map

```
                     ┌──────────────────────────────────┐
                     │ Incoming Transaction / Simulation │
                     └─────────────────┬────────────────┘
                                       │
                                       ▼
                   ┌──────────────────────────────────────┐
                   │    FastAPI Serving Layer (Port 8000)  │
                   │                                      │
                   │  • POST /transaction/score           │
                   │  • POST /investigation/generate-dossier
                   │  • GET  /stream/transactions (SSE)   │
                   │  • GET  /metrics (Prometheus)        │
                   │  • GET  /health & /ready             │
                   └───────┬──────────────────────┬───────┘
                           │                      │
             ┌─────────────┴────────┐     ┌───────┴──────────────┐
             ▼                      ▼     ▼                      ▼
  ┌─────────────────────┐  ┌────────────┐ ┌───────────────┐ ┌────────────────┐
  │   Redis Service     │  │   Neo4j    │ │ Canonical     │ │ Investigation  │
  │                     │  │  Service   │ │ Trust Engine  │ │ Agent + RAG    │
  │ • 16D GNN Embeddings│  │            │ │               │ │                │
  │ • Feature Context   │  │ • 1-Hop    │ │ • XGBoost P5  │ │ • Zero-        │
  │ • Ring Fast Lookup  │  │ • Collusion│ │ • Conformal   │ │   Hallucination│
  │ • Fallback to Disk  │  │ • Temporal │ │   Coverage    │ │ • Evidence-    │
  │                     │  │ • Parameter│ │ • Shannon H2  │ │   Grounded     │
  │                     │  │   Cypher   │ │               │ │   Dossier      │
  └─────────────────────┘  └────────────┘ └───────────────┘ └────────────────┘
             │                      │
             ▼                      ▼
  ┌─────────────────────────────────────┐
  │   Fallback Layer: Disk Artifacts   │
  │ models/*.joblib & in-memory graph   │
  └─────────────────────────────────────┘
```

---

## 2. Component Design & Contracts

### A. Redis Service (`backend/services/redis_service.py`)
- **Connection**: `redis.Redis.from_url` with timeout bounds (`socket_connect_timeout=2.0`, `socket_timeout=2.0`).
- **Embedding Format**: 16-dimensional float vector matching `models/buyer_embeddings.joblib` and `models/seller_embeddings.joblib`.
- **Key Schema**:
  - `trustshield:embedding:{entity_type}:{entity_id}` (canonical)
  - `buyer:{id}:emb` / `seller:{id}:emb` (backward-compatibility with `seed_mesh.py`)
  - `trustshield:context:{entity_type}:{entity_id}` (cached feature contexts)
- **Graceful Degradation**:
  - If Redis is offline, `is_available` returns `False`.
  - Lookups fall back transparently to `store.buyer_embeddings` / `store.seller_embeddings` on disk.
  - Returns metadata indicating `source: "redis" | "disk_artifact" | "unavailable"`.
  - Fallback triggers `trustshield_fallback_total{subsystem="redis"}` counter.

### B. Neo4j Service (`backend/services/neo4j_service.py`)
- **Driver**: Official `neo4j.GraphDatabase.driver` with connection pooling.
- **Security & Parameterization**:
  - 100% Parameterized Cypher queries (Zero string interpolation).
  - Explicit parameter bindings: `$seller_id`, `$buyer_id`, `$days`, `$cutoff_time`.
- **Temporal Invariant**:
  - All queries strictly enforce `event_time < decision_time` to prevent future leakage.
- **Graceful Degradation**:
  - If Neo4j is offline, falls back to `models/fraud_rings.joblib` and in-memory `Neo4jInvestigator` graph.
  - Explicit metadata: `graph_source: "neo4j" | "disk_artifact" | "unavailable"`.

### C. Forensic Dossier API (`POST /investigation/generate-dossier`)
- **Input**:
  ```json
  {
    "entity_type": "transaction",
    "entity_id": "ORD_78901",
    "transaction_data": { ... optional explicit feature override ... },
    "include_graph": true
  }
  ```
- **Execution Pipeline**:
  1. Retrieve entity / transaction context (from payload, Redis context, or disk seeds).
  2. Evaluate through `CanonicalTrustEngine` & active model.
  3. Extract multi-hop neighborhood via `Neo4jService`.
  4. Pass to `GenAIInvestigationAgent` with `ForensicRAGIndex`.
  5. Run Automated Hallucination Guard to verify mathematical consistency.
  6. Return structured dossier with separated OBSERVED EVIDENCE, MODEL INFERENCE, and ACTIONS.

### D. Server-Sent Events (SSE) Stream (`GET /stream/transactions`)
- **Endpoint**: `GET /stream/transactions` (text/event-stream).
- **Architecture**:
  - Async generator with keepalive heartbeat comments (`: keep-alive\n\n`) every 15s.
  - Client disconnect detection via `await request.is_disconnected()`.
  - Bounded concurrency with active connection tracking in Prometheus.
- **Payload Schema**:
  ```json
  {
    "event_id": "EVT_90218",
    "timestamp": "2026-10-08T01:50:00Z",
    "order_id": "ORD_78921",
    "buyer_id": "BUYER_RING_MEMBER_04",
    "seller_id": "SELLER_RING_LEADER_01",
    "amount": 890.00,
    "risk_score": 0.942,
    "calibrated_risk": 0.938,
    "decision": "BLOCK",
    "trust_score": 6.2,
    "reason_codes": ["HIGH_DEVICE_COLLISION", "RING_COLLUSION_SUSPECT"],
    "source": "simulation_stream"
  }
  ```
- **Frontend Wiring**:
  - Update `frontend/src/app/transactions/feed/page.tsx` and `/dashboard` with auto-reconnecting `EventSource`.

### E. Prometheus & Grafana Observability
- **Endpoint**: `GET /metrics`.
- **Metrics Hierarchy**:
  - Requests: `trustshield_requests_total`, `trustshield_request_duration_seconds`
  - Scoring: `trustshield_decisions_total`, `trustshield_model_inference_seconds`
  - Infrastructure: `trustshield_redis_latency_seconds`, `trustshield_neo4j_latency_seconds`
  - Fallbacks: `trustshield_fallback_total{subsystem, reason}`
  - Investigations: `trustshield_dossier_generation_total`, `trustshield_dossier_duration_seconds`
  - SSE: `trustshield_sse_active_clients`, `trustshield_sse_events_total`
- **Docker Orchestration**:
  - Add Prometheus (`prom/prometheus:v2.51.0`) scraping `trustshield-api:8000/metrics`.
  - Add Grafana (`grafana/grafana:10.4.0`) with provisioned datasource and pre-built 12-panel dashboard.

---

## 3. Implementation Phases & Quality Gates

| Phase | Milestone | Acceptance Criteria |
|---|---|---|
| **Phase A** | Redis & Neo4j Services | Connection management, parameterized Cypher, 16D embedding cache, disk fallback, health/ready probes |
| **Phase B** | Dossier API | `POST /investigation/generate-dossier` with structured schema, hallucination guard, 404/422 safety |
| **Phase C** | SSE Live Stream | `GET /stream/transactions`, keepalive, disconnect detection, live model scoring, Next.js UI integration |
| **Phase D** | Prometheus & Grafana | `GET /metrics`, latency histograms, fallback counters, docker-compose orchestration, dashboard JSON |
| **Phase E** | Full Integration & Tests | Zero regressions (174+ tests pass), new unit/integration/temporal tests, benchmarks, docs sync |
