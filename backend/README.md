# TrustShield Serving Layer (FastAPI)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2.0+-e92063.svg)](https://docs.pydantic.dev)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-51%20Backend%20Passing-brightgreen.svg)](https://github.com/Rajiv107ai/Trustshield)
[![Explainability](https://img.shields.io/badge/Explainability-TreeSHAP%20Exact-8A2BE2.svg)](../trustshield_project/shap_explainer.py)

Production-grade serving gateway for the **TrustShield AI** fraud intelligence platform. Built with FastAPI, Pydantic v2, and ASGI server optimizations, delivering sub-20ms inference latency, strict boundary schema validation, isotonic probability calibration, and orchestrator readiness probes.

---

## 🏛️ Architecture Overview

The serving layer acts as the ingestion boundary and decision dispatcher:

```mermaid
graph LR
    Client[Client / Gateway] -->|HTTP Request| FastAPI[FastAPI App (backend/main.py)]
    FastAPI -->|Validate Schema| Pydantic[Pydantic v2 Schemas (schemas.py)]
    FastAPI -->|Load Artifacts| Store[ModelStore (model_loader.py)]
    Store --> Models[(Joblib Artifacts & Calibrators)]
    Store --> CLIP[(FAISS & CLIP Cache)]
    FastAPI -->|Ensemble Inference| Engine[Trust Engine Decisioning]
    Engine -->|HTTP Response| Client
```

---

## 🚀 Endpoints & Route Reference

### 1. Health & Orchestrator Probes
- **`GET /health`**  
  **Purpose:** Kubernetes/ECS liveness probe.  
  **Status Code:** `200 OK`  
  **Response:** Returns service status, version, and component availability flags.
  ```json
  {
    "status": "healthy",
    "version": "2.0.0",
    "models_loaded": true,
    "components": {
      "phase5_hybrid": true,
      "phase5_calibrator": true,
      "phase3_fallback": true,
      "fraud_rings": true,
      "clip_cache": true
    }
  }
  ```

- **`GET /ready`**  
  **Purpose:** Kubernetes readiness probe.  
  **Status Codes:** `200 OK` when ready, `503 Service Unavailable` if uninitialized.  
  **Response:** Returns `ready`, `degraded`, or `not_ready` with detailed sub-system diagnostics.

---

### 2. Real-Time Transaction Scoring
- **`POST /transaction/score`**  
  **Purpose:** Primary low-latency scoring endpoint for inbound transactions. Evaluates tabular behavioral signals, graph topology features, and optional multimodal embeddings.  
  **Key Features:**
  - Strict Pydantic boundary schema validation.
  - Transparent support for either `amount` or `order_amount` (with explicit HTTP 400 validation if conflicting values are supplied).
  - Production Isotonic probability calibration (`models/phase5_calibrator.joblib`).
  - Binary Shannon entropy calculation for epistemic confidence estimation ($1 - H_2(p)$).
  - Decision routing: `ALLOW`, `REVIEW`, `HOLD`, `BLOCK`.
  - Informative diagnostic reason codes (`HIGH_VELOCITY`, `RING_COLLUSION_SUSPECT`, `CROSS_SELLER_IMAGE_REUSE`, etc.).

#### Example Request:
```bash
curl -X POST http://localhost:8000/transaction/score \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "ORD_78910",
    "buyer_id": "BUYER_429",
    "seller_id": "SELLER_102",
    "amount": 250.00,
    "buyer_orders_before": 12,
    "buyer_returns_before": 1,
    "share_degree": 2,
    "buyer_pagerank": 0.0015
  }'
```

#### Example Response:
```json
{
  "order_id": "ORD_78910",
  "overall_fraud_probability": 0.0842,
  "risk_label": "low",
  "decision": "ALLOW",
  "trust_score": 91.58,
  "confidence": 0.923,
  "model_used": "Hybrid GNN + XGBoost (Phase 5, tabular+graph+GNN embeddings)",
  "model_version": "phase5-hybrid-calibrated",
  "cold_start": false,
  "model_disagreement": 0.031,
  "reason_codes": []
}
```

---

---

### 3. TreeSHAP Explainability Engine
- **`POST /transaction/explain`**  
  **Purpose:** Computes sub-10ms exact local Shapley attributions for any transaction, highlighting features driving fraud risk higher vs. legitimate mitigating factors.  
  **Key Features:**
  - Fast `TreeExplainer` computing local feature contributions.
  - Automatically splits into `top_positive_drivers` (risk amplifiers) and `top_negative_dampeners` (trust factors).
  - Translates machine feature keys to human-friendly names (e.g. `share_degree` → `"Hardware / Address Sharing Degree"`).
  - Synthesizes professional, evidence-backed natural language summaries for fraud operations.

#### Example Request:
```bash
curl -X POST http://localhost:8000/transaction/explain \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "ORD_98124",
    "amount": 890.0,
    "base_price": 120.0,
    "share_degree": 6.0,
    "share_component_size": 7.0,
    "seller_age_days": 4.0,
    "buyer_return_rate_before": 0.75
  }'
```

#### Example Response:
```json
{
  "order_id": "ORD_98124",
  "base_value": -1.2450,
  "overall_fraud_probability": 0.8421,
  "decision": "BLOCK",
  "top_positive_drivers": [
    {
      "feature_name": "share_degree",
      "friendly_name": "Hardware / Address Sharing Degree",
      "feature_value": 6.0,
      "shap_value": 1.482,
      "abs_impact": 1.482
    }
  ],
  "top_negative_dampeners": [],
  "all_attributions": { "share_degree": 1.482, "amount": 0.312 },
  "investigator_narrative": "Elevated risk is primarily propelled by: Hardware / Address Sharing Degree (6.0, SHAP: +1.482)."
}
```

---

### 4. Forensic Investigation Dossier API
- **`POST /investigation/generate-dossier`**  
  **Purpose:** Generates comprehensive, evidence-grounded forensic investigation packages combining entity facts, model inference, graph topology, TreeSHAP attributions, and policy RAG guidelines.  
  **Key Features:**
  - Automated evidence-grounding verification cross-checking generated assertions against recorded telemetry.
  - Reconstructed graph topology context (shared device collisions, multi-account clusters).
  - Verifiable hypothesis synthesis and concrete investigator next steps.

---

### 5. Real-Time Transaction SSE Stream
- **`GET /stream/transactions?interval=2.0`**  
  **Purpose:** Server-Sent Events (SSE) streaming endpoint for live fraud operations dashboards (`frontend/src/app/transactions/feed/`).  
  **Key Features:**
  - Pushes live scored transactions with calibrated probabilities and decision routing.
  - Automatic 15-second heartbeat keepalive frames preventing proxy timeouts.
  - Clean client disconnect cancellation lifecycle management.
  - Bounded concurrency guarantees.

---

### 6. Production Metrics & Observability
- **`GET /metrics`**: Prometheus text format exporter tracking scoring request counters, p50/p95/p99 latency histograms (`scoring_latency_seconds`), decision counters (`decisions_total`), and error fallbacks.
- **`GET /system/benchmark`**: High-resolution empirical hardware benchmark returning percentile latencies across model inference, Trust Engine, Redis vector lookup, and Neo4j graph traversal.

---

### 7. Collusion Fraud Rings Intelligence
- **`GET /fraud-rings`**  
  **Query Parameters:**
  - `min_risk_score` (float, default: `0.0`): Filter rings above a composite risk threshold.
  - `limit` (int, default: `50`): Maximum number of ring clusters returned.
  **Response:** Pre-ranked graph clusters with member nodes, collision counts, device/address densities, and burstiness statistics.

---

### 8. Multimodal Listing Integrity
- **`POST /listing/analyze`**  
  **Purpose:** Analyzes product listings for catalog counterfeits and cross-seller image theft using CLIP visual-semantic embeddings and FAISS nearest-neighbor indexing.  
  **Response:** Semantic similarity score, near-duplicate collision count, and cross-seller reuse flags.

---

### 9. Return Abuse Evaluation
- **`POST /return/analyze`**  
  **Purpose:** Evaluates return request risk based on historical return velocities, claim history, and seller return exposure.

---

## 🏗️ Services Integration Architecture (`backend/services/`)

The backend encapsulates auxiliary infrastructure and domain logic into modular service singletons:

1. **`CacheService` (`backend/services/cache_service.py`)**:
   - Manages connection pooling to the Redis feature store (`redis:6379`).
   - Caches 16-dimensional GNN node embeddings and candidate fraud ring membership sets.
   - Built-in circuit breaker fallback: seamlessly degrades to local memory cache if Redis is unavailable.

2. **`GraphService` (`backend/services/graph_service.py`)**:
   - Manages official Neo4j driver connections (`bolt://neo4j:7687`).
   - Executes parameterized Cypher queries to extract 2-hop ego graphs and collusion paths.
   - Graceful fallback: returns structured synthetic topology if the graph database is offline.

3. **`AuditLogService` (`backend/services/audit_service.py`)**:
   - Append-only structured audit logger recording transaction decisions, model versions, and investigator interventions.
   - Thread-safe in-memory ring buffer with optional persistent file backend.

4. **`InvestigationService` (`backend/services/investigation_service.py`)**:
   - Orchestrates multi-source evidence extraction (scoring, graph, TreeSHAP).
   - Generates verified facts, executive summaries, and investigator action items.
   - Enforces strict evidence grounding guard preventing unverified assertions.

---

## ⚙️ Model Artifact Store (`model_loader.py`)

The `ModelStore` singleton lazily initializes and caches production artifacts from `models/`:
- `models/hybrid_model.joblib`: Phase 5 Hybrid XGBoost model.
- `models/phase5_calibrator.joblib`: Phase 5 fitted Isotonic Probability Calibrator.
- `models/combined_graph_model.joblib`: Phase 3 Tabular + Graph Random Forest fallback.
- `models/calibrator.joblib`: Phase 3 fitted Isotonic Probability Calibrator.
- `models/fraud_rings.joblib`: Pre-computed candidate fraud ring clusters.
- `models/buyer_embeddings.joblib` & `seller_embeddings.joblib`: Pre-computed GNN node representation lookups.
- `models/clip_cache/`: Offline cached CLIP embeddings with cryptographic metadata fingerprint verification.

---

## 🌐 Orchestration & Service Mesh Integration

The serving layer operates standalone or orchestrated within a containerized microservice mesh (`docker-compose.yml`):
- **FastAPI Gateway (`trustshield-api`)**: Sub-20ms inference server exposing REST and OpenAPI documentation.
- **Neo4j Property Graph (`trustshield-neo4j`)**: Multi-relational graph store mapping collusion communities, shared devices, and address networks.
- **Redis In-Memory Feature Store (`trustshield-redis`)**: High-throughput sub-millisecond key-value lookup for 16-dimensional GNN embeddings and fraud ring member sets.
- **Prometheus (`trustshield-prometheus`)**: Metrics collection scraping `/metrics` every 15s.
- **Grafana (`trustshield-grafana`)**: Live dashboard visualization for latency percentiles and throughput.

### Environment Configuration (`.env.example`)
Configure runtime settings by copying `.env.example`:
```bash
cp .env.example .env
```
Key configuration keys:
```ini
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
ENVIRONMENT=production
LOG_LEVEL=INFO

NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=trustshield_secret

REDIS_URL=redis://redis:6379/0
PROMETHEUS_ENABLED=true
```

---

## 🎨 Frontend Contract Alignment (`frontend_master_prompts/`)

The Pydantic v2 schemas defined in `backend/schemas.py` (`HealthResponse`, `ReadyResponse`, `TransactionScoreRequest`, `TransactionScoreResponse`, `FraudRingsResponse`, `ListingScoreRequest`, `ReturnScoreRequest`, `ExplainResponse`, `InvestigationDossierResponse`) are 100% matched in TypeScript definitions under `frontend_master_prompts/02_API_SCHEMAS_TYPESCRIPT.md` and `frontend/src/lib/types/api.ts`, guaranteeing end-to-end type safety between backend responses and UI components.

---

## 🧪 Testing the Serving Layer

Execute complete backend test suites and static typing validation (51 tests passing):
```bash
# Run all backend endpoint and schema tests (29 tests)
pytest backend/test_backend.py -v

# Run new extension tests: Dossier, SSE, Metrics (9 tests)
pytest backend/test_new_extensions.py -v

# Run live driver & service fallback tests (5 tests)
pytest backend/test_services.py -v

# Run TreeSHAP explainability unit & integration tests (8 tests)
pytest trustshield_project/test_shap_explainer.py -v

# Run entire backend & serving test suite together (51 tests)
pytest backend trustshield_project/test_shap_explainer.py -v

# Run Pyright static type checker across backend
npx --yes pyright backend/

# Run Flake8 syntax and undefined symbol linter
flake8 backend/ --count --select=E9,F63,F7,F82 --show-source

# Run smoke test client against live API
python scripts/smoke_test_api.py --port 8000
```
