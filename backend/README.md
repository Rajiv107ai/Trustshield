# TrustShield Serving Layer (FastAPI)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2.0+-e92063.svg)](https://docs.pydantic.dev)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-Backend%20Passing-brightgreen.svg)](https://github.com/Rajiv107ai/Trustshield)

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

### 3. Collusion Fraud Rings Intelligence
- **`GET /fraud-rings`**  
  **Query Parameters:**
  - `min_risk_score` (float, default: `0.0`): Filter rings above a composite risk threshold.
  - `limit` (int, default: `50`): Maximum number of ring clusters returned.
  **Response:** Pre-ranked graph clusters with member nodes, collision counts, device/address densities, and burstiness statistics.

---

### 4. Multimodal Listing Integrity
- **`POST /listing/analyze`**  
  **Purpose:** Analyzes product listings for catalog counterfeits and cross-seller image theft using CLIP visual-semantic embeddings and FAISS nearest-neighbor indexing.  
  **Response:** Semantic similarity score, near-duplicate collision count, and cross-seller reuse flags.

---

### 5. Return Abuse Evaluation
- **`POST /return/analyze`**  
  **Purpose:** Evaluates return request risk based on historical return velocities, claim history, and seller return exposure.

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

## 🧪 Testing the Serving Layer

Execute backend integration tests:
```bash
# Run all backend endpoint and schema tests
pytest backend/test_backend.py -v

# Run smoke test client against live API
python scripts/smoke_test_api.py --port 8000
```
