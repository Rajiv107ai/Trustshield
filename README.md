# TrustShield AI — E-Commerce Fraud Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10+_%7C_3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-16_(React_19)-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![Turbopack](https://img.shields.io/badge/Turbopack-Enabled-0070F3?logo=vercel&logoColor=white)](https://turbo.build/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-HeteroGNN-EE4C2C?logo=pytorch&logoColor=white)](https://pyg.org)
[![Cryptography](https://img.shields.io/badge/Cryptography-AES--256--GCM_AEAD-4B32C3?logo=security&logoColor=white)](https://cryptography.io)
[![Tests Passing](https://img.shields.io/badge/Tests-519_Passed_(100%25)-brightgreen)](trustshield_project/)
[![Pyright Type Check](https://img.shields.io/badge/Pyright-0_Errors-brightgreen)](pyrightconfig.json)
[![Stage 3.5.5 Reconciled](https://img.shields.io/badge/Audit-Stage_3.5.5_Reconciled-blue)](reports/phase355_artifact_provenance_audit.md)
[![Redis](https://img.shields.io/badge/Redis-8.1_Feature_Store-DC382D?logo=redis&logoColor=white)](docs/INFRASTRUCTURE.md)
[![Neo4j](https://img.shields.io/badge/Neo4j-6.4_Graph_DB-45818E?logo=neo4j&logoColor=white)](docs/INFRASTRUCTURE.md)

**TrustShield AI** is an enterprise-grade, end-to-end fraud intelligence and evaluation governance platform engineered for modern e-commerce marketplaces. Rather than analyzing transactions as isolated point events, TrustShield maps buyers, sellers, listings, devices, bank accounts, and physical addresses into a dynamic, temporal **Trust Graph** to detect coordinated collusion rings, catalog theft, wardrobing return abuse, and synthetic identities in real time.

---

## 📑 Table of Contents

- [The Core Problem & Architecture](#-the-core-problem--architecture)
- [System Architecture & Data Flow](#-system-architecture--data-flow)
- [Production Capabilities & API Endpoints](#-production-capabilities--api-endpoints)
- [Enterprise Next.js 16 Web Console](#-enterprise-nextjs-16-web-console)
- [Scientific Performance & Benchmarks](#-scientific-performance--benchmarks)
- [Stage 3.5 Cryptographic Evaluation Gate & Governance](#-stage-35-cryptographic-evaluation-gate--governance)
- [Repository Organization](#-repository-organization)
- [Quickstart Guide](#-quickstart-guide)
- [2-Minute Technical Demonstration](#-2-minute-technical-demonstration)
- [Verification & Automated Test Suite](#-verification--automated-test-suite)
- [Integrity, Ethics & Limitations](#-integrity-ethics--limitations)

---

## 💡 The Core Problem & Architecture

Conventional rule-based engines and isolated tabular classifiers fail against organized e-commerce syndicates:

1. **Collusion & Review Rings**: Fraudulent seller clusters orchestrate synthetic transaction rings using coordinated device pools, recycled IP addresses, and shared proxy hubs to inflate merchant ratings.
2. **Catalog & Image Theft**: Rogue merchants scrape authentic product photography and technical specifications from reputable sellers, list clone items at steep discounts, and default on fulfillment.
3. **Wardrobing & Serial Return Abuse**: Sybil buyer accounts order luxury goods, substitute them with counterfeits, and abuse automated dispute-resolution windows.
4. **Temporal Data Leakage**: Standard ML models routinely overfit by allowing future graph topology or delayed chargeback labels to bleed backward in time.

TrustShield eliminates these vulnerabilities through **strict point-in-time graph feature extraction** (`event_time < decision_time`), **multimodal CLIP representation alignment**, **TreeSHAP local attributions**, and an **independently audited, single-shot evaluation gate**.

---

## 🏗️ System Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion & Boundary Validation"]
        A[Inbound Transaction / Listing Request] --> B[FastAPI Gateway :8000]
        B --> C[Pydantic v2 Schema Validation]
        B --> AUTH[RBAC: Bearer Token / API Key]
    end

    subgraph MultiLayer["Multi-Layer Intelligence Engine"]
        C --> D[Point-in-Time Behavioral Features]
        C --> E[Dynamic Trust Graph<br/>NetworkX + Neo4j Cypher]
        C --> F[Multimodal CLIP & FAISS<br/>Text-Image Cross-Attention]
        C --> G[Heterogeneous GNN<br/>PyTorch Geometric HeteroData]
    end

    subgraph Decision["Trust Engine & Calibrated Routing"]
        D & E & F & G --> H[Stacking Meta-Learner]
        H --> I[Isotonic Probability Calibrator]
        I --> J{Decision Router}
        J -->|p < 0.20| K[ALLOW: Instant Clearance]
        J -->|0.20 <= p <= 0.70| L[REVIEW / HOLD: Manual Queue]
        J -->|p > 0.70| M[BLOCK: Decline & Freeze]
    end

    subgraph Forensic["Forensic Observability & Audit"]
        L --> N[Sub-10ms TreeSHAP Explainer]
        N --> O[Automated Anti-Hallucination Dossier]
        O --> P[Next.js 16 Enterprise Console :3000]
        H --> Q[Redis 8.1 16D Feature Store]
        E --> R[Neo4j 6.4 Graph Database]
        B --> S[Prometheus Registry & Grafana :3001]
    end
```

---

## ⚡ Production Capabilities & API Endpoints

| Endpoint | Method | Latency (p95) | Description |
| :--- | :---: | :---: | :--- |
| `/transaction/score` | `POST` | `< 15ms` | Real-time multi-signal scoring, cold-start confidence attenuation, and automated decision routing (`ALLOW`, `REVIEW`, `HOLD`, `BLOCK`). |
| `/transaction/explain` | `POST` | `< 10ms` | Instant TreeSHAP exact local attributions generating human-readable risk drivers and mitigating factors. |
| `/stream/transactions` | `GET` | Real-time | Server-Sent Events (SSE) live feed delivering continuous decisions with 15s keepalive frames and bounded memory queues. |
| `/listing/analyze` | `POST` | `< 25ms` | Multimodal visual-textual consistency scoring using OpenAI CLIP-ViT and vector search against genuine catalog embeddings. |
| `/investigation/generate-dossier`| `POST` | `< 50ms` | 2-hop topological investigation generator synthesizing verified entity history into structured investigative dossiers. |
| `/fraud-rings` | `GET` | `< 20ms` | Topological cluster intelligence identifying coordinated syndicates by HHI concentration, burstiness, and device overlap. |
| `/metrics` | `GET` | `< 2ms` | Enterprise Prometheus exposition endpoint tracking throughput, latency histograms, error rates, and model divergence. |
| `/health` & `/ready` | `GET` | `< 1ms` | Liveness and readiness probes with dependency checks for Redis, Neo4j, model stores, and disk caches. |

---

## 🖥️ Enterprise Next.js 16 Web Console

The TrustShield console is built on **Next.js 16 App Router**, **React 19**, **Turbopack**, and **Tailwind CSS**, featuring **14 dedicated interactive views**:

```
frontend/src/app/
├── (overview)             # Executive summary, key metric ribbons, and risk breakdown
├── dashboard/             # Operational hub with real-time health telemetry & action queues
├── transactions/          # Interactive transaction simulator with 1-click test scenarios
├── transactions/feed/     # Chronological live SSE transaction stream with filterable feed
├── fraud-rings/           # Coordinated syndicate clusters with one-click entity freezing
├── trust-graph/           # Full-screen interactive relational graph explorer with node filters
├── listings/              # Multimodal CLIP image and text mismatch studio with visual cards
├── returns/               # Wardrobing and return abuse pattern detector
├── investigations/        # Case management studio and AI-generated forensic dossiers
├── models/                # Model observatory with calibration curves (ECE) and ROC/PR curves
├── evaluation/            # Stage 3.5 holdout cryptographic gate verification and audit trail
├── monitoring/            # Real-time Prometheus telemetry and microservice status probes
└── settings/              # Gateway URL switcher, API credentials, and environment overrides
```

### Key UI Features
* **Dual-Audience Switcher**: Instantly toggle between **Executive Story View** (high-level risk scores, plain-English reasons, business impacts) and **Deep AI Inspector** (16D GNN embeddings, SHAP waterfall charts, conformal uncertainty intervals).
* **Command Palette (`⌘K` / `Ctrl+K`)**: Universal fuzzy search modal to jump to transactions, buyer IDs, seller IDs, or fraud rings.
* **Offline Resilience**: Automatically connects to local disk fixtures if backend microservices are offline.

---

## 📊 Scientific Performance & Benchmarks

All models were evaluated under strict **temporal isolation** (Months 1–8 training, Months 9–10 validation, Months 11–12 out-of-time test). Future data is rigorously blocked from historical feature states.

| Model / Architecture | Out-of-Time ROC-AUC | Out-of-Time PR-AUC | ECE (Calibration Error) | Inference Latency (p95) |
| :--- | :---: | :---: | :---: | :---: |
| **Tabular Baseline (Random Forest / XGBoost)** | 0.651 | 0.265 | 0.082 → 0.021 | 4.2 ms |
| **Graph-Degraded Fallback (Zero Graph Topology)** | 0.603 | 0.242 | 0.078 → 0.026 | 5.1 ms |
| **Tabular + NetworkX Graph Features (Phase 3)** | 0.789 | 0.442 | 0.066 → 0.026 | 8.6 ms |
| **Hybrid (Tabular + Graph + GraphSAGE GNN, Phase 5)** | **0.765** | **0.419** | **0.065 → 0.033** | **14.2 ms** |
| **Stacking Trust Engine (Final Ensemble)** | **0.792** | **0.465** | **0.021 (Calibrated)** | **14.8 ms** |

*Note: All models employ isotonic calibration fitted strictly on validation split predictions to ensure output probabilities represent true empirical frequencies.*

---

## 🔒 Stage 3.5 Cryptographic Evaluation Gate & Governance

TrustShield introduces an immutable, replay-resistant holdout evaluation gate ([evaluation_gate.py](trustshield_project/evaluation_gate.py)) preventing premature data leaks, test set snooping, and metric manipulation:

```mermaid
stateDiagram-v2
    [*] --> AUTHORIZED: Check Authorization Token & Manifest
    AUTHORIZED --> CLAIMED: Atomic CAS Run Lock Claimed
    CLAIMED --> PREDICTIONS_COMMITTED: 10-Artifact SHA-256 Digest Bound
    PREDICTIONS_COMMITTED --> EVALUATING: Nonce & AES-256-GCM Envelope Verified
    EVALUATING --> COMPLETED: Scoring Executed & Metrics Logged
    EVALUATING --> FAILED: Fail-Closed On Discrepancy
    FAILED --> [*]
    COMPLETED --> [*]
```

### Evaluation Gate Guarantees
1. **10-Artifact Cryptographic Commitment**: Binds predictions array, model artifact hash, source commit, input manifest hash, container digest, evaluation code, feature schema, policy hash, and unique execution run ID.
2. **AES-256-GCM Envelope Protection**: Holdout ground-truth labels are sealed with hardware-accelerated authenticated encryption (`cryptography>=42.0`). Wrong keys or tampered payloads fail closed immediately with zero label leakage.
3. **Optimistic Concurrency & Crash Safety**: Thread-safe spinlocks and durable JSON ledgers guarantee atomic state transitions and eliminate double-evaluation replays.
4. **Capacity Constraint Precision**: Strict floor rounding ($C = \lfloor b \times N \rfloor$) with deterministic tie-breaking ensures identical metric calculations across all evaluators.

---

## 📁 Repository Organization

```
trustshield_full_handoff/
├── backend/                          # FastAPI REST API & Microservice Layer
│   ├── auth.py                       # RBAC authorization engine (Bearer + API Key)
│   ├── main.py                       # REST API endpoints, SSE streaming, health probes
│   ├── model_loader.py               # Pre-trained artifact store and warm-cache manager
│   ├── schemas.py                    # Strict Pydantic v2 schemas and validation bounds
│   ├── services/                     # Isolated service modules (Redis, Neo4j, Metrics, SSE)
│   └── test_*.py                     # Serving layer test suite (60 tests passed)
│
├── frontend/                         # Next.js 16 Enterprise Console
│   ├── src/app/                      # 14 App Router dashboard views
│   ├── src/components/               # UI components, layout headers, ⌘K search modal
│   ├── src/context/                  # ViewModeContext (Executive vs Deep AI) & RealtimeContext
│   └── src/lib/                      # Typed API client and interactive scenario presets
│
├── trustshield_project/              # Core ML, Graph, GNN & Cryptographic Security Engine
│   ├── advanced_ring_intelligence.py # HHI concentration and cluster burstiness
│   ├── advanced_trust_engine.py      # Stacking meta-learner and uncertainty bounds
│   ├── baseline_model.py             # Point-in-time tabular baseline models
│   ├── calibration.py                # Isotonic regression and Platt scaling calibrators
│   ├── entity_generator.py           # Synthetic generator for 8 e-commerce entity types
│   ├── evaluation_gate.py            # Stage 3.5 cryptographic evaluation gate & durable ledger
│   ├── fraud_injection.py            # Layered injection for 4 adversarial fraud topologies
│   ├── graph_features.py             # Point-in-time snapshot graph topological extraction
│   ├── hetero_gnn.py                 # PyTorch Geometric heterogeneous GNN
│   ├── multimodal_clip_faiss.py      # CLIP embeddings and FAISS similarity indexing
│   ├── shap_explainer.py             # Sub-10ms TreeSHAP attribution engine
│   └── test_*.py                     # Unit, security, and adversarial test suites (459 tests passed)
│
├── scripts/                          # Automation, Reproduction & Benchmarking Harnesses
│   ├── benchmark_scoring_http.py     # HTTP throughput, latency, and p50/p95 benchmarking
│   ├── e2e_smoke_validation.py       # 9-case operational end-to-end validation suite
│   ├── generate_realistic_*.py       # Synthetic dataset generation engines (v2 & v2.1)
│   ├── reproduce_all.py              # Master single-command reproduction pipeline
│   ├── run_delayed_feedback_*.py     # Label latency and delayed ground-truth experiments
│   ├── run_robustness_experiments.py # Adversarial noise and feature missingness experiments
│   ├── run_scalability_experiment.py # QPS throughput and concurrency scalability tests
│   └── seed_mesh.py                  # Neo4j Bolt and Redis feature store seeding script
│
├── models/                           # Serialized Joblib Models & Holdout Baselines
│   ├── combined_graph_model.joblib   # Tabular + graph Random Forest
│   ├── hybrid_model.joblib           # GNN + Tabular XGBoost hybrid
│   ├── calibrator.joblib             # Fitted isotonic calibrator
│   └── stage34/                      # Frozen Stage 3.5.5 candidate models & calibrators
│
├── data/                             # Versioned Synthetic Datasets & Temporal Splits
│   └── synthetic_v2_1/               # Validated v2.1 tables, manifest, and split metadata
│
├── docker/                           # Production Service Mesh Configurations
│   ├── Dockerfile.evaluator          # Sealed independent evaluator container
│   ├── prometheus/prometheus.yml     # Prometheus metrics scraping configuration
│   └── grafana/                      # Pre-provisioned dashboards and datasources
│
├── docs/                             # Engineering Specifications & Model Cards
│   ├── SPECIFICATIONS.md             # Unified system architecture specifications
│   ├── INFRASTRUCTURE.md             # Redis & Neo4j deployment and schema guides
│   ├── OBSERVABILITY.md              # Prometheus metrics and Grafana setup guide
│   └── MODEL_CARD.md                 # System model card, ethical use & governance notes
│
├── reports/                          # Audit Trail & Verification Checklists
│   ├── phase352_gate_security_audit.json
│   ├── phase354_independent_ledger_audit.md
│   └── phase355_artifact_provenance_audit.md
│
├── pyrightconfig.json                # Strict static type checker configuration
├── pytest.ini                        # Test runner configuration
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Quickstart Guide

### 1. Python Backend Setup

```bash
# Clone the repository
git clone https://github.com/Rajiv107ai/Trustshield.git
cd Trustshield

# Create and activate Python virtual environment
python -m venv .venv

# Windows (PowerShell - Direct execution without policy changes):
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Alternative (Activate environment first):
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser  # Run once if scripts are disabled
# .\.venv\Scripts\Activate.ps1
# uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Linux / macOS:
# source .venv/bin/activate
# uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
* **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
* **Health Check**: `http://localhost:8000/health`
* **Readiness Probe**: `http://localhost:8000/ready`

### 2. Next.js Frontend Console Setup

```bash
# Open a new terminal in the frontend directory
cd frontend

# Install Node dependencies
npm install

# Start development server with Turbopack
npm run dev
```
* **Console URL**: `http://localhost:3000` (Use `⌘K` / `Ctrl+K` for global navigation)

### 3. Full 5-Service Docker Mesh (Optional)

To start the complete production stack (FastAPI + Redis + Neo4j + Prometheus + Grafana):

```bash
# Copy environment configuration
cp .env.example .env

# Start container mesh
docker compose up -d

# Seed Neo4j graph and Redis feature store
python scripts/seed_mesh.py
```
* **Grafana Dashboards**: `http://localhost:3001` (Credentials: `admin` / `trustshield_admin`)
* **Prometheus Metrics**: `http://localhost:9090`
* **Neo4j Browser**: `http://localhost:7474` (Credentials: `neo4j` / `trustshield_dev_secret`)

---

## 💻 2-Minute Technical Demonstration

Verify the entire system end-to-end with pre-trained models and verified fixtures:

### Step 1: Run the 9-Case Operational Validation Suite
```powershell
python scripts/e2e_smoke_validation.py
```
*Validates normal transactions, high-risk collusion, cold-start seller/buyer, image catalog theft, isolated entity, input validation, temporal boundaries, and fraud rings.*

### Step 2: Measure Live HTTP Scoring Throughput & Latency
```powershell
python scripts/benchmark_scoring_http.py --requests 20 --warmup 5
```
*Demonstrates sub-20ms p50 latency and sub-25ms p95 latency under in-process FastAPI execution.*

### Step 3: Trigger Live Scoring Request via cURL
```powershell
curl -X POST "http://127.0.0.1:8000/transaction/score" `
     -H "Content-Type: application/json" `
     -d "{\"order_id\":\"DEMO_001\",\"buyer_id\":\"BUYER_000001\",\"seller_id\":\"SELLER_000001\",\"amount\":49.99,\"base_price\":49.99,\"category_median_price\":49.99,\"buyer_orders_before\":12,\"seller_total_listings_before\":25,\"buyer_age_days\":180,\"seller_age_days\":365,\"device_shared_buyer_count\":1}"
```

---

## 🧪 Verification & Automated Test Suite

TrustShield is backed by **519 passing automated tests** with **100% pass rate** across all subsystems:

```bash
# 1. Run Core ML, GNN, Security & Evaluation Gate Test Suite (459 tests)
pytest trustshield_project/ -v

# 2. Run FastAPI Backend Serving & Authentication Security Suite (60 tests)
pytest backend/ -v

# 3. Verify Pyright Static Type Checking (0 Errors)
pyright trustshield_project backend

# 4. Verify Next.js Frontend Production Build & TypeScript Types (16/16 routes)
cd frontend
npm run build
npm run lint
```

### Verification Matrix Summary

| Test Domain | Target Files | Tests | Result | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Holdout Evaluation Gate Security** | `test_stage351` through `test_stage354` | 188 | 188 Passed | **PASS** |
| **Model Policy & Protocol Reconciliation** | `test_stage32` through `test_stage34` | 74 | 74 Passed | **PASS** |
| **Backend API & Role-Based Auth** | `backend/test_*.py` | 60 | 60 Passed | **PASS** |
| **Graph Models, GNN & TreeSHAP** | `test_graph_and_models.py`, `test_shap_explainer.py` | 65 | 65 Passed | **PASS** |
| **Temporal Anti-Leakage & Data Integrity**| `test_leakage.py`, `test_data_integrity.py` | 51 | 51 Passed | **PASS** |
| **Synthetic Generators & Injection** | `test_synthetic_*.py`, `test_fraud_injection.py` | 81 | 81 Passed | **PASS** |
| **Total Automated Tests** | **All 24 test suites** | **519** | **519 Passed** | **100% PASS** |

---

## ⚖️ Integrity, Ethics & Limitations

- **Local Test Evidence**: All tests in this repository represent local automated test evidence on verified synthetic fixtures. No live banking gateway or external production network is deployed.
- **Commitment Scheme**: The evaluation gate commitment is a hash-based cryptographic binding scheme guaranteeing artifact immutability, not an asymmetric digital signature.
- **Synthetic Data**: Synthetic Dataset v2.1 validates data contracts, feature extraction pipelines, calibration, and serialization mechanics. Real-market generalization requires live shadow evaluation with production transaction streams.
- **Privacy & Compliance**: All synthetic identities, graph relations, and transaction vectors are generated without real personal identifiable information (PII).

---

<div align="center">
  <sub>Developed for research, education, and marketplace integrity. Built with FastAPI, Next.js, PyTorch Geometric, and Cryptography.</sub>
</div>
