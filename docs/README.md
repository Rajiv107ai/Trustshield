# TrustShield AI — Documentation & Governance Hub

This directory serves as the centralized repository for all engineering audits, scientific research reports, regulatory model cards, architectural specifications, and bug tracking inventories across the **TrustShield AI** platform.

---

## 🧭 Master Documentation Index

### 🛠️ 1. Technical Repair & Verification (Current State)
- [**Master Technical Repair Report (`FINAL_REPAIR_REPORT.md`)**](FINAL_REPAIR_REPORT.md)  
  *Authoritative record of the engineering repair campaign.* Documents the resolution of all 22 cataloged defects (`TS-001` to `TS-022`), temporal leakage elimination, production probability calibration, and empirical verification (192 passing tests across unit, regression, serving, and mesh seeder suites).
- [**Repair Baseline State (`REPAIR_BASELINE.md`)**](REPAIR_BASELINE.md)  
  Pre-repair vulnerability baseline, checklist, and architectural baseline.
- [**Tracked Bug Registry (`BUG_INVENTORY.json`)**](BUG_INVENTORY.json)  
  Machine-readable JSON registry of tracked defects, files modified, and verification evidence.

---

### 🔬 2. Scientific Audits & Empirical Benchmarks
- [**Model Robustness & Perturbation Report (`ROBUSTNESS_REPORT.md`)**](ROBUSTNESS_REPORT.md)  
  Empirical stress testing under Gaussian feature noise, missingness injection, extreme dollar value perturbations, and topology disconnects.
- [**High-Throughput Scalability Report (`SCALABILITY_REPORT.md`)**](SCALABILITY_REPORT.md)  
  Empirical benchmarking measuring QPS throughput, p50/p95/p99 inference latencies, worker thread scaling, and saturation profiles.
- [**Delayed Feedback Simulation Report (`DELAYED_FEEDBACK_REPORT.md`)**](DELAYED_FEEDBACK_REPORT.md)  
  Rigorous simulation of 7-day, 14-day, 30-day, and 60-day fraud label latency, measuring PR-AUC decay and evaluating streaming mitigation strategies.
- [**49-Point Scientific Re-Audit (`49_POINT_REAUDIT.md`)**](49_POINT_REAUDIT.md)  
  Exhaustive line-by-line audit examining statistical validity, temporal leakage, graph isolation, and mathematical proofs.
- [**Final Before/After Benchmark Report (`FINAL_BEFORE_AFTER.md`)**](FINAL_BEFORE_AFTER.md)  
  Empirical comparison of legacy vs. repaired pipelines under strict chronological test set evaluation.
- [**Phase 0: Baseline Audit (`BASELINE_AUDIT.md`)**](BASELINE_AUDIT.md)  
  Initial codebase inspection, identifying early architectural shortcomings and data leakage risks.
- [**Full Project Audit Baseline (`FULL_PROJECT_AUDIT_BASELINE.md`)**](FULL_PROJECT_AUDIT_BASELINE.md)  
  Holistic whole-project inventory across all modules and scripts.
- [**Full Project Audit Report (`FULL_PROJECT_AUDIT_REPORT.md`)**](FULL_PROJECT_AUDIT_REPORT.md)  
  Detailed findings from full-system static and dynamic verification.

---

### 🏛️ 3. Advanced Research, Infrastructure & Architecture
- [**Enterprise Architecture & Extensions Blueprint (`ARCHITECTURE_AND_EXTENSIONS_PLAN.md`)**](ARCHITECTURE_AND_EXTENSIONS_PLAN.md)  
  Holistic technical plan detailing the microservices layer, Redis 16D vector store, Neo4j temporal graph, SSE event streaming, TreeSHAP explainability, and observability.
- [**Real-Time Streaming Architecture (`REALTIME_ARCHITECTURE.md`)**](REALTIME_ARCHITECTURE.md)  
  Specifications for Server-Sent Events (SSE) `/stream/transactions`, 15-second heartbeat keepalive frames, bounded worker queues, and client disconnect lifecycles.
- [**Infrastructure & Service Mesh (`INFRASTRUCTURE.md`)**](INFRASTRUCTURE.md)  
  Redis 16-dimensional GNN embedding store, Neo4j temporal Cypher property graph, Docker Compose 5-service orchestration, and zero-downtime offline fallbacks.
- [**Forensic Investigation API Specification (`INVESTIGATION_API.md`)**](INVESTIGATION_API.md)  
  API contract for `/investigation/generate-dossier`, combining entity telemetry, 2-hop graph topology, policy RAG guidelines, and automated evidence grounding verification.
- [**Phase 2: Final Advanced Research Report (`PHASE2_FINAL_REPORT.md`)**](PHASE2_FINAL_REPORT.md)  
  Comprehensive mathematical synthesis of Heterogeneous GNNs (`HeteroData`), $Time2Vec$ continuous-time edge learning, multimodal CLIP/FAISS vector retrieval, and Conformal Prediction uncertainty quantification.
- [**Phase 2: Pre-Implementation Baseline (`PHASE2_BASELINE.md`)**](PHASE2_BASELINE.md)  
  Baseline state before deploying Phase 2 advanced research algorithms.
- [**Phase 2: Production Roadmap (`PHASE2_ROADMAP.md`)**](PHASE2_ROADMAP.md)  
  Strategic engineering design for high-throughput streaming (Apache Kafka, Apache Flink) and enterprise graph databases (Neo4j).
- [**System Design & Core Specifications (`SPECIFICATIONS.md`)**](SPECIFICATIONS.md)  
  Unified foundational specification covering the 8-entity relational model, pipeline dependency DAG, synthetic fraud typologies, snapshot graph design, anti-leakage invariants, and verification protocols.

---

### 📋 4. Governance, Compliance, Observability & Safety
- [**Production Observability Specification (`OBSERVABILITY.md`)**](OBSERVABILITY.md)  
  Prometheus metric registry (`/metrics`), latency histograms, Grafana dashboard templates, and low-cardinality guardrails.
- [**System Model Card (`MODEL_CARD.md`)**](MODEL_CARD.md)  
  Standardized AI governance card detailing intended use, prohibited applications, training data distributions, fairness evaluations, and ethical considerations.
- [**Technical Limitations & Disclosure (`LIMITATIONS.md`)**](LIMITATIONS.md)  
  Explicit boundary conditions, cold-start limitations, throughput boundaries, and operational assumptions.

---

### 🎨 5. Enterprise Frontend & UX Specification
- [**Frontend Implementation Report (`../frontend/FRONTEND_IMPLEMENTATION_REPORT.md`)**](../frontend/FRONTEND_IMPLEMENTATION_REPORT.md)  
  Enterprise console architecture, 12 Next.js views, Dual-Audience switch, and zero-fake-data backend grounding.
- [**Frontend Master Prompts Package (`../frontend_master_prompts/README.md`)**](../frontend_master_prompts/README.md)  
  Production-grade frontend specifications for autonomous UI creation:
  - `00_MASTER_FRONTEND_PROMPT.md`: 12 enterprise console views with strict Zero-Fake-Data backend grounding.
  - `01_TECH_STACK_AND_TOKENS.md`: Dark console palette, decision color tokens, and JetBrains Mono typography.
  - `02_API_SCHEMAS_TYPESCRIPT.md`: TypeScript contracts mirroring 100% of Pydantic v2 schemas.
  - `03_SCENARIOS_AND_DUAL_MODE.md`: 4 interactive demo scenarios and plain-English reason translations.

