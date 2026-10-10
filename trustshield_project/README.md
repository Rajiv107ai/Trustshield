# TrustShield Core Intelligence & Research Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-HeteroGNN-EE4C2C.svg)](https://pyg.org)
[![FAISS](https://img.shields.io/badge/FAISS-VectorSearch-00599C.svg)](https://github.com/facebookresearch/faiss)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4+-eb5424.svg)](https://xgboost.ai)
[![Test Suite](https://img.shields.io/badge/Tests-214%20Passed-brightgreen.svg)](https://github.com/Rajiv107ai/Trustshield)
[![Explainability](https://img.shields.io/badge/Explainability-TreeSHAP%20Exact-8A2BE2.svg)](shap_explainer.py)
[![Temporal Safety](https://img.shields.io/badge/Temporal%20Invariant-event__time%20%3C%20decision__time-blue.svg)](../docs/FINAL_REPAIR_REPORT.md)

This directory contains the core machine learning models, relational graph neural networks, continuous-time edge dynamics, multimodal vector indices, and information-theoretic decisioning algorithms powering the **TrustShield AI** platform.

---

## 🔬 Module Architecture & Breakdown

### 1. Graph Neural Networks, Topology & Temporal Invariants
- **`temporal_utils.py`**:  
  Canonical temporal invariant enforcement. Implements strict historical inequality (`event_time < decision_time`), chronological DataFrame filtering, temporal graph edge slicing, and validation boundary isolation.
- **`hetero_gnn.py`**:  
  Heterogeneous relational Graph Neural Network built on PyTorch Geometric `HeteroData`. Distinguishes node types (`buyer`, `seller`, `device`, `address`) and edge relations (`uses_device`, `shares_address`, `transacts_with`) with strict historical cutoff boundaries.
- **`temporal_gnn.py`**:  
  Continuous-time edge learning with $Time2Vec$ harmonic positional encodings. Enforces strict temporal directionality ($\Delta t \ge 0$ asserted) to eliminate retrospective leakage.
- **`advanced_ring_intelligence.py`**:  
  Candidate collusion cluster extraction on multi-relational graphs. Computes cluster-level topological features:
  - Hardware / device collision density
  - Merchant concentration using Herfindahl-Hirschman Index (HHI)
  - Inter-order temporal arrival burstiness (coefficient of variation of arrival intervals)
  - Coordinated return velocities
- **`graph_features.py`**:  
  NetworkX topological metrics (PageRank, degree centrality, connected component size) computed with strict chronological cutoff boundaries (`cutoff_date <= TRAIN_END` for training).

### 2. Decisioning, Calibration & Conformal Coverage
- **`advanced_trust_engine.py`**:  
  Stacking meta-learner combining tabular gradient boosted models with relational GNN embeddings. Implements split conformal uncertainty guarantees to yield valid prediction sets ($\{0\}$, $\{1\}$, or $\{0, 1\}$) with $\ge 95\%$ coverage guarantees ($\alpha = 0.05$).
- **`trust_engine.py`**:  
  Unified Trust Engine orchestrating multi-detector risks into an aggregate trust score ($0 - 100$), binary Shannon entropy confidence ($1 - H_2(p)$), model disagreement spread, reason codes, and operational routing tiers (`ALLOW`, `REVIEW`, `HOLD`, `BLOCK`).
- **`calibration.py`**:  
  Post-hoc probability calibration utilizing validation-fitted Isotonic Regression and Platt scaling. Evaluates Expected Calibration Error (ECE) and Brier Score.

### 3. TreeSHAP Explainability & Narrative Synthesis
- **`shap_explainer.py`**:  
  Production TreeSHAP explainability engine computing exact local Shapley attributions in sub-10ms. Separates positive risk contributors (amplifying fraud likelihood) from negative risk mitigations (protective factors). Translates feature keys to human-friendly definitions (including GNN latent dimensions) and synthesizes plain-English investigator summaries.
- **`test_shap_explainer.py`**:  
  Unit and integration test suite (8 tests) verifying explainer initialization, driver rankings, friendly name resolution, DataFrame/Series formats, and FastAPI integration.

### 4. Multimodal & Forensic Investigation
- **`multimodal_clip_faiss.py`**:  
  CLIP vision-language feature extraction combined with FAISS sub-millisecond similarity index. Features strict self-match exclusion by listing identity and detects cross-seller image reuse and title-image semantic divergence.
- **`investigation_agent.py`**:  
  Autonomous fraud ops agent generating human-readable forensic dossiers. Features an automated evidence grounding verification guard that cross-references all claims against extracted graph evidence.
- **`investigation_rag.py`**:  
  Dense vector retrieval-augmented generation (RAG) indexing fraud typology playbooks and past incident resolutions.
- **`neo4j_investigator.py`**:  
  Cypher query generators extracting 2-hop ego networks and collusion paths for forensic review.

### 5. Robustness, Splitting & Testing
- **`splits.py`**:  
  Strict chronological data partitioning (`TRAIN_END`, `VAL_END`) preventing temporal snooping.
- **`missingness.py`**:  
  Missing data handling with explicit indicator flags preserving missing-not-at-random (MNAR) signals.
- **`robustness.py`**:  
  Statistical validation through non-parametric bootstrapping and multi-seed sensitivity analysis.
- **`reproducibility.py`**:  
  Deterministic RNG seed isolation, hardware fingerprinting, and environment capture.
- **`versioning.py`**:  
  SHA-256 artifact hashing and catalog manifest generation for deployed models.
- **`mlops_pipeline.py`**:  
  Offline-to-online feature store simulation and population stability index (PSI) drift monitoring.
- **`test_repair_pipeline_regression.py`**:  
  20-point temporal invariant regression suite asserting no future leakage, monotonic stacking, FAISS self-match exclusion, and offline-online scoring parity.
- **`test_seed_mesh.py`**:  
  Unit test suite for Neo4j Cypher constraint/collusion generation and Redis RESP serialization protocols.

---

## 🧪 Running the Intelligence Test Suite

```bash
# 1. Run master technical repair regression suite (20 tests)
pytest trustshield_project/test_repair_pipeline_regression.py -v

# 2. Run TreeSHAP explainability unit & integration tests (8 tests)
pytest trustshield_project/test_shap_explainer.py -v

# 3. Run comprehensive Phase 2 suite (16 tests)
pytest trustshield_project/test_phase2_suite.py -v

# 4. Run Neo4j & Redis mesh seeder tests (5 tests)
pytest trustshield_project/test_seed_mesh.py -v

# 5. Run temporal leakage and data integrity guards
pytest trustshield_project/test_leakage.py -v

# 6. Run audit regression tests
pytest trustshield_project/test_audit_fixes.py -v
pytest trustshield_project/test_core_fixes.py -v
```

---

## 📚 Governance & Research Documentation

Detailed scientific audit logs and architectural reports are maintained in [`docs/`](../docs/README.md):
- [**System Design & Core Specifications**](../docs/SPECIFICATIONS.md)
- [**Master Technical Repair Report**](../docs/FINAL_REPAIR_REPORT.md)
- [**Repair Baseline State**](../docs/REPAIR_BASELINE.md)
- [**Phase 0: Baseline Audit**](../docs/BASELINE_AUDIT.md)
- [**Phase 1: 49-Point Scientific Re-Audit**](../docs/49_POINT_REAUDIT.md)
- [**Phase 1: Final Before/After Report**](../docs/FINAL_BEFORE_AFTER.md)
- [**Phase 2: Baseline State Inspection**](../docs/PHASE2_BASELINE.md)
- [**Phase 2: Advanced Final Research Report**](../docs/PHASE2_FINAL_REPORT.md)
- [**System Model Card**](../docs/MODEL_CARD.md)
- [**Technical Limitations & Disclosure**](../docs/LIMITATIONS.md)
- [**Tracked Bug Registry**](../docs/BUG_INVENTORY.json)