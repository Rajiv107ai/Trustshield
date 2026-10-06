# TrustShield AI — Phase 2 Baseline Audit & Inspection Report

**Inspection Date:** 2026-10-07  
**Auditor:** Senior ML Researcher + Backend Systems Architect  
**Source Repository:** `https://github.com/Rajiv107ai/Trustshield`  
**Test Suite State:** 125 passing tests (100% green)  
**Environment:** Python 3.14.7 | PyTorch 2.14.1 | PyG (torch-geometric) 2.8.0 | FAISS 1.15.1 | XGBoost 3.4.1 | Scikit-Learn 1.9.1  

---

## 1. Baseline Verification & Current System State

Prior to introducing any Phase 2 extensions, the Phase 1 codebase was inspected and audited:
- **Test Integrity:** 125 automated unit and regression tests pass with zero errors.
- **Temporal Correctness:** All historical joins utilize backward-asof matching without exact equality (`allow_exact_matches=False`). Graph construction for training is strictly partitioned at `cutoff_date <= TRAIN_END` to eliminate temporal leakage.
- **Anti-Feedback Guards:** Forbidden model outputs (`trust_score`, `risk_score`, `overall_fraud_probability`, `decision`, etc.) are blocked by assert checks in `leakage_audit()` and `TrustEngine`.

---

## 2. Relational Schema & Entity Map

TrustShield models an 8-entity e-commerce relational structure:
```
Seller ──creates──> Listing
Buyer ──places──> Order
Order ──contains──> Listing
Buyer ──uses──> Device
Seller ──uses──> Device
Buyer ──uses──> Address
Seller ──uses──> Address
Order ──generates──> Return
```
- **Sellers:** 1,000 entities with onboarding dates, tenure, rating, registration IP.
- **Buyers:** 10,000 entities with account creation dates, age, orders before, returns before.
- **Catalog & Listings:** 5,000 products, prices, base category median prices, text descriptions, image paths.
- **Orders & Returns:** Chronologically ordered transactions over a 12-month simulation window.

---

## 3. Baseline Model Performance Benchmark

Evaluated on the frozen holdout temporal test partition (`order_date > VAL_END`):

| Architecture / Model | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | Review Rate | FP / 1,000 | Latency (p95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 Tabular Baseline (RF)** | 0.742 | 0.418 | 0.612 | 0.540 | 0.574 | 8.8% | 34.1 | 4.2 ms |
| **Phase 3 Tabular + Graph (RF)** | 0.814 | 0.528 | 0.704 | 0.648 | 0.675 | 11.2% | 27.2 | 8.6 ms |
| **Existing Standalone GNN (2-layer GCN)** | 0.519 | 0.114 | 0.220 | 0.190 | 0.204 | 14.5% | 88.0 | 38.5 ms |
| **Phase 5 Hybrid GNN + XGBoost** | 0.696 | 0.384 | 0.589 | 0.512 | 0.548 | 9.4% | 41.6 | 26.2 ms |
| **Phase 1 Unified Trust Engine** | **0.841** | **0.586** | **0.748** | **0.692** | **0.719** | **7.5%** | **19.4** | **11.4 ms** |

---

## 4. Architectural Deficits to Address in Phase 2

1. **Homogeneous Graph Collapse:** The Phase 3 and Phase 5 graphs collapse multi-relational edges into homogeneous adjacency. A buyer sharing an address with a seller is treated identically to a buyer purchasing an item. A true **Heterogeneous GNN (`HeteroData`)** is required to preserve typed relational semantics.
2. **Static Snapshot Limitation:** Existing graph embeddings ignore dynamic edge timestamps. A **Temporal GNN** study is needed to evaluate whether time-decayed attention or temporal edge sequences yield measurable signal over static features.
3. **Multimodal Heuristics:** The Fake Listing Detector currently falls back to synthetic cosine similarity ($0.85$) when product images are missing. Real **CLIP embeddings + FAISS vector indexing** must be integrated to enable real nearest-neighbor visual anomaly detection.
4. **Trust Engine Stacking:** The Phase 1 Trust Engine utilizes weighted linear aggregation. Phase 2 must evaluate **Stacking / Meta-Learners** against weighted ensembles and provide conformal prediction intervals.
5. **Community-Level Ring Intelligence:** Connected components flag broad clusters without scoring ring risk. Phase 2 must implement **candidate community burstiness and sub-graph risk scoring**.
6. **Investigative Explanations:** The API provides raw metrics. A **GenAI Forensic Investigation Agent + RAG** layer is needed to synthesize structured evidence into investigator dossiers without determining the fraud probability.
