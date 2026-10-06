# TrustShield AI — Full Project Audit Baseline

**Document Revision:** 2026-10-07  
**Role:** Master Project Audit, Bug Fix & Validation Agent  
**Source Repository:** `https://github.com/Rajiv107ai/Trustshield`  

---

## 1. Repository Structure & Artifact Inventory

The repository contains 84 active files organized into the following functional layout:
- **`trustshield_project/`**: Core simulation pipeline, feature generation, specialized models, graph topology, GNN architectures, Trust Engine, and test suites.
- **`backend/`**: FastAPI production gateway, Pydantic v2 schemas, model loader, and endpoint unit tests.
- **`models/`**: Persisted model artifacts (`combined_graph_model.joblib`, `hybrid_model.joblib`, `fake_listing_model.joblib`, `return_fraud_model.joblib`, `fraud_rings.joblib`, embeddings, CLIP caches).
- **`scripts/`**: Offline training and embedding generation utility scripts.
- **`docs/`**: Technical audits, model cards, architectural roadmaps, and reports.

---

## 2. Main Components & Responsibilities

| Component | Files | Primary Responsibility |
| :--- | :--- | :--- |
| **Data Generation** | `entity_generator.py`, `product_listing_generator.py`, `order_return_generator.py` | Synthesize 8 marketplace entities over a 12-month timeline with realistic onboarding and organic return rates. |
| **Fraud Injection** | `fraud_injection.py` | Inject 4 distinct fraud typologies (fake listings, return abuse, hardware rings, seller-buyer collusion). |
| **Feature Engineering** | `baseline_model.py`, `graph_features.py` | Generate behavioral, ratio, and topological graph features with backward-asof temporal safety. |
| **Specialized Detectors** | `phase2_specialized_models.py`, `multimodal_scoring.py` | Train and score fake listing (text-image) and return fraud detectors. |
| **Graph & GNN** | `graph_features.py`, `gnn_model.py`, `hetero_gnn.py`, `temporal_gnn.py` | Construct multi-relational graphs, train GNN message passing, and export embeddings. |
| **Trust Engine** | `trust_engine.py`, `advanced_trust_engine.py` | Calibrate and aggregate component risks into operational decisions (`ALLOW`, `REVIEW`, `HOLD`, `BLOCK`). |
| **API & Inference** | `backend/main.py`, `backend/model_loader.py`, `backend/schemas.py` | FastAPI gateway serving `/health`, `/ready`, `/transaction/score`, `/fraud-rings`, `/listing/analyze`. |
| **Forensic Forensics** | `neo4j_investigator.py`, `investigation_rag.py`, `investigation_agent.py` | Subgraph queries, policy retrieval, and evidence-grounded forensic dossier synthesis. |

---

## 3. Data & Execution Flow

```
1. Base Entities (Sellers, Buyers, Devices, Addresses)
   ↓
2. Catalog & Listings (Products, Listings, Pricing)
   ↓
3. Transactions (Orders, Returns with organic baseline)
   ↓
4. Fraud Injection (4 Scenarios with ground-truth ledger)
   ↓
5. Feature Construction (Strict temporal backward joins)
   ↓
6. Chronological Partitioning (Train <= 2024-02-15 < Val <= 2024-03-01 < Test)
   ↓
7. Model Training (Tabular RF/XGB, HeteroGNN, Specialized Models)
   ↓
8. Model Persistence (Serialized to models/*.joblib)
   ↓
9. Online Serving (FastAPI loads artifacts → Trust Engine routes decisions)
```

---

## 4. Dependencies & Runtime Environment

- **Python:** 3.14.7 (Windows x86_64)
- **PyTorch:** 2.14.1
- **PyTorch Geometric (PyG):** 2.8.0.post1
- **FAISS:** 1.15.1 (faiss-cpu)
- **Scikit-Learn:** 1.9.1
- **XGBoost:** 3.4.1
- **FastAPI / Pydantic:** 0.141.1 / 2.13.5
- **NetworkX:** 3.6.1

---

## 5. Potential Failure Points & Suspicious Areas for Deep Audit

1. **Feature Ordering Mismatch in FastAPI:** Verify whether columns extracted by `_build_feature_row()` in `backend/main.py` match the exact feature names and sequence in `feature_meta["all_feature_cols"]`.
2. **Missing Input Normalization:** Verify what happens when raw inputs (`order_amount` vs `amount`) are both supplied or when values are `None`.
3. **Temporal Graph Cutoff Consistency:** Ensure all graph builders consistently isolate training graphs strictly before `TRAIN_END` and evaluation graphs before `VAL_END`.
4. **Silent NaNs in GNN Feature Projections:** Ensure missing entity features (e.g. buyers with zero orders) are imputed cleanly without introducing NaNs into PyTorch tensors.
5. **Trust Engine Aggregation Invariance:** Ensure no single component detector can crash the aggregator if an optional detector score is omitted.
