# TrustShield — Phase 0 Baseline Audit

> **Audit date:** 2026-10-05
> **Status:** READ-ONLY — no source code was modified during this phase
> **Baseline test run:** pytest 71 selected tests (18 CLIP deselected), Python 3.14.7

---

## 1. Repository Structure

```
trustshield_full_handoff/
├── trustshield_project/          ← core ML pipeline (flat layout)
│   ├── entity_generator.py       ← 8-entity synthetic marketplace generation
│   ├── product_listing_generator.py
│   ├── order_return_generator.py
│   ├── fraud_injection.py        ← 4 fraud scenarios
│   ├── baseline_model.py         ← Phase 1C: combined tabular models
│   ├── phase2_specialized_models.py ← Phase 2: specialized detectors
│   ├── graph_features.py         ← Phase 3: graph features + ring detection
│   ├── gnn_model.py              ← standalone GraphSAGE
│   ├── phase5_hybrid_model.py    ← Phase 5: hybrid GraphSAGE + XGBoost
│   ├── multimodal_scoring.py     ← Phase 4: CLIP + TF-IDF scorer
│   ├── export_dataset.py         ← CSV/Parquet/PostgreSQL export
│   ├── utils.py                  ← evaluate(), find_cost_optimal_threshold()
│   └── conftest.py               ← pytest fixtures
├── backend/
│   ├── main.py                   ← FastAPI 4-endpoint serving layer
│   ├── model_loader.py           ← joblib model store
│   └── schemas.py                ← Pydantic v2 schemas
├── docs/, models/, scripts/
└── requirements.txt, pytest.ini
```

---

## 2. Current Architecture

### 2.1 Data Generation

8-entity relational marketplace, 12-month synthetic timeline (Jan–Dec 2025).

| Entity | Count | Notes |
|--------|-------|-------|
| Address | ~4,950 | 30 regions, first_seen_date |
| Device | 5,000 | fingerprints, first_seen_date |
| Seller | 500 | 10 categories, staggered signup |
| Buyer | 5,000 | staggered signup |
| Product | ~3,000 | ABO real catalog or synthetic fallback |
| Listing | ~20,000 | per run |
| Order | ~50,000 | per run |
| Return | subset | of orders |

Fully deterministic from RNG_SEED=42. Reproducibility verified by md5 checksums.

### 2.2 Fraud Injection — 4 Scenarios

| Scenario | Target | Mechanism | ~Proportion |
|----------|--------|-----------|------------|
| Fake Listing | Listings | Price undercut + image/text mismatch | 40% of fraud |
| Return Abuse | Returns | High return rate, fast (1-5 days) | 30% of fraud |
| Coordinated Fraud | Orders | Device-sharing rings, 3-7 day burst | 20% of fraud |
| Seller-Buyer Collusion | Orders | Seller+buyer group, 5-14 day burst | 10% of fraud |

Overall fraud rate: ~7%. fraud_ring_id NEVER in any feature table (separate ledger).

### 2.3 Current Models

#### Phase 1C — Combined Transaction Classifier
Features (10): price_vs_base_price_ratio, price_vs_category_median_ratio, seller_age_days,
seller_total_listings_before, buyer_age_days, buyer_orders_before, buyer_returns_before,
buyer_return_rate_before, device_shared_buyer_count, amount.
Models: Naive rule, Logistic Regression, Random Forest, XGBoost.
Top feature: buyer_return_rate_before (XGBoost importance: 0.363).

#### Phase 2 — Specialized Detectors
- Fake Listing Detector: listing-level, 5 features incl. multimodal_similarity_score
- Return Fraud Detector: return-level, 10+ features incl. days_to_return
Both use cost-optimal threshold search on validation set.

#### Phase 3 — Graph-Augmented Combined Model
Graph features (8): share_degree, share_component_size, buyer_seller_degree,
buyer_pagerank, seller_buyer_degree, seller_pagerank, seller_buyer_concentration_hhi,
buyer_seller_edge_weight_before.
Total: 18 features. Graph library: NetworkX (homogeneous, undirected).

#### Phase 4 — CLIP Multimodal Scorer
- CLIPMultimodalScorer: real CLIP (openai/clip-vit-base-patch32) image-text cosine similarity
- MultimodalScorer: TF-IDF fallback (CPU-only, no extra deps)
- Embedding cache: SHA-256 fingerprinted, compressed NumPy archives

#### Phase 5 — Hybrid GraphSAGE + XGBoost
- GraphSAGEEncoder: 2-layer SAGEConv (7->32->16), 50 epochs, best-val-AUC checkpoint
- EdgeClassifier: MLP on concatenated embeddings + edge features
- XGBClassifier: 18 phase-3 features + 32 GNN dims = 50 total features
- Temporal safety: encoder trained on <=TRAIN_END graph, test embeddings from <=VAL_END

### 2.4 Current Graph Implementation

| Graph | Library | Nodes | Edges | Temporal? |
|-------|---------|-------|-------|-----------|
| Relationship graph | NetworkX undirected | Buyers | Shared addresses + devices | Yes — first_seen_date cutoff |
| Bipartite snapshot | NetworkX | Buyers + Sellers | Order edges | Yes — monthly cumulative |
| GNN graph | PyTorch Geometric | Buyers + Sellers | Orders + sharing | Yes — TRAIN_END/VAL_END cutoff |

GRAPH TYPE: HOMOGENEOUS. Entity types not typed in GNN — relationship semantics partially lost.
Ring detection: connected components only. Uses "suspected rings" terminology (correct).

### 2.5 Current GNN Implementation

GraphSAGEEncoder:
  SAGEConv(7->32) -> ReLU -> SAGEConv(32->16)
EdgeClassifier:
  Linear(34->32) -> ReLU -> Linear(32->1)
Node features: [is_buyer, is_seller, age/365, log1p(orders), log1p(avg_amount), degree, component_size]
Edge features: [price_vs_base_price_ratio, log1p(amount)]
Loss: BCE with pos_weight. Optimizer: Adam lr=0.002, gradient clipping 1.0.

### 2.6 Current API

Framework: FastAPI 0.141.1 + Uvicorn 0.52.4 + Pydantic v2.13.5

| Method | Endpoint | Model | Description |
|--------|----------|-------|-------------|
| GET | /health | — | Liveness + model load status |
| POST | /transaction/score | Phase 5 hybrid (fallback: Phase 3 RF) | Transaction fraud probability |
| GET | /fraud-rings | Phase 3 RF | Pre-computed ring list |
| POST | /listing/analyze | Phase 4 CLIP + Phase 2 XGBoost | Fake listing + narrative |

NO Trust Engine. NO ALLOW/REVIEW/HOLD/BLOCK. Risk label is 3-band only (low/medium/high).

### 2.7 Feature Pipeline — Temporal Safety

Mechanism: pd.merge_asof(direction="backward", allow_exact_matches=False)
Applied in: _cumulative_count_asof(), _asof_cumulative_from_events(), add_edge_weight_before(), _strict_prior_cumcount()
Category median: computed on training set only (no leakage).
Device shared buyer count: computed on training set only (no leakage).

Train/Val/Test split:
  Train: order_date <= 2025-08-31
  Val:   2025-09-01 to 2025-10-31
  Test:  > 2025-10-31

### 2.8 Current Metrics (README — real reproduced numbers)

Phase 1C XGBoost (test set):
  Precision: 0.424 | Recall: 0.220 | F1: 0.289 | ROC-AUC: 0.654 | PR-AUC: 0.270

Phase 2 Return Fraud XGBoost (test set):
  Default threshold: P=0.919, R=0.605, F1=0.729, ROC-AUC=0.951
  Cost-optimal (0.01): P=0.734, R=0.905, F1=0.810

Phase 3 Graph Ablation (XGBoost, test set):
  Tabular-only:    P=0.520, R=0.190, F1=0.278, ROC-AUC=0.651, PR-AUC=0.265
  Tabular+Graph:   P=0.605, R=0.203, F1=0.304, ROC-AUC=0.680, PR-AUC=0.302
  (Graph adds: +0.029 ROC-AUC, +0.037 PR-AUC, +14% precision)

Phase 5 Hybrid (test set):
  Standalone GraphSAGE: ROC-AUC=0.519, P=0.091, R=0.939, F1=0.167
  Hybrid GNN+XGBoost:   ROC-AUC=0.696, P=0.580, R=0.180, F1=0.275 (Val ROC-AUC: 0.792)

### 2.9 Current Test Suite

Total: 89 collected (71 run, 18 CLIP deselected without ABO data)

| File | Tests | Coverage |
|------|-------|----------|
| test_data_integrity.py | ~15 | Entity counts, dtypes, column presence |
| test_fraud_injection.py | ~8 | Fraud rates, ring IDs, leakage guards |
| test_graph_and_models.py | ~20 | Graph, ring detection, model evaluation |
| test_leakage.py | ~15 | Temporal safety, same-timestamp regression, graph cutoff |
| test_multimodal_clip.py | ~18 | CLIP, TF-IDF, embedding cache (CLIP-marked) |
| test_backend.py | ~13 | API smoke tests, schema, response fields |

### 2.10 Current Dependencies

```
numpy>=1.24, pandas>=2.0, scikit-learn>=1.3, networkx>=3.1, xgboost>=2.0
pyarrow>=14.0, transformers>=4.37, Pillow>=10.0, torch>=2.0, torch-geometric>=2.4
fastapi>=0.110, uvicorn[standard]>=0.29, pydantic>=2.0, httpx>=0.27.0
joblib>=1.3, pytest>=8.0
# Optional: sqlalchemy>=2.0, psycopg2-binary>=2.9
# Commented out: shap>=0.44, mlflow>=2.10
```

Installed versions: XGBoost 3.4.1, PyTorch 2.13.0 CPU, PyG 2.8.0, Python 3.14.7
PINNING GAP: requirements.txt uses >= only — no lock file, no upper bounds.

---

## 3. Current Limitations

### Architecture
1. HOMOGENEOUS GRAPH: No typed nodes/edges in GNN. Relationship semantics partially lost.
2. NO TRUST ENGINE: No unified ALLOW/REVIEW/HOLD/BLOCK decision layer.
3. NO CALIBRATION: Raw model probabilities — not calibrated to true posteriors.
4. NO SHAP: shap commented out. Investigator narrative is rule-based, not SHAP-driven.
5. NO MODEL DISAGREEMENT: Hybrid model does not surface inter-model disagreement.
6. FLAT MODULE LAYOUT: All files in trustshield_project/ — no src/ separation.

### Evaluation
7. NO BOOTSTRAP CIs: Point estimates only — no metric uncertainty.
8. NO MULTI-SEED: All results from seed=42 only.
9. NO ROBUSTNESS EXPERIMENTS: No changed-fraud-regime testing.
10. NO PREVALENCE SENSITIVITY: Only ~7% fraud rate tested.
11. NO TEMPORAL DRIFT ANALYSIS: No per-period degradation reporting.
12. NO FAIRNESS/SEGMENT ANALYSIS: No stratification by seller/buyer tenure.

### Infrastructure
13. NO MLFLOW: No experiment tracking.
14. NO DOCKER: No container.
15. NO CI/CD: No GitHub Actions.
16. NO DVC: No large file versioning.
17. DEPENDENCY PINNING GAP: >= bounds only, no lock file.
18. IN-MEMORY ONLY: No database. All data in DataFrames per run.

### Graph
19. TEMPORAL LEAKAGE RISK: run_phase_3() calls build_relationship_graph() without
    cutoff_date for the ring detection path — uses ALL future sharing relationships.
    (The GNN path in phase5_hybrid_model.py correctly uses cutoff_date.)
20. SIMPLE RING DETECTION: Connected components only. No Louvain/Leiden. No ring-level
    features (temporal burst score, density, shared device count per ring).
21. NO COLD-START CONFIDENCE PENALTY: Cold-start entities receive zero GNN embeddings
    but no confidence reduction is exposed in the API response.
