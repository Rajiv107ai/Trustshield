# TrustShield AI — System Design & Core Specifications

**Platform:** TrustShield AI: Relational E-Commerce Fraud Intelligence Platform  
**Specification Version:** 2.0 (Consolidated Architecture & Standards)  
**Status:** Authoritative Foundation Document  

---

## 1. Executive Overview & Problem Definition

### 1.1 Mission & Research Problem
**TrustShield AI** is an advanced, technically defensible e-commerce fraud-intelligence platform. It unifies transaction-level risk, seller catalog integrity, return abuse detection, and entity-relationship collusion discovery into an interconnected **Trust Graph**.

The primary scientific research inquiry driving this architecture:
> **Central Research Question:** Does combining behavioral, multimodal, and graph-based relational information improve the detection of coordinated e-commerce fraud compared to isolated transaction-level models?

Secondary research questions resolved by the system:
- **Graph Lift:** Does topological graph structure provide statistically significant lift over rich tabular behavioral features?
- **Multimodal Lift:** Does multimodal cross-seller image-text alignment flag counterfeit and catalog abuse that purely numerical features miss?
- **Collusion Discovery:** Can coordinated fraud rings and bursty sharing clusters be detected systematically rather than treating transactions in isolation?
- **Relational vs. Homogeneous Learning:** Does a typed Heterogeneous GNN (`HeteroData`) outperform homogeneous graph representations?
- **Decision Boundary & Cost Asymmetry:** How do asymmetric false-positive vs. false-negative cost assumptions alter the optimal operational decision threshold?

### 1.2 Technology Stack
The platform leverages a purpose-built, modular stack:
- **Core ML & Data:** Python 3.10+, Pandas, NumPy, Scikit-learn, XGBoost.
- **Deep Representation & Graph Learning:** PyTorch, PyTorch Geometric (`torch_geometric`), Transformers (CLIP), FAISS vector similarity.
- **Serving Layer:** FastAPI, Pydantic v2 (boundary schemas), Uvicorn.
- **Graph & Cache Infrastructure:** Neo4j (Cypher property graph), Redis (16D embedding store).
- **Observability:** Prometheus metrics exporter, Grafana live dashboards.
- **Explainability:** TreeSHAP exact local Shapley attributions, LLM forensic dossier synthesis.
- **Containerization & CI/CD:** Docker multi-stage builds, Docker Compose service mesh, GitHub Actions CI.

---

## 2. Entity-Relationship Schema & Architecture

### 2.1 The 8 Core Entities
TrustShield models marketplace dynamics across 8 distinct relational entities:

```
[Seller] ──creates──> [Listing] ──based_on──> [Product]
   │                      │
   │                      ▼
   │                  [Order] ──generates──> [Return]
   │                      │
   ▼                      ▼
[Address] <─────────── [Buyer] ───────────> [Device]
```

1. **Seller:** Merchants with onboarding timestamps, tenure, store rating, and registration IP.
2. **Buyer:** Purchasing accounts with account creation timestamps, order history, and historical return velocities.
3. **Product:** Universal catalog items with manufacturer specifications, reference images, and baseline category median prices.
4. **Listing:** Seller offerings tying a product to a seller, price, textual description, and uploaded image.
5. **Order:** Financial transactions linking a buyer, listing, quantity, dollar amount, device, and timestamp.
6. **Return:** Post-purchase dispute/return events with claim reasons and outcome states.
7. **Device:** Unique client hardware fingerprint identifiers.
8. **Address:** Physical delivery and billing postal locations.

> **Ground Truth Isolation Rule:**  
> The generator maintains an isolated `fraud_ring_id` and ground-truth scenario ledger. This metadata is strictly decoupled and **never stored as a feature** in any model-facing table, preventing circular target lookup.

### 2.2 Pipeline Dependency DAG
The end-to-end data and model pipeline follows a strict topological execution DAG:

```
entity_generator.py (Addresses, Devices, Sellers, Buyers)
        │
        ▼
product_listing_generator.py (Product Catalog, Seller Listings)
        │
        ▼
order_return_generator.py (Chronological Orders, Organic Returns)
        │
        ▼
fraud_injection.py (Layered Injections & Ground-Truth Ledger)
        │
        ├──> baseline_model.py (Phase 1C: Tabular Baseline)
        ├──> phase2_specialized_models.py (Phase 2: Listing & Return Models)
        ├──> graph_features.py (Phase 3: Snapshot Graph Topological Features)
        ├──> hetero_gnn.py / temporal_gnn.py (Phase 5: HeteroData & Time2Vec GNN)
        ├──> multimodal_clip_faiss.py (CLIP & FAISS Vector Index)
        └──> advanced_trust_engine.py (Stacking Meta-Learner & Conformal Bounds)
```

Each module can execute deterministically from upstream artifacts or rebuild its lineage independently.

---

## 3. Synthetic Fraud Design & Invariants

### 3.1 Scenario Typologies & Distribution
Marketplace fraud is injected synthetically to ensure precise causal evaluation across 4 distinct typologies, calibrated to a realistic ~6–8% aggregate fraud prevalence:

1. **Fake Listings (~40% of fraud):**
   - Dramatically discounted or inflated pricing relative to category medians ($price\_ratio \ll 1.0$ or $\gg 1.0$).
   - Textual description discordance and image-text semantic mismatch.
   - Cross-seller catalog image duplication.
2. **Return Abuse & Wardrobing (~30% of fraud):**
   - High return frequency shortly after receipt.
   - Claim discrepancies (e.g., claiming "damaged" or "wrong item" systematically).
   - High refund amounts relative to buyer transaction tenure.
3. **Coordinated Device Rings (~20% of fraud):**
   - Collusion clusters where multiple distinct buyer accounts share identical hardware device fingerprints.
   - High inter-order velocity and burstiness.
4. **Seller-Buyer Collusion (~10% of fraud):**
   - Rapid reciprocal orders placed between tightly coupled buyer and seller nodes sharing geographic addresses or IP ranges.
   - Artificial rating inflation and fake transaction turnover.

### 3.2 Topological Sharing Rates
To prevent sharing signals from becoming trivial shortcut features, sharing rates are intentionally conservative:
- **Address Sharing:** ~12% of buyers share addresses (70% legitimate multi-resident households, 30% fraud-ring linked).
- **Device Sharing:** ~8% of buyers share hardware (50% legitimate shared family devices, 50% fraud-ring linked).

### 3.3 Strict Temporal Split Boundaries
Data generation spans a simulated 12-month chronological horizon with non-shuffled temporal partitioning:
- **Training Interval:** Month 1 through Month 8 ($t \le 2024\text{-}02\text{-}15$)
- **Validation Interval:** Month 9 through Month 10 ($2024\text{-}02\text{-}15 < t \le 2024\text{-}03\text{-}01$)
- **Test Holdout Interval:** Month 11 through Month 12 ($t > 2024\text{-}03\text{-}01$)

---

## 4. Engineering Principles & Leakage Discipline

### 4.1 Strict Temporal Safety Invariant
The fundamental governing invariant across all features, embeddings, and graphs:
$$\text{event\_time} < \text{decision\_time}$$

All historical cumulative statistics (e.g., `buyer_orders_before`, `buyer_returns_before`) must be computed using backward-asof merges without exact match equality:
```python
pd.merge_asof(
    events,
    historical_aggregates,
    on="timestamp",
    by="entity_id",
    direction="backward",
    allow_exact_matches=False,
)
```

### 4.2 Graph Snapshot Isolation
- **Static Relationship Graph:** Derived from structural hardware and address sharing. Never includes edge ground-truth labels.
- **Dynamic Transaction Graph:** Built strictly from historical monthly snapshots. An order occurring at time $t$ in month $M$ only queries graph topology state compiled as of the end of month $M-1$. Future order relationships are completely invisible to the graph engine.

### 4.3 Anti-Feedback & Circularity Guards
- Models must never receive outputs of downstream or concurrent models as features (e.g., `trust_score`, `risk_score`, `overall_fraud_probability`, or `decision`).
- Fraud injection mechanism flags (e.g., `is_price_anomaly`, `is_copied_image`) are internal ground truth and strictly barred from feature sets.

### 4.4 Cost-Asymmetric Decision Thresholds
Rather than assuming an arbitrary default decision threshold of 0.5:
- **False Negative Cost ($C_{FN}$):** Proportionate to transaction dollar loss ($order\_amount$).
- **False Positive Cost ($C_{FP}$):** Human operational review cost (calibrated at flat \$50 review cost).
- Operating thresholds are tuned on the validation set to minimize expected commercial cost before deployment.

---

## 5. Testing & Verification Framework

### 5.1 Layered Verification Strategy
The platform enforces a four-tier testing hierarchy (214 automated tests):
1. **Generator & Integrity Tests (`test_data_integrity.py`, `test_fraud_injection.py`):**
   - Exact volume counts, referential foreign-key integrity, non-negative entity tenure.
   - Ordering monotonicity ($t_{order} \ge t_{listing} \ge t_{signup}$; $t_{return} \ge t_{order}$).
2. **Leakage & Temporal Regression Tests (`test_leakage.py`, `test_repair_pipeline_regression.py`):**
   - Executable assertions confirming zero future timestamps in graph nodes or tabular features.
   - Backward-asof boundary verifications.
3. **Model & Explainability Tests (`test_graph_and_models.py`, `test_phase2_suite.py`, `test_shap_explainer.py`):**
   - Bounded prediction ranges, isotonic calibration monotonicity, TreeSHAP attribution completeness ($\sum \phi_i = f(x) - E[f(x)]$).
4. **Serving Layer & Service Mesh Tests (`test_backend.py`, `test_new_extensions.py`, `test_services.py`):**
   - Schema conformance, live fallback under Neo4j/Redis disconnection, SSE streaming lifecycle, and Prometheus metric formatting.

---

## 6. Phased Evolution & Delivery Roadmap

| Phase | Milestone Name | Focus & Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **Phase 1** | Foundation & Baseline | Synthetic generator, IEEE-CIS + ABO datasets, combined tabular baseline. | **Complete** |
| **Phase 2** | Specialized Detectors | Dedicated Listing Detector, Return Fraud Detector, cost-optimal tuning. | **Complete** |
| **Phase 3** | Graph Topological Learning | Trust Graph, snapshot isolation, NetworkX degree/PageRank features, fraud ring detection. | **Complete** |
| **Phase 4** | Multimodal Intelligence | ABO catalog integration, CLIP image/text embeddings, FAISS sub-millisecond search. | **Complete** |
| **Phase 5** | Heterogeneous & Temporal GNN | PyTorch Geometric `HeteroData`, $Time2Vec$ continuous time, XGBoost hybrid stacking. | **Complete** |
| **Phase 6** | Enterprise Serving Layer | FastAPI gateway, Pydantic v2 schemas, live Neo4j + Redis mesh, TreeSHAP explainer, Next.js console. | **Complete** |
| **Phase 7** | Cloud-Native MLOps & Governance | Prometheus latency histograms, Grafana dashboards, empirical robustness & scalability reports, CI/CD. | **Complete** |
