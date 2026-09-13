# TrustShield AI — Architecture

## Entity-relationship schema (locked, Phase 1B)
8 entities: **Seller, Buyer, Product, Listing, Order, Return, Device,
Address**.

Relationships:
- Seller `creates` Listing
- Listing `based_on` Product
- Buyer `purchases` Order
- Order `contains` Listing
- Buyer `requests` Return
- Return `belongs_to` Order
- Buyer `uses` Device
- Order `used_device` Device
- Buyer / Seller `associated_with` Address

`fraud_ring_id` is a **ground-truth-only generator label**, never a stored
feature on any of the above tables — kept in a separate ledger to avoid
leakage (a model must never be able to look up the answer via an ID
column).

## Pipeline (implementation order = dependency order)

```
entity_generator.py          Address, Device, Seller, Buyer
        |
product_listing_generator.py Product catalog + Listings
        |
order_return_generator.py    Orders + Returns
        |
fraud_injection.py           4 fraud scenarios layered on top
        |
        +-- baseline_model.py            Phase 1C: combined tabular baseline
        +-- phase2_specialized_models.py Phase 2: listing/return specialized detectors
        +-- graph_features.py            Phase 3: timestamp-snapshotted graph features
        +-- gnn_model.py                 Phase 3 (GNN upgrade): GraphSAGE via PyTorch Geometric
```

Each phase script rebuilds the full upstream pipeline itself — there is no
shared database or cache between phases. This keeps every phase
independently runnable and independently verifiable.

## Trust Graph design
Two feature families, each with its own temporal-safety mechanism:

1. **Static relationship graph** (device sharing + address sharing) — no
   per-edge timestamp exists in the schema (a documented limitation), so
   these edges are treated as available for a buyer's whole lifetime.
   Built from share EXISTENCE only, regardless of `share_type`
   (fraud_linked vs. legitimate) — using the type label would leak the
   ground truth into a "structural" feature.
2. **Monthly snapshot graph** (buyer-seller order edges) — genuinely
   temporal. An order in month M only ever uses the graph snapshot as of
   the END of month M-1, never its own month.

Graph features currently implemented (Phase 3, networkx-based): degree,
PageRank, connected-component size, buyer-seller edge weight, and
seller-buyer concentration (HHI). A GNN (GraphSAGE) upgrade path exists in
`gnn_model.py`, using the same underlying graph structures.

## Data strategy
Real product catalog (images + text — Amazon Berkeley Objects) for
realism, with fraud ground truth generated entirely synthetically via
injected corruption/anomalies (price anomaly, image-text mismatch, copied
images, different-item returns). Kept as **one single consistent domain**
across tabular, image, and text data, rather than mixing in a separate
counterfeit-specific dataset — prioritizing a simple, cohesive system over
maximizing any single component's sophistication in isolation.

## Scale
Developed/debugged at small scale (~500 sellers, ~5,000 buyers, ~20k
listings, ~50k orders) for fast iteration. A scalability
section/experiment (e.g. runtime at 500 vs. 5,000 sellers) plus
architecture notes on production scaling (Neo4j migration path, batch
processing) belong in the final report — the pipeline itself is not
actually run at large scale.

## ⚠️ Restored sections (present in the original design, missing from this
file as handed off — needed for Phase 6, and for the MVP-level basic API
now due per phases.md's correction)

### Database tables (target schema — currently CSV via export_dataset.py)
sellers, buyers, products, listings, orders, returns, devices, addresses,
risk_scores, fraud_clusters, alerts, model_predictions,
trust_score_history. PostgreSQL preferred; SQLite acceptable for local
dev. Synthetic IDs only, no real PII.

### Backend endpoints (target — MVP needs a small real subset, not all of these)
```
POST /listing/analyze
POST /return/analyze
POST /transaction/score      <- MVP: wrap baseline_model.py / phase2_specialized_models.py output
GET  /seller/{id}
GET  /buyer/{id}
GET  /transaction/{id}
GET  /fraud-rings            <- MVP: needs detect_fraud_rings() (see phases.md Phase 3 gap)
GET  /graph/{entity_id}
GET  /alerts
```
Frontend/API responses must consume real model output — never hardcoded
demo data.

### Folder structure (target)
```
trustshield/
├── data/{raw,processed,synthetic,external}/
├── notebooks/{01_eda,02_listing,03_returns,04_behavior,05_graph,06_multimodal,07_gnn}/
├── src/{data,features,listing,returns,behavior,multimodal,graph,risk_engine,explainability}/
   (current flat trustshield_project/*.py layout should migrate here once
   the pipeline stabilizes — flat layout was fine for fast iteration
   during Phase 1-3, but src/ separation matters once a backend imports
   from it)
├── models/
├── backend/
├── frontend/
├── experiments/{baselines,multimodal,graph,gnn,ablation}/
├── tests/
├── docs/        <- the 7 files in this docs/ folder
├── scripts/
├── requirements.txt, docker-compose.yml, .env.example, README.md
```

### Frontend screens (stretch, minimal — not a priority)
1. **Overview** — transactions analyzed, high-risk transactions, fraud
   rings, suspicious listings/returns, estimated money at risk.
2. **Transaction Investigation** — overall/listing/seller/buyer/return/
   graph risk + reasons.
3. **Fraud Network** — interactive graph (Seller, Listing, Buyer, Order,
   Return, Device, Address).
