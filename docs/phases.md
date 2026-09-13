# TrustShield AI — Phases

Estimated ~6-9 weeks part-time across 7 phases. **MVP scope** = Phase 1-3 +
a *basic* FastAPI (a handful of real endpoints returning real model output,
no polish) + README/report with real metrics. Phases 4-7 are stretch
goals, so the project stays complete and demo-able at every checkpoint.

**⚠️ Correction (was silently contradicted below — fixed):** a basic
working API is part of the non-negotiable MVP, not something deferred
entirely to Phase 6. Phase 6 below is the *polished/productionized*
version (PostgreSQL, full endpoint set, dashboard) — that part is stretch.
The MVP-level API is a much smaller ask: 2-3 endpoints
(`/transaction/score`, `/fraud-rings`) wrapping the already-trained Phase
2/3 models, returning real predictions as JSON. See architecture.md for
the endpoint contract.

## Phase 1 — Foundation (MVP)
- 1A: Dataset research/selection (IEEE-CIS + ABO).
- 1B: Synthetic relational marketplace dataset design + generator
  (entities, fraud injection).
- 1C: Simplest defensible transaction-level baseline model (naive rule +
  Logistic Regression + Random Forest/XGBoost).

**Status: COMPLETE.**

## Phase 2 — Core ML (MVP)
Specialized listing/return baseline models with evaluation, split out of
the Phase 1C combined baseline: Fake Listing Detector (listing-level),
Return Fraud Detector (return-level).

**Status: COMPLETE.**

## Phase 3 — Graph (MVP)
Trust Graph + graph features + fraud ring detection. Timestamp-snapshotted
graph state (not the final full graph) to avoid leakage.

**Status: graph-topological-feature version (networkx) COMPLETE, with an
explicit tabular-only vs. tabular+graph ablation. GNN (GraphSAGE via
PyTorch Geometric) upgrade path written but not yet run/verified (sandbox
environment constraint — see Limitations).**

**⚠️ Gap found and flagged:** the MVP scope for this phase names three
things — graph, graph *features*, and fraud ring *detection*. Only the
first two are actually implemented. `graph_features.py` produces
ring-*membership* features (share_degree, share_component_size, etc.) that
feed the classifier, but there is no function that outputs an actual list
of suspected fraud rings (e.g. connected components above a suspicion
threshold, ranked). This is needed to complete Phase 3 as originally
scoped and to power the `/fraud-rings` MVP API endpoint above — add a
`detect_fraud_rings()` function (e.g. connected components on the
fraud-linked-sharing graph + a per-component risk score from member
predictions) before calling Phase 3 fully done.

## Phase 4 — Multimodal (stretch)
Real ABO product images + text; image-text mismatch scoring (e.g.
CLIP-style). Includes the optional Gen AI item: LLM converts SHAP
explanations into natural language for investigators.

**Status: Real ABO product catalog is now LIVE and fully integrated (Session 7). The actual multimodal *signal extraction* (image-text mismatch scoring via CLIP) remains an open stretch goal.**

## Phase 5 — GNN (stretch, complete)
Formal GraphSAGE/GCN model, evaluated against the simpler graph-feature
approach.

**Status: COMPLETE.** A hybrid architecture combining GraphSAGE embeddings with tabular features via XGBoost was successfully trained, significantly outperforming the pure GNN approach (Test ROC-AUC: 0.696 vs 0.519).

## Phase 6 — Productization (stretch)
The *polished/production* version of the API and serving layer:
PostgreSQL (currently CSV files via export_dataset.py), the full endpoint
set from architecture.md, fraud-ring visualization, alerts. The
*basic* API (a few endpoints on the trained models) is MVP — see the
correction at the top of this file — not this phase.

## Phase 7 — MLOps + Research (stretch)
MLflow/Docker/GitHub Actions, final report with scalability write-up and
"what I'd do differently at scale" section.
