# TrustShield AI

An end-to-end synthetic e-commerce fraud intelligence pipeline — built as an AI/ML/Data Science portfolio project, not a production system. It generates a realistic relational marketplace dataset with injected fraud, then progressively layers behavioral, tabular, and graph-based detection on top of it, measuring what each layer actually adds.

## Core idea

Most fraud-detection portfolio projects stop at "tabular ML on a labeled dataset." This project instead:

1. **Builds its own labeled dataset** — an 8-entity relational marketplace (Seller, Buyer, Product, Listing, Order, Return, Device, Address) simulated over a 12-month timeline with realistic onboarding growth, then injects 4 distinct fraud scenarios (not one generic "is_fraud" flag).
2. **Treats temporal leakage as a first-class concern** — every engineered feature is audited to prove it only uses information that existed strictly *before* the moment a decision would actually be made, not just "before the label was known."
3. **Measures the value of relational/graph structure explicitly** — a tabular-only baseline vs. a tabular+graph-features ablation, with real before/after numbers, rather than assuming graph features help.

## Architecture

```
entity_generator.py          -> Address, Device, Seller, Buyer (12-month onboarding, device/address sharing)
product_listing_generator.py -> Product catalog + Listings (ABO-catalog-shaped, synthetic fallback)
order_return_generator.py    -> Orders + Returns (organic baseline only, temporally consistent)
fraud_injection.py           -> 4 fraud scenarios layered on top of the clean base data
baseline_model.py            -> Phase 1C: naive rule + Logistic Regression + Random Forest (combined target)
phase2_specialized_models.py -> Phase 2: Fake Listing Detector + Return Fraud Detector (specialized)
graph_features.py            -> Phase 3: timestamp-snapshotted graph-topological features + ablation
```

Each script rebuilds the full upstream pipeline in its own `if __name__ == "__main__"` block — there's no shared database or cache. Running `graph_features.py` alone regenerates everything from `entity_generator.py` onward, deterministically (see Reproducibility below).

## The 4 fraud scenarios

Rather than one generic fraud flag, four distinct, realistically-distinguishable patterns are injected in fixed proportion (~40/30/20/10% of a ~7% overall fraud rate):

| Scenario | Mechanism |
|---|---|
| **Fake Listing** | Price undercut (partial overlap with normal price variance, not perfectly separable) + image/description mismatch on a stratified sample of listings across traffic tiers |
| **Return Abuse** | A buyer (or a fraud-linked address-sharing pair) returns an unusually high share of their orders, fast (1-5 days vs. organic 1-21) |
| **Coordinated Fraud** | Device-sharing rings (connected components over fraud-linked device links) whose orders are rescheduled into a short time-burst (3-7 days) |
| **Seller-Buyer Collusion** | A seller + a small repeat-buyer group, orders burst-clustered (5-14 days) with near-automatic refunds |

`fraud_ring_id` is a ground-truth-only label kept in a separate ledger (`fraud_ground_truth`) — it is never merged into any feature table, so it can't leak into a model by being present as a column.

## Design principles enforced throughout

- **Temporal safety, mechanically enforced, not just promised.** History features use `pd.merge_asof(direction="backward", allow_exact_matches=False)` — the join itself guarantees "strictly earlier," rather than relying on a comment saying so.
- **Leakage-audit checklists as executable assertions**, not documentation. These caught real bugs during development (see Known issues found & fixed).
- **No trivial/circular features.** Early versions of this pipeline hit suspicious 1.000 precision/recall scores — investigation found the fraud-injection mechanism itself was leaking into the label-adjacent field (e.g., a return `reason` string used only by fraudulent returns). Fixed by drawing fraud-path values from the same distribution as organic ones.
- **Reproducibility verified, not assumed.** A fixed `numpy.random.default_rng(42)` seed is used everywhere — including patching two subtle sources of hidden non-determinism (Python's per-process string-hash randomization affecting `set()` iteration order, and pandas `.sample()` silently drawing from numpy's *unseeded* global random state instead of the seeded generator). Verified via md5 checksums of output across independent process runs.
- **Cost-aware thresholding**, not just a fixed 0.5 cutoff — an explicit FN=order-amount / FP=flat-review-cost model is swept on the validation set only, and can shift recall dramatically (e.g., Return Fraud Detector: 0.55 recall at 0.5 threshold vs. 0.97 at the cost-optimal threshold).

## Results

> All numbers below are real, reproduced runs — not fabricated. See `docs/memory.md` for the session-by-session verification log.

**Phase 1C — combined baseline** (single order-level target, all 4 fraud types blended):

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Naive rule-based | 0.286 | 0.456 | 0.351 | — | — |
| Random Forest | 0.551 | 0.163 | 0.251 | 0.633 | 0.248 |
| **XGBoost** | **0.424** | **0.220** | **0.289** | **0.654** | **0.270** |

XGBoost's top feature: `buyer_return_rate_before` (importance 0.363).

**Phase 2 — specialized detectors** (listing-level / return-level, at default 0.5 / cost-optimal threshold):

| Detector | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Fake Listing RF (default) | 0.859 | 0.488 | 0.622 | 0.754 |
| Fake Listing XGB (default) | 0.451 | 0.512 | 0.480 | 0.764 |
| Return Fraud RF (default) | 0.948 | 0.549 | 0.696 | 0.951 |
| Return Fraud RF (cost-opt @0.13) | 0.677 | 0.950 | 0.790 | — |
| **Return Fraud XGB (default)** | **0.919** | **0.605** | **0.729** | **0.951** |
| Return Fraud XGB (cost-opt @0.01) | 0.734 | 0.905 | 0.810 | — |

**Phase 3 — does graph structure actually help?** (tabular-only vs. tabular+graph, XGBoost, same test set):

| Metric | Tabular-only | Tabular + Graph |
|---|---|---|
| Precision | 0.520 | 0.605 |
| Recall | 0.190 | 0.203 |
| F1 | 0.278 | 0.304 |
| ROC-AUC | 0.651 | 0.680 |
| PR-AUC | 0.265 | 0.302 |

`share_degree` is the #1 feature (importance 0.168) — above all tabular features including `buyer_return_rate_before`. Known ring members scored measurably higher (avg predicted probability 0.341 → 0.374) once graph features were added. Fraud ring detection: 691 suspected rings, 107 high-risk (avg ≥ 0.3), 67 very high-risk (avg ≥ 0.5).

**Phase 5 — Hybrid GNN (GraphSAGE + XGBoost)**:

| Model | Val ROC-AUC | Test ROC-AUC | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|---|
| Standalone GraphSAGE | — | 0.519 | 0.091 | 0.939 | 0.167 |
| **Hybrid (GNN embeddings + XGBoost)** | **0.792** | **0.696** | **0.580** | **0.180** | **0.275** |

The standalone GNN result (barely above random) confirmed the expected limitation: pure graph structure, without tabular behavioral features, is insufficient for this dataset. The hybrid architecture (GraphSAGE encodes structural context → XGBoost classifies using both embeddings and tabular features) achieves meaningful lift. GNN encoder Val ROC-AUC: 0.708.

## Known issues found & fixed during development

Documented here deliberately — the debugging process is part of the project's engineering evidence, not something to hide:

1. Two leakage bugs caught by suspicious perfect (1.000) Phase 2 scores: a fraud-only return `reason` value, and a fake-listing price factor with zero overlap with normal price variance.
2. Two temporal-integrity bugs caught by the leakage-audit checklist: burst-rescheduling not checking buyer signup dates, and not excluding orders that already had a return on file (both could put a timestamp before its own prerequisite).
3. A reproducibility bug caught by re-running the pipeline twice and comparing checksums: `set()` hash-randomization and un-seeded `pandas.sample()` calls silently broke determinism despite a fixed top-level seed.
4. A structural-realism gap caught by inspecting actual ring sizes: `return_abuse` fraud rings were 100% solo-buyer (0 genuine multi-account rings) until address-sharing pairs were explicitly prioritized as co-abusers.
5. A cost-threshold column-name mismatch in `graph_features.py`: the code checked for `order_amount` (absent) instead of `amount` (always present), silently falling back to the flat $100 proxy. Fixed.

## Library stack

All libraries below are now installed and verified:

| Library | Version | Role |
|---|---|---|
| scikit-learn | latest | Baseline models, RF fallback |
| XGBoost | 3.4.1 | Primary boosted-tree classifier |
| PyTorch | 2.13.0 CPU | GNN training |
| PyTorch Geometric | 2.8.0 | GraphSAGE (SAGEConv) |
| FastAPI | 0.141.1 | REST API serving layer |
| Uvicorn | 0.52.4 | ASGI server |
| Pydantic | 2.13.5 | Request/response schemas |

## Setup

```cmd
C:\Python314\python.exe -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```cmd
cd trustshield_project
python entity_generator.py
python product_listing_generator.py
python order_return_generator.py
python fraud_injection.py
python baseline_model.py
python phase2_specialized_models.py
python graph_features.py
```

All data is generated in memory and printed by the scripts; no database is required.

To train all models and start the API:

```cmd
python scripts/train_and_save_models.py   # generates .joblib artifacts
cd backend
uvicorn main:app --reload                 # API docs at http://localhost:8000/docs
```

To run the Phase 5 hybrid GNN:

```cmd
python trustshield_project/phase5_hybrid_model.py
```

To run the test suite:

```cmd
cd trustshield_project
python -m pytest -v
```

## Real product data (ABO)

The pipeline automatically detects the Amazon Berkeley Objects dataset if placed at `trustshield_project/data/external/abo/`. When present, it loads 3,000 real product records with real image paths. When absent, a synthetic placeholder catalog with identical schema is used automatically — no code changes needed either way.

## Current State

What **exists and is verified**:
- **Phases 1–3**: Full data pipeline, specialized detectors, graph feature ablation — all with real XGBoost numbers.
- **Phase 5**: Hybrid GraphSAGE + XGBoost model, trained and evaluated.
- **FastAPI serving layer** (`backend/`): Three endpoints (`/health`, `/transaction/score`, `/fraud-rings`) — smoke-tested locally.
- **Real ABO product data**: Live in the pipeline (auto-detected from `data/external/abo/`).
- **Test suite** (pytest): 44 tests across data integrity, fraud injection logic, and temporal-leakage audits — all passing.

What **remains as stretch goals**:
- **Real multimodal signal** — CLIP-style image-text mismatch scoring (Phase 4). The ABO images are present; the embedding comparison step is not yet built.
- **Scalability experiment** — runtime comparison at 500 vs. 5,000 sellers.
- **MLOps layer** — MLflow experiment tracking, Docker, GitHub Actions CI (Phase 7).
