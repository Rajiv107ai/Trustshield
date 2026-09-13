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

**Phase 1C — combined baseline** (single order-level target, all 4 fraud types blended):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Naive rule-based | 0.286 | 0.456 | 0.351 |
| Random Forest | 0.550 | 0.157 | 0.244 |

**Phase 2 — specialized detectors** (split by entity, at default 0.5 / at cost-optimal threshold):

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| Fake Listing (RF, default) | 0.859 | 0.488 | 0.622 |
| Return Fraud (RF, default) | 0.948 | 0.549 | 0.696 |
| Return Fraud (RF, cost-optimal @0.12) | 0.660 | 0.972 | 0.786 |

**Phase 3 — does graph structure actually help?** (tabular-only vs. tabular+graph, same RF, same test set):

| Metric | Tabular-only | Tabular + Graph |
|---|---|---|
| Recall | 0.161 | 0.302 |
| F1 | 0.250 | 0.309 |
| ROC-AUC | 0.632 | 0.699 |
| PR-AUC | 0.246 | 0.291 |

Device/address-sharing graph features (`share_degree`, `share_component_size`) landed in the top-3 most important features of the combined model — and known `coordinated_fraud`/`collusion` ring members scored measurably higher (0.341 → 0.376 avg predicted probability) once graph features were added.

## Known issues found & fixed during development

Documented here deliberately — the debugging process is part of the project's engineering evidence, not something to hide:

1. Two leakage bugs caught by suspicious perfect (1.000) Phase 2 scores: a fraud-only return `reason` value, and a fake-listing price factor with zero overlap with normal price variance.
2. Two temporal-integrity bugs caught by the leakage-audit checklist: burst-rescheduling not checking buyer signup dates, and not excluding orders that already had a return on file (both could put a timestamp before its own prerequisite).
3. A reproducibility bug caught by re-running the pipeline twice and comparing checksums: `set()` hash-randomization and un-seeded `pandas.sample()` calls silently broke determinism despite a fixed top-level seed.
4. A structural-realism gap caught by inspecting actual ring sizes: `return_abuse` fraud rings were 100% solo-buyer (0 genuine multi-account rings) until address-sharing pairs were explicitly prioritized as co-abusers.

## Environment limitations (this sandbox specifically)

- **XGBoost, PyTorch, PyTorch Geometric, or DGL were initially unavailable** — the project was built using scikit-learn and networkx stand-ins, but has since been run and verified with the real libraries locally. Both the fallback and the real library paths remain in the code (auto-detected).

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

## Current State & Remaining Work

What **already exists**:
- **FastAPI serving layer** (`backend/`) with three endpoints: `/health`, `/transaction/score`, `/fraud-rings`.
- **GNN stub** (`gnn_model.py`) — a complete GraphSAGE implementation written against stable PyTorch Geometric APIs, but not yet run against real hardware (torch/torch_geometric unavailable in the sandbox). Run locally with `pip install torch torch_geometric` and report back.

What **still needs work**:
- **Run the GNN locally** and compare its test-set metrics against the `graph_features.py` tabular+graph ablation (the purpose of `gnn_model.py`).
- **Real multimodal signal** — actual ABO product images + CLIP-style image-text mismatch scoring, replacing the current placeholder categorical swap.
- **Test suite** (pytest) formalizing the manual print-based validation checks used throughout development.

