# TrustShield AI — Testing Strategy

Layered testing approach, plus an explicit note on what's deliberately
NOT over-tested (see bottom).

## 1. Generator tests
- **Volume checks** — exact target counts (e.g. 20,000 listings, 50,000
  orders) hit as expected.
- **Referential-integrity checks** — every foreign key (seller_id,
  buyer_id, product_id, listing_id, order_id) resolves to a real row in
  its parent table.
- **Rate checks** — sharing rates (~12% address, ~8% device), fraud rate
  (~6-8%), fraud-type proportions (~40/30/20/10%) land within tolerance
  of their targets.
- **Temporal checks** — no order before its listing's listing_date, no
  order before the buyer's signup_date, no return before its own order,
  no negative account ages, no return count exceeding order count for a
  buyer/seller as of any date.

## 2. Data-leakage audit checklist
Executable, printed assertions (not just documentation) run before any
model is trained, verifying:
- Every feature is computed from information strictly before the
  decision point in question.
- No fraud-injection mechanism column (price_anomaly, image_mismatch,
  fraud_ring_id, etc.) appears in the feature set.
- Cumulative/history features never show an impossible value (e.g.
  returns-before exceeding orders-before for the same entity).

## 3. Model / evaluation tests
- Real, measured precision/recall/F1/ROC-AUC/PR-AUC on a held-out
  TEMPORAL test split — never a random split.
- Confusion matrices reported alongside summary metrics.
- Reproducibility verified via checksum comparison across independent
  runs (not just "a seed is set").
- Before/after ablations (e.g. tabular-only vs. tabular+graph) to
  demonstrate that added complexity earns its place.

## 4. API tests (stretch)
Once a FastAPI serving layer exists: basic request/response contract
tests, not a priority until that layer is built.

## Deliberately NOT over-tested
- UI — not a priority for this project (see project-requirement.md).
- Load testing beyond the one planned scalability experiment (500 vs.
  5,000 sellers runtime comparison).
- Stretch features (multimodal, GNN, MLOps) — tested once actually built,
  not speculatively ahead of time.
