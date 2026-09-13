# TrustShield AI — Design Decisions

## Phase 1A — dataset shortlist
- **IEEE-CIS Fraud Detection** (Kaggle) — for tabular behavioral realism
  only, NOT for fraud labels (it's payment fraud, not marketplace fraud).
- **Amazon Berkeley Objects (ABO)** — for product images/text.
- PaySim dropped as redundant with IEEE-CIS.

## Fraud design (locked, Phase 1B)
- Overall fraud rate: **~6-8%** — a deliberate deviation from the
  realistic 1-3%, documented explicitly, to keep enough fraud cases for
  graph patterns and ML learning at this dataset scale.
- 4 fraud scenario types implemented from the start (not added
  incrementally), in unequal proportion **~40/30/20/10%**:
  1. Fake Listing
  2. Return Abuse
  3. Coordinated Fraud (shared device)
  4. Seller-Buyer Collusion

## Sharing design
- ~12% of buyers share an address with another buyer (70% legitimate
  family, 30% fraud-ring-linked).
- ~8% share a device (50% legitimate, 50% fraud-ring-linked).
- Kept conservative deliberately so shared-device/address doesn't become
  an overfit shortcut signal.

## Temporal design (locked, Phase 1B)
- 12-month simulated timeline, gradual seller/buyer onboarding, growing
  order volume.
- Fraud rings cluster in **time-bursts**, not uniformly across the year.
- Temporal (non-random) split: **Month 1-8 train / 9-10 val / 11-12
  test**.
- Graph features must be built from **timestamp-snapshotted graph state**
  (never the final full graph) to avoid leakage — flagged from the start
  as the trickiest implementation part, handled as a dedicated module
  (Phase 3).

## Scale strategy
~500 sellers, ~5,000 buyers, ~20,000 listings, ~50,000 orders for
development/iteration speed, with a separate scalability write-up planned
rather than actually running the pipeline at production scale.

## Refinements woven into existing phases (not separate phases)
- Naive rule-based baseline before the ML baseline.
- Explicit FP/FN cost-asymmetry matrix for threshold tuning.
- A concrete before/after fraud-ring case study in the graph section.
- An explicit "Limitations" section in the final report.
- Disciplined phase-wise git commit history.
- An explicit data-leakage-audit checklist (features only use pre-cutoff
  data) — implemented as executable assertions, not just documentation.
- Per-model documentation (model-card style).
- Showing prediction confidence/calibration alongside risk scores.
- A "what I'd do differently at scale" section in the final report.

## Gen AI / Agentic AI decision
- DL is already sufficiently represented (multimodal embeddings,
  GraphSAGE) — no need to add more just for resume weight.
- One optional Gen AI stretch item added to Phase 4: an LLM converts SHAP
  explanations into natural language for investigators (bounded,
  non-chatbot use).
- Agentic AI deliberately excluded from this project — conflicts with MVP
  discipline and the "assist, not auto-decide" principle. To be explored
  in a separate project if wanted.

## Phase 5 — GNN Hybrid Design (locked)
- A pure GNN model underperforms on this dataset because it lacks the dense behavioral features.
- The hybrid architecture (GraphSAGE embeddings concatenated with tabular features into XGBoost) combines the structural relational signal of the GNN with the rich tabular context.
- Cold-start nodes (nodes missing from the train/val graph) are gracefully handled by assigning them a zero vector embedding via index `0`.
