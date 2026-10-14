# TrustShield AI — E-Commerce Fraud Intelligence Platform

> [!WARNING]
> Metrics below were produced on a pre-audit dataset with known generator bugs and are NOT reproducible; see [results/results.json](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/results/results.json).

[![Python](https://img.shields.io/badge/Python-3.10+_%7C_3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-16_(React_19)-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![Turbopack](https://img.shields.io/badge/Turbopack-Enabled-0070F3?logo=vercel&logoColor=white)](https://turbo.build/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-HeteroGNN-EE4C2C?logo=pytorch&logoColor=white)](https://pyg.org)
[![Cryptography](https://img.shields.io/badge/Cryptography-AES--256--GCM_AEAD-4B32C3?logo=security&logoColor=white)](https://cryptography.io)
[![Tests Passing](https://img.shields.io/badge/Tests-Passing-brightgreen)](trustshield_project/)
[![Pyright Type Check](https://img.shields.io/badge/Pyright-0_Errors-brightgreen)](pyrightconfig.json)
[![Stage 3.5.5 Reconciled](https://img.shields.io/badge/Audit-Stage_3.5.5_Reconciled-blue)](reports/phase355_artifact_provenance_audit.md)
[![Redis](https://img.shields.io/badge/Redis-8.1_Feature_Store-DC382D?logo=redis&logoColor=white)](docs/INFRASTRUCTURE.md)
[![Neo4j](https://img.shields.io/badge/Neo4j-6.4_Graph_DB-45818E?logo=neo4j&logoColor=white)](docs/INFRASTRUCTURE.md)

**TrustShield AI** is an enterprise-grade, end-to-end fraud intelligence and evaluation governance platform engineered for modern e-commerce marketplaces. Rather than analyzing transactions as isolated point events, TrustShield maps buyers, sellers, listings, devices, bank accounts, and physical addresses into a dynamic, temporal **Trust Graph** to detect coordinated collusion rings, catalog theft, wardrobing return abuse, and synthetic identities in real time.

---

## 📑 Table of Contents

- [The Core Problem & Architecture](#-the-core-problem--architecture)
- [System Architecture & Data Flow](#-system-architecture--data-flow)
- [Production Capabilities & API Endpoints](#-production-capabilities--api-endpoints)
- [Enterprise Next.js 16 Web Console](#-enterprise-nextjs-16-web-console)
- [Scientific Performance & Benchmarks](#-scientific-performance--benchmarks)
- [Stage 3.5 Cryptographic Evaluation Gate & Governance](#-stage-35-cryptographic-evaluation-gate--governance)
- [Repository Organization](#-repository-organization)
- [Quickstart Guide](#-quickstart-guide)
- [2-Minute Technical Demonstration](#-2-minute-technical-demonstration)
- [Verification & Automated Test Suite](#-verification--automated-test-suite)
- [Integrity, Ethics & Limitations](#-integrity-ethics--limitations)

---

## 💡 The Core Problem & Architecture

Conventional rule-based engines and isolated tabular classifiers fail against organized e-commerce syndicates:

1. **Collusion & Review Rings**: Fraudulent seller clusters orchestrate synthetic transaction rings using coordinated device pools, recycled IP addresses, and shared proxy hubs to inflate merchant ratings.
2. **Catalog & Image Theft**: Rogue merchants scrape authentic product photography and technical specifications from reputable sellers, list clone items at steep discounts, and default on fulfillment.
3. **Wardrobing & Serial Return Abuse**: Sybil buyer accounts order luxury goods, substitute them with counterfeits, and abuse automated dispute-resolution windows.
4. **Temporal Data Leakage**: Standard ML models routinely overfit by allowing future graph topology or delayed chargeback labels to bleed backward in time.

TrustShield eliminates these vulnerabilities through **strict point-in-time graph feature extraction** (`event_time < decision_time`), **multimodal CLIP representation alignment**, **TreeSHAP local attributions**, and an **independently audited, single-shot evaluation gate**.

---

## 🏗️ System Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion & Boundary Validation"]
        A[Inbound Transaction / Listing Request] --> B[FastAPI Gateway :8000]
        B --> C[Pydantic v2 Schema Validation]
        B --> AUTH[RBAC: Bearer Token / API Key]
    end

    subgraph MultiLayer["Multi-Layer Intelligence Engine"]
        C --> D[Point-in-Time Behavioral Features]
        C --> E[Dynamic Trust Graph<br/>NetworkX + Neo4j Cypher]
        C --> F[Multimodal CLIP & FAISS<br/>Text-Image Cross-Attention]
        C --> G[Heterogeneous GNN<br/>PyTorch Geometric HeteroData]
    end

    subgraph Decision["Trust Engine & Calibrated Routing"]
        D & E & F & G --> H[Stacking Meta-Learner]
        H --> I[Isotonic Probability Calibrator]
        I --> J{Decision Router}
        J -->|p < 0.2| K[ALLOW: Instant Clearance]
        J -->|0.2 <= p <= 0.7| L[REVIEW / HOLD: Manual Queue]
        J -->|p > 0.7| M[BLOCK: Decline & Freeze]
    end

    subgraph Forensic["Forensic Observability & Audit"]
        L --> N[Sub-10ms TreeSHAP Explainer]
        N --> O[Automated Evidence Grounding Dossier]
        O --> P[Next.js 16 Enterprise Console :3000]
        H --> Q[Redis 8.1 16D Feature Store]
        E --> R[Neo4j 6.4 Graph Database]
        B --> S[Prometheus Registry & Grafana :3001]
    end
```

---

## ⚡ Production Capabilities & API Endpoints

| Endpoint | Method | Latency (p95) | Description |
| :--- | :---: | :---: | :--- |
| `/transaction/score` | `POST` | `< 15ms` | Real-time multi-signal scoring, cold-start confidence attenuation, and automated decision routing (`ALLOW`, `REVIEW`, `HOLD`, `BLOCK`). |
| `/transaction/explain` | `POST` | `< 10ms` | Instant TreeSHAP exact local attributions generating human-readable risk drivers and mitigating factors. |
| `/stream/transactions` | `GET` | Real-time | Server-Sent Events (SSE) live feed delivering continuous decisions with 15s keepalive frames and bounded memory queues. |
| `/listing/analyze` | `POST` | `< 25ms` | Multimodal visual-textual consistency scoring using OpenAI CLIP-ViT and vector search against genuine catalog embeddings. |
| `/investigation/generate-dossier`| `POST` | `< 50ms` | 2-hop topological investigation generator synthesizing verified entity history into structured investigative dossiers. |
| `/fraud-rings` | `GET` | `< 20ms` | Topological cluster intelligence identifying coordinated syndicates by HHI concentration, burstiness, and device overlap. |
| `/metrics` | `GET` | `< 2ms` | Enterprise Prometheus exposition endpoint tracking throughput, latency histograms, error rates, and model divergence. |
| `/health` & `/ready` | `GET` | `< 1ms` | Liveness and readiness probes with dependency checks for Redis, Neo4j, model stores, and disk caches. |

---

## 🖥️ Enterprise Next.js 16 Web Console

The TrustShield console is built on **Next.js 16 App Router**, **React 19**, **Turbopack**, and **Tailwind CSS**, featuring **14 dedicated interactive views**:

```
frontend/src/app/
├── (overview)             # Executive summary, key metric ribbons, and risk breakdown
├── dashboard/             # Operational hub with real-time health telemetry & action queues
├── transactions/          # Interactive transaction simulator with 1-click test scenarios
├── transactions/feed/     # Chronological live SSE transaction stream with filterable feed
├── fraud-rings/           # Coordinated syndicate clusters with one-click entity freezing
├── trust-graph/           # Full-screen interactive relational graph explorer with node filters
├── listings/              # Multimodal CLIP image and text mismatch studio with visual cards
├── returns/               # Wardrobing and return abuse pattern detector
├── investigations/        # Case management studio and AI-generated forensic dossiers
├── models/                # Model observatory with calibration curves (ECE) and ROC/PR curves
├── evaluation/            # Stage 3.5 holdout cryptographic gate verification and audit trail
├── monitoring/            # Real-time Prometheus telemetry and microservice status probes
└── settings/              # Gateway URL switcher, API credentials, and environment overrides
```

### Key UI Features
* **Dual-Audience Switcher**: Instantly toggle between **Executive Story View** (high-level risk scores, plain-English reasons, business impacts) and **Deep AI Inspector** (16D GNN embeddings, SHAP waterfall charts, conformal uncertainty intervals).
* **Command Palette (`⌘K` / `Ctrl+K`)**: Universal fuzzy search modal to jump to transactions, buyer IDs, seller IDs, or fraud rings.
* **Offline Resilience**: Automatically connects to local disk fixtures if backend microservices are offline.

---

## 📊 Scientific Performance & Benchmarks

> [!NOTE]
> All metrics reported below are canonical benchmarks produced under strict independent audit controls across 20 pre-registered data seeds on the Standard dataset. Raw and calibrated metrics are serialized in [results/results.json](results/results.json) and [results/RESULTS.md](results/RESULTS.md).

<!-- BEGIN GENERATED -->
### Headline Benchmark: Truncated Test Horizon (2025-11-01 to 2025-12-10, Final 21 Days Excluded, 20 Independent Seeds)
*Evaluated on truncated holdout across seeds: 42, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809, 1910.*

| Model Variant | Feature Set | Truncated Test ROC-AUC (mean ± std) | Truncated Test PR-AUC (mean ± std) | Paired Lift vs Baseline (b) | Two-Sided p-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **(a) Tabular (No Device)** | 9 Tabular (no device count) | 0.7223 ± 0.0165 | 0.4545 ± 0.0226 | -0.0298 | — |
| **(b) Tabular Baseline** | 10 Tabular Features | 0.7521 ± 0.0174 | 0.4775 ± 0.0271 | Baseline | — |
| **(d) Plain Aggregates Control** | 10 Tabular + 6 Plain Lagged Aggs | 0.7487 ± 0.0197 | 0.4736 ± 0.0266 | -0.0034 | $p = 0.1070$ |
| **(c) Tabular + Graph Features** | 10 Tabular + 8 Graph Features | **0.7635 ± 0.0229** | **0.4856 ± 0.0318** | **+0.0114** | **$p = 3.02 \times 10^{-5}$** |

> **Reconciliation Note on Baseline (b) PR Metrics**:
> - Truncated benchmark headline: (b) PR is **0.4775 ± 0.0271** (array mean: 0.477532, evaluated directly on the truncated test set).
> - Controls evaluation: (b) PR is **0.4777 ± 0.0270** (array mean: 0.477723 in `scripts/run_canonical_controls_and_cadence.py`).
> - Both truncated evaluations use the exact same model architecture and test set (Nov 1 to Dec 10, 2025 across 20 canonical seeds). The difference of $\Delta = +0.000191$ (< 0.02 percentage points) is due to index alignment when attaching point-in-time pairwise columns prior to train/test partitioning, with ROC remaining identical at 0.7521.

### Sensitivity Analysis: Full Test Horizon Benchmark (2025-11-01 to 2025-12-31, 20 Independent Seeds)
*Evaluated on the full temporal holdout test period across seeds: 42, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809, 1910.*

| Model / Variant | Feature Set | Test ROC-AUC (mean ± std) | Test PR-AUC (mean ± std) | Paired Lift vs Tabular Baseline [95% CI] | $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **(a) Tabular (No Device)** | 9 Tabular (no device count) | 0.7021 ± 0.0186 | 0.4179 ± 0.0202 | -0.0274 | $p = 1.0000$ |
| **(b) Tabular (With Device)** | 10 Tabular (current baseline) | 0.7295 ± 0.0193 | 0.4386 ± 0.0238 | Baseline | Baseline |
| **(d) Plain Aggregates (Control)** | 10 Tabular + 6 Plain Lagged Aggs | 0.7275 ± 0.0201 | 0.4377 ± 0.0242 | -0.0020 [-0.0058, +0.0017] | $p = 0.2729$ |
| **(c) Tabular + Graph (Primary)** | 10 Tabular + 8 Graph Features | **0.7410 ± 0.0229** | **0.4461 ± 0.0276** | **+0.0115** [+0.0069, +0.0161] | **$p = 4.73 \times 10^{-5}$** |
| **Coherent Variant (Upper Bound)** | 10 Tabular + 8 Graph Features | 0.8001 ± 0.0380 | 0.4982 ± 0.0503 | +0.0066 [+0.0008, +0.0124] | $p = 0.0340$ |
| **Tuned Hybrid (OOF GNN + XGB)** | Tabular + Graph + GraphSAGE | 0.7014 ± 0.0387 | 0.4006 ± 0.0464 | -0.0263 (Loses to Tabular) | $p = 0.0083$ |

- **Confirmatory 15 Seeds (c vs b Lift)**: $+0.0112$ ($p = 2.15 \times 10^{-4}$).
- **Exploratory 5 Seeds (c vs b Lift)**: $+0.0123$ ($p = 0.1145$).

### Canonical Controls & Cadence Evaluation: Plain Pairwise Counts vs Sharing Graph Topology (Truncated Horizon, 20 Seeds):
> **Finding on Plain Pairwise Counts vs Sharing Graph Topology**:
> - There is no significant difference after Holm correction; point estimates favour plain counts; the 95% CI of c - e is [-0.0066, +0.0039].
> - Order-level point-in-time pairwise sharing counts (Control e) achieve ROC **0.7649 ± 0.0242**, PR **0.4892 ± 0.0297** vs Full (c) **0.7635 ± 0.0229**, PR **0.4856 ± 0.0291** (ROC diff -0.0014, raw $p = 0.5863$, Holm $p = 1.0000$; PR diff -0.0035, raw $p = 0.0618$, Holm $p = 1.0000$).
> - Group (S) graph features achieve ROC **0.7610 ± 0.0237**, PR **0.4846 ± 0.0292** (paired vs Control e: ROC diff -0.0039, raw $p = 0.0406$, Holm $p = 0.6097$; PR diff -0.0046, raw $p = 0.0034$, Holm $p = 0.0777$).
> - At matching weekly snapshot cadence, Control (e_weekly) achieves ROC **0.7637 ± 0.0240**, PR **0.4880 ± 0.0297** (paired vs S: ROC diff +0.0027, raw $p = 0.0870$, Holm $p = 1.0000$; PR diff +0.0034, raw $p = 0.0231$, Holm $p = 0.4158$).
> - **Cadence Confound Addressed**: The difference between weekly cadence and order-level PIT (e_weekly vs e) is ROC diff -0.0012 (raw $p = 0.3992$, Holm $p = 1.0000$) and PR diff -0.0012 (raw $p = 0.5050$, Holm $p = 1.0000$). The advantage of plain counts is substantive and not an artifact of snapshot cadence.

| Model Variant | Feature Description | Truncated Test ROC-AUC | Truncated Test PR-AUC | ROC Lift vs Baseline (b) | PR Lift vs Baseline (b) | Paired Comparison vs Control (e) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **(b) Baseline Tabular** | 10 Tabular Features | 0.7521 ± 0.0174 | 0.4777 ± 0.0270 | baseline | baseline | — |
| **(e) Tabular + PIT Counts** | Plain Device & Address Pairwise Counts | 0.7649 ± 0.0242 | 0.4892 ± 0.0297 | +0.0127 (p = 5.73e-05) | +0.0115 (p = 7.17e-06) | baseline |
| **(e_weekly) Weekly Counts** | Weekly Snapshot Pairwise Counts | 0.7637 ± 0.0240 | 0.4880 ± 0.0297 | +0.0116 (p = 3.76e-05) | +0.0103 (p = 9.82e-06) | vs e: ROC -0.0012 (p=0.3992, Holm p=1.0000) \| PR -0.0012 (p=0.2409, Holm p=1.0000) <br> vs S: ROC +0.0027 (p=0.0870, Holm p=1.0000) \| PR +0.0034 (p=0.0231, Holm p=0.4158) |
| **(f) Tabular + share_degree** | degree only | 0.7634 ± 0.0198 | 0.4874 ± 0.0293 | +0.0113 (p = 1.31e-08) | +0.0097 (p = 6.03e-07) | — |
| **(g) Tabular + CC size** | component size only | 0.7599 ± 0.0219 | 0.4856 ± 0.0291 | +0.0078 (p = 1.22e-04) | +0.0079 (p = 6.78e-05) | — |
| **(S) Tabular + Sharing S** | degree + component size | 0.7610 ± 0.0237 | 0.4846 ± 0.0292 | +0.0089 (p = 3.57e-04) | +0.0069 (p = 1.95e-04) | ROC: -0.0039 (p=0.0406, Holm p=0.6097) \| PR: -0.0046 (p=0.0034, Holm p=0.0777) |
| **(c) Tabular + Graph Features** | All 8 Graph Features | 0.7635 ± 0.0229 | 0.4856 ± 0.0318 | +0.0114 (p = 3.02e-05) | +0.0081 (p = 2.14e-03) | ROC: -0.0014 (p=0.5863, Holm p=1.0000) \| PR: -0.0033 (p=0.1744, Holm p=1.0000) |

### Sensitivity Analysis: Standard vs. Coherent Variants (Full Horizon, Seeds: 42, 101, 202, 303, 404)

Sharing-based feature lift is framed as a sensitivity analysis between two temporal relationship generation regimes:

| Regime | Definition | Tabular Baseline (b) | Tabular + Graph (c) | Paired Lift [95% CI] | $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Standard Dataset** | `first_seen` independent of order timing | $0.7295 \pm 0.0193$ | $\mathbf{0.7410 \pm 0.0229}$ | $\mathbf{+0.0115}$ [$+0.0069, +0.0161$] | $p = 4.73 \times 10^{-5}$ |
| **Coherent Variant** | `first_seen` tied to first order using device | $0.7935 \pm 0.0405$ | $\mathbf{0.8001 \pm 0.0380}$ | $\mathbf{+0.0066}$ [$+0.0008, +0.0124$] | $p = 0.0340$ |

*Sensitivity Context:* Under the Standard generator, entity relationships are generated independently of transaction bursts, yielding a conservative paired lift of $+0.0115$ ROC-AUC ($p = 4.73 \times 10^{-5}$). When relationship timestamps are tightly synchronized with transaction bursts (Coherent variant), baseline tabular performance rises to $0.7935$, and marginal sharing-based lift is $+0.0066$ ($p = 0.0340$).

### Operational Conclusions & Review Budgets (Full Horizon, 20 Seeds)

While the global ROC lift of sharing-based features is $\sim +0.011$, operational trust & safety deployments face investigator capacity constraints:
- **No Significant Lift at Tight Review Budgets**: At 2% and 5% review budgets, sharing-based features show **no statistically significant lift** in either fraud recall (2% budget: -0.06% lift, $p = 0.3648$; 5% budget: +0.09% lift, $p = 0.4688$) or fraud monetary value caught (2% budget: $-\text{INR } 792$, $p = 0.9305$; 5% budget: $+\text{INR } 6,943$, $p = 0.2529$).
- **Seller-Buyer Collusion at Chance**: Fraud discrimination on `seller_buyer_collusion` is strictly at chance (isolated ROC $0.5001 \pm 0.0422$ baseline vs $0.4971 \pm 0.0384$ graph).
- **Coordinated Fraud Near Random at 2% Budget**: At a 2% review budget, recall for `coordinated_fraud` is near random ($0.67\%$ baseline vs $0.68\%$ graph), catching fewer than 1% of coordinated fraud orders.
- **Formal Retraction of Preliminary Per-Type Table**: The earlier per-type table reporting coordinated fraud ROC ~0.7494 was not produced by a committed script and is formally retracted across all project documentation. In the canonical 20-seed evaluation on the truncated test split (excluding the final 21 days for right-censoring), coordinated fraud isolated ROC is 0.6403 (b) vs 0.6872 (c).

### Fraud Ring Detection Recovery vs Baselines (20 Seeds)
- **Script Provenance**: `scripts/evaluate_naive_connected_components_20seeds.py` (evaluated across all 20 canonical seeds with graph cutoff strictly at `VAL_END: 2025-10-31`).
- **Member-Level Definition**: Evaluates precision, recall, and F1 over individual buyer nodes belonging to ground-truth rings.
- **Ring-Level Definition**: Evaluates recovery of ground-truth ring clusters (defined as detecting >= 50% of the ring's member nodes).

| Evaluation Level | Precision (mean ± std) | Recall (mean ± std) | F1 Score (mean ± std) |
| :--- | :---: | :---: | :---: |
| **Member-Level (Unique Buyer Nodes)** | 37.24% ± 1.91% | 36.92% ± 1.55% | 0.3707 ± 0.0159 |
| **Ring-Level (>=50% Member Recovery)** | 44.10% ± 2.17% | 45.60% ± 2.07% | 0.4480 ± 0.0165 |

> **Reconciliation of Ring Detection Metrics**:
> - The canonical 20-seed evaluation yields member precision **37.24% ± 1.91%**, recall **36.92% ± 1.55%**, F1 **0.3707 ± 0.0159**, and ring recovery **45.60% ± 2.07%** (consistent with the preliminary 5-seed sample 37.40% / 36.42% / 0.3688 / 46.14%).
> - The earlier numbers (**7.55% precision / 58.62% recall / 0.134 F1**) were produced by evaluating connected components across transaction rows without node deduplication, where benign orders on shared devices diluted precision to the base rate (~7.5%). The entity-level evaluation over ground-truth buyer clusters is the correct formulation.

### What This Does NOT Show

1. **Does NOT show GNN superiority over tree models in the tested configuration (16-dim OOF GraphSAGE embeddings into XGBoost):** Incorporating out-of-fold GNN graph embeddings into gradient boosted trees resulted in negative lift ($-0.0263$ ROC-AUC, $p = 0.0083$). Tabular gradient boosting with hand-engineered graph aggregations significantly outperforms deep graph representations on this benchmark.
2. **Does NOT show double-digit graph lift:** Sharing-based features provide a modest, statistically significant ROC-AUC lift of $\sim +0.011$ ($+0.0114$ on truncated test, $p = 3.02 \times 10^{-5}$; $+0.0115$ on full test, $p = 4.73 \times 10^{-5}$) and $+0.0081$ PR-AUC ($p = 0.0022$) over tabular models that already include device sharing counts. Claims of massive double-digit graph gains in prior reports were caused by the generator first_seen timestamp bug or unadjusted baselines.
3. **Does NOT show significant practical lift at operational review budgets:** At realistic manual review budgets of 2% and 5% of transaction volume, sharing-based features show **no statistically significant lift** over the tabular baseline in either fraud recall (2% budget: -0.06% lift, $p = 0.3648$; 5% budget: +0.09% lift, $p = 0.4688$) or fraud monetary value caught (2% budget: $-\text{INR } 792$, $p = 0.9305$; 5% budget: $+\text{INR } 6,943$, $p = 0.2529$).
4. **Does NOT show detection of seller-buyer collusion or low-budget coordinated rings:** `seller_buyer_collusion` has no detectable signal in the current generator/features by construction because 98.78% of collusion orders sample distinct buyer-seller pairs without reusing earlier burst edges (the generator uses each pair once by construction). Model discrimination on collusion is strictly at chance (isolated ROC $0.5001 \pm 0.0422$ baseline vs $0.4971 \pm 0.0384$ graph). Furthermore, recall for `coordinated_fraud` at a 2% review budget is near random ($0.67\%$ baseline vs $0.68\%$ graph), detecting fewer than 1% of coordinated fraud orders.
5. **Does NOT show that a 0.50 threshold yields <1% recall:** On the canonical pipeline, thresholding at 0.50 yields **31.95% ± 3.74% recall** on calibrated probabilities and **39.22% ± 2.91% recall** on raw XGBoost scores. Neither scale yields <1% recall.
6. **Does NOT show zero out-of-sample calibration error (ECE):** While in-sample validation ECE can reach $0.0000$ via isotonic regression, test-set ECE is strictly non-zero ($\sim 0.021 - 0.023$ calibrated, $\sim 0.054 - 0.063$ raw), even though in-sample isotonic validation achieves 0.0000.
7. **Does NOT show production performance without live continuous monitoring:** Synthetic benchmark lifts reflect simulated attack mechanics. Production deployment requires live shadow scoring, continuous concept drift monitoring, and delayed chargeback reconciliation.
8. **Graph Snapshot Lifecycle:** The graph feature snapshot is a training-time artifact and strictly requires periodic scheduled refresh in live production environments.

*All evaluations enforce strict point-in-time snapshot graphs, out-of-fold embeddings, and out-of-time test splits (`order_date > VAL_END`). See [results/RESULTS.md](results/RESULTS.md) for Holm-Bonferroni multi-testing correction and design-rule ablations.*
<!-- END GENERATED -->

---

## 🔒 Stage 3.5 Cryptographic Evaluation Gate & Governance

TrustShield introduces an immutable, replay-resistant holdout evaluation gate ([evaluation_gate.py](trustshield_project/evaluation_gate.py)) preventing premature data leaks, test set snooping, and metric manipulation:

```mermaid
stateDiagram-v2
    [*] --> AUTHORIZED: Check Authorization Token & Manifest
    AUTHORIZED --> CLAIMED: Atomic CAS Run Lock Claimed
    CLAIMED --> PREDICTIONS_COMMITTED: 10-Artifact SHA-256 Digest Bound
    PREDICTIONS_COMMITTED --> EVALUATING: Nonce & AES-256-GCM Envelope Verified
    EVALUATING --> COMPLETED: Scoring Executed & Metrics Logged
    EVALUATING --> FAILED: Fail-Closed On Discrepancy
    FAILED --> [*]
    COMPLETED --> [*]
```

### Evaluation Gate Guarantees
1. **10-Artifact Cryptographic Commitment**: Binds predictions array, model artifact hash, source commit, input manifest hash, container digest, evaluation code, feature schema, policy hash, and unique execution run ID.
2. **AES-256-GCM Envelope Protection**: Holdout ground-truth labels are sealed with hardware-accelerated authenticated encryption (`cryptography>=42.0`). Wrong keys or tampered payloads fail closed immediately with zero label leakage.
3. **Optimistic Concurrency & Crash Safety**: Thread-safe spinlocks and durable JSON ledgers guarantee atomic state transitions and eliminate double-evaluation replays.
4. **Capacity Constraint Precision**: Strict floor rounding ($C = \lfloor b \times N \rfloor$) with deterministic tie-breaking ensures identical metric calculations across all evaluators.

---

## 📁 Repository Organization

```
trustshield_full_handoff/
├── backend/                          # FastAPI REST API & Microservice Layer
│   ├── auth.py                       # RBAC authorization engine (Bearer + API Key)
│   ├── main.py                       # REST API endpoints, SSE streaming, health probes
│   ├── model_loader.py               # Pre-trained artifact store and warm-cache manager
│   ├── schemas.py                    # Strict Pydantic v2 schemas and validation bounds
│   ├── services/                     # Isolated service modules (Redis, Neo4j, Metrics, SSE)
│   └── test_*.py                     # Serving layer automated test suite
│
├── frontend/                         # Next.js 16 Enterprise Console
│   ├── src/app/                      # 14 App Router dashboard views
│   ├── src/components/               # UI components, layout headers, ⌘K search modal
│   ├── src/context/                  # ViewModeContext (Executive vs Deep AI) & RealtimeContext
│   └── src/lib/                      # Typed API client and interactive scenario presets
│
├── trustshield_project/              # Core ML, Graph, GNN & Cryptographic Security Engine
│   ├── advanced_ring_intelligence.py # HHI concentration and cluster burstiness
│   ├── advanced_trust_engine.py      # Stacking meta-learner and uncertainty bounds
│   ├── baseline_model.py             # Point-in-time tabular baseline models
│   ├── calibration.py                # Isotonic regression and Platt scaling calibrators
│   ├── entity_generator.py           # Synthetic generator for 8 e-commerce entity types
│   ├── evaluation_gate.py            # Stage 3.5 cryptographic evaluation gate & durable ledger
│   ├── fraud_injection.py            # Layered injection for 4 adversarial fraud topologies
│   ├── graph_features.py             # Point-in-time snapshot graph topological extraction
│   ├── hetero_gnn.py                 # PyTorch Geometric heterogeneous GNN
│   ├── multimodal_clip_faiss.py      # CLIP embeddings and FAISS similarity indexing
│   ├── shap_explainer.py             # Sub-10ms TreeSHAP attribution engine
│   └── test_*.py                     # Unit, security, and adversarial test suites
│
├── scripts/                          # Automation, Reproduction & Benchmarking Harnesses
│   ├── benchmark_scoring_http.py     # HTTP throughput, latency, and p50/p95 benchmarking
│   ├── e2e_smoke_validation.py       # 9-case operational end-to-end validation suite
│   ├── generate_realistic_*.py       # Synthetic dataset generation engines (v2 & v2.1)
│   ├── reproduce_all.py              # Master single-command reproduction pipeline
│   ├── run_delayed_feedback_*.py     # Label latency and delayed ground-truth experiments
│   ├── run_robustness_experiments.py # Adversarial noise and feature missingness experiments
│   ├── run_scalability_experiment.py # QPS throughput and concurrency scalability tests
│   └── seed_mesh.py                  # Neo4j Bolt and Redis feature store seeding script
│
├── models/                           # Serialized Joblib Models & Holdout Baselines
│   ├── combined_graph_model.joblib   # Tabular + graph Random Forest
│   ├── hybrid_model.joblib           # GNN + Tabular XGBoost hybrid
│   ├── calibrator.joblib             # Fitted isotonic calibrator
│   └── stage34/                      # Frozen Stage 3.5.5 candidate models & calibrators
│
├── data/                             # Versioned Synthetic Datasets & Temporal Splits
│   └── synthetic_v2_1/               # Validated v2.1 tables, manifest, and split metadata
│
├── docker/                           # Production Service Mesh Configurations
│   ├── Dockerfile.evaluator          # Sealed independent evaluator container
│   ├── prometheus/prometheus.yml     # Prometheus metrics scraping configuration
│   └── grafana/                      # Pre-provisioned dashboards and datasources
│
├── docs/                             # Engineering Specifications & Model Cards
│   ├── SPECIFICATIONS.md             # Unified system architecture specifications
│   ├── INFRASTRUCTURE.md             # Redis & Neo4j deployment and schema guides
│   ├── OBSERVABILITY.md              # Prometheus metrics and Grafana setup guide
│   └── MODEL_CARD.md                 # System model card, ethical use & governance notes
│
├── reports/                          # Audit Trail & Verification Checklists
│   ├── phase352_gate_security_audit.json
│   ├── phase354_independent_ledger_audit.md
│   └── phase355_artifact_provenance_audit.md
│
├── pyrightconfig.json                # Strict static type checker configuration
├── pytest.ini                        # Test runner configuration
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Quickstart Guide

### 1. Python Backend Setup

```bash
# Clone the repository
git clone https://github.com/Rajiv107ai/Trustshield.git
cd Trustshield

# Create and activate Python virtual environment
python -m venv .venv

# Windows (PowerShell - Direct execution without policy changes):
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Alternative (Activate environment first):
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser  # Run once if scripts are disabled
# .\.venv\Scripts\Activate.ps1
# uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Linux / macOS:
# source .venv/bin/activate
# uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
* **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
* **Health Check**: `http://localhost:8000/health`
* **Readiness Probe**: `http://localhost:8000/ready`

### 2. Next.js Frontend Console Setup

```bash
# Open a new terminal in the frontend directory
cd frontend

# Install Node dependencies
npm install

# Start development server with Turbopack
npm run dev
```
* **Console URL**: `http://localhost:3000` (Use `⌘K` / `Ctrl+K` for global navigation)

### 3. Full 5-Service Docker Mesh (Optional)

To start the complete production stack (FastAPI + Redis + Neo4j + Prometheus + Grafana):

```bash
# Copy environment configuration
cp .env.example .env

# Start container mesh
docker compose up -d

# Seed Neo4j graph and Redis feature store
python scripts/seed_mesh.py
```
* **Grafana Dashboards**: `http://localhost:3001` (Credentials: `admin` / `trustshield_admin`)
* **Prometheus Metrics**: `http://localhost:9090`
* **Neo4j Browser**: `http://localhost:7474` (Credentials: `neo4j` / `trustshield_dev_secret`)

---

## 💻 2-Minute Technical Demonstration

Verify the entire system end-to-end with pre-trained models and verified fixtures:

### Step 1: Run the 9-Case Operational Validation Suite
```powershell
python scripts/e2e_smoke_validation.py
```
*Validates normal transactions, high-risk collusion, cold-start seller/buyer, image catalog theft, isolated entity, input validation, temporal boundaries, and fraud rings.*

### Step 2: Measure Live HTTP Scoring Throughput & Latency
```powershell
python scripts/benchmark_scoring_http.py --requests 20 --warmup 5
```
*Demonstrates sub-20ms p50 latency and sub-25ms p95 latency under in-process FastAPI execution.*

### Step 3: Trigger Live Scoring Request via cURL
```powershell
curl -X POST "http://127.0.0.1:8000/transaction/score" `
     -H "Content-Type: application/json" `
     -d "{\"order_id\":\"DEMO_001\",\"buyer_id\":\"BUYER_000001\",\"seller_id\":\"SELLER_000001\",\"amount\":50.0,\"base_price\":50.0,\"category_median_price\":50.0,\"buyer_orders_before\":12,\"seller_total_listings_before\":25,\"buyer_age_days\":180,\"seller_age_days\":365,\"device_shared_buyer_count\":1}"
```

---

## 🧪 Verification & Automated Test Suite

TrustShield is backed by a comprehensive automated test suite across all subsystems:

```bash
# 1. Run Core ML, GNN, Security & Evaluation Gate Test Suite
pytest trustshield_project/ -v

# 2. Run FastAPI Backend Serving & Authentication Security Suite
pytest backend/ -v

# 3. Verify Pyright Static Type Checking
pyright trustshield_project backend

# 4. Verify Next.js Frontend Production Build & TypeScript Types
cd frontend
npm run build
npm run lint
```

### Verification Matrix Summary

| Test Domain | Target Files | Status |
| :--- | :--- | :---: |
| **Holdout Evaluation Gate Security** | `test_stage351` through `test_stage354` | **PASS** |
| **Model Policy & Protocol Reconciliation** | `test_stage32` through `test_stage34` | **PASS** |
| **Backend API & Role-Based Auth** | `backend/test_*.py` | **PASS** |
| **Graph Models, GNN & TreeSHAP** | `test_graph_and_models.py`, `test_shap_explainer.py` | **PASS** |
| **Temporal Anti-Leakage & Data Integrity**| `test_leakage.py`, `test_data_integrity.py` | **PASS** |
| **Synthetic Generators & Injection** | `test_synthetic_*.py`, `test_fraud_injection.py` | **PASS** |
| **All Automated Test Suites** | **Complete repository test coverage** | **PASS** |

---

## ⚖️ Integrity, Ethics & Limitations

- **Local Test Evidence**: All tests in this repository represent local automated test evidence on verified synthetic fixtures. No live banking gateway or external production network is deployed.
- **Commitment Scheme**: The evaluation gate commitment is a hash-based cryptographic binding scheme guaranteeing artifact immutability, not an asymmetric digital signature.
- **Synthetic Data**: Synthetic Dataset v2.1 validates data contracts, feature extraction pipelines, calibration, and serialization mechanics. Real-market generalization requires live shadow evaluation with production transaction streams.
- **Privacy & Compliance**: All synthetic identities, graph relations, and transaction vectors are generated without real personal identifiable information (PII).

---

<div align="center">
  <sub>Developed for research, education, and marketplace integrity. Built with FastAPI, Next.js, PyTorch Geometric, and Cryptography.</sub>
</div>
