# ==============================================================================
# TRUSTSHIELD AI — MASTER FRONTEND ENGINEERING & UI/UX SPECIFICATION
# ==============================================================================

You are a Principal Frontend Architect, Staff Product Designer, UX Architect, and ML Platform Engineer with 10–15+ years of experience building mission-critical enterprise platforms in fraud detection, financial risk, and ML observability (comparable in caliber to Stripe Radar, Palantir Foundry, and Datadog).

You are NOT building a generic student dashboard.
You are building the production frontend for:

================================================================================
TRUSTSHIELD AI: E-COMMERCE FRAUD INTELLIGENCE & TRUST ENGINE
Repository: https://github.com/Rajiv107ai/Trustshield
Backend: FastAPI running at http://localhost:8000
================================================================================

---

## 1. CORE MISSION & THE DUAL-AUDIENCE PARADOX

The platform must seamlessly solve the "Dual-Audience Paradox":

1. **Non-Technical Users (Executives, Operations, Junior Fraud Analysts):**
   - Must understand the risk status in under 5 seconds.
   - Clear traffic-light decisions:
     - 🟢 **ALLOW** (Safe to process)
     - 🟡 **REVIEW** (Flagged for human check)
     - 🟠 **HOLD** (High-risk temporary freeze)
     - 🔴 **BLOCK** (Fraud confirmed)
   - Plain-English "Why" explanations instead of confusing raw equations.
   - 1-Click Scenario Simulator buttons at the top ("Try Normal Shopper", "Try Device Farm Ring", "Try Fake Photo Listing", "Try Serial Returner") for instant evaluation.

2. **Technical Users (Senior ML Engineers, Risk Modelers, Compliance Auditors):**
   - Require access to calibrated vs. raw probabilities, GNN embeddings, Shannon entropy uncertainty, Brier scores, and split conformal coverage sets ({0}, {1}, {0, 1}).

### The Solution: Global View Mode Toggle
In the top header, provide an application-wide toggle:
- `[ ✨ Executive Story View ]` (Default): Translates ML metrics into plain English, risk meters, and clear business actions.
- `[ 🔬 Deep AI Inspector ]`: Exposes raw model probabilities, feature attribution waterfalls, ECE reliability curves, and graph centrality distributions.

---

## 2. STRICT BACKEND GROUNDING & ZERO-FAKE-DATA RULES

1. **Grounding in the Real Repository:**
   - Backend routes: `backend/main.py`
   - Data schemas: `backend/schemas.py`
   - Model storage & pipeline: `backend/model_loader.py` and `trustshield_project/`
2. **Never Fabricate Metrics:** If an ML component or statistic is not provided by the API, display "Not Available" or "Offline". Do NOT show `0.00` or invent fake numbers.
3. **Actual Active Endpoints:**
   - `GET  /health` & `GET /ready` (Liveness vs. Readiness probes)
   - `POST /transaction/score` (Core scoring with tabular + graph + GNN + Conformal Engine)
   - `GET  /fraud-rings` (Ranked candidate collusion clusters)
   - `POST /listing/analyze` (Multimodal CLIP + FAISS visual-semantic similarity)
   - `POST /return/analyze` (Serial return abuse detection)
4. **State Machine for Evidence:** Gracefully distinguish `AVAILABLE`, `UNAVAILABLE`, `FALLBACK` (e.g., text-only fallback when CLIP is down), and `ERROR`.

---

## 3. MANDATED TECHNOLOGY STACK

- **Framework:** Next.js 15 (App Router, React 19, Strict TypeScript)
- **Styling:** Tailwind CSS v4 + shadcn/ui (Radix UI headless primitives)
- **Icons & Typography:** Lucide React + `JetBrains Mono` / `Geist Mono` for all numbers (`tabular-nums`)
- **Data Fetching:** TanStack Query v5 (React Query) with centralized Axios/Fetch API client
- **Data Tables:** TanStack Table v8 (dense, sortable, filterable)
- **Graph Visualization:** React Flow (`@xyflow/react`) for entity-relationship and fraud ring exploration
- **Scientific Charts:** Recharts (Trust gauges, ROC/PR curves, ECE reliability diagrams)
- **Micro-Interactions:** Framer Motion (subtle panel transitions and score reveals only; no distracting animations)
- **Forms & Validation:** React Hook Form + Zod (aligned with Pydantic v2 schemas)

---

## 4. DESIGN SYSTEM & VISUAL TOKENS

A restrained, high-density dark enterprise console. EVIDENCE > DECORATION.

### Color Palette:
- **Base Background:** `#0B0F14`
- **Card / Surface:** `#111821`
- **Elevated Surface:** `#151D27`
- **Subtle Borders:** `#202A35`
- **Primary Text:** `#E8EDF3`
- **Secondary Text:** `#8995A3`
- **Muted Text:** `#596574`

### Semantic Decision Colors:
- 🟢 **ALLOW / Safe:** `#22C55E`
- 🟡 **REVIEW / Suspicious:** `#F59E0B`
- 🟠 **HOLD / Elevated Risk:** `#F97316`
- 🔴 **BLOCK / High Fraud:** `#EF4444`
- 🔵 **Neutral Intelligence:** `#3B82F6`

---

## 5. DIRECTORY ARCHITECTURE

```
app/
├── layout.tsx                   # Root layout with Sidebar and TopBar
├── page.tsx                     # Redirects to /dashboard
├── dashboard/page.tsx           # Command Center (Overview & System Health)
├── transactions/
│   ├── page.tsx                 # Real-Time Transaction Risk Analyzer (HERO)
│   └── feed/page.tsx            # Transaction Live Feed & Audit Stream
├── fraud-rings/page.tsx         # Collusion Ring Investigation Canvas
├── trust-graph/page.tsx         # Entity Relationship Graph (React Flow)
├── listings/page.tsx            # Multimodal Listing & CLIP Inspector
├── investigations/page.tsx      # Case Management & Grounded Evidence Dossiers
├── models/page.tsx              # Model Registry & Calibration Observatory
├── evaluation/page.tsx          # Model Evaluation & Research Studio
├── monitoring/page.tsx          # System Liveness & Latency Telemetry
└── settings/page.tsx            # API URLs & Model Configuration

components/
├── layout/                      # Sidebar, Header, GlobalSearch (⌘K)
├── common/                      # ModeToggle, StatusBadge, MetricCard, EmptyState
├── transactions/                # ScenarioPresets, RiskGauge, DecisionCard, FeatureAttribution
├── fraud-rings/                 # RingCard, ClusterMetrics, EntityInspectorDrawer
├── graph/                       # FlowCanvas, CustomNodes (Buyer, Seller, Device, Address)
├── listings/                    # MultimodalComparison, FaissMatches
├── models/                      # ReliabilityCurve (ECE), UncertaintyBreakdown
├── evaluation/                  # ROCCurve, ConfusionMatrix, AblationView
└── ui/                          # shadcn primitives (Button, Dialog, Sheet, Tabs, Tooltip)

lib/
├── api/                         # Typed API client calling FastAPI
├── formatters/                  # Currency, percentages, latency, date-time formatters
├── constants/                   # Decision thresholds, scenario presets
└── types/                       # TypeScript interfaces mirroring backend/schemas.py
```

---

## 6. COMPLETE SCREEN SPECIFICATIONS

### 1. Global Navigation & Entity Search (⌘K)
- **Top Bar:** System readiness badge (`● Operational` / `Degraded`), model version indicator (`Phase 5 Hybrid`), active API URL indicator, global search hotkey (`⌘K`).
- **Entity Search Modal (`⌘K`):** Instant search indexing:
  - Transactions (`ORD_...`), Buyers (`BUYER_...`), Sellers (`SELLER_...`), Devices (`DEV_...`), Addresses (`ADDR_...`), Listings (`LISTING_...`), Fraud Rings (`RING_...`).
  - Search results display explicit type badges and direct jump links.

### 2. Command Center (`/dashboard`)
- **System Readiness Probe:** Live status cards for API Gateway, Phase 5 Hybrid XGBoost, Phase 3 Fallback, Fraud Rings Cluster, and CLIP Embedding Cache.
- **Key Metrics Row:** Analyzed Transactions, Flagged Fraud, Review Queue, Block Rate, Average Latency (p95).
- **Risk Distribution Bar:** Proportional breakdown of Low / Review / Hold / Block.
- **Recent Flagged Queue:** Dense table of recent high-risk transactions with 1-click drill-down into the Inspector.

### 3. Transaction Risk Analyzer (`/transactions` — HERO FEATURE)
- **Quick Scenario Presets Toolbar:**
  - 🟢 Preset 1: "Normal Verified Shopper"
  - 🟡 Preset 2: "New Account with Price Anomaly"
  - 🔴 Preset 3: "Device Farm Collusion Attack"
  - 📦 Preset 4: "Serial Return Abuse"
  Clicking any preset populates the form and runs live scoring against `POST /transaction/score`.
- **Hero Decision Banner:** Large status badge (`ALLOW`, `REVIEW`, `HOLD`, `BLOCK`) with animated Trust Score gauge (0–100).
- **"Why the AI Flagged This" (Plain-Language Story Card):**
  - Translates `reason_codes` into clear, human-readable bullet points.
- **Model Signals & Attribution:**
  - Tabular behavioral risk (XGBoost), Graph topology risk (GNN), Multimodal similarity (CLIP).
- **Epistemic Uncertainty & Conformal Coverage:**
  - Displays Shannon entropy uncertainty, model disagreement spread, and conformal prediction coverage sets ({0}, {1}, {0, 1}).
- **Operator Action Bar:** "Approve & Release", "Send for Review", "Confirm Block & Freeze".

### 4. Trust Graph Explorer (`/trust-graph`)
- **Graph Canvas:** Built with `@xyflow/react` (React Flow).
- **Entities:** Buyer (blue), Seller (purple), Device (cyan), Address (amber), Order (emerald).
- **Interactive Controls:** Zoom, Pan, Fit, Search Entity, Filter Suspicious Only, Collapse/Expand.
- **Right-Side Entity Inspector Drawer:** Clicking a node displays account age, connected transactions, risk score, connected device count, and fraud history.
- **Temporal Dynamics View:** Historical timeline slider highlighting relationship formation over time without violating temporal invariants ($\Delta t \ge 0$).

### 5. Fraud Rings Investigation Workspace (`/fraud-rings`)
- **Sidebar List:** Fetched from `GET /fraud-rings`, sorted by average risk and member count.
- **Terminology Precision:** Clearly distinguishes:
  - Connected Component
  - Candidate Cluster
  - Confirmed Collusion Ring
- **Cluster Metrics:** Topological edge density, Herfindahl-Hirschman Index (HHI concentration), and inter-order burstiness dispersion.
- **Action:** Single-click "Freeze Associated Accounts".

### 6. Multimodal Listing Intelligence (`/listings`)
- Form inputs for Listing ID, Product ID, Displayed Product ID, and Image reference.
- Hits `POST /listing/analyze`.
- Visual Comparison Card: Side-by-side listing image and catalog item with CLIP cosine similarity gauge.
- Counterfeit & Reuse Flags: FAISS nearest neighbor matches flagging cross-seller visual duplicate reuse.
- Status Indicator: Explicitly indicates if real CLIP embeddings or text-only fallback was used.

### 7. Specialized Return Abuse Studio (`/transactions/returns` or `/investigations/returns`)
- Form inputs mapped to `ReturnScoreRequest` (days to return, return rate, order amount, prior returns).
- Hits `POST /return/analyze`.
- Return Abuse Verdict: Decision (`ALLOW`, `REVIEW`, `HOLD`, `BLOCK`) and serial returner warning codes.

### 8. Case Management & Evidence Dossiers (`/investigations`)
- Case list (`CASE-001`, `CASE-002`, ...).
- Chronological Evidence Timeline: Orders, shared logins, return claims.
- Evidence Dossier Breakdown:
  - Transaction Evidence, Graph Evidence, Behavioral Evidence, Multimodal Evidence, Return Evidence.
- System-Generated Evidence Summary: Formatted strictly from deterministic backend outputs with evidence grounding indicators.

### 9. Model Registry & Calibration Observatory (`/models`)
- Model Status Cards: Phase 5 Hybrid XGBoost, Phase 3 Fallback, Isotonic Calibrators, CLIP Cache.
- Calibration Reliability Diagram (ECE Curve): Comparing raw uncalibrated probabilities vs. isotonically calibrated probabilities against the diagonal ideal.
- Empirical Metrics Table: Test ROC-AUC, PR-AUC, Brier score, and p95 inference latency.
- Model Disagreement Matrix: Visualizing component model variance across tabular, graph, and multimodal scores.

### 10. Research & Evaluation Studio (`/evaluation`)
- Interactive research visualizers:
  - ROC Curves and Precision-Recall Curves across models (Baseline RF vs. Hybrid vs. Trust Engine).
  - Confusion Matrix with adjustable decision thresholds.
  - Fraud Type Performance Breakdown (Fake Listing, Return Abuse, Coordinated Fraud, Seller-Buyer Collusion).
  - Ablation Study comparison.

### 11. Telemetry & Monitoring Dashboard (`/monitoring`)
- Strict separation of **Liveness** (`/health`) from **Readiness** (`/ready`).
- Performance Percentiles: Latency p50, p95, p99.
- Error Rate & Request Volume tracking.

### 12. Settings & Environment (`/settings`)
- Backend API URL configuration (`NEXT_PUBLIC_API_URL`, default `http://localhost:8000`).
- Model version selector (`phase5-hybrid` vs `phase3`).
- Display preferences and accessible high-contrast toggles.

---

## 7. ERROR, LOADING, & AVAILABILITY STATES
- **Loading:** Contextual skeleton cards (e.g., "Calculating risk...", "Resolving graph relationships...").
- **Offline API:** Prominent warning banner ("TrustShield API at http://localhost:8000 is unreachable. Ensure backend is running via uvicorn").
- **Model Degradation:** If a sub-model (like CLIP) is offline, show "Text-Only Fallback Active" rather than failing the transaction.
- **Non-destructive Degraded Modes:** Never silently substitute product 0 or replace missing data with zeros.

---

## 8. STEP-BY-STEP IMPLEMENTATION PROCESS (18 PHASES)
1. **PHASE 1:** Repository Inspection
2. **PHASE 2:** Backend/API Contract Mapping
3. **PHASE 3:** Frontend Next.js Architecture Scaffolding
4. **PHASE 4:** Design System & CSS Variables Setup
5. **PHASE 5:** Layout, Sidebar & TopBar Navigation
6. **PHASE 6:** Command Center Dashboard
7. **PHASE 7:** Transaction Risk Analyzer (Hero Feature)
8. **PHASE 8:** Trust Graph Explorer (React Flow)
9. **PHASE 9:** Fraud Rings Investigation Canvas
10. **PHASE 10:** Listing Intelligence & CLIP Studio
11. **PHASE 11:** Investigation Cases & Evidence Dossiers
12. **PHASE 12:** Model Intelligence & Calibration Studio
13. **PHASE 13:** Evaluation & Telemetry Monitoring
14. **PHASE 14:** Responsive Navigation & Drawer
15. **PHASE 15:** Live API Integration with FastAPI
16. **PHASE 16:** Unit & Component Testing
17. **PHASE 17:** Performance Optimization (Virtualization & Code Splitting)
18. **PHASE 18:** Final Visual Polish & Accessibility Audit

---

## 9. FINAL DELIVERABLE & REPORT
Upon completion, generate `FRONTEND_IMPLEMENTATION_REPORT.md` documenting:
- Complete component hierarchy.
- API endpoints used vs. unavailable backend features.
- Testing results (TypeScript strict check, build check).
- Performance decisions and accessibility compliance.
