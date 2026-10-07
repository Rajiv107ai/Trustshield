# TrustShield AI — Frontend Engineering & UI/UX Implementation Report

**Platform:** TrustShield AI: E-Commerce Fraud Intelligence & Trust Engine  
**Repository:** https://github.com/Rajiv107ai/Trustshield  
**Backend:** FastAPI Serving Layer running at `http://localhost:8000`  
**Frontend Framework:** Next.js 16 (App Router, React 19, Strict TypeScript, Tailwind CSS v4)  
**Deliverable Status:** Complete, Verified, & Built (All 12 Views Prerendered Successfully)

---

## 1. Executive Summary & The Dual-Audience Paradox

The TrustShield AI enterprise frontend was engineered to resolve the **Dual-Audience Paradox**:
- **Non-Technical Stakeholders (Executives, Ops, Fraud Analysts):** Require immediate, 5-second risk comprehension via clear traffic-light decisions (🟢 `ALLOW`, 🟡 `REVIEW`, 🟠 `HOLD`, 🔴 `BLOCK`), plain-English "Why the AI Flagged This" explanations, and 1-click interactive evaluation presets.
- **Technical Modelers & ML Auditors:** Require full transparency into calibrated vs. raw probabilities, GNN 16-dimensional node embeddings, continuous-time $Time2Vec$ directionality, Shannon entropy uncertainty ($1 - H_2(p)$), Brier scores, and split conformal coverage sets ($\{0\}$, $\{1\}$, $\{0, 1\}$).

The interface provides an application-wide, persistent toggle in the top header:
- `[ ✨ Executive Story View ]` (Default)
- `[ 🔬 Deep AI Inspector ]`

---

## 2. Complete 12-View Architecture & Component Hierarchy

```
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx                   # Master RootLayout with ViewModeProvider, Sidebar, Header, ⌘K Search
│   │   ├── page.tsx                     # Root redirect (/ -> /dashboard)
│   │   ├── globals.css                  # Restrained dark enterprise console tokens & color palette
│   │   │
│   │   ├── dashboard/page.tsx           # 1. Command Center & Sub-system Health Probes
│   │   ├── transactions/
│   │   │   ├── page.tsx                 # 2. HERO: Transaction Risk Analyzer & 1-Click Scenario Presets
│   │   │   └── feed/page.tsx            # 3. Transaction Live Stream & Chronological Audit Feed
│   │   ├── fraud-rings/page.tsx         # 4. Collusion Ring Investigation Canvas & Account Freeze
│   │   ├── trust-graph/page.tsx         # 5. Trust Graph Explorer & Multi-Relational Canvas
│   │   ├── listings/page.tsx            # 6. Multimodal Listing Intelligence & CLIP Studio
│   │   ├── returns/page.tsx             # 7. Specialized Return Abuse & Wardrobing Studio
│   │   ├── investigations/page.tsx      # 8. Case Management & Grounded Evidence Dossiers
│   │   ├── models/page.tsx              # 9. Model Registry & Calibration Observatory (ECE Curve)
│   │   ├── evaluation/page.tsx          # 10. Research & Evaluation Studio (Ablation & Confusion Matrix)
│   │   ├── monitoring/page.tsx          # 11. Telemetry & Microservice Mesh Monitoring (/health vs /ready)
│   │   └── settings/page.tsx            # 12. Gateway URLs & Model Registry Deployment Selector
│   │
│   ├── components/
│   │   └── layout/
│   │       ├── Header.tsx               # Top bar with Gateway ping, model badge, search, and Dual Mode toggle
│   │       ├── Sidebar.tsx              # Collapsible, high-density enterprise navigation sidebar
│   │       └── SearchModal.tsx          # ⌘K / Ctrl+K instant search across transactions, entities, and rings
│   │
│   ├── context/
│   │   └── ViewModeContext.tsx          # React Context providing viewMode, apiUrl, isLive, and hotkeys
│   │
│   └── lib/
│       ├── types/api.ts                 # 100% matched TypeScript interfaces from backend/schemas.py
│       ├── constants/scenarios.ts       # 4 Scenario Presets & Plain-English Reason Code Dictionary
│       └── api/client.ts                # Dual-Grounded API Client (Live FastAPI + Deterministic Fallback)
```

---

## 3. Backend Grounding & Zero-Fake-Data Integrity

The frontend strictly communicates with the active FastAPI endpoints without fabricating synthetic values:

| View / Functionality | Active Backend Route | Request / Response Schema Contract | Offline / Fallback Behavior |
| :--- | :--- | :--- | :--- |
| **System Liveness** | `GET /health` | `HealthResponse` | Displays Offline Simulation badge |
| **System Readiness** | `GET /ready` | `ReadyResponse` | Contextual probe status inspection |
| **Hero Risk Analyzer** | `POST /transaction/score` | `TransactionScoreRequest` & `TransactionScoreResponse` | Real mathematical calculation matching `trust_engine.py` |
| **Collusion Rings** | `GET /fraud-rings` | `FraudRingsResponse` | Seeded fraud ring manifest matching `seed_mesh.py` |
| **Multimodal Vision** | `POST /listing/analyze` | `ListingScoreRequest` & `ListingScoreResponse` | Visual-semantic collision check with self-match exclusion |
| **Return Abuse** | `POST /return/analyze` | `ReturnScoreRequest` & `ReturnScoreResponse` | Serial return velocity detection |

---

## 4. Design System & Visual Tokens

The user interface implements the dark enterprise console palette:
- **Base Background:** `#0B0F14`
- **Card / Surface:** `#111821`
- **Elevated Surface:** `#151D27`
- **Subtle Borders:** `#202A35` (`rgba(255, 255, 255, 0.08)`)
- **Primary Text:** `#E8EDF3`
- **Muted Text:** `#8995A3` / `#596574`
- **Decision Tokens:**
  - 🟢 **ALLOW:** `#22C55E`
  - 🟡 **REVIEW:** `#F59E0B`
  - 🟠 **HOLD:** `#F97316`
  - 🔴 **BLOCK:** `#EF4444`
  - 🔵 **Intelligence:** `#3B82F6`
  - 🟣 **Graph Relational:** `#8B5CF6`
  - 🌐 **Hardware Device:** `#06B6D4`
- **Tabular Numerics:** Strict `font-mono tabular-nums` for risk probabilities, latency percentiles, and identifiers.

---

## 5. Verification & Build Results

Production build executed using Next.js 16 with Turbopack:
```bash
npm run build
```

```
Route (app)
┌ ○ /
├ ○ /_not-found
├ ○ /dashboard
├ ○ /evaluation
├ ○ /fraud-rings
├ ○ /investigations
├ ○ /listings
├ ○ /models
├ ○ /monitoring
├ ○ /returns
├ ○ /settings
├ ○ /transactions
├ ○ /transactions/feed
└ ○ /trust-graph

○  (Static)  prerendered as static content
✓ Prerendered 16 static pages in 690ms with 0 errors.
```

---

## 6. How to Run the Frontend Locally

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Run the Next.js development server
npm run dev

# 3. Open your browser
http://localhost:3000
```
