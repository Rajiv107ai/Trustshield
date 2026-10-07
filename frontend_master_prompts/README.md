# TrustShield AI — Frontend Master Prompts Package

This directory contains the production-grade frontend architecture, design tokens, TypeScript schemas, and prompts for building the **TrustShield AI** enterprise user interface.

## Package Contents:

1. **`00_MASTER_FRONTEND_PROMPT.md`**  
   The complete, exhaustive master prompt for your AI coding assistant (Cursor, Claude 3.7 Sonnet, or Antigravity). Covers:
   - The Dual-Audience Paradox (`Executive Story View` ⟷ `Deep AI Inspector`).
   - Absolute Zero-Fake-Data & Backend Grounding policies.
   - Mandated tech stack (`Next.js 15` + `TypeScript` + `Tailwind` + `shadcn/ui` + `TanStack Query` + `React Flow` + `Recharts` + `Zod`).
   - Detailed specifications for all 12 views: Command Center, Transaction Risk Analyzer, Trust Graph, Fraud Rings, Listing Intelligence, Return Abuse, Investigation Cases, Model Registry, Evaluation Studio, Telemetry, and Settings.
   - The 18-Phase Implementation Plan and Quality Verification Suite.

2. **`01_TECH_STACK_AND_TOKENS.md`**  
   The design system tokens:
   - Dark enterprise console color palette (`#0B0F14`, `#111821`, `#151D27`, `#202A35`).
   - Semantic traffic-light decision tokens (`#22C55E`, `#F59E0B`, `#F97316`, `#EF4444`).
   - Typography standards (`JetBrains Mono` for tabular numbers).

3. **`02_API_SCHEMAS_TYPESCRIPT.md`**  
   TypeScript interfaces mirroring 100% of the active Pydantic v2 schemas in `backend/schemas.py`:
   - `HealthResponse` & `ReadyResponse`
   - `TransactionScoreRequest` & `TransactionScoreResponse`
   - `FraudRingItem` & `FraudRingsResponse`
   - `ListingScoreRequest` & `ListingScoreResponse`
   - `ReturnScoreRequest` & `ReturnScoreResponse`

4. **`03_SCENARIOS_AND_DUAL_MODE.md`**  
   - 4 One-click interactive test scenarios (`Verified Repeat Buyer`, `Price Anomaly`, `Device Farm Collusion`, `Serial Return Abuse`).
   - Plain-English reason code translation dictionary to make ML outputs self-explanatory to non-technical users.

---

## How to Build the Frontend:
1. Open your AI coding assistant (Cursor, Claude, or Antigravity).
2. Point it to `00_MASTER_FRONTEND_PROMPT.md`.
3. It will scaffold and build the entire frontend inside a dedicated `frontend/` directory connecting to `http://localhost:8000` without modifying any backend source code.
