# TrustShield AI — API & Frontend Integration Verification Matrix

---

## 1. Backend API Endpoint Inventory

| Endpoint | Method | Auth Required | Minimum Role | Backend Handler | Frontend Client Method | Live / Fallback Integration |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| `/health` | `GET` | No | None | `main.py:health` | `TrustShieldApi.getHealth()` | Live endpoint with offline simulation fallback |
| `/ready` | `GET` | No | None | `main.py:ready` | `TrustShieldApi.getReady()` | Live endpoint with offline simulation fallback |
| `/metrics` | `GET` | No | None | `main.py:prometheus_metrics` | Direct Prometheus scrape | Live exposition |
| `/transaction/score` | `POST` | **Yes** | Any Authenticated | `main.py:score_transaction` | `TrustShieldApi.scoreTransaction()` | **Blocked if auth enabled (ISSUE-02)** |
| `/transaction/explain`| `POST` | **Yes** | Any Authenticated | `main.py:explain_transaction` | `TrustShieldApi.explainTransaction()` | **Blocked if auth enabled (ISSUE-02)** |
| `/fraud-rings` | `GET` | **Yes** | Any Authenticated | `main.py:get_fraud_rings` | `TrustShieldApi.getFraudRings()` | **Blocked if auth enabled (ISSUE-02)** |
| `/listing/analyze` | `POST` | **Yes** | Any Authenticated | `main.py:analyze_listing` | `TrustShieldApi.scoreListing()` | **Blocked if auth enabled (ISSUE-02)** |
| `/return/analyze` | `POST` | **Yes** | Any Authenticated | `main.py:analyze_return` | `TrustShieldApi.scoreReturn()` | **Blocked if auth enabled (ISSUE-02)** |
| `/investigation/generate-dossier` | `POST` | **Yes** | **OPERATOR** | `main.py:generate_dossier` | `TrustShieldApi.generateDossier()` | **Blocked if auth enabled (ISSUE-02)** |
| `/system/benchmark` | `GET` | **Yes** | **ADMIN** | `main.py:system_benchmark` | `TrustShieldApi.getBenchmark()` | **Blocked if auth enabled (ISSUE-02)** |
| `/stream/transactions` | `GET` (SSE) | **Yes** | Any Authenticated | `main.py:stream_transactions` | `TrustShieldApi.createTransactionEventSource()` | **Blocked (ISSUE-02 & ISSUE-03)** |

---

## 2. Frontend Page Integration Map

| Frontend Page Route | Key User Action | Target Backend API | Observed Behavior (Auth Disabled) | Observed Behavior (Auth Enabled) |
| :--- | :--- | :--- | :--- | :--- |
| `/dashboard` | View metrics, health & action queues | `/health`, `/ready` | Live data | Live data (endpoints are public) |
| `/transactions` | 1-click test scenarios, score transactions | `/transaction/score`, `/transaction/explain` | Real FastAPI scoring + TreeSHAP | **Fails (401) -> Falls back to client mock** |
| `/transactions/feed` | Real-time SSE streaming live feed | `/stream/transactions` | Live SSE feed (2.0s interval) | **Fails (401) -> Falls back to 4s interval timer** |
| `/fraud-rings` | Browse rings, freeze member entities | `/fraud-rings` | Real pre-computed rings (708 rings) | **Fails (401) -> Falls back to static list** |
| `/trust-graph` | Relational 2-hop graph explorer | Uses rings data | Live visual graph canvas | Mock visual graph |
| `/listings` | Multimodal CLIP image/text audit | `/listing/analyze` | Real CLIP-ViT cosine similarity | **Fails (401) -> Falls back to static cards** |
| `/returns` | Serial return pattern analysis | `/return/analyze` | Real return risk scoring | **Fails (401) -> Falls back to static cards** |
| `/investigations` | AI forensic case dossier generator | `/investigation/generate-dossier` | Real Grounded RAG Dossier | **Fails (401) -> Falls back to mock dossier** |
| `/monitoring` | Prometheus probe status & system latency | `/ready`, `/system/benchmark` | Latency histograms & service state | **Benchmark fails (401) -> Mock latency** |
| `/settings` | Switch API URL & model version | Local state update | Saves to `localStorage` | URL saved, but cannot supply API key |

---

## 3. Disconnect Summary

Because `client.ts` never attaches an `X-API-Key` or `Bearer` token, the frontend's Dual Grounded Mode seamlessly masks HTTP 401 failures by falling back to deterministic local mock routines. This creates the illusion that the platform is operating, while in reality no backend intelligence is being reached under secure environments.
