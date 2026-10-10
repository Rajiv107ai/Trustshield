# TrustShield AI — Complete File Inventory & Module Inspection Status

**Total Inventoried Files (Excluding `.git`, `.venv`, `node_modules`, `.next`, cache):** ~325 project files (plus ~350,000 raw external ABO dataset images in `data/external/abo/`).

---

## 1. Directory Structure & Module Breakdown

| Directory / Package | File Count | Primary Functionality | Inspection Status |
| :--- | :---: | :--- | :--- |
| `backend/` | 11 | FastAPI scoring gateway, authentication, schemas, services, tests | **INSPECTED (100%)** |
| `backend/services/` | 6 | Redis feature store, Neo4j graph driver, dossier agent, Prometheus | **INSPECTED (100%)** |
| `frontend/src/app/` | 14 | Next.js 16 App Router pages (Dashboard, Transactions, Feed, Rings, Graph, etc.) | **INSPECTED (100%)** |
| `frontend/src/components/` | 12 | Navigation headers, sidebar, command palette modal, badges, UI elements | **INSPECTED (100%)** |
| `frontend/src/context/` | 1 | Global `ViewModeContext` managing API URLs, offline fallbacks, view mode | **INSPECTED (100%)** |
| `frontend/src/lib/` | 4 | API client (`client.ts`), mock fallback generators, TypeScript types | **INSPECTED (100%)** |
| `trustshield_project/` | 28 | Core ML models, feature extractors, evaluation gate, leakage tests | **INSPECTED (100%)** |
| `models/` | 28 | Pre-trained model artifacts (`.joblib`, `.npy`, `.json`, CLIP embeddings) | **INSPECTED (100%)** |
| `data/synthetic_v2/` | 7 | Synthetic marketplace tables (orders, listings, returns, sellers, buyers) | **INSPECTED (100%)** |
| `data/synthetic_v2_1/` | 19 | Generalized synthetic benchmark tables, perturbation logs, manifest | **INSPECTED (100%)** |
| `scripts/` | 14 | Data generators, HTTP benchmarks, mesh seeder, evaluation runners | **INSPECTED (100%)** |
| `docker/` | 5 | Evaluator Dockerfile, Grafana provisioning, Prometheus config | **INSPECTED (100%)** |
| `docs/` | 28 | Architectural blueprints, baseline audits, model cards, specifications | **INSPECTED (100%)** |
| `reports/` | 24 | Cryptographic holdout audits, experiment results, ledger verifications | **INSPECTED (100%)** |
| `frontend_master_prompts/` | 5 | Frontend design guidelines, tokens, schema specifications | **INSPECTED (100%)** |
| Root Configuration | 12 | `docker-compose.yml`, `Dockerfile`, `requirements.txt`, `.gitattributes`, etc. | **INSPECTED (100%)** |

---

## 2. Key Entry Points & Module Tracing

1. **Backend Application Gateway:**
   - Entry Point: `backend/main.py:app`
   - Command: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`
   - Health probes: `GET /health`, `GET /ready`
   - Scoring pipeline: `POST /transaction/score` -> `_build_hybrid_feature_row` -> `store.hybrid_model.predict_proba` -> `store.phase5_calibrator` -> `TrustEngine.score` -> `TreeSHAP` -> response.

2. **Frontend Operational Console:**
   - Entry Point: `frontend/src/app/page.tsx` -> redirects to `/dashboard`
   - Development server: `npm run dev` (Turbopack on port 3000)
   - Production bundle: `next build` -> 16 static routes

3. **Autonomous End-to-End Validator:**
   - Entry Point: `scripts/e2e_smoke_validation.py`
   - Executes 9 distinct operational scenarios without external dependencies.

4. **Cryptographic Holdout Gate:**
   - Entry Point: `trustshield_project/evaluation_gate.py`
   - Enforces 6-state execution ledger, SHA-256 commit-reveal binding, and AES-256-GCM label protection.
