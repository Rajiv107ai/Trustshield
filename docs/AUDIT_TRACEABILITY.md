# Audit Fix Traceability

This matrix closes the remaining audit-number ambiguity. It maps the original
Master Repair/reaudit fix numbers to the current lean serving architecture,
implementation, and regression evidence. The synthetic research lane remains
separate from production governance work.

| Fix | Current interpretation | Implementation | Evidence | Status |
|---:|---|---|---|---|
| 1 | Production environment/security propagation | `backend/auth.py`, `backend/main.py`, `docker-compose.yml`, `docs/DEPLOYMENT.md` | `backend/test_security.py`, release/deploy checks | implemented; Docker runtime gate remains external |
| 2 | PIT history readiness and packaging | `.dockerignore`, `Dockerfile`, `backend/model_loader.py`, `backend/routers/system.py` | `backend/test_pit_readiness.py`, E2E readiness | implemented; external container verification pending |
| 3 | Server graph/derived-feature ownership | `backend/features.py`, `backend/routers/scoring.py` | `backend/test_feature_boundary.py`, backend suite | implemented for strict production boundary; historical feature service remains governed by PIT availability |
| 4 | API-key transport/security | `backend/auth.py`, `frontend/src/lib/api/client.ts`, `backend/test_security.py` | query-key rejection and SSE header smoke | implemented |
| 5 | Retire unavailable Phase 5 serving claims | `backend/model_loader.py`, `backend/README.md`, frontend model page | backend/API smoke and docs review | implemented |
| 6 | Probability calibration | `models/calibrator.joblib`, `backend/model_loader.py`, `backend/routers/scoring.py`, `trustshield_project/calibration.py` | calibration tests, production invariance, API smoke | implemented |
| 7 | Simulation attribution | `backend/routers/operations.py`, frontend stream status/UI | E2E/API smoke and response assertions | implemented |
| 8 | Live operator safety/operational boundary | `backend/routers/actions.py`, frontend action integration | `backend/test_actions_api.py`, action receipts | implemented as durable outbox boundary |
| 9 | Evidence provenance/grounding | `backend/services/investigation_service.py`, response schemas | backend dossier tests and provenance fields | implemented with caller-declared warnings |
| 10 | Graph snapshot freshness disclosure | `backend/model_loader.py`, readiness/version metadata | model-loader output and readiness checks | implemented disclosure; per-row fresh topology remains a future provider integration |
| 11 | Metrics/readiness serialization | `backend/services/metrics_service.py`, `backend/routers/system.py` | API smoke `/metrics`, readiness tests | implemented |
| 12 | Conditional conformal coverage | `backend/schemas.py`, `backend/routers/scoring.py`, conformal engine | backend/core tests and production invariance | implemented |
| 13 | API input range validation | `backend/schemas.py` | `backend/test_schema_validation.py` | implemented |
| 14 | Demo mode event attribution | `frontend/src/lib/api/client.ts` and stream UI | frontend build/lint and E2E | implemented |
| 15 | Benchmark estimate disclosure | `frontend/src/lib/api/client.ts`, benchmark metadata | frontend build and API smoke | implemented; network load remains external |
| 16 | Threshold centralization | `trustshield_project/policy_config.py`, `frontend/src/lib/constants/thresholds.ts` | policy/frontend checks and build | implemented |
| 17 | Settings/monitoring/trust-graph routes | frontend app routes | Next build route output | implemented |
| 18 | API/frontend contract parity | `scripts/check_openapi_contract.py`, `backend/test_openapi_contract.py` | OpenAPI guard | implemented |

The former “unknown” Fix 6 and Fix 13 are therefore named and evidenced here:
Fix 6 is the calibrated probability path; Fix 13 is strict Pydantic input
range validation.

## Audit release rule

The matrix is not a substitute for deployment evidence. A release still needs
the exact manifest, clean checkout, Docker runtime check, external HTTP load
test, provider sandbox receipts, and restore drill described in
`docs/RELIABILITY.md`.
