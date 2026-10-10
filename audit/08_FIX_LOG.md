# TrustShield AI — Remediation & Fix Log

**Current Status:** **AUDIT-ONLY MODE COMPLETED (0 CODEBASE MODIFICATIONS APPLIED)**

In strict adherence to user instructions (*"don't make any changes in codebase, only list the issues if present"*), no source code, configuration files, model artifacts, datasets, or Git branches have been altered.

---

## Planned Remediation Specification

When remediation is approved, the following minimal, targeted fixes should be executed in priority order:

### 1. Fix ISSUE-01 (P1): Re-align Canonical LF Hashes
- **Target Files:** `data/synthetic_v2_1/dataset_manifest.json`, test files in `trustshield_project/`.
- **Change:**
  1. Ensure all CSV and JSON files in `data/synthetic_v2_1/` have LF line endings.
  2. Compute canonical LF SHA-256 hashes and update `data/synthetic_v2_1/dataset_manifest.json`.
  3. Update expected hash constants in the 6 test files to match the canonical LF hashes.
- **Verification:** Run `pytest trustshield_project/ -k "manifest or virgin or unmutated"` to confirm 100% pass rate.

### 2. Fix ISSUE-02 (P1): Add Frontend Authentication Support
- **Target Files:** `frontend/src/lib/api/client.ts`, `frontend/src/app/settings/page.tsx`.
- **Change:**
  1. Add `trustshield_api_key` state management in Settings page.
  2. Update `apiFetch` in `client.ts` to include `headers["X-API-Key"] = getApiKey()`.
- **Verification:** Enable backend auth (`TRUSTSHIELD_AUTH_ENABLED=true`), configure key in UI, and verify live scoring succeeds with HTTP 200.

### 3. Fix ISSUE-03 (P1): Support Query-Param / Fetch SSE Authentication
- **Target Files:** `backend/auth.py`, `frontend/src/lib/api/client.ts`.
- **Change:**
  1. Add `api_key: Optional[str] = Query(default=None)` in `backend/auth.py:authenticate_request`.
  2. In `client.ts:createTransactionEventSource`, append `?api_key=${apiKey}` to the EventSource URL.
- **Verification:** Verify live transaction feed connects and displays events under authenticated mode.

### 4. Fix ISSUE-04 (P2): Resolve ESLint `set-state-in-effect` in ViewModeContext
- **Target Files:** `frontend/src/context/ViewModeContext.tsx`.
- **Change:**
  Replace synchronous `setIsMounted(true)` in `useEffect` with proper hydration state handling.
- **Verification:** Run `npm run lint` and verify 0 errors.

### 5. Fix ISSUE-05 (P2): Add Explicit Typing Casts in `backend/main.py`
- **Target Files:** `backend/main.py:806-811`.
- **Change:**
  Wrap pandas row values with explicit scalar casts: `_risk_label(float(row["avg_risk_score"]))`.
- **Verification:** Run `pyright trustshield_project backend` and verify 0 errors.

### 6. Fix ISSUE-06 (P2): Align Default Development Neo4j Password
- **Target Files:** `backend/services/neo4j_service.py`, `README.md`, `docs/INFRASTRUCTURE.md`.
- **Change:**
  Unify default fallback password to `"trustshield_secret"` matching `docker-compose.yml` and `.env.example`.
- **Verification:** Run `scripts/e2e_smoke_validation.py` against running Neo4j container without auth failure logs.
