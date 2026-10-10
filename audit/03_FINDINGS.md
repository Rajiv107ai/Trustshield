# TrustShield AI — Detailed Findings & Issue Catalog

Every issue documented below has been verified with concrete evidence, code references, and reproduction steps.

---

### ISSUE-01: Cryptographic Manifest & Test Hash Failures (CRLF vs. LF Line Endings)
- **Issue ID:** `ISSUE-01`
- **Severity:** `P1 (High)`
- **Category:** `Data Integrity / Testing / DevOps`
- **Affected Files:**
  - `data/synthetic_v2_1/dataset_manifest.json`
  - `trustshield_project/test_stage32_audit.py` (line 81)
  - `trustshield_project/test_stage32_final_gate.py` (line 286)
  - `trustshield_project/test_stage351_environment_sealing.py` (line 229)
  - `trustshield_project/test_stage35_temporal_holdout_readiness.py` (lines 229, 455)
  - `trustshield_project/test_synthetic_v2_1.py` (line 89)
  - `.gitattributes` (lines 1-10)
- **Explanation:**
  The synthetic dataset tables in `data/synthetic_v2_1/` were originally generated on Windows, where files were written with CRLF (`\r\n`) line endings. The generator recorded the binary SHA-256 hashes of those CRLF files into `dataset_manifest.json` and hardcoded those hashes into the test suites. However, the repository contains a `.gitattributes` file explicitly enforcing `text eol=lf` on `*.csv` and `*.json`. When the repository is cloned or checked out, Git normalizes all CSV and JSON files to standard Unix LF line endings. Consequently, `hashlib.sha256(open(f, "rb").read())` computes the hash of the LF bytes, which mismatches the hardcoded CRLF hash.
- **Evidence:**
  - Python verification proves that computing SHA-256 with CRLF yields the exact expected test hash:
    - `orders.csv`: LF = `b5c7f2fb...`, CRLF = `325597a6...` (matches expected test hash).
    - `addresses.csv`: LF = `23790290...`, CRLF = `6bd18108...` (matches expected test hash).
    - `dataset_manifest.json`: LF = `0f635d58...`, CRLF = `b0aba240...` (matches expected test hash).
  - Test run output: 6 tests failed with `AssertionError: SHA-256 mismatch`.
- **Expected vs. Actual Behavior:**
  - *Expected:* Test suite runs cleanly with 100% pass rate across any OS or Git clone.
  - *Actual:* 6 tests fail on clean checkout due to line ending conversion.
- **Root Cause:**
  Binary hash checks performed on text files (`.csv`, `.json`) without line-ending normalization or canonical serialization prior to hashing.
- **Recommended Fix:**
  Normalize the dataset files to LF (`\n`) once on disk, recalculate the canonical LF SHA-256 hashes in `dataset_manifest.json`, and update the 6 test assertion constants to match the canonical LF hashes.
- **Verification Status:** `CONFIRMED`

---

### ISSUE-02: Next.js Frontend Client Lacks Authentication Header Support
- **Issue ID:** `ISSUE-02`
- **Severity:** `P1 (High)`
- **Category:** `Full-Stack Integration / Security`
- **Affected Files:**
  - `frontend/src/lib/api/client.ts` (lines 43-60)
  - `frontend/src/app/settings/page.tsx`
  - `backend/main.py` (line 404, 665, 767, etc.)
  - `backend/auth.py` (lines 65-95, 126-162)
- **Explanation:**
  The FastAPI backend is protected by a Role-Based Access Control framework in `backend/auth.py`. When `TRUSTSHIELD_AUTH_ENABLED=true` (or when `TRUSTSHIELD_ENV=production`), all endpoints require either an `X-API-Key` header or an `Authorization: Bearer` token. However, `frontend/src/lib/api/client.ts`'s `apiFetch` helper only includes `Content-Type: application/json`. There is no API key configured, stored in `localStorage`, or appended to outgoing requests.
- **Evidence:**
  - In `frontend/src/lib/api/client.ts`:
    ```typescript
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    }
    ```
  - Grep search for `api_key` or `X-API-Key` in `frontend/src/` returns 0 results.
  - When backend auth is active, any fetch from the frontend returns `HTTP 401 Unauthorized: {"detail": "Authentication required. Provide 'X-API-Key' header or Bearer token."}`.
- **Expected vs. Actual Behavior:**
  - *Expected:* Frontend allows user to configure an API key (e.g. in Settings) and automatically attaches `X-API-Key` to backend requests.
  - *Actual:* Real requests are rejected with HTTP 401; frontend silently falls back to client-side simulated data.
- **Root Cause:**
  Frontend was developed assuming unauthenticated local development mode without integrating the RBAC layer introduced in Phase 2.
- **Recommended Fix:**
  Add an API Key input field in `frontend/src/app/settings/page.tsx`, store `trustshield_api_key` in `localStorage`, and inject `X-API-Key` into the headers object in `frontend/src/lib/api/client.ts:apiFetch`.
- **Verification Status:** `CONFIRMED`

---

### ISSUE-03: Browser EventSource Cannot Connect to SSE Stream Under Authentication
- **Issue ID:** `ISSUE-03`
- **Severity:** `P1 (High)`
- **Category:** `Backend / Streaming / Full-Stack Integration`
- **Affected Files:**
  - `backend/main.py` (line 1062)
  - `backend/auth.py` (lines 126-162)
  - `frontend/src/lib/api/client.ts` (lines 534-650)
  - `frontend/src/app/transactions/feed/page.tsx`
- **Explanation:**
  The real-time transaction stream route `@app.get("/stream/transactions")` is guarded with `dependencies=[Depends(authenticate_request)]`. In `backend/auth.py`, `authenticate_request` only accepts credentials via the `X-API-Key` HTTP header or the `Authorization` Bearer header. However, the standard W3C `EventSource` browser API does not support adding custom headers to HTTP GET requests. Because the backend does not allow token authentication via query parameter (`?api_key=...` or `?token=...`), no web browser can connect to the authenticated live stream.
- **Evidence:**
  - In `backend/main.py:1062`: `@app.get("/stream/transactions", ..., dependencies=[Depends(authenticate_request)])`.
  - In `backend/auth.py:126-143`:
    ```python
    def authenticate_request(
        api_key: Optional[str] = Security(_api_key_header),
        bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    ) -> AuthUser:
    ```
  - In `frontend/src/lib/api/client.ts:637`: `es = new EventSource(url)`.
- **Expected vs. Actual Behavior:**
  - *Expected:* Frontend browser receives live Server-Sent Events when authenticated.
  - *Actual:* Browser connection fails with HTTP 401; frontend falls back to synthetic periodic interval timer.
- **Root Cause:**
  Protocol mismatch: `EventSource` cannot send custom request headers, and backend auth dependency lacks query-parameter token resolution.
- **Recommended Fix:**
  Update `backend/auth.py:authenticate_request` to optionally inspect an `api_key` query parameter (`api_key: Optional[str] = Query(default=None)`), or replace native browser `EventSource` with a fetch-based readable stream reader (`fetchEventSource`).
- **Verification Status:** `CONFIRMED`

---

### ISSUE-04: ESLint React Hook Error in Frontend ViewModeContext
- **Issue ID:** `ISSUE-04`
- **Severity:** `P2 (Medium)`
- **Category:** `Frontend / Code Quality`
- **Affected Files:**
  - `frontend/src/context/ViewModeContext.tsx` (line 46)
- **Explanation:**
  In `frontend/src/context/ViewModeContext.tsx`, `setIsMounted(true)` is invoked synchronously inside a `useEffect` callback to guard against SSR hydration mismatches. Under React 19 and ESLint 9 (`eslint-plugin-react-hooks`), calling `setState` directly inside an effect body without a subscription or external event triggers the `react-hooks/set-state-in-effect` error.
- **Evidence:**
  - Command output from `npm.cmd run lint`:
    ```
    C:\Users\rajiv_pis9z8x\Downloads\trustshield_full_handoff\frontend\src\context\ViewModeContext.tsx:46:5
      44 |
      45 |   useEffect(() => {
    > 46 |     setIsMounted(true);
         |     ^^^^^^^^^^^^ Avoid calling setState() directly within an effect
      47 |
      48 |     try {
      49 |       const saved = localStorage.getItem("trustshield_view_mode");  react-hooks/set-state-in-effect
    ✖ 86 problems (1 error, 85 warnings)
    ```
- **Expected vs. Actual Behavior:**
  - *Expected:* `npm run lint` passes with 0 errors.
  - *Actual:* Lint check exits with code 1.
- **Root Cause:**
  Synchronous state setter in effect violating React compiler rules.
- **Recommended Fix:**
  Use `useSyncExternalStore` or initialize `isMounted` state lazily or defer via `queueMicrotask` / `requestAnimationFrame`.
- **Verification Status:** `CONFIRMED`

---

### ISSUE-05: Pyright Type Mismatches in `backend/main.py` Fraud Ring Query Handler
- **Issue ID:** `ISSUE-05`
- **Severity:** `P2 (Medium)`
- **Category:** `Backend / Static Typing`
- **Affected Files:**
  - `backend/main.py` (lines 806-811)
- **Explanation:**
  In `backend/main.py`, the `get_fraud_rings` route iterates through `store.rings_df.iterrows()`. When reading Series values like `row["avg_risk_score"]`, pandas typing stubs return `Series | Any`. Passing these values directly to `_risk_label(row["avg_risk_score"])` (which expects `float`) causes Pyright to raise `reportArgumentType` errors.
- **Evidence:**
  - Running `.\.venv\Scripts\python.exe -m pyright trustshield_project backend` yields 12 errors:
    ```
    backend/main.py:811:36 - error: Argument of type "Self@Series | Unknown | Any | Series | ndarray" cannot be assigned to parameter "score" of type "float" in function "_risk_label"
    12 errors, 0 warnings
    ```
- **Expected vs. Actual Behavior:**
  - *Expected:* Pyright passes with 0 errors as claimed by the README badge.
  - *Actual:* 12 type errors reported.
- **Root Cause:**
  Missing `float(...)` cast in `_risk_label(float(row["avg_risk_score"]))`.
- **Recommended Fix:**
  Wrap the argument with `float(row["avg_risk_score"])`.
- **Verification Status:** `CONFIRMED`

---

### ISSUE-06: Neo4j Password Inconsistency Between Docker, Service Fallback, and Documentation
- **Issue ID:** `ISSUE-06`
- **Severity:** `P2 (Medium)`
- **Category:** `Infrastructure / DevOps / Configuration`
- **Affected Files:**
  - `backend/services/neo4j_service.py` (line 66)
  - `docker-compose.yml` (lines 30, 59)
  - `.env.example` (line 26)
  - `docs/INFRASTRUCTURE.md` (line 66)
  - `README.md` (line 301)
- **Explanation:**
  Three conflicting passwords exist:
  - In `backend/services/neo4j_service.py:66`: fallback is `"trustshield_secret_dev_only"`
  - In `docker-compose.yml:59` and `.env.example:26`: password is `"trustshield_secret"`
  - In `docs/INFRASTRUCTURE.md:66` and `README.md:301`: documented password is `"trustshield_dev_secret"`
- **Evidence:**
  When executing `scripts/e2e_smoke_validation.py` and `scripts/benchmark_scoring_http.py`, the driver connects to local port 7687 using its fallback password `"trustshield_secret_dev_only"` and fails:
  `Neo4j unavailable at bolt://localhost:7687 ({neo4j_code: Neo.ClientError.Security.Unauthorized} {message: The client is unauthorized due to authentication failure.})`.
- **Expected vs. Actual Behavior:**
  - *Expected:* Consistent default password across documentation, Docker compose, and backend fallback.
  - *Actual:* Auth failure when connecting without an explicitly configured `.env`.
- **Root Cause:**
  Incomplete password consolidation across Phase 2 updates.
- **Recommended Fix:**
  Unify the development default password to `"trustshield_secret"` across `neo4j_service.py`, `README.md`, and `docs/INFRASTRUCTURE.md`.
- **Verification Status:** `CONFIRMED`

---

### ISSUE-07: Scikit-Learn Unpickling Version Mismatch Warnings
- **Issue ID:** `ISSUE-07`
- **Severity:** `P3 (Low)`
- **Category:** `ML Engineering / Artifact Integrity`
- **Affected Files:**
  - `models/stage34/stage34_tabular_model.joblib`
  - `models/stage34/stage34_tabular_calibrator.joblib`
  - `trustshield_project/test_stage33_evaluation.py`
  - `requirements.txt` (line 7)
- **Explanation:**
  Serialized joblib models were trained with `scikit-learn==1.9.0`. The current environment installs `scikit-learn 1.9.1`. Unpickling triggers `InconsistentVersionWarning`. While `docker/requirements-evaluator.lock` pinned `scikit-learn==1.9.0`, the root `requirements.txt` has `scikit-learn>=1.3`.
- **Evidence:**
  Pytest output: `InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.9.0 when using version 1.9.1`.
- **Recommended Fix:**
  Pin `scikit-learn==1.9.0` in root `requirements.txt` or re-serialize artifacts under the target version with reproducible test validation.
- **Verification Status:** `CONFIRMED`

---

### ISSUE-08: Redis Binary vs. JSON Embedding Serialization Documentation Discrepancy
- **Issue ID:** `ISSUE-08`
- **Severity:** `P3 (Low)`
- **Category:** `Documentation / Infrastructure`
- **Affected Files:**
  - `docs/INFRASTRUCTURE.md` (line 18)
  - `backend/services/redis_service.py` (line 223)
  - `scripts/seed_mesh.py` (line 314)
- **Explanation:**
  `docs/INFRASTRUCTURE.md` specifies that 16D embeddings are stored as packed binary floats (`np.float32.tobytes()`). In reality, `scripts/seed_mesh.py` and `redis_service.py` use JSON string serialization (`json.dumps` and `json.loads`) with `decode_responses=True`.
- **Recommended Fix:**
  Update `docs/INFRASTRUCTURE.md` to document JSON string serialization, or convert the implementation to binary buffer storage.
- **Verification Status:** `CONFIRMED`

---

### ISSUE-09: Unused Variables and Dead Imports in Next.js Pages
- **Issue ID:** `ISSUE-09`
- **Severity:** `P3 (Low)`
- **Category:** `Frontend / Code Quality`
- **Affected Files:**
  - `frontend/src/app/models/page.tsx`
  - `frontend/src/app/monitoring/page.tsx`
  - `frontend/src/app/returns/page.tsx`
  - `frontend/src/app/settings/page.tsx`
  - `frontend/src/app/transactions/page.tsx`
  - `frontend/src/app/trust-graph/page.tsx`
- **Explanation:**
  ESLint reports 85 `@typescript-eslint/no-unused-vars` warnings across 10 pages and layout components.
- **Recommended Fix:**
  Prune unused lucide icon imports and unused local state flags.
- **Verification Status:** `CONFIRMED`
