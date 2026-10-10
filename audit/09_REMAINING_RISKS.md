# TrustShield AI — Remaining Risks & Architecture Limitations

---

## 1. Unresolved Technical Risks

1. **Authentication Silo Between Frontend and Backend (ISSUE-02 & ISSUE-03):**
   Until the frontend is updated with API key support, running in production mode (`TRUSTSHIELD_ENV=production`) renders the web console unable to communicate with the FastAPI serving layer, forcing it into offline mock mode.
2. **Line Ending Fragility Across Operating Systems (ISSUE-01):**
   Hashing raw text files byte-by-byte without canonical line-ending normalization introduces persistent risk of false alarms during integrity audits whenever developers clone or check out the repository across different operating systems.
3. **External Service Dependency Availability:**
   Running tests without Docker containers running causes Redis and Neo4j queries to fall back to disk joblib artifacts. While this fallback is intentionally designed and verified to operate gracefully, live Cypher query latency and dynamic Neo4j graph traversals remain unverified until the Docker mesh is running.

---

## 2. Environment Limitations & Assumptions

1. **Python 3.14 Environment:**
   The workspace runs Python 3.14.7. While core dependencies (FastAPI, PyTorch, Scikit-Learn, XGBoost) execute cleanly, certain older libraries or binary wheels may experience deprecation warnings or subtle behaviors compared to the Docker target of Python 3.11.
2. **Node / PowerShell Execution Policy:**
   PowerShell execution policies on Windows block running `npm.ps1`, requiring the use of `npm.cmd` for direct terminal execution.
