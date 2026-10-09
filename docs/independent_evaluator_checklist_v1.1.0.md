# TrustShield — Independent Evaluator Readiness & Execution Checklist

**Document Version**: 1.1.0  
**Status**: RECONCILED AUDIT PROTOCOL FOR INDEPENDENT EVALUATOR SIGN-OFF  
**Target Evaluation**: Blind-Holdout Evaluation (Synthetic Dataset v2.2)  
**Governance Scope**: Independent Execution Boundary, Cryptographic Commitment, and Enforceable Gates  
**Mandatory Notice**: *Completion or possession of this document does NOT constitute authorization to generate, decrypt, inspect, or unblind Synthetic Dataset v2.2. Formal authorization requires independent stakeholder sign-off as defined in Section 1. No holdout labels or decryption keys are accessible within this environment.*

---

## 1. Roles, Custody, and Governance Separation

To guarantee scientific objectivity and prevent adaptive tuning or target leakage, the evaluation protocol strictly enforces the following separation of responsibilities:

| Role | Designee / Authority | Mandatory Scope & Governance Responsibility |
| :--- | :--- | :--- |
| **Executive Authorizer** | TrustShield Governance Board | Sole authority to grant formal written authorization for holdout dataset generation, remote ledger registration, and final promotion sign-off. Approves numerical promotion thresholds prior to holdout evaluation. |
| **Independent Evaluator** | External / Independent Auditor | Exclusive custodian of holdout random seed, AES-256-GCM label decryption key, and evaluation credentials. Approves future input manifest, verifies OCI image digest, claims remote ledger run, and executes one-shot scoring. |
| **MLOps Engineer** | Deployment Custodian | Builds OCI container image, verifies build reproducibility, deploys authenticated remote ledger service, provides sealed execution environment. |
| **ML Research Engineer** | Model Developer | Submits frozen candidate model artifact, calibrator artifact, and prediction commitment. Generates predictions solely from unblinded transaction features. **Has zero access to holdout labels, random seeds, or decryption keys.** |

### Key Custody & Execution Boundary Invariant
```
+-----------------------------------------------------------------------------------------+
|                                 DEVELOPER BOUNDARY                                      |
|                                                                                         |
|  Unblinded Features  -----> [ Frozen Model + Calibrator ] -----> Canonical Predictions   |
|  (holdout_orders.csv)        (No Label Access Ever)               (<f8 Little-Endian)    |
|                                                                            |            |
|                                                                            v            |
|                                                                   10-Artifact Commitment|
+----------------------------------------------------------------------------|------------+
                                                                             | Published
+----------------------------------------------------------------------------v------------+
|                                EVALUATOR BOUNDARY                                       |
|                                                                                         |
|  10-Artifact Verification  ====> [ Authoritative Remote Ledger ]                        |
|  Manifest & Environment                                  |                              |
|                                                          v                              |
|  AES-256-GCM Key (Evaluator Memory Only) ==> [ Ephemeral One-Shot Scorer ]              |
|                                                          |                              |
|                                                          v                              |
|                                                Aggregate Metrics Only                   |
|                                            (Zero Instance Leakage to Dev)               |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Assurance Level Classification System

Every technical requirement in this checklist is explicitly tagged with its current assurance status:

- `[IMPLEMENTED]` — Code exists in the repository.
- `[LOCALLY_TESTED]` — Verified by local automated unit/adversarial pytest fixtures.
- `[INTEGRATION_TESTED]` — Verified across multi-module workflow fixtures.
- `[INDEPENDENTLY_VERIFIED]` — Cryptographic hashes, code boundaries, and protocol contracts independently verified.
- `[DEPLOYED]` — Service, daemon, or image is actively running in authorized infrastructure.
- `[FORMALLY_AUTHORIZED]` — Formally signed off by the Executive Governance Board.

---

## 3. Five-Phase Verification Checklist

### Phase I: Pre-Generation & Environment Sealing

- [x] **1.1. Candidate Model & Calibrator Verification** `[LOCALLY_TESTED]` *(Reconciled Baseline; Awaiting Independent Evaluator Audit)*
  - Candidate Model: [`models/stage34/stage34_tabular_model.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_model.joblib) (1,025,950 bytes)
  - Reconciled Actual SHA-256: `1310b43c3c8d23a1d4054200a5321f815de113b1a07793bb0d17f508c6cac22a`
  - Calibrator Artifact: [`models/stage34/stage34_tabular_calibrator.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_calibrator.joblib) (1,468 bytes)
  - Reconciled Actual SHA-256: `152f287e8f4d478285eeac13d380f4312ceb453bed85b3ff1f99d799424148a8`
  - Historical Discrepancy Note: The prior Checklist v1.1.0 hashes (`1310b43c3c8ddb66...` and `152f287e8f4da138...`) were test fixture mock strings transcribed into the report, not computed from file bytes. Physical on-disk files match the approved Stage 3.5.3 baseline bit-for-bit.
  - Feature Schema: Strictly verified against the exact 10 ordered tabular features (`price_vs_base_price_ratio`, `price_vs_category_median_ratio`, `seller_age_days`, `seller_total_listings_before`, `buyer_age_days`, `buyer_orders_before`, `buyer_returns_before`, `buyer_return_rate_before`, `device_shared_buyer_count`, `amount`).
- [x] **1.2. Source Code & Evaluation Gate Lock** `[LOCALLY_TESTED]` *(Awaiting Independent Evaluator Audit)*
  - Baseline Git Commit: `4615a53602d0a4ffe447d4d2fd14bd69a70bf740`
  - Git Branch: `phase-3-data-generalization`
  - Evaluation Gate Module: [`trustshield_project/evaluation_gate.py`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/trustshield_project/evaluation_gate.py)
- [ ] **1.3. OCI Container Build & Digest Pinning** `[IMPLEMENTED]` *[BLOCKER: Docker Engine Inactive]*
  - Evaluator Dockerfile: [`docker/Dockerfile.evaluator`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/docker/Dockerfile.evaluator)
  - Dependency Lockfile: [`docker/requirements-evaluator.lock`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/docker/requirements-evaluator.lock) (scikit-learn pinned to 1.9.0)
  - Requirement: Evaluator must execute a reproducible container build and extract the immutable OCI image digest (`docker-image@sha256:...`). Mutable tags (e.g., `:latest`) are strictly prohibited.
  - Verification Status: Docker daemon is currently stopped on the host (`failed to connect to the docker API`). Container digest remains an open technical blocker. Dependency lockfile and Dockerfile exist, but container build and image digest remain unexecuted.
- [ ] **1.4. Pre-Registration of Promotion Thresholds & Acceptance Rules** `[IMPLEMENTED]` *[Awaiting Governance Board Authorization]*
  - Governance Rule: Numerical promotion thresholds (PR-AUC minimum delta, capacity recall floors, Brier/ECE tolerances) and acceptance rules MUST be formally registered and signed off before holdout evaluation.
  - The evaluator and developer must NEVER invent thresholds after holdout unblinding.

---

### Phase II: Authorized Holdout Generation & Key Escrow

- [ ] **2.1. Holdout Generation Authorization** `[LOCALLY_TESTED]` *[Awaiting Governance Board]*
  - Requires signed cryptographic authorization token matching authorized token SHA-256 hash.
  - Token must include unexpired `expires_at_utc` timestamp.
  - Notice: Generation of Synthetic Dataset v2.2 is strictly prohibited until formal sign-off.
- [ ] **2.2. CSPRNG Seed Escrow** `[LOCALLY_TESTED]` *[Evaluator Custody]*
  - Seed generated using a cryptographically secure pseudorandom number generator (CSPRNG).
  - Seed is escrowed exclusively by the Independent Evaluator; never shared with developers or committed to Git.
- [ ] **2.3. Temporal Horizon & Return-Label Observation Cutoff** `[IMPLEMENTED]`
  - Temporal Horizon: Holdout orders simulate future period (e.g., 2026-02-01 to 2026-02-28).
  - Strict 21-day maturity cutoff: orders placed on or after `2026-02-08 00:00:00` are right-censored to prevent forward-looking observation leakage.
- [ ] **2.4. Independent Approval of Future Input Manifest** `[IMPLEMENTED]`
  - The evaluator approves the exact future input manifest (`dataset_manifest.json`) *after* authorized dataset generation and *before* prediction commitment and label access.
  - Reusing the historical v2.1 manifest is strictly prohibited.
  - Closed-dataset policy enforced: extra unmanifested files, duplicate primary keys, null primary keys, and path traversal strings are strictly rejected.
- [ ] **2.5. AES-256-GCM Envelope Encryption & Label Escrow** `[LOCALLY_TESTED]`
  - Ground-truth holdout labels encrypted by Evaluator into an authenticated AES-256-GCM envelope (`labels_encrypted.bin`).
  - Envelope format: `nonce (12 bytes) || ciphertext || auth_tag (16 bytes)` (minimum 28 bytes).
  - Nonce generation: fresh 96-bit CSPRNG nonce per encryption.
  - Symmetric 256-bit key remains in evaluator custody. Developer environment receives ONLY unblinded transaction features (`orders.csv`).

---

### Phase III: Prediction Submission & Ten-Artifact Cryptographic Commitment

- [ ] **3.1. Developer Prediction Generation** `[IMPLEMENTED]`
  - Developer runs frozen candidate model on unblinded transaction features.
  - Decryption keys are NEVER injected into the developer environment.
- [ ] **3.2. Canonical Prediction Serialization & Hashing** `[LOCALLY_TESTED]`
  - Format Specification:
    - Array Shape: Strictly 1D numpy array (`ndim == 1`, length $N$).
    - Precision & Endianness: IEEE 754 64-bit float, little-endian: `np.dtype('<f8')`.
    - Memory Layout: Contiguous byte stream: `np.ascontiguousarray(predictions, dtype='<f8').tobytes()`.
    - Bounds & Finiteness: All values strictly in $[0.0, 1.0]$; zero NaNs, Infinities, or negative values.
    - Row Ordering: Exactly matches holdout `order_id` sequence.
  - Prediction Digest: `prediction_sha256 = sha256(canonical_bytes).hexdigest()`.
- [ ] **3.3. Ten-Artifact Cryptographic Commitment Publication** `[LOCALLY_TESTED]`
  - Developer creates `prediction_commitment.json` binding all ten execution-identity fields via cryptographic SHA-256 hashes:
    1. `prediction_sha256`: SHA-256 of canonical `<f8` little-endian prediction byte stream.
    2. `bound_model_sha256`: SHA-256 of frozen candidate model artifact.
    3. `bound_calibrator_sha256`: SHA-256 of frozen calibrator artifact.
    4. `bound_source_commit`: SHA-1 of baseline Git commit.
    5. `bound_evaluation_code_sha256`: SHA-256 of evaluation gate module.
    6. `bound_feature_schema_hash`: SHA-256 of canonical feature schema.
    7. `bound_input_manifest_sha256`: SHA-256 of evaluator-approved future input manifest.
    8. `bound_environment_digest`: Immutable OCI container image digest.
    9. `bound_evaluation_policy_sha256`: SHA-256 of approved evaluation policy.
    10. `bound_run_id`: Unique authorized evaluation run identifier.
  - Verification & Signature Architecture Note: The commitment is an integrity hash-binding scheme verified by digest equality against the approved manifest dictionary. No asymmetric digital signature (e.g., Ed25519/ECDSA private key signing) is currently implemented; signer authentication relies on authenticated ledger session transport.
  - Commitment published to the authoritative remote ledger BEFORE label access.

---

### Phase IV: Contract Validation, Remote Ledger Claiming & One-Shot Scoring

- [ ] **4.1. Independent Remote Ledger Claiming & Mutual Exclusion** `[LOCALLY_TESTED]` *[Deployment Pending]*
  - Remote vs Local Distinction: Local ledger (`DurableFileLedger`) is strictly for offline development. Real evaluation requires a deployed, authenticated remote ledger (`RemoteLedgerAdapter`).
  - Bearer token authentication required.
  - Optimistic Concurrency Control (CAS): Server-side monotonic versioning and atomic claim locking.
  - Mutual exclusion: Simultaneous claims by competing processes are rejected with HTTP 409 Conflict.
- [ ] **4.2. Pre-Scoring Contract & Commitment Verification** `[LOCALLY_TESTED]`
  - Evaluator executes `execute_holdout_evaluation_gate()`:
    1. Authorization token verification (expiry, hash).
    2. Environment digest and evaluation policy verification.
    3. Input manifest integrity and dataset file validation (hash match, no extra files, unique non-null primary keys, atomic TOCTOU load-and-verify).
    4. Prediction contract validation (exact row count, exact order ID matching, score bounds $[0, 1]$, zero NaNs).
    5. Complete 10-artifact commitment verification.
  - **Fail-Closed Invariant**: If ANY check fails, execution terminates immediately. A spy function verifies that label scoring is NEVER invoked on precondition failure.
- [ ] **4.3. Atomic Transition to EVALUATING State** `[LOCALLY_TESTED]`
  - Ledger transitions to `EVALUATING` and records `labels_accessed = True` immediately before key injection.
- [ ] **4.4. Crash Safety, Timeout & No-Retry Rule** `[LOCALLY_TESTED]`
  - If a network partition, evaluator crash, timeout, or scoring exception occurs AFTER labels may have been accessed:
    - Ledger records terminal `FAILED` state.
    - **NO automatic retry is permitted.** Repeated execution of the run ID is blocked permanently.
    - Re-evaluation requires formal governance invalidation, post-mortem audit, fresh run ID, and fresh holdout generation.
- [ ] **4.5. Ephemeral In-Memory Label Decryption** `[LOCALLY_TESTED]`
  - Evaluator decrypts `labels_encrypted.bin` in ephemeral memory.
  - AES-256-GCM authentication tag is verified in constant time. Any tag mismatch, bit corruption, or wrong key fails closed with `PermissionError`.
  - Runtime Memory Limits Note: While ground-truth arrays are deleted and garbage-collected post-scoring, Python's runtime (pymalloc allocator and object pooling) does not guarantee physical RAM zeroization. True memory sanitization and anti-forensic security rely on ephemeral process exit and container termination.
- [ ] **4.6. Metric Computation Standards** `[LOCALLY_TESTED]`
  - **PR-AUC / Average Precision**: $\text{AP} = \sum_k (R_k - R_{k-1}) P_k$ computed from continuous calibrated probabilities.
  - **Natural Prevalence Baseline**: $\pi = \frac{1}{N} \sum_{i=1}^N y_i$ reported as the mandatory uninformative reference baseline for PR-AUC. Unlike ROC-AUC where random ranking yields $0.50$, random guessing under PR-AUC yields the positive class prevalence $\pi$. An uninformative model does NOT score $0.50$ in Average Precision.
  - **ROC-AUC**: Continuous score discrimination across all thresholds (random baseline = $0.50$).
  - **Brier Score**: $\frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$.
  - **Expected Calibration Error (ECE)**: 10 uniform probability bins in $[0, 1]$. Empty bins contribute 0. Measured on an independent cohort.
  - **Capacity-Constrained Metric**: Recall at review volume $\le \lfloor b \cdot N \rfloor$ for operating points $b \in \{0.01, 0.02, 0.05, 0.10\}$.
    - Strict floor integer rounding enforced.
    - Deterministic tie-breaking: tied probabilities sorted by `order_id` ascending. Strict cap is never breached.
  - **Expected Operational Cost**: $\text{Cost} = \text{FP} \cdot \$10 + \text{FN} \cdot \$150 + \text{TP} \cdot \$10$.

---

### Phase V: Post-Evaluation Audit, Reporting & Promotion Gate

- [ ] **5.1. Strict Aggregate Metrics Release** `[IMPLEMENTED]`
  - Only high-level confusion matrices, summary metric tables, and calibration curves may be published.
  - Instance-level predictions, individual false negative IDs, and raw ground-truth labels are NEVER disclosed to model developers.
- [ ] **5.2. Audit Logging & Non-Repudiation** `[LOCALLY_TESTED]`
  - Ledger marks run state as `COMPLETED`.
  - Tamper-evident audit log records timestamp, evaluator identity, container digest, commitment SHA-256, and evaluation outcome.
- [ ] **5.3. Synthetic Holdout Limitation Disclaimer** `[IMPLEMENTED]`
  - Protocol explicitly states: **Synthetic holdout results do NOT establish real-market generalization.**
  - Satisfying holdout promotion criteria validates algorithmic soundness and synthetic robustness, but does NOT replace production shadow testing or fraud-ring behavior in live market conditions.
- [ ] **5.4. Final Promotion Decision** `[IMPLEMENTED]` *[Awaiting Governance Board Authorization]*
  - Promotion granted ONLY if candidate model satisfies pre-registered criteria without ad-hoc threshold tuning.

---

## 4. Failure & Abort Procedures

If any of the following failure modes occur, the evaluation run is immediately terminated and marked **FAILED / AUDIT REJECTED**:

1. Authorization token missing, invalid, or expired.
2. Incomplete or substituted 10-artifact commitment.
3. Prediction shape, NaN/Inf, bounds, row count, or order ID contract violation.
4. Input manifest digest mismatch, missing files, or closed-dataset violations (extra files, duplicate PKs).
5. TOCTOU file replacement detected during atomic data loading.
6. Container digest mismatch against approved image digest.
7. Remote ledger CAS version conflict or replay attempt of consumed run ID.
8. Evaluator crash, network timeout, or exception during label access (fail-closed, no automatic retry).
9. AES-256-GCM authentication tag verification failure.

---

## 5. Sign-Off & Attestation Block

*This section must be completed and cryptographically signed prior to holdout evaluation.*

| Role | Signatory Name | Organization / Key Fingerprint | Signature Date | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Executive Authorizer** | *Pending Governance Board* | `ED25519:...` | *Pending* | `AWAITING SIGN-OFF` |
| **Independent Evaluator** | *Pending External Auditor* | `ED25519:...` | *Pending* | `AWAITING SIGN-OFF` |
| **MLOps Engineer** | *Pending Deployment Custodian* | `ED25519:...` | *Pending* | `AWAITING DEPLOYMENT` |
| **ML Research Engineer** | Antigravity AI Auditor | `4615a53602d0...` (Git Baseline) | 2026-10-09 | `RECONCILED AUDIT COMPLETE (STAGE 3.5.5)` |
