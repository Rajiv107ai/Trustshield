# TrustShield — Independent Evaluator Readiness & Execution Checklist

**Document Version**: 1.0.0  
**Status**: DRAFT PROTOCOL FOR INDEPENDENT EVALUATOR SIGN-OFF  
**Target Evaluation**: Blind-Holdout Evaluation (Synthetic Dataset v2.2)  
**Notice**: *Completion or possession of this document does NOT constitute authorization to generate or unblind Synthetic Dataset v2.2. Formal authorization requires independent stakeholder sign-off as defined in Section 1.*

---

## 1. Roles, Custody, and Governance Separation

| Role | Designee / Authority | Mandatory Scope & Governance Responsibility |
| :--- | :--- | :--- |
| **Executive Authorizer** | TrustShield Governance Board | Sole authority to grant formal authorization for holdout generation and unblinding. |
| **Independent Evaluator** | External / Independent Auditor | Controls holdout random seed, holds AES-256-GCM label decryption key, verifies container digest, runs one-shot scoring. |
| **MLOps Engineer** | Deployment Custodian | Builds OCI container image, verifies build reproducibility, provides sealed execution environment. |
| **ML Research Engineer** | Model Developer | Submits frozen candidate model artifact, preprocessing pipeline, and prediction commitment hash. **Has zero access to holdout labels or decryption keys.** |

---

## 2. Stage-by-Stage Verification Checklist

### Phase I: Pre-Generation & Environment Sealing
- [ ] **1.1. Candidate Model Verification**:
  - Model Artifact: [`models/stage34/stage34_tabular_model.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_model.joblib)
  - Verified SHA-256: `1310b43c3c8d23a1d4054200a5321f815de113b1a07793bb0d17f508c6cac22a`
  - Calibrator Artifact: [`models/stage34/stage34_tabular_calibrator.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_calibrator.joblib)
  - Verified SHA-256: `152f287e8f4d478285eeac13d380f4312ceb453bed85b3ff1f99d799424148a8`
  - Model Schema: Exactly 10 ordered tabular features. No graph features.
- [ ] **1.2. Git Baseline Lock**:
  - Repository HEAD Commit: `4615a53602d0a4ffe447d4d2fd14bd69a70bf740`
  - Branch: `phase-3-data-generalization`
- [ ] **1.3. OCI Container Build & Digest Pinning**:
  - Evaluator Dockerfile: [`docker/Dockerfile.evaluator`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/docker/Dockerfile.evaluator)
  - Lockfile: [`docker/requirements-evaluator.lock`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/docker/requirements-evaluator.lock) (scikit-learn pinned to 1.9.0)
  - Extracted Immutable OCI Digest: Recorded as `docker-image@sha256:...` (cannot be mutable `:latest` tag).
  - Smoke Test Passed: Model and calibrator load inside container with 0 unpickling warnings.

### Phase II: Authorized Holdout Generation & Key Escrow
- [ ] **2.1. Generation Authorization Token**:
  - Signed authorization token verified by Independent Evaluator.
- [ ] **2.2. Seed Escrow**:
  - Secure random seed (entropy source: cryptographically secure CSPRNG) generated and escrowed exclusively by Independent Evaluator.
  - Seed is NEVER shared with developers or committed to source control.
- [ ] **2.3. Return-Label Observation Cutoff Verification**:
  - Simulation end defined (e.g., `2026-02-28 23:59:59`).
  - Strict 21-day maturity cutoff enforced: orders placed on or after `2026-02-08 00:00:00` are right-censored.
- [ ] **2.4. Label Encryption & Custody**:
  - Evaluator encrypts ground-truth labels using AES-256-GCM (`labels_encrypted.bin`).
  - Symmetric 256-bit key remains in Evaluator custody.
  - Public unblinded transaction features (`holdout_orders_unblinded.csv`) provided to Model Developer.

### Phase III: Prediction Submission & Commit-Reveal Binding
- [ ] **3.1. Developer Prediction Generation**:
  - Developer runs frozen Tabular-Only model on `holdout_orders_unblinded.csv` inside sealed container.
  - Generates `holdout_predictions.npy`.
- [ ] **3.2. Cryptographic Prediction Commitment**:
  - Developer computes `sha256(holdout_predictions.npy)` and generates `prediction_commitment.json` using [`trustshield_project/evaluation_gate.py`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/trustshield_project/evaluation_gate.py).
  - Commitment binds:
    - `prediction_sha256`
    - `bound_model_sha256`: `1310b43c3c8d...`
    - `bound_source_commit`: `4615a53602d0...`
    - `bound_environment_digest`: `sha256:...`
  - Commitment published to tamper-evident audit ledger BEFORE key access.

### Phase IV: Contract Validation & One-Shot Evaluation
- [ ] **4.1. Pre-Scoring Contract Verification**:
  - `validate_prediction_contract()` executed by Evaluator:
    - Length matches holdout order count exactly ($N$).
    - Zero duplicate order IDs.
    - Exact 1-to-1 matching order ID sequence.
    - Zero NaNs, Infs, or scores outside $[0.0, 1.0]$.
  - *Action on Failure*: Abort run immediately. No partial scoring.
- [ ] **4.2. Single-Shot Ledger Enforcement**:
  - `enforce_single_shot_evaluation()` checks run ID against persistent execution ledger.
  - If run ID was previously executed, execution fails closed with `RuntimeError`.
- [ ] **4.3. One-Shot Decryption & Evaluation**:
  - Evaluator injects decryption key in container memory.
  - Labels decrypted in ephemeral memory and discarded immediately after metric computation.
  - Metrics computed using approved implementations:
    - Primary Ranking: PR-AUC (Average Precision) + Positive Fraud Prevalence baseline.
    - Secondary Ranking: ROC-AUC.
    - Calibration: Brier Score + ECE (10 uniform bins, independent cohort).
    - Operational Decision: Expected loss under $\text{Cost} = \text{FP} \cdot \$10 + \text{FN} \cdot \$150 + \text{TP} \cdot \$10$.
    - Capacity Compliance: Review volume $\le \lfloor bN \rfloor$ for $b \in \{0.01, 0.02, 0.05, 0.10\}$.

### Phase V: Post-Evaluation Audit & Reporting
- [ ] **5.1. Aggregate Metrics Release Only**:
  - Only high-level aggregate summary tables and confusion matrices are released.
  - Zero row-level ground-truth labels, residual errors, or instance-level false negative IDs may be logged or disclosed to developers.
- [ ] **5.2. Audit Logging**:
  - Security audit log records: Evaluator identity, timestamp, container digest, prediction hash, and validation outcomes.
- [ ] **5.3. Model Promotion Gate**:
  - Candidate model is accepted for deployment ONLY if holdout PR-AUC and capacity-constrained recall meet pre-declared criteria.
  - Re-tuning thresholds or models on holdout results is strictly banned.

---

## 3. Failure & Abort Procedures

If any of the following occur, the evaluation run is immediately terminated and marked **FAILED / AUDIT REJECTED**:
1. Prediction row count does not match holdout order count.
2. Missing or duplicate order IDs.
3. Prediction SHA-256 does not match the pre-committed hash.
4. Container digest does not match the frozen image digest.
5. Missing or invalid authorization token.
6. Repeated evaluation attempt detected.
