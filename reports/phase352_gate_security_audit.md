# TrustShield Stage 3.5.2 — Container Build Verification & Evaluation-Gate Security Audit Report

**Audit Date**: October 9, 2026  
**Auditor Roles**: Senior MLOps Security Engineer, ML Reproducibility Auditor, Cryptographic Protocol Reviewer, Python Test Engineer  
**Active Git Branch**: `phase-3-data-generalization`  
**Verified Baseline Commit**: `4615a53602d0a4ffe447d4d2fd14bd69a70bf740` (Active HEAD)  
**Readiness Status**: **CONDITIONALLY READY** (Evaluation Gates Enforced in Code; Container Digest Unverified due to Inactive Docker Engine)  

---

## 1. Executive Summary & Baseline State

TrustShield Stage 3.5.2 executes a rigorous, code-level security audit and adversarial verification of the evaluation gates and execution environment.

### Safeguards Enforced
- **Synthetic Dataset v2.2 was not generated**.
- **No future holdout labels were created, decrypted, inspected, or unblinded**.
- **No models or calibrators were refitted or modified**.
- **Production APIs and transaction decision engines remain untouched**.
- **All uncommitted user work in the repository was preserved**.
- **Zero fabricated container digests or build results were recorded**.

### Repository Baseline
- **Branch**: `phase-3-data-generalization`
- **HEAD Commit**: `4615a53602d0a4ffe447d4d2fd14bd69a70bf740`
- **Working Tree State**: Controlled audit state. Uncommitted modifications in `backend/`, `frontend/`, `docs/`, and `README.md` were preserved without mutation.

---

## 2. Workstream A — Container Build Verification & Local Smoke Test

### 1. Docker Build Execution
Attempting to compile the evaluator container image via the existing [`docker/Dockerfile.evaluator`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/docker/Dockerfile.evaluator):
```powershell
docker build -f docker/Dockerfile.evaluator -t trustshield-evaluator:candidate .
```
- **CLI Detected**: `Docker version 29.8.2, build 7fc2dff`
- **Exit Code**: `1`
- **Actual Error Output**:
  ```text
  ERROR: failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; 
  check if the path is correct and if the daemon is running: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
  ```
- **Audit Assessment**: The Docker Desktop daemon engine is inactive on this Windows host. An OCI image digest cannot be compiled on this machine during this turn.
- *Safeguard Applied*: **No image digest was fabricated.** In accordance with protocol, the image status is documented as `BLOCKED_DAEMON_INACTIVE_ON_HOST`.

### 2. Local Deterministic Inference Smoke Test
To verify model reproducibility on permitted non-holdout inputs, the candidate model ([`stage34_tabular_model.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_model.joblib)) and calibrator ([`stage34_tabular_calibrator.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_calibrator.joblib)) were executed across **5 repeated runs** on a fixed synthetic fixture:
- **Comparison Method**: Bitwise array equality (`np.array_equal(run_0, run_i)`).
- **Maximum Absolute Difference**: Exactly `0.0`.
- **Outcome**: All 5 runs yielded **bitwise identical** outputs (`[0.06252812, 0.03658537, 0.03658537, 0.06252812, 0.125]`).

---

## 3. Workstream B — Prediction Commitment Adversarial Audit

In Stage 3.5.2, [`trustshield_project/evaluation_gate.py`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/trustshield_project/evaluation_gate.py) was enhanced to bind the prediction array to **9 distinct cryptographic artifacts**:
1. `prediction_sha256` (SHA-256 of raw float64 prediction bytes)
2. `bound_model_sha256` ([`stage34_tabular_model.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_model.joblib))
3. `bound_calibrator_sha256` ([`stage34_tabular_calibrator.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_calibrator.joblib))
4. `bound_source_commit` (`4615a53602d0a4ffe447d4d2fd14bd69a70bf740`)
5. `bound_evaluation_code_sha256`
6. `bound_feature_schema_hash`
7. `bound_input_manifest_sha256` ([`dataset_manifest.json`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/dataset_manifest.json))
8. `bound_environment_digest`
9. `bound_evaluation_policy_sha256`

An executable verification function, `verify_prediction_commitment(commitment, approved_manifest, predictions=None)`, was implemented and subjected to adversarial tests:
- **Tampered Predictions Array**: Detected and rejected even when a single prediction score was modified by `1e-12` (`ValueError: Submitted predictions hash does not match committed digest`).
- **Substituted Model Hash**: Fails closed (`ValueError: bound_model_sha256 mismatch`).
- **Substituted Source Commit**: Fails closed (`ValueError: bound_source_commit mismatch`).
- **Substituted Environment Digest**: Fails closed (`ValueError: bound_environment_digest mismatch`).
- **Substituted Input Manifest**: Fails closed (`ValueError: bound_input_manifest_sha256 mismatch`).
- **Missing Required Fields**: Fails closed (`ValueError: Missing mandatory field`).
- **Malformed Commitment Type**: Fails closed on non-dict payload.

> [!WARNING]
> **Cryptographic Scope Limitation**: A SHA-256 prediction commitment proves that submitted predictions match a committed hash. It does **not** prove that the developer did not inspect labels through side channels or that predictions were generated exclusively by approved code. Strong security requires independent key custody and single-shot evaluation.

---

## 4. Workstream C — Single-Shot Enforcement Audit

The persistence mechanism of `enforce_single_shot_evaluation()` was audited and hardened against failure modes:

| Test Scenario | Evaluated Behavior | Implementation Status |
| :--- | :--- | :---: |
| **First Permitted Run** | Run ID appended to persistent JSON ledger | **VERIFIED** |
| **Repeated Run ID** | Raises `RuntimeError: Repeated holdout evaluation is strictly prohibited` | **VERIFIED** |
| **Corrupted Ledger Content** | Non-list or invalid JSON fails closed with `RuntimeError` (never silently overwrites) | **VERIFIED** |
| **Crash Safety** | Writes to `.ledger_tmp_{pid}_{time}.json` then executes atomic `os.replace()` | **VERIFIED** |
| **Concurrency Protection** | Directory spinlock (`ledger.json.lock`) mutually excludes simultaneous threads | **VERIFIED** |
| **Tamper Resistance** | Local JSON file can be edited/deleted by local machine administrator | **PARTIAL (Host-Bound)** |
| **Cross-Machine Replay** | Copying code to a fresh machine or container resets the local ledger | **LIMITATION** |

### Security Assurance Classification
1. **Duplicate Detection**: **VERIFIED** (Reliably blocks repeated run IDs within the same ledger environment).
2. **Crash-Safe State Transitions**: **VERIFIED** (Atomic temporary file replacement prevents truncated or corrupted state).
3. **Concurrency Safety**: **VERIFIED** (Directory spinlock guarantees mutual exclusion).
4. **Tamper Evidence**: **PARTIAL** (Protected within standard user permissions; vulnerable to root/administrator deletion).
5. **Cross-Machine Replay Resistance**: **MISSING / HOST-BOUND** (Requires evaluator-controlled external state or remote ledger).
6. **Independent Auditability**: **PARTIAL** (Audit log is local JSONL; requires signed timestamping for external verifiability).

---

## 5. Workstream D — Authorization Enforcement Audit

The authorization gate [`trustshield_project/evaluation_gate.py::verify_holdout_authorization`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/trustshield_project/evaluation_gate.py) was audited across distinct assurance levels:

### Assurance Level Analysis
1. **Unit Test Level**: **VERIFIED**. Missing tokens, empty strings, invalid hashes, and expired tokens strictly raise `PermissionError` fail-closed.
2. **Integration Test Level**: **PARTIALLY VERIFIED**. `evaluation_gate.py` enforces authorization strictly before prediction verification, ensuring no label access can occur without valid credentials.
3. **Operational Control Level**: **SPECIFIED IN CHECKLIST**. Independent Evaluator holds AES-256-GCM label decryption keys in escrow away from developer access.
4. **Generation Entry Point Level**: **PROTECTED BY CODE ABSENCE**. The Synthetic Dataset v2.2 generator does not exist in the repository. Unauthorized generation is prevented by the absence of execution code.

---

## 6. Workstream E — Log Confidentiality & Audit Trail

Inspection of `_record_audit_event()` and evaluation logging confirms:
- **No Ground-Truth Labels Logged**: Zero row-level labels or fraud identities are printed.
- **No Individual Error IDs**: False Positive or False Negative order IDs are suppressed.
- **No Raw Secrets Logged**: Full authorization tokens are never logged. Only an 8-character SHA-256 digest prefix is logged for security audits.
- **Audit Format**: Structured JSONL with UTC ISO 8601 timestamps and failure reasons (`MISSING_TOKEN`, `INVALID_TOKEN_HASH`, `EXPIRED_TOKEN`).
- **Limitation**: Local JSONL audit logs can be modified by host administrators without cryptographic proof. Remote logging or signed ledgers are recommended for production evaluations.

---

## 7. Workstream F — Capacity Policy & Contract Consistency

The capacity policy was audited across edge cases:
- **Formula**: $C = \lfloor bN \rfloor$ strictly enforced across all budgets ($b \in \{0.01, 0.02, 0.05, 0.10\}$).
- **Small Datasets**: For $N = 10$, $5\%$ review budget yields $C = \lfloor 0.5 \rfloor = 0$ orders (no review permitted). For $N = 20$, $C = 1$.
- **Zero Capacity**: $b = 0 \implies C = 0$.
- **Deterministic Tie-Breaking**: When multiple orders share identical probability scores at the threshold boundary, orders are sorted deterministically by `(score DESC, order_date ASC, order_id ASC)` to strictly obey budget $C$ without arbitrary selection.
- **Policy Separation Reconfirmed**:
  - Unconstrained theoretical minimum: Score step 11 ($u = 0.067633$, cost **$\$101,090.00$**, volume $43.03\%$).
  - Actionable capacity operating point: $5\%$ budget $\implies t = 0.15$ ($477$ orders flagged, $3.93\%$ volume, catching $99$ frauds).

---

## 8. Regression Test Execution & Pass Counts

All cumulative Phase 3 regression tests were executed in a single command:
```powershell
.\.venv\Scripts\pytest.exe trustshield_project/test_stage32_audit.py `
                          trustshield_project/test_stage32_final_gate.py `
                          trustshield_project/test_stage33_evaluation.py `
                          trustshield_project/test_stage331_audit.py `
                          trustshield_project/test_stage34_graph_ablation.py `
                          trustshield_project/test_stage341_policy_audit.py `
                          trustshield_project/test_stage342_audit_reconciliation.py `
                          trustshield_project/test_stage35_temporal_holdout_readiness.py `
                          trustshield_project/test_stage351_environment_sealing.py `
                          trustshield_project/test_stage352_gate_security.py
```

### Actual Execution Outcome
```text
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\rajiv_pis9z8x\Downloads\trustshield_full_handoff
configfile: pytest.ini
plugins: anyio-4.15.1
collected 126 items

trustshield_project\test_stage32_audit.py ..............                 [ 11%]
trustshield_project\test_stage32_final_gate.py ........                  [ 17%]
trustshield_project\test_stage33_evaluation.py .........                 [ 24%]
trustshield_project\test_stage331_audit.py ......                        [ 29%]
trustshield_project\test_stage34_graph_ablation.py ..............        [ 40%]
trustshield_project\test_stage341_policy_audit.py ..........             [ 48%]
trustshield_project\test_stage342_audit_reconciliation.py .............  [ 58%]
trustshield_project\test_stage35_temporal_holdout_readiness.py ......... [ 65%]
.........                                                                [ 73%]
trustshield_project\test_stage351_environment_sealing.py ..............  [ 84%]
trustshield_project\test_stage352_gate_security.py ....................  [100%]

======================= 126 passed, 2 warnings in 3.88s =======================
```
- **Exit Code**: `0`
- **Total Tests Executed**: **126**
- **Passed**: **126** (100%)
- **Failed**: **0**
- **Warnings**: `2` (the previously documented scikit-learn unpickle version warning in test_stage33; remediated in `docker/requirements-evaluator.lock`).

---

## 9. Protected Artifact Verification

| Protected Artifact | Verified SHA-256 | Status |
| :--- | :--- | :---: |
| [`scripts/generate_realistic_synthetic_data_v2_1.py`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/scripts/generate_realistic_synthetic_data_v2_1.py) | `0098dad245d724997560c5a5bc4bf70c1bcb98057a4b32d42f85560a4e08bd55` | **UNMUTATED** |
| [`data/synthetic_v2_1/dataset_manifest.json`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/dataset_manifest.json) | `b0aba240f1439c1d83c1c9773977334ddf39321feed0e95f53ef30b1c138e74b` | **UNMUTATED** |
| [`data/synthetic_v2_1/orders.csv`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/orders.csv) | `325597a6e0c2afa533c01d26895457cba166cd0bc2c0f9ce4f12c43355741b8f` | **UNMUTATED** |
| [`models/stage34/stage34_tabular_model.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_model.joblib) | `1310b43c3c8d23a1d4054200a5321f815de113b1a07793bb0d17f508c6cac22a` | **UNMUTATED** |
| [`models/stage34/stage34_tabular_calibrator.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_calibrator.joblib) | `152f287e8f4d478285eeac13d380f4312ceb453bed85b3ff1f99d799424148a8` | **UNMUTATED** |
| `Synthetic Dataset v2.2` | Generator script and dataset directory do not exist | **UNGENERATED & LOCKED** |

---

## 10. Remaining Blockers & Critical Vulnerabilities

| Blocker ID | Description | Severity | Responsible Party | Required Remediation |
| :---: | :--- | :---: | :--- | :--- |
| **BLK-01** | Docker Daemon Inactive | **Blocking for Unblinding** | MLOps Security Engineer | Start Docker daemon on build host, compile [`docker/Dockerfile.evaluator`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/docker/Dockerfile.evaluator), and extract the immutable `@sha256:...` digest. |
| **BLK-02** | Local Ledger Replay Vulnerability | **Security Limitation** | MLOps Security Engineer & Evaluator | Local file ledger can be reset if evaluation runs in ephemeral/fresh containers without persistent volume binding. Ledger must be escrowed on Evaluator's persistent host. |
| **BLK-03** | Holdout Generation Authorization | **Blocking for Generation** | Governance Board & Evaluator | Formal sign-off on the Evaluator Checklist, CSPRNG seed generation, and generation of v2.2. |

---

## 11. Explicit Readiness Decision

### **CONDITIONALLY READY**

The evaluation gate, 9-artifact prediction commitment, crash-safe atomic ledger, and deterministic capacity constraints are **implemented and verified in executable code**.

However, the evaluation environment **cannot be marked fully sealed** until the Docker daemon builds the container image and extracts the immutable OCI digest on an active build host.

*Synthetic Dataset v2.2 remains ungenerated, unblinded, and locked.*
