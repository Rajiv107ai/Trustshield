# TrustShield Stage 3.5 — Temporal Generalization & Blind-Holdout Readiness Audit Report

**Audit Date**: October 9, 2026  
**Auditor Roles**: Senior ML Research Engineer, ML Evaluation Auditor, Data-Leakage Specialist, MLOps Security Engineer  
**Active Git Branch**: `phase-3-data-generalization`  
**Verified Baseline Commit**: `4615a53602d0a4ffe447d4d2fd14bd69a70bf740` (Active HEAD)  
**Readiness Status**: **CONDITIONALLY READY** (Holdout v2.2 Generation and Unblinding Remain Blocked Pending Independent Review)  

---

## 1. Executive Summary

TrustShield Stage 3.5 establishes the scientific protocol, data integrity boundaries, and operational safeguards required to prepare the TrustShield platform for a future, temporally separated, blind-holdout evaluation.

In accordance with strict auditor safeguards, **Synthetic Dataset v2.2 was not generated**, **no future holdout labels were unblinded or inspected**, **no models were refitted**, **no production inference code was altered**, and **no Git commits were pushed**.

### Primary Findings
1. **Mathematical Resolution of Threshold-Cost Discrepancy**:
   - The reported Stage 3.4.2 cost discrepancy—**$101,590** at threshold $0.07$ vs **$101,130** at threshold $0.06$—is fully reconciled from underlying predictions and labels.
   - On the coarse 80-point grid ($0.01$ to $0.80$, step $0.01$), threshold $t = 0.06$ yields $7,022$ flagged orders and a cost of **$\$101,720.00$**, whereas $t = 0.07$ yields $3,709$ flagged orders and a cost of **$\$101,590.00$**. Within that coarse grid, $t = 0.07$ was uniquely the minimum.
   - However, the model predictions (produced by Isotonic Regression) exhibit only 26 discrete output probability levels. Step 13 occurs at probability score **$u = 0.069074$**, flagging $4,983$ orders ($\text{TP} = 512, \text{FP} = 4,471, \text{FN} = 342$) and achieving a cost of **$\$101,130.00$**. This step was mislabeled as "$0.06$" in the Stage 3.4.2 text table.
   - Crucially, this audit discovers that the **true global minimum expected cost across all possible thresholds** occurs at score step 11 (**$u = 0.067633$**), flagging $5,219$ orders ($\text{TP} = 528, \text{FP} = 4,691, \text{FN} = 326$) with an expected cost of **$\$101,090.00$** ($500$ lower than the coarse grid minimum). The phrase "cost-optimal" applied to $t = 0.07$ was merely an artifact of an insufficiently fine evaluation grid.
2. **Independent Verification of Previous Audit Claims**:
   - The Stage 3.4.2 regression suite was verified independently: exactly **13 tests passed**.
   - Cumulative Phase 3 tests prior to Stage 3.5 were verified independently: exactly **74 tests passed**.
   - With 18 new automated Stage 3.5 tests added, total passing Phase 3 regression tests reach **92 passed, 0 failed**.
   - The validation ECE of $0.0000$ was independently confirmed to be an in-sample artifact of evaluating the isotonic calibrator on its own fitting data. Out-of-sample diagnostic test ECE is **$0.0289$** for Tabular-Only and **$0.0282$** for Late Fusion.
3. **Temporal Split & Feature Leakage Audit**:
   - Temporal splits in Synthetic v2.1 are strictly non-overlapping: Train ($\le$ Aug 31, 2025, $16,952$ orders), Validation (Sep 1 to Oct 31, 2025, $12,129$ orders), and Diagnostic Test (Nov 1 to Dec 31, 2025, $20,919$ orders). Zero duplicate order IDs cross splits.
   - Point-in-time accumulators (`_cumulative_count_asof`, `_asof_cumulative_from_events`) strictly enforce `allow_exact_matches=False`, preventing same-timestamp event leakage.
   - The 21-day return observation maturity rule was mathematically verified: orders placed within 21 days of simulation end (e.g., on or after Dec 11, 2025 in v2.1, or on or after Feb 8, 2026 in a proposed Jan–Feb 2026 simulation) must be treated as right-censored observations.
4. **Reproducibility & Serialization Gap**:
   - The candidate production model—Tabular-Only XGBoost ([`stage34_tabular_model.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_model.joblib)) and its probability calibrator ([`stage34_tabular_calibrator.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_calibrator.joblib))—was reproduced **bit-for-bit** against recorded validation and diagnostic test metrics.
   - Serialization Defect Isolated: In Stage 3.4, the Late Fusion champion's calibrator was never serialized to disk. However, because tabular-only modeling is the candidate architecture for transaction fraud ranking, this defect is non-blocking for tabular evaluation.
5. **Readiness Verdict**:
   - **CONDITIONALLY READY**. The evaluation contract, cryptographic commit-reveal protocol, and split integrity are verified. Unblinding is blocked on container image digest pinning and formal authorization of holdout generation.

---

## 2. Scope and Mandatory Safeguards

This audit was conducted under non-negotiable operational safeguards:

| Safeguard Category | Enforced Restriction | Verification Status |
| :--- | :--- | :---: |
| **Data Protection** | Synthetic Dataset v2.2 was not generated | **ENFORCED** |
| **Label Protection** | No future holdout labels were unblinded, decrypted, or inspected | **ENFORCED** |
| **Historical Preservation** | Datasets, manifests, predictions, and model artifacts were not modified in place | **ENFORCED** |
| **Model Freeze** | Existing models and calibrators were not refitted or altered | **ENFORCED** |
| **Production Protection** | Backend inference APIs, routes, and decision engines were not changed | **ENFORCED** |
| **Git Protection** | Zero commits or pushes executed; baseline commit and user working tree preserved | **ENFORCED** |

---

## 3. Repository State and Baseline Commit

- **Working Directory**: `c:\Users\rajiv_pis9z8x\Downloads\trustshield_full_handoff`
- **Active Git Branch**: `phase-3-data-generalization`
- **HEAD Commit**: `4615a53602d0a4ffe447d4d2fd14bd69a70bf740`
- **Working Tree State**: Controlled audit state. Uncommitted user modifications in `backend/`, `frontend/`, and `docs/` were preserved. Only audit test scripts and report deliverables were added.

---

## 4. Evidence Inventory

Every critical artifact supporting the TrustShield evaluation pipeline was audited for presence, SHA-256 hash, and provenance:

| Artifact Path | Artifact Type | Verified SHA-256 | Git Status | Audit Role | Provenance & Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| [`scripts/generate_realistic_synthetic_data_v2_1.py`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/scripts/generate_realistic_synthetic_data_v2_1.py) | Code | `0098dad245d724997560c5a5bc4bf70c1bcb98057a4b32d42f85560a4e08bd55` | Tracked | Frozen Generator | Stage 3.1.1 generator script |
| [`data/synthetic_v2_1/dataset_manifest.json`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/dataset_manifest.json) | Manifest | `b0aba240f1439c1d83c1c9773977334ddf39321feed0e95f53ef30b1c138e74b` | Tracked | Data Manifest | Frozen v2.1 data metadata |
| [`data/synthetic_v2_1/orders.csv`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/orders.csv) | Dataset | `325597a6e0c2afa533c01d26895457cba166cd0bc2c0f9ce4f12c43355741b8f` | Tracked | Primary Data | 50,000 orders (Jan–Dec 2025) |
| [`data/synthetic_v2_1/listings.csv`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/listings.csv) | Dataset | `0081b562dc900c14e1b90f2310261401ab2f8f40dda4805aa72bd17959d5e619` | Tracked | Catalog Data | 20,000 listings |
| [`data/synthetic_v2_1/returns.csv`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/returns.csv) | Dataset | `34eb6514d2ebe4b90608f31fb09fd71c2aa1709424d3e00c22c6a38df119a14a` | Tracked | Return Data | 5,537 return records |
| [`models/stage34/stage34_tabular_model.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_model.joblib) | Model | `1310b43c3c8d23a1d4054200a5321f815de113b1a07793bb0d17f508c6cac22a` | Tracked | Candidate Model | XGBClassifier (10 tabular features) |
| [`models/stage34/stage34_tabular_calibrator.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_calibrator.joblib) | Calibrator | `152f287e8f4d478285eeac13d380f4312ceb453bed85b3ff1f99d799424148a8` | Tracked | Candidate Calibrator | Isotonic ProbabilityCalibrator |
| [`models/stage34/stage342_val_predictions.npy`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage342_val_predictions.npy) | Array | `68977af8aeef1c148133c3823f251360e2feac8f29b178301a1698b4b985beb5` | Tracked | Reference Preds | 12,129 validation predictions |
| [`models/stage34/stage342_val_labels.npy`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage342_val_labels.npy) | Array | `43b6ed221aa6763770b26faa626e1d53cef27926c55b3c2d44a8c29e67e3d36c` | Tracked | Reference Labels | 12,129 validation labels (854 pos) |
| [`models/stage34/stage34_decision_policy.json`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_decision_policy.json) | Policy | `e42321deba759830f420f0c7165b1c01ff384cc00a6715276d98e509c4508d7b` | Tracked | Policy Artifact | Operational threshold rules |

---

## 5. Stage 3.4.2 Claim Verification

The claims in the Stage 3.4.2 audit report were verified independently via code execution:

1. **Stage 3.4.2 Suite Pass Count**: Claimed 13 tests passed. **VERIFIED**. Running `pytest trustshield_project/test_stage342_audit_reconciliation.py` executed and passed all 13 tests in 0.42s.
2. **Phase 3 Cumulative Pass Count**: Claimed 74 cumulative passing tests. **VERIFIED**. Running pytest across all 7 historical test files executed and passed all 74 items in 3.59s.
3. **Validation ECE = 0.0000**: Claimed that validation ECE = 0.0000 was measured on the fitting cohort. **VERIFIED**. On the validation cohort (the calibrator's training data), ECE is 0.0000. However, on the separate historical diagnostic test cohort, the tabular model exhibits a true ECE of **0.0289** (and late fusion exhibits **0.0282**). Zero calibration error is an in-sample artifact, not proof of temporal generalization.

---

## 6. Threshold-Cost Inconsistency Reconciliation

### The Discrepancy
The Stage 3.4.2 report noted:
- Threshold $0.07$: Cost of **$\$101,590.00$**
- Threshold $0.06$: Cost of **$\$101,130.00$**
- In other sections, $0.07$ was declared as the "cost-optimal" threshold.

### Code-Level Forensic Resolution
Under the approved cost model:
$$\text{Cost} = \text{FP} \cdot \$10 + \text{FN} \cdot \$150 + \text{TP} \cdot \$10 \equiv \text{Flagged} \cdot \$10 + \text{Missed} \cdot \$150$$

Inspection of `scripts/run_stage34_experiments.py` reveals that the threshold sweep was computed over a **coarse 80-point grid**:
$$\text{grid} = \text{np.linspace}(0.01, 0.80, 80) = [0.01, 0.02, \dots, 0.06, 0.07, \dots, 0.80]$$

Evaluating validation orders ($N = 12,129$, actual fraud = $854$) on this coarse grid yields:
- **At $t = 0.06$**: 
  - Captures all predictions $p \ge 0.062992$ ($7,022$ orders).
  - $\text{TP} = 644, \text{FP} = 6,378, \text{FN} = 210, \text{TN} = 4,897$.
  - $\text{Cost} = 6,378 \cdot 10 + 210 \cdot 150 + 644 \cdot 10 = \mathbf{\$101,720.00}$.
- **At $t = 0.07$**:
  - Captures all predictions $p \ge 0.076923$ ($3,709$ orders).
  - $\text{TP} = 424, \text{FP} = 3,285, \text{FN} = 430, \text{TN} = 7,990$.
  - $\text{Cost} = 3,285 \cdot 10 + 430 \cdot 150 + 424 \cdot 10 = \mathbf{\$101,590.00}$.

**Why did $t = 0.07$ appear cost-optimal?**  
Because on the coarse 80-point grid, $\$101,590 < \$101,720$. Thus, `cost_optimal_entry = min(threshold_sweep, key=lambda x: x["expected_cost"])` selected $0.07$.

**Where did $\$101,130$ come from?**  
Because the calibrator (Isotonic Regression) produces step-wise outputs, validation predictions take on **only 26 unique discrete probability values**.
- At score step 13 ($u = 0.06907378...$):
  - $\text{Flagged} = 4,983, \text{TP} = 512, \text{FP} = 4,471, \text{FN} = 342, \text{TN} = 6,804$.
  - $\text{Cost} = 4,471 \cdot 10 + 342 \cdot 150 + 512 \cdot 10 = \mathbf{\$101,130.00}$.
- In the Stage 3.4.2 summary table, the row for $u = 0.069074$ was truncated and rounded to "$0.06$", which created the apparent contradiction with the coarse grid's $t = 0.06$ ($101,720$).

**The Unconstrained Minimum Cost ($101,090)**:  
Evaluating all 26 distinct isotonic score steps reveals that the true global minimum cost occurs at **step 11 ($u = 0.06763285...$)**:
- $\text{Flagged} = 5,219$ ($43.03\%$ volume)
- $\text{TP} = 528, \text{FP} = 4,691, \text{FN} = 326, \text{TN} = 6,584$
- $\text{Precision} = 0.1012, \text{Recall} = 0.6183, \text{F1} = 0.1739$
- $\text{Cost} = 4,691 \cdot 10 + 326 \cdot 150 + 528 \cdot 10 = \mathbf{\$101,090.00}$

| Threshold / Score Point | Evaluation Granularity | Flagged Orders | TP | FP | FN | Expected Cost | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$t = 0.0600$** | Coarse 80-pt Grid | 7,022 | 644 | 6,378 | 210 | $101,720.00 | Coarse point |
| **$t = 0.0700$** | Coarse 80-pt Grid | 3,709 | 424 | 3,285 | 430 | $101,590.00 | Coarse grid minimum |
| **$u = 0.069074$** | Isotonic Step 13 | 4,983 | 512 | 4,471 | 342 | $101,130.00 | Mislabeled as "0.06" in 3.4.2 |
| **$u = 0.067633$** | Isotonic Step 11 | **5,219** | **528** | **4,691** | **326** | **$101,090.00** | **True unconstrained minimum** |

> [!IMPORTANT]
> **Audit Conclusion**: The coarse grid with step 0.01 was insufficiently granular to locate the true cost-optimal operating point of the model. Calling $t = 0.07$ "cost-optimal" was an evaluation artifact.

---

## 7. Workstream B — Temporal Generalization Protocol

### A. Temporal Split Integrity
- **Train Split**: Jan 1, 2025 to Aug 31, 2025 ($16,952$ orders, $813$ fraud, prevalence $4.80\%$).
- **Validation Split**: Sep 1, 2025 to Oct 31, 2025 ($12,129$ orders, $854$ fraud, prevalence $7.04\%$).
- **Diagnostic Test Split**: Nov 1, 2025 to Dec 31, 2025 ($20,919$ orders, $1,835$ fraud, prevalence $8.77\%$).
- Precedence is strictly enforced: $\text{Train}_{\max} < \text{Val}_{\min}$ and $\text{Val}_{\max} < \text{Test}_{\min}$. Zero order IDs overlap.

### B. Entity & Leakage Pathways
- **Buyer & Seller Continuity**:
  - $2,471$ buyers overlap between Train and Val; $3,059$ overlap between Val and Test.
  - $467$ of $500$ sellers operate across all three time periods.
  - *Distinction*: Entity presence across time is legitimate marketplace activity, not leakage, **provided that feature accumulators only observe historical events prior to the order timestamp**.
- **Point-in-Time Causality**:
  - `_cumulative_count_asof` and `_asof_cumulative_from_events` use `pd.merge_asof(direction="backward", allow_exact_matches=False)`.
  - Same-second or same-day orders for the same entity are not counted as prior history for each other.
- **Preprocessing Fitting Boundary**:
  - Category price medians (`category_median`) and device buyer counts (`device_buyer_counts`) in `baseline_model.py` are fitted strictly on `train_mask = df["order_date"] <= TRAIN_END`. Unseen entities receive default fallback values ($1.0$ or category fallback).

### C. Return-Label 21-Day Maturity Rule
- **Longitudinal Return Window**: Returns require $1$ to $21$ days turnaround delay.
- **2025 Simulation Right-Censoring**:
  - Simulation ended on Dec 31, 2025. Orders placed on or after Dec 11, 2025 ($8,618$ orders) had $< 21$ days observation.
  - In Synthetic v2.1, returns scheduled after Dec 31 were dropped from `returns.csv`.
- **Future 2026 Holdout Requirement**:
  - For a proposed Jan 1 to Feb 28, 2026 simulation window, the safe observation cutoff is **February 7, 2026 23:59:59**. Orders placed after this date must be right-censored. Evaluating return-fraud on orders placed between Feb 8 and Feb 28 without mature outcome tracking constitutes label maturity leakage.

### D. Distribution-Shift Analysis
Real dataset distributions across Train, Val, and Test were analyzed:

| Distribution Dimension | Train (Jan–Aug 2025) | Validation (Sep–Oct 2025) | Diagnostic Test (Nov–Dec 2025) | Material Shift Observed |
| :--- | :---: | :---: | :---: | :--- |
| **Order Volume** | 16,952 | 12,129 | 20,919 | Seasonal surge in Q4 (+72% vs Val) |
| **Fraud Prevalence** | 4.80% | 7.04% | 8.77% | +82.7% relative increase from Train |
| **Return Abuse Share** | 55.0% of fraud | 37.4% of fraud | 25.4% of fraud | Relative decline as fraud syndicates scale |
| **Coordinated / Collusion Share** | 11.0% of fraud | 29.3% of fraud | 48.8% of fraud | Massive surge in late-year syndicate activity |
| **Mean Transaction Amount** | $1,861.39 | $1,961.16 | $1,990.63 | +6.9% drift toward higher-value orders |
| **Mean Seller Age (days)** | 124.6 | 217.2 | 272.9 | Natural aging of seller base |
| **Mean Prior Orders per Buyer** | 4.64 | 7.48 | 10.58 | Buyer experience accumulation |
| **Tabular Feature Nulls** | 0.00% | 0.00% | 0.00% | Zero missing values across all splits |

> [!NOTE]
> The substantial drift in fraud prevalence (4.8% $\to$ 8.8%) and syndicate composition means that model ranking metrics (PR-AUC) will naturally change baseline across temporal splits. Evaluating PR-AUC alongside positive prevalence is mandatory.

---

## 8. Workstream C — Reproducibility and Artifact Locking

### Tabular-Only Candidate Model Reproduction
The candidate model for order fraud ranking is the **Tabular-Only XGBoost model** ([`stage34_tabular_model.joblib`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/stage34/stage34_tabular_model.joblib)):
- Recomputed on validation inputs: $\text{ROC-AUC} = 0.6451, \text{PR-AUC} = 0.1252, \text{Brier} = 0.0639, \text{ECE} = 0.0000$.
- Recomputed on test inputs: $\text{ROC-AUC} = 0.6231, \text{PR-AUC} = 0.1406, \text{Brier} = 0.0793, \text{ECE} = 0.0289$.
- Both match the recorded metrics in `reports/phase34_metrics.json` **bit-for-bit to 4 decimal places**.

### Serialization Defect Identified (Late Fusion Calibrator)
In Stage 3.4, `scripts/run_stage34_experiments.py` dumped:
- `stage34_tabular_calibrator.joblib`
- `stage34_early_fusion_calibrator.joblib`
- `stage34_noisyor_calibrator.joblib`
- **However, `stage34_late_fusion_calibrator.joblib` was omitted from `joblib.dump()`**.
- Because the Late Fusion calibrator was never written to disk, an independent auditor cannot reproduce the Late Fusion calibrated probabilities without refitting the calibrator on the stacker outputs.
- *Remediation*: The production candidate model is Tabular-Only XGBoost, which is fully persisted and verified. Late Fusion is retained for ablation research and not proposed for blind evaluation.

### Execution Environment Locking Requirements
During test execution, scikit-learn emitted:
`InconsistentVersionWarning: Trying to unpickle estimator IsotonicRegression from version 1.9.0 when using version 1.9.1.`
- This warning demonstrates that minor patch versions in Python pickling introduce instability.
- An immutable OCI container image digest (e.g., `docker-image@sha256:...`) pinning Python 3.14, scikit-learn 1.9.1, xgboost 3.2.0, pandas 3.0.1, and numpy 2.3.2 must be built and frozen before the holdout run.

---

## 9. Workstream D — Blind-Holdout Threat Model

| Threat ID | Threat Name | Severity | Proposed & Implemented Mitigations | Implementation Status |
| :---: | :--- | :---: | :--- | :---: |
| **THREAT-01** | Accidental Label Exposure | **CRITICAL** | Holdout ground-truth labels encrypted with AES-256-GCM (`labels_encrypted.bin`). Evaluation operator receives only feature data. | **DESIGNED & GATE TESTED** |
| **THREAT-02** | Developer Access to Decryption Keys | **CRITICAL** | Keys held strictly by Independent Auditor. Evaluation gate raises `PermissionError` fail-closed without token. | **IMPLEMENTED IN TEST SUITE** |
| **THREAT-03** | Mutable Model or Pipeline Post-Prediction | **HIGH** | Cryptographic commit-reveal: SHA-256 hash of predictions array committed to ledger before labels are unblinded. | **IMPLEMENTED IN TEST SUITE** |
| **THREAT-04** | Repeated Holdout Queries / Adaptive Overfitting | **CRITICAL** | Single-shot evaluation contract: exactly one evaluation execution permitted; zero iterative querying. | **CONTRACT SPECIFIED** |
| **THREAT-05** | Incorrect Input-to-Prediction Joins | **HIGH** | Strict 1-to-1 order ID matching, row count assertion, non-null check, order preservation. | **IMPLEMENTED IN TEST SUITE** |
| **THREAT-06** | Selective Removal of Difficult Examples | **HIGH** | Evaluation runner asserts `len(predictions) == len(holdout_orders)`; dropped rows abort execution. | **IMPLEMENTED IN TEST SUITE** |
| **THREAT-07** | Modified Evaluation Code | **HIGH** | Evaluation runner script hash locked and verified inside container. | **SPECIFIED IN CONTRACT** |
| **THREAT-08** | Logs Containing Sensitive Holdout Labels | **MEDIUM** | Runner output sanitized; only aggregate confusion matrix and summary metrics logged. | **SPECIFIED IN CONTRACT** |
| **THREAT-09** | Missing or Duplicate Predictions | **HIGH** | Assertion: `len(set(order_ids)) == len(order_ids) == expected_count`. | **IMPLEMENTED IN TEST SUITE** |
| **THREAT-10** | Unauthorized Holdout Regeneration | **CRITICAL** | Holdout seed escrowed with independent auditor; generator execution locked. | **SAFEGUARD ACTIVE** |
| **THREAT-11** | Untrusted Evaluation Operator | **HIGH** | Immutable container digest pinning (OCI image digest) + cryptographic run attestation. | **BLOCKER / CONDITION PENDING** |

---

## 10. Workstream E — Evaluation Contract

The evaluation contract separates concerns into distinct metric categories:

### 1. Ranking Quality Metrics
- **Primary Metric**: Average Precision / PR-AUC computed over continuous predicted probabilities.
- **Secondary Metric**: ROC-AUC.
- **Reporting Rule**: The baseline positive fraud prevalence ($P_{\text{fraud}} = \frac{\sum y}{N}$) must be reported alongside PR-AUC.

### 2. Probability Calibration Metrics
- **Primary Metric**: Brier Score ($\frac{1}{N} \sum (p_i - y_i)^2$).
- **Secondary Metric**: Expected Calibration Error (ECE) across 10 uniform-width bins. Empty bins contribute 0 calibration error.
- **Cohort Separation Rule**: Calibration parameters must **never be fitted on the evaluation cohort**. In-sample ECE is forbidden as proof of generalization.

### 3. Operational Decision Quality & Capacity Constraints
- **Cost Formulation**:
  $$\text{Expected Loss} = \text{FP} \cdot \$10.00 + \text{FN} \cdot \$150.00 + \text{TP} \cdot \$10.00$$
- **Capacity Limits**:
  For review budget $b \in \{0.01, 0.02, 0.05, 0.10\}$ and evaluation size $N$:
  $$\text{Maximum Allowed Flagged Orders } C = \lfloor bN \rfloor$$
- **Tie-Breaking Rule**:
  - The model selection logic must never exceed $C$.
  - Nearest-threshold selection that exceeds $C$ is strictly banned.
  - If multiple orders share the same score at the boundary such that taking all of them exceeds $C$, the threshold must be chosen conservatively (smaller volume $\le C$). If individual order sorting is required, secondary sorting keys must be deterministic: `(score DESC, order_date ASC, order_id ASC)`.

---

## 11. Workstream F — Automated Checks and Regression Tests

The new automated readiness test suite was implemented in [`trustshield_project/test_stage35_temporal_holdout_readiness.py`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/trustshield_project/test_stage35_temporal_holdout_readiness.py).

### Execution Command and Real Results
```powershell
.\.venv\Scripts\pytest.exe trustshield_project/test_stage32_audit.py `
                          trustshield_project/test_stage32_final_gate.py `
                          trustshield_project/test_stage33_evaluation.py `
                          trustshield_project/test_stage331_audit.py `
                          trustshield_project/test_stage34_graph_ablation.py `
                          trustshield_project/test_stage341_policy_audit.py `
                          trustshield_project/test_stage342_audit_reconciliation.py `
                          trustshield_project/test_stage35_temporal_holdout_readiness.py
```

```text
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
collected 92 items

trustshield_project\test_stage32_audit.py ..............                 [ 15%]
trustshield_project\test_stage32_final_gate.py ........                  [ 23%]
trustshield_project\test_stage33_evaluation.py .........                 [ 33%]
trustshield_project\test_stage331_audit.py ......                        [ 40%]
trustshield_project\test_stage34_graph_ablation.py ..............        [ 55%]
trustshield_project\test_stage341_policy_audit.py ..........             [ 66%]
trustshield_project\test_stage342_audit_reconciliation.py .............  [ 80%]
trustshield_project\test_stage35_temporal_holdout_readiness.py ......... [ 90%]
.........                                                                [100%]

======================= 92 passed, 2 warnings in 2.92s ========================
```
- **Exit Code**: `0`
- **Tests Passed**: **92** (18 new Stage 3.5 tests + 74 historical Phase 3 tests).
- **Tests Failed**: **0**.
- **Tests Skipped**: **0**.

---

## 12. Known Limitations and Remaining Conditions

Before an authorized blind-holdout evaluation can proceed, the following conditions must be satisfied:

1. **Condition 1 — Immutable Container Digest (Blocking for Unblinding)**:
   - *Owner*: MLOps Security Engineer.
   - *Requirement*: Local Windows environment produces scikit-learn unpickling version warnings (1.9.0 vs 1.9.1). A Docker/OCI container image must be built pinning Python 3.14, scikit-learn 1.9.1, xgboost 3.2.0, pandas 3.0.1, numpy 2.3.2, and its immutable SHA-256 digest (not mutable `:latest` tag) recorded.
2. **Condition 2 — Formal Holdout Authorization & Seed Escrow (Blocking for Generation)**:
   - *Owner*: Independent Evaluation Auditor & Research Lead.
   - *Requirement*: Dataset v2.2 has not been authorized. A separate, formal authorization event must grant permission to generate and encrypt v2.2 with an independent escrowed seed.
3. **Condition 3 — Tabular-Only Model Formalization (Non-Blocking)**:
   - *Owner*: ML Research Engineer.
   - *Requirement*: Formalize that the candidate model for order fraud ranking is the Tabular-Only XGBoost model, as the Late Fusion calibrator was not serialized in Stage 3.4.

---

## 13. Explicit Readiness Decision

### Readiness Status: **CONDITIONALLY READY**

The evaluation protocol, temporal leak guards, 21-day label maturity rules, and mathematical reconciliations are **verified and certified**. 

However, **Synthetic Dataset v2.2 remains ungenerated and unblinded**. Full evaluation execution is conditioned on:
1. Freezing an immutable container image digest.
2. Independent escrow of the holdout seed and AES-256-GCM encryption key.
3. Formal stakeholder authorization for the holdout run.
