# TrustShield Phase 3 — Stage 3.2 Evaluation Protocol

**Document Version:** 1.0  
**Effective Date:** 2026-10-09  
**Stage:** 3.2 — Data Contract, Feature Validity, and Evaluation Readiness Audit  
**Status:** **FROZEN (STRICT AUDIT LOCK)**  
**Target Dataset:** [`data/synthetic_v2_1/`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1)  
**Auditor / Engineers:** Senior ML Research Auditor, Backend Security Architect, Data Quality Engineer  
**Git Branch:** `phase-3-data-generalization`  
**Baseline Git Commit:** `4615a53602d0a4ffe447d4d2fd14bd69a70bf740`  

---

## 1. Executive Summary & Purpose

This protocol establishes the **frozen out-of-time evaluation methodology** for TrustShield under the audited [`data/synthetic_v2_1/`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1) benchmark.

Following the forensic audits of Phase 1, Phase 2, and Stage 3.1, earlier evaluation metrics were confounded by:
1. Shortcut learning (e.g., cross-catalog product mismatch producing near-zero text-image similarity).
2. Simulation-end aggregate leakage (`total_orders`, `total_returns` computed across future transactions).
3. Right-censoring bias (treating orders placed near simulation end with no observed return as confirmed legitimate non-returns).
4. Circular evaluation (evaluating models on the same distributions used to tune perturbation heuristics).

This document codifies the temporal partitions, sample counts, label prevalence, entity turnover, right-censoring constraints, and model selection criteria for all subsequent modeling stages. Per mandatory safeguards, **no models are trained, no thresholds are tuned, and no inference code is executed in Stage 3.2**.

---

## 2. Historical Baseline Reference (v1 and v2 Datasets)

Historical test metrics from v1 and v2 benchmarks are preserved strictly for comparative baseline reference. They represent the legacy performance envelope before generator debiasing.

### 2.1 Historical Model Performance Summary

*Source: [`models/reproduced_evaluation_report.json`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/reproduced_evaluation_report.json) & [`reports/phase2_final_verification.md`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/reports/phase2_final_verification.md)*

| Model | Evaluation Split | ROC-AUC (Raw) | ROC-AUC (Calibrated) | PR-AUC (Calibrated) | Precision | Recall | F1 | ECE | Test N |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 3 Combined Graph (XGBoost)** | Historical Test | 0.7897 | 0.7890 | 0.4418 | 0.6857 | 0.3212 | 0.4374 | 0.0265 | 16,886 |
| **Phase 3 Zero-Graph Degraded** | Historical Test | 0.6043 | 0.6029 | 0.2419 | 0.9055 | 0.1186 | 0.2097 | 0.0619 | 16,886 |
| **Phase 5 Hybrid Model (Stacking)** | Historical Test | 0.7666 | 0.7652 | 0.4191 | 0.8143 | 0.2515 | 0.3843 | 0.0330 | 16,886 |
| **Fake-Listing Detector** | Pure Months 11–12 | 0.9496 | 0.9496 | 0.7406 | 0.7152 | 0.6835 | 0.6990 | 0.0104 | 6,237 |
| **Return-Fraud Detector** | Pure Months 11–12 | 0.9235 | 0.9235 | 0.8806 | 0.8277 | 0.6786 | 0.7457 | 0.0841 | 1,917 |

> [!NOTE]
> The fake-listing detector ROC-AUC (0.9496) on v1 data was heavily inflated by arbitrary cross-catalog swaps (e.g., electronics swapped with shoes), creating an unrealistic multimodal shortcut. On [`data/synthetic_v2_1/`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1), the category-constrained perturbation yields a single-feature diagnostic ROC-AUC of **0.7599**. Retrained models must be evaluated against this realistic diagnostic baseline.

---

## 3. Proposed Out-of-Time Temporal Split Specification (v2.1)

All evaluations under `data/synthetic_v2_1/` are governed by a strictly chronological, non-overlapping 3-way temporal split. Random k-fold cross-validation is strictly prohibited due to future-information leakage.

### 3.1 Split Boundary Definitions

| Partition | Window Start (Inclusive) | Window End (Inclusive) | Span | Operational Role |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `2025-01-01 00:00:00` | `2025-08-31 23:59:59` | 8 Months | Model fitting, representation learning, graph structure construction |
| **Validation** | `2025-09-01 00:00:00` | `2025-10-31 23:59:59` | 2 Months | Hyperparameter optimization, probability calibration, threshold tuning |
| **Test (FROZEN)** | `2025-11-01 00:00:00` | `2025-12-31 23:59:59` | 2 Months | Final out-of-time benchmark evaluation (**UNTOUCHED**) |

### 3.2 Partition Sizing & Distribution Statistics

#### Orders Table (`orders.csv` — 50,000 total)

| Metric | Train Split | Validation Split | Test Split (FROZEN) | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Total Orders** | 16,952 (33.90%) | 12,129 (24.26%) | 20,919 (41.84%) | 50,000 |
| **Fraud Orders** | 813 | 854 | 1,835 | 3,502 |
| **Fraud Rate** | **4.80%** | **7.04%** | **8.77%** | **7.00%** |
| - *Return Abuse* | 447 (54.98%) | 319 (37.35%) | 467 (25.45%) | 1,233 |
| - *Fake Listing* | 277 (34.07%) | 285 (33.37%) | 473 (25.78%) | 1,035 |
| - *Coordinated Fraud* | 76 (9.35%) | 167 (19.56%) | 445 (24.25%) | 688 |
| - *Seller-Buyer Collusion* | 13 (1.60%) | 83 (9.72%) | 450 (24.52%) | 546 |

#### Listings Table (`listings.csv` — 20,000 total)

| Metric | Train Split | Validation Split | Test Split (FROZEN) | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Total Listings** | 13,489 (67.45%) | 3,825 (19.12%) | 2,686 (13.43%) | 20,000 |
| **Fake Listings** | 301 | 143 | 56 | 500 |
| **Fake Listing Rate** | **2.23%** | **3.74%** | **2.08%** | **2.50%** |

#### Returns Table (`returns.csv` — 5,537 total)

| Metric | Train Split | Validation Split | Test Split (FROZEN) | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Total Returns** | 1,758 (31.75%) | 1,333 (24.07%) | 2,446 (44.18%) | 5,537 |
| **Fraudulent Returns** | 416 | 351 | 785 | 1,552 |
| **Return Fraud Rate** | **23.66%** | **26.33%** | **32.09%** | **28.03%** |

---

## 4. Entity Overlap & Generalization Dynamics

A robust evaluation must explicitly quantify the ratio of returning entities versus unseen ("cold-start") entities across temporal splits. Models that overfit to entity identifiers will experience severe performance degradation on cold-start cohorts.

### 4.1 Buyer Entity Demographics

| Split | Unique Buyers | Returning Buyers (Seen in Prior Splits) | Cold-Start Buyers (Unseen in Prior Splits) | Cold-Start Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 3,032 | N/A (Baseline cohort) | 3,032 | 100.0% |
| **Validation** | 3,546 | 2,471 (seen in Train) | 1,075 (unseen in Train) | **30.3%** |
| **Test (FROZEN)** | 4,338 | 3,446 (seen in Train or Val) | 892 (completely unseen) | **20.6%** |

### 4.2 Seller Entity Demographics

| Split | Unique Sellers | Returning Sellers (Seen in Prior Splits) | Cold-Start Sellers (Unseen in Prior Splits) | Cold-Start Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 468 | N/A (Baseline cohort) | 468 | 100.0% |
| **Validation** | 487 | 467 (seen in Train) | 20 (unseen in Train) | **4.1%** |
| **Test (FROZEN)** | 499 | 487 (seen in Train or Val) | 12 (completely unseen) | **2.4%** |

---

## 5. Right-Censoring & Observation Window Protocol

### 5.1 The Right-Censoring Horizon

In the simulation, returns are permitted up to **21 days** following order placement.
- Simulation window: `[2025-01-01 00:00:00, 2025-12-31 23:59:59]`.
- Orders placed after **`2025-12-10 23:59:59`** have return turnaround windows extending into calendar year 2026.
- A total of **8,961 orders** occur between Dec 10 and Dec 31.
- In `data/synthetic_v2_1/`, exactly **350 return events** scheduled for 2026 were truncated from `returns.csv`.

### 5.2 Mandatory Evaluation Rules for Return-Fraud Models

1. **Negative Outcome Fallacy Prohibited**:
   An order placed on or after `2025-12-10` that does not appear in `returns.csv` **MUST NOT** be labeled as a verified non-return or legitimate outcome. Doing so introduces severe survival bias.
2. **Evaluation Window Restriction**:
   When evaluating binary return-fraud detectors on the Test split:
   - **Primary Protocol (Censored Boundary)**: Restrict the evaluation cohort to orders placed between `2025-11-01 00:00:00` and `2025-12-10 23:59:59` (full 21-day observation maturity).
   - **Secondary Protocol (Survival Analysis)**: If orders after Dec 10 are included, observations must be treated as right-censored with survival time $T_{censor} = \text{2025-12-31} - T_{order}$.
3. **Label Disambiguation**:
   - `has_return_event`: The observable behavioral outcome (did the customer initiate a return?).
   - `is_return_abuse`: The fraud ground truth (was the return abusive/fraudulent?).
   Evaluation metrics must evaluate `is_return_abuse` conditional on `has_return_event == 1`, or explicitly model the two-stage hazard.

---

## 6. Point-in-Time Feature Isolation Rules

To eliminate data leakage, feature extraction code must comply with the following architectural rules:

### 6.1 Banned Feature Catalog

The following fields from `data/synthetic_v2_1/` are **strictly forbidden** from entering candidate feature sets:

| Banned Field | Table | Reason for Exclusion |
| :--- | :--- | :--- |
| `total_orders` | `buyers.csv` | Full-simulation aggregate. Contains transactions occurring after time $T$. |
| `total_returns` | `buyers.csv` | Full-simulation aggregate. Contains returns occurring after time $T$. |
| `total_listings` | `sellers.csv` | Full-simulation aggregate. Contains listings created after time $T$. |
| `total_orders_received` | `sellers.csv` | Full-simulation aggregate. Contains orders received after time $T$. |
| `trust_score_current` | `buyers`, `sellers` | Static initialization artifact (uncalibrated 70.0). |
| `is_fraudulent` | Operational tables | Target label. |
| `fraud_type` | Operational tables | Target label / subtype metadata. |
| `persona` | `generator_personas` | Generator internal control metadata. |
| `fraud_ring_id` | `fraud_ground_truth` | Evaluation ground truth. |

### 6.2 Point-in-Time Construction Mechanics

- Historical aggregates for buyer order count, buyer return count, and seller listing count must be computed using backward merge (`pandas.merge_asof` with `direction='backward'` and strict `<` inequalities).
- Static entity characteristics (e.g., category-level median prices, device sharing rates) must be computed **exclusively on the Train partition** (`order_date <= 2025-08-31`) and joined to Validation and Test partitions as immutable lookup tables.

---

## 7. Model Selection, Calibration & Decision Governance

### 7.1 Primary and Secondary Metrics

For all fraud detection models, the primary decision criteria are:
1. **Precision-Recall AUC (PR-AUC)**: Primary ranking metric reflecting performance on highly imbalanced fraud classes.
2. **ROC-AUC**: Secondary discrimination diagnostic.
3. **Brier Score & Expected Calibration Error (ECE)**: Required to ensure predicted probabilities correspond to true empirical risk. ECE must be evaluated across 10 uniform probability bins.
4. **F1-Score at Operating Threshold**: Operational decision efficiency.

### 7.2 Decision Threshold Policy

- **Threshold Selection**: Optimal classification thresholds ($T^*$) must be tuned solely on the **Validation partition** to maximize F1 or satisfy minimum precision constraints (e.g., Precision $\ge 0.70$).
- **Freezing**: Once $T^*$ is selected on Validation, it is strictly frozen.
- **Test Evaluation**: Frozen $T^*$ is applied to the Test partition exactly once. Threshold re-tuning on the Test partition is strictly prohibited.

### 7.3 Test Set Isolation Enforcement

The final Test partition (`2025-11-01` to `2025-12-31`) is under strict lock:
- No model architecture exploration on Test.
- No hyperparameter tuning on Test.
- No feature selection iterations on Test.
- Test metrics may only be generated during the official Stage 3.3 model evaluation audit.

---

## 8. Verification & Test Suite Evidence

The evaluation protocol rules defined above are codified and programmatically verified in [`trustshield_project/test_stage32_audit.py`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/trustshield_project/test_stage32_audit.py).

### 8.1 Automated Test Execution

```powershell
.\.venv\Scripts\pytest.exe trustshield_project/test_stage32_audit.py -v
```

**Results:**
- `TestDatasetContract::test_manifest_hashes_match_disk`: **PASSED**
- `TestDatasetContract::test_primary_key_uniqueness_and_zero_nulls`: **PASSED**
- `TestDatasetContract::test_foreign_key_referential_integrity`: **PASSED**
- `TestDatasetContract::test_cross_table_relationship_consistency`: **PASSED**
- `TestDatasetContract::test_category_constraints_hold_on_all_listings`: **PASSED**
- `TestPointInTimeFeatures::test_unsafe_fields_banned_from_model_features`: **PASSED**
- `TestPointInTimeFeatures::test_late_arriving_returns_isolated_at_order_time`: **PASSED**
- `TestPointInTimeFeatures::test_temporal_invariance_under_future_event_insertion`: **PASSED**
- `TestRightCensoring::test_missing_return_near_simulation_end_is_not_confirmed_negative`: **PASSED**
- `TestRightCensoring::test_label_disambiguation_abuse_vs_return_event`: **PASSED**
- `TestMultimodalEvidence::test_reproduced_similarity_distributions_and_auc`: **PASSED**
- `TestMultimodalEvidence::test_image_availability_and_fallback_counts`: **PASSED**
- `TestEvaluationProtocol::test_temporal_split_boundaries_and_order_counts`: **PASSED**
- `TestEvaluationProtocol::test_test_set_frozen_isolation`: **PASSED**

**Overall Test Suite Status:** `14 passed in 4.77s` (100% PASS).
