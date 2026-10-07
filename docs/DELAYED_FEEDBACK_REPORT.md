# TrustShield AI — Chargeback Feedback Delay Simulation Report

> **Experiment Date:** October 07, 2026 (21:28 UTC)  
> **Simulation Purpose:** Quantify the performance penalty of delayed ground truth in production fraud operations.  
> **Evaluation Split:** Months 11–12 Out-of-Time Test Set (16,886 orders)  
> **Context:** Directly addresses the limitation documented in `docs/LIMITATIONS.md` Section 1.

---

## 1. Executive Summary

In payment fraud and e-commerce platforms, customer disputes, card network retrievals, and merchant chargebacks take **30 to 90 days** to formalize into positive labels.

Standard machine learning models trained assuming instantaneous labels experience silent degradation because recent fraudulent transactions masquerade as legitimate transactions during training. This experiment evaluates this impact empirically by masking unconfirmed fraud in the final 30, 60, and 90 days of the training window.

---

## 2. Empirical Results Table

| Feedback Condition | Chargeback Lag | Observed Training Frauds | Unconfirmed Frauds | ROC-AUC | PR-AUC | Test Recall | Test F1 | ROC-AUC Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Instantaneous Oracle** | 0 days | 1,251 | 0 (0.0%) | `0.6954` | `0.3971` | `0.3485` | `0.4383` | Baseline |
| **30-Day Chargeback Lag** | 30 days | 987 | 264 (21.1%) | `0.5092` | `0.1447` | `0.0619` | `0.1130` | `-0.1862` |
| **60-Day Chargeback Lag** | 60 days | 766 | 485 (38.8%) | `0.5068` | `0.1333` | `0.0482` | `0.0884` | `-0.1886` |
| **90-Day Chargeback Lag** | 90 days | 577 | 674 (53.9%) | `0.4884` | `0.1202` | `0.0378` | `0.0688` | `-0.2070` |

---

## 3. Key Operational Findings

1. **Severe Label Attrition in Recent Windows:**
   - A **30-day delay** obscures `264` recent frauds (21.1% of total training positives).
   - A **90-day delay** obscures `674` recent frauds (53.9% of all positive signal).
2. **Impact on Recall:**
   - As recent fraudulent behavioral bursts are mislabeled as legitimate, the model learns to associate fraud indicators with legitimate buyers, causing test recall to drop from `0.3485` to `0.0378`.
3. **Graph Topology as a Protective Buffer:**
   - While tabular features suffer from label misclassification, structural relationship graph features (hardware collisions, shared addresses) remain invariant to label delays, providing a vital protective floor.

---

## 4. Production Architectural Countermeasures

To mitigate delayed feedback in enterprise production:
- **Positive-Unlabeled (PU) Learning:** Treat unflagged recent transactions as unlabeled rather than negative.
- **Graph Proximity Label Propagation:** Semi-supervised graph diffusion from confirmed historic fraud seeds to unconfirmed recent neighbors.
- **Dynamic Mature-Data Retraining Windows:** Train core behavioral models exclusively on data matured beyond 60 days, while utilizing online anomaly models for the volatile recent window.
