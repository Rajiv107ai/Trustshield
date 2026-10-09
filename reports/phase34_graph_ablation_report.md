# TrustShield Stage 3.4 — Controlled Graph Ablation Report

**Date**: October 9, 2026  
**Auditor / Researcher**: Independent Evaluation & Verification Agent  
**Baseline Commit**: `4615a53602d0a4ffe447d4d2fd14bd69a70bf740`  
**Active Git Branch**: `phase-3-data-generalization`  
**Dataset Version**: Synthetic Dataset v2.1 (`data/synthetic_v2_1/`)  
**Status**: **COMPLETED — CONTROLLED EVALUATION CERTIFIED**

---

## Executive Summary

Stage 3.4 investigates whether graph-topological features provide genuine, out-of-time predictive lift for transaction fraud detection, or whether earlier reported performance was an artifact of shortcut learning, feature dilution, or flawed zero-feature ablation methodology.

### Definitive Findings
1. **Graph Features Provide Zero Incremental Lift**:
   - The **Graph-Only Model** achieved a Validation ROC-AUC of **0.5041** (95% CI: `[0.4839, 0.5253]`) and a Validation PR-AUC of **0.0751** (matching the baseline positive prevalence of `0.0704`). It is mathematically indistinguishable from random guessing.
   - On the Historical Diagnostic Test set, the Graph-Only model collapsed to an ROC-AUC of **0.4658** (worse than random guessing).
2. **Early Fusion Actively Degrades Performance**:
   - Concatenating 8 graph features onto 10 tabular features (**Early-Fusion Model**) reduced Validation ROC-AUC from **0.6395** (Tabular-Only) to **0.6272** (-0.0123) and Validation PR-AUC from **0.1273** to **0.1212** (-0.0061).
   - On the Historical Diagnostic Test set, Early Fusion dropped Test ROC-AUC from **0.6246** to **0.6076** (-0.0170).
   - *Mechanism*: The addition of high-sparsity, drifting graph features diluted gradient boosting tree splits, leading to suboptimal feature selections and impaired generalization.
3. **Validation-Tuned Late Fusion Discards the Graph Model**:
   - A grid search across convex combination weights $\alpha \cdot P_{\text{tab}} + (1-\alpha) \cdot P_{\text{graph}}$ on validation data converged to **$\alpha = 1.00$** (100% tabular weight, 0% graph weight).
4. **Refutation of Legacy Ablation Claims**:
   - In Stage 2 and Stage 3.3, setting graph features to zero on a graph-trained model caused a performance drop from 0.6043 to 0.5603. Our controlled ablation proves this drop was **not** evidence of graph utility, but rather the disruption of tree split thresholds from out-of-distribution inputs (feature mutilation). When models are trained without graph features from scratch, they outperform the graph-trained model.

---

## Experimental Protocol and Cohorts

### 1. Temporal Splits & Cohort Properties
All models were trained and evaluated on identical partitions from the frozen `data/synthetic_v2_1/` dataset:
- **Training Cohort**: `order_date <= 2025-08-31 23:59:59`  
  - Total Orders: $16,952$ | Fraudulent Orders: $813$ | Positive Prevalence: **4.80%**
- **Validation Cohort** (used for all model selection, tuning, and thresholding): `2025-08-31 23:59:59 < order_date <= 2025-10-31 23:59:59`  
  - Total Orders: $12,129$ | Fraudulent Orders: $854$ | Positive Prevalence: **7.04%**
- **Historical Diagnostic Test Cohort** (strictly diagnostic; never used for selection): `order_date > 2025-10-31 23:59:59`  
  - Total Orders: $20,919$ | Fraudulent Orders: $1,835$ | Positive Prevalence: **8.77%**

### 2. Candidate Architectures
All tree-based models shared identical tuning budgets and hyperparameters (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=neg/pos`, `random_state=42`).
1. **Tabular-Only XGBoost**: Trained on 10 behavioral and velocity features.
2. **Graph-Only XGBoost**: Trained on 8 topological, centrality, and edge weight features.
3. **Early-Fusion XGBoost**: Trained on the concatenated 18-dimensional feature vector.
4. **Late-Fusion Ensemble**: Combines outputs of Tabular-Only and Graph-Only models via validation-tuned convex blend or validation-fit logistic regression stacker.

---

## Forensic Inspection of Graph Features

### 1. Sparsity and Default Value Prevalence
Inspection reveals extreme sparsity across the graph feature space in all splits:

| Feature Name | Feature Type | Default / Null Value | Train Sparsity (%) | Val Sparsity (%) | Test Sparsity (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `share_degree` | Entity Sharing Degree | `0.0` | 66.36% | 65.81% | 66.27% |
| `share_component_size` | Sharing Cluster Size | `1.0` | 66.36% | 65.81% | 66.27% |
| `buyer_seller_degree` | Monthly Bipartite Degree | `0.0` | 36.06% | 24.82% | 26.24% |
| `buyer_pagerank` | Monthly PageRank Score | `0.0` | 36.06% | 24.82% | 26.24% |
| `seller_buyer_degree` | Seller In-Degree | `0.0` | 8.06% | 1.45% | 0.64% |
| `seller_pagerank` | Seller PageRank Score | `0.0` | 8.06% | 1.45% | 0.64% |
| `seller_buyer_concentration_hhi` | Concentration Index | `0.0` | 8.06% | 1.45% | 0.64% |
| `buyer_seller_edge_weight_before`| Cumulative Prior Orders | `0.0` | **97.97%** | **97.35%** | **96.19%** |

- **Address & Device Sharing**: Over **65% of all buyers** have never shared an address or device with another buyer. Fraud rings represent a small, highly concentrated cluster, offering negligible signal for general marketplace orders.
- **Repeat Buyer-Seller Pairs**: Over **96% of transactions** represent first-time interactions between that specific buyer and seller (`buyer_seller_edge_weight_before == 0`).

### 2. Temporal Drift and Scale Expansion
Graph centrality metrics computed over growing cumulative or lagged snapshots suffer from significant distributional shift over time:

| Feature Name | Train Mean $\pm$ Std | Validation Mean $\pm$ Std | Diagnostic Test Mean $\pm$ Std | Temporal Trend |
| :--- | :---: | :---: | :---: | :--- |
| `buyer_seller_degree` | $3.46 \pm 5.11$ | $5.81 \pm 7.32$ | $7.27 \pm 8.99$ | +110% expansion |
| `seller_buyer_degree` | $28.49 \pm 31.79$ | $71.29 \pm 60.40$ | $121.35 \pm 97.52$ | +326% expansion |
| `seller_buyer_concentration_hhi` | $0.1121 \pm 0.1966$ | $0.0353 \pm 0.0684$ | $0.0179 \pm 0.0363$ | -84% decay |
| `buyer_pagerank` | $0.0003 \pm 0.0006$ | $0.0002 \pm 0.0002$ | $0.0001 \pm 0.0001$ | Dilution with graph growth |

- Because the marketplace graph accumulates nodes and edges over chronological time, seller degrees expand four-fold, while PageRank and HHI concentration compress toward zero. Fixed tree split cutoffs learned during the training period become obsolete during future inference.

### 3. Entity Cold-Start Dynamics
- **Buyer Cold-Start Rate**:
  - Validation Split: **35.72%** of orders originated from buyers unseen in the training period.
  - Test Split: **65.00%** of orders originated from buyers unseen in the training period.
- **Seller Cold-Start Rate**:
  - Validation Split: **3.31%** | Test Split: **6.00%**.
- High buyer turnover severely impairs graph-based buyer profiling in e-commerce settings where guest checkouts or episodic purchasing predominate.

---

## Controlled Graph Ablation Results

### 1. Validation Performance (Primary Decision Criterion)

All models evaluated on the frozen Validation cohort ($N = 12,129$, Prevalence: $7.04\%$):

| Architecture | ROC-AUC | 95% Bootstrap CI | PR-AUC | 95% Bootstrap CI | Precision | Recall | F1 | Brier Score | ECE (10-bin) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tabular-Only** | **0.6395** | `[0.6211, 0.6599]` | **0.1273** | `[0.1141, 0.1437]` | 0.1501 | 0.2728 | **0.1937** | **0.0639** | **0.0000** |
| **Graph-Only** | 0.5041 | `[0.4839, 0.5253]` | 0.0751 | `[0.0685, 0.0836]` | 0.0717 | **0.9614** | 0.1334 | 0.0653 | 0.0000 |
| **Early-Fusion** | 0.6272 | `[0.6087, 0.6468]` | 0.1212 | `[0.1082, 0.1402]` | 0.1197 | 0.3607 | 0.1797 | 0.0641 | 0.0000 |
| **Late-Fusion (Stacker)** | 0.6388 | `[0.6204, 0.6594]` | 0.1256 | `[0.1144, 0.1441]` | **0.1502** | 0.2717 | 0.1934 | **0.0639** | 0.0000 |

*Note: Precision, Recall, and F1 are evaluated at validation-optimal F1 thresholds. Brier and ECE reflect validation-fitted Isotonic probability calibration.*

### 2. Historical Diagnostic Test Performance (Informational Only)

Evaluated strictly as an informational diagnostic on the previously exposed test cohort ($N = 20,919$, Prevalence: $8.77\%$):

| Architecture | Test ROC-AUC | Test PR-AUC | Test Precision | Test Recall | Test F1 | Test Brier | Test ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tabular-Only** | **0.6246** | **0.1470** | **0.1986** | 0.1700 | **0.1832** | **0.0793** | 0.0289 |
| **Graph-Only** | 0.4658 | 0.0803 | 0.0881 | **0.9030** | 0.1605 | 0.0806 | **0.0204** |
| **Early-Fusion** | 0.6076 | 0.1309 | 0.1461 | 0.1815 | 0.1619 | 0.0798 | 0.0282 |
| **Late-Fusion (Stacker)** | 0.6244 | 0.1417 | 0.1973 | 0.1700 | 0.1827 | **0.0793** | 0.0282 |

```
PR-AUC Comparison across Architectures (Validation):
Tabular-Only:       [█████████████████████████] 0.1273  <-- CHAMPION
Late-Fusion:        [████████████████████████ ] 0.1256
Early-Fusion:       [███████████████████████  ] 0.1212  (-4.8% vs Tabular)
Graph-Only:         [██████████████           ] 0.0751  (Zero lift over prior 0.0704)
```

---

## Detailed Architectural Analysis

### 1. Why Graph-Only Fails
The Graph-Only model achieves PR-AUC 0.0751 on validation (prevalence: 0.0704) and 0.0803 on test (prevalence: 0.0877). It fails because:
- 66% of buyers have 0 sharing degree and component size 1.
- Over 96% of buyer-seller pairs have no prior transaction history.
- The few buyers involved in dense rings in the synthetic generation account for a tiny fraction of total marketplace fraud.
- In out-of-time test periods, buyer turnover (65% new buyers) ensures the graph model has no prior topological history for the majority of perpetrators.

### 2. Why Early Fusion Degrades Tabular Signal
Early fusion concatenates all 18 features into XGBoost. In gradient boosting:
- At each split, the algorithm evaluates feature candidates across random subsets (`colsample_bytree=0.8`).
- When 8 out of 18 features (44%) are uninformative or non-stationary graph metrics, nearly half of the candidate features in any tree split are noise.
- This results in split dilution: trees frequently branch on spurious graph thresholds rather than informative tabular behavioral features (e.g., buyer return velocity, amount anomaly, price-to-base ratio).
- Consequently, Early Fusion exhibits inferior validation PR-AUC (0.1212 vs 0.1273) and inferior test PR-AUC (0.1309 vs 0.1470).

### 3. Late Fusion Behavior
- The convex blend search strictly identified $\alpha = 1.00$ as the optimal validation weight. Any non-zero weight assigned to the graph model degraded the ensemble's PR-AUC.
- The logistic regression stacker assigned a near-zero coefficient to graph predictions, effectively replicating the Tabular-Only model's predictions.

---

## Proposal for Untouched Future Holdout (Dataset v2.2)

Because the v2.1 test set has now been inspected across Stages 3.2, 3.3, 3.3.1, and 3.4, it can no longer serve as a pristine final test set for future production readiness claims. We propose the following protocol for **Synthetic Dataset v2.2**:

### 1. Specification and Horizon Extension
- **Simulation Horizon**: Extend the simulation timeline by 60 additional days:
  - Simulation Period: `2026-01-01 00:00:00` through `2026-02-28 23:59:59`.
  - Observation Window: Strict 21-day maturity cutoff enforced on all return tracking.
- **Estimated Volume**: $\approx 25,000$ new orders, generated under the existing realistic generator parameters.

### 2. Cryptographic Access Controls
1. **Blind Generation**: The generator script outputs features to `data/synthetic_v2_2/orders_features.csv`, but writes true labels (`is_fraudulent`, `fraud_type`) to an encrypted archive (`data/synthetic_v2_2/labels.enc`) using AES-256-GCM.
2. **SHA-256 Manifest**: Generate a tamper-evident `dataset_manifest_v2_2.json` containing the cryptographic hash of all feature files and the encrypted label blob.
3. **Unblinded Evaluation Gate**: Model inference code must generate and serialize prediction arrays (`p_test.npy`) *before* the decryption key is supplied to compute final audit metrics.
4. **Single-Use Policy**: Once unblinded, the dataset becomes historical diagnostic data and cannot be used for retraining or hyperparameter selection.

---

## Conclusion & Architectural Recommendations

1. **Retire Standalone Graph Features from Order-Level Scoring**:
   - The current 8 graph features should be removed from the transaction scoring model.
   - Tabular behavioral features remain the primary driver of out-of-time transaction fraud detection.
2. **Repurpose Graph Signals for Specialized Ring Review**:
   - Rather than forcing sparse graph features into the transactional inference pipeline, relationship graphs should be used exclusively as post-scoring investigative overlays (e.g., for batch ring identification in the investigation console).
3. **Champion Model Selection**:
   - The **Tabular-Only Model** is formally certified as the champion transaction detector on the v2.1 benchmark (Validation PR-AUC: 0.1273, ROC-AUC: 0.6395).

---
*Report certified by Independent Audit Subsystem.*
