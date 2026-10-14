<!-- BEGIN GENERATED -->
# TrustShield AI — Audited Evaluation Results & Scientific Benchmarks

> **Provenance Notice**: Every metric in this document is dynamically generated from `results/results.json` via `scripts/generate_results_md.py`. No numbers are hard-coded.

- **Execution Timestamp**: `2026-10-10T11:44:13Z`
- **Git Commit**: `91b0cbb4fda537d8e0528568d0730d7759342010`
- **Primary Hypothesis Sample Size**: 20 independent seeds
- **Exploratory Sample Size**: 5 independent seeds

## 1. Headline Benchmark: Standard Dataset (20 Independent Seeds)

All models evaluated on the temporal holdout test period (`order_date > 2025-10-31`).

| Model Variant | Feature Set | ROC-AUC (mean +/- std) | PR-AUC (mean +/- std) | ROC Lift vs Tabular Baseline (b) | Two-Sided p-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **(a) Tabular (without device)** | 9 Tabular Features | 0.7021 +/- 0.0186 | 0.4179 +/- 0.0202 | +0.0000 | p = 1.0000e+00 |
| **(b) Tabular (with device, baseline)** | 10 Tabular Features | 0.7295 +/- 0.0193 | 0.4386 +/- 0.0238 | baseline (+0.0000) | — |
| **(d) Tabular + Plain Aggregates (control)** | 10 Tabular + 6 Plain Lagged Aggs | 0.7275 +/- 0.0201 | 0.4377 +/- 0.0242 | -0.0020 | p = 2.7292e-01 |
| **(c) Tabular + Graph Features** | 10 Tabular + 8 Graph Features | 0.7410 +/- 0.0229 | 0.4461 +/- 0.0276 | **+0.0115** | **p = 4.7314e-05** |

## 2. Non-Graph Control Analysis (Variant d vs c and b)

Control Variant (d) attaches 6 month-lagged plain aggregates computed with identical point-in-time conventions as the graph features, but with **zero graph computation**:
- `buyer_prev_month_order_count`
- `buyer_distinct_sellers_prior`
- `seller_prev_month_order_count`
- `seller_distinct_buyers_prior`
- `seller_top_buyer_share_prior`
- `buyer_seller_edge_weight_before`

### Paired Comparisons vs Control:
| Comparison | Metric | Paired Mean Diff | 95% Confidence Interval | t-statistic | p-value | Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **(c vs d)** Graph vs Plain Aggregates | ROC-AUC | +0.0136 | [+0.0081, +0.0190] | t = 5.24 | p = 4.6636e-05 | Graph retains genuine predictive signal over plain aggregates |
| | PR-AUC | +0.0084 | [+0.0038, +0.0131] | t = 3.78 | p = 1.2662e-03 | Precision lift remains positive over plain aggregates |
| **(d vs b)** Plain Aggregates vs Tabular | ROC-AUC | -0.0020 | [-0.0058, +0.0017] | t = -1.13 | p = 2.7292e-01 | Plain lagged counts alone add zero lift over tabular baseline |
| | PR-AUC | -0.0008 | [-0.0033, +0.0016] | t = -0.71 | p = 4.8370e-01 | No PR improvement from plain aggregates |

### Topological PageRank vs Plain Order Counts Correlation:
- **`buyer_pagerank` vs `buyer_orders_before`**: Pearson r = 0.8430 +/- 0.0357 | Spearman rho = 0.7562 +/- 0.0258 (High correlation: PageRank is heavily driven by plain order counts)
- **`seller_pagerank` vs seller ORDER COUNT**: Pearson r = 0.9899 +/- 0.0066 | Spearman rho = 0.9696 +/- 0.0082 (Near-perfect correlation: seller PageRank is collinear with total seller order volume)
- **Scope & Attribution**: This conclusion is scoped strictly to bipartite PageRank (`buyer_pagerank` and `seller_pagerank`), where topological centrality heavily tracks raw activity. Furthermore, control (d) demonstrates that month-lagged plain activity aggregates provide zero predictive lift over baseline tabular features, confirming that the genuine lift originates from sharing-based features (S: device and address clusters).

## 2b. Canonical Controls & Cadence Evaluation: Plain Pairwise Counts vs Sharing Graph Topology

**Evaluation Split**: Truncated Test Horizon (`2025-11-01` to `2025-12-10`, final 21 days excluded, $N=20$ independent seeds: 42, 101, 202, ..., 1910).

> **Headline Finding: Plain Pairwise Sharing Counts Match or Beat Graph Topology Features**
> - Order-level point-in-time pairwise sharing counts (Control e) achieve ROC **0.7649 +/- 0.0242** vs Group S **0.7610 +/- 0.0237** (paired diff -0.0039, p = 0.0406) and Full (c) **0.7635 +/- 0.0229**.
> - At matching weekly snapshot cadence, Control (e_weekly) achieves ROC **0.7637 +/- 0.0240**, matching or slightly beating topological features (paired diff vs S: +0.0027 ROC, +0.0034 PR, p = 0.0231).
> - **Cadence Confound Addressed**: The difference between weekly cadence and order-level PIT (e_weekly vs e) is small and not statistically significant (ROC diff -0.0012, p = 0.3992). The advantage of plain counts is **NOT** due to snapshot cadence; plain pairwise counts capture the fraud signal directly without graph abstraction overhead.

| Model Variant | Feature Description | Truncated Test ROC-AUC | Truncated Test PR-AUC | Paired Lift vs Baseline (b) | Paired Lift vs Control (e) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **(b) Baseline Tabular** | Sharing Features | 0.7521 +/- 0.0174 | 0.4777 +/- 0.0270 | baseline | — |
| **(e) Tabular + PIT Pairwise Counts** | Sharing Features | 0.7649 +/- 0.0242 | 0.4892 +/- 0.0297 | +0.0127 (p = 5.7335e-05) | baseline |
| **(e_weekly) Tabular + Weekly Pairwise Counts** | Sharing Features | 0.7637 +/- 0.0240 | 0.4880 +/- 0.0297 | +0.0116 (p = 3.7606e-05) | -0.0012 (p = 3.9923e-01) |
| **(f) Tabular + share_degree only** | Sharing Features | 0.7634 +/- 0.0198 | 0.4874 +/- 0.0293 | +0.0113 (p = 1.3111e-08) | — |
| **(g) Tabular + share_component_size only** | Sharing Features | 0.7599 +/- 0.0219 | 0.4856 +/- 0.0291 | +0.0078 (p = 1.2177e-04) | — |
| **(S) Tabular + S (degree + CC size)** | Sharing Features | 0.7610 +/- 0.0237 | 0.4846 +/- 0.0292 | +0.0089 (p = 3.5722e-04) | -0.0039 (p = 4.0649e-02) |
| **(c) Full Tabular + Graph (8 features)** | Sharing Features | 0.7635 +/- 0.0229 | 0.4858 +/- 0.0318 | +0.0114 (p = 3.0153e-05) | -0.0014 (p = 5.8628e-01) |

## 3. Group Ablation Analysis (Main Ablation over 20 Seeds)

Feature groups defined as:
- **S (Sharing Graph)** = `{'share_degree', 'share_component_size'}`
- **B (Bipartite Graph)** = `{'buyer_seller_degree', 'buyer_pagerank', 'seller_buyer_degree', 'seller_pagerank', 'seller_buyer_concentration_hhi'}`
- **E (Edge History)** = `{'buyer_seller_edge_weight_before'}`

| Group Configuration | ROC-AUC (mean +/- std) | ROC Lift vs Tabular (b) | 95% Confidence Interval | Paired t-stat | p-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tabular + S** | 0.7385 +/- 0.0212 | +0.0090 | [+0.0051, +0.0130] | t = 4.77 | p = 1.3257e-04 |
| **Tabular + B** | 0.7340 +/- 0.0199 | +0.0045 | [+0.0004, +0.0086] | t = 2.32 | p = 3.1829e-02 |
| **Tabular + E** | 0.7335 +/- 0.0203 | +0.0040 | [+0.0012, +0.0069] | t = 2.93 | p = 8.5518e-03 |
| **Tabular + S+B (Drop E)** | 0.7422 +/- 0.0206 | +0.0127 | [+0.0085, +0.0168] | t = 6.42 | p = 3.7505e-06 |
| **Tabular + S+E (Drop B)** | 0.7395 +/- 0.0201 | +0.0100 | [+0.0065, +0.0134] | t = 6.02 | p = 8.6438e-06 |
| **Tabular + B+E (Drop S)** | 0.7323 +/- 0.0210 | +0.0027 | [-0.0014, +0.0069] | t = 1.37 | p = 1.8628e-01 |
| **Full Variant (c) [S+B+E]** | 0.7410 +/- 0.0229 | +0.0115 | [+0.0069, +0.0161] | t = 0.00 | p = 4.7314e-05 |
| **Baseline Tabular (b)** | 0.7295 +/- 0.0193 | +0.0000 | [0.0000, 0.0000] | — | 1.0000 |

## 4. Pre-Registered Primary Analysis & Seed Partitions

To ensure confirmatory rigor, the 20 seeds are partitioned into an initial exploratory subset and a pre-registered confirmatory holdout:

| Partition | Seeds Evaluated | Tabular Baseline (b) ROC | Tabular + Graph (c) ROC | Paired Lift | 95% Confidence Interval | p-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **exploratory_5_seeds** | 5 seeds ([42, 101, 202]...) | 0.7276 +/- 0.0296 | 0.7399 +/- 0.0315 | **+0.0123** | [-0.0047, +0.0293] | **p = 1.1453e-01** |
| **confirmatory_15_seeds** | 15 seeds ([505, 606, 707]...) | 0.7301 +/- 0.0160 | 0.7414 +/- 0.0206 | **+0.0112** | [+0.0064, +0.0161] | **p = 2.1478e-04** |
| **full_20_seeds** | 20 seeds ([42, 101, 202]...) | — | — | **+0.0115** | [+0.0069, +0.0161] | **p = 4.7314e-05** |

## 5. Step-Down Holm-Bonferroni Correction (Audited Exploratory Family)

Controls Family-Wise Error Rate (FWER <= 0.05) across all pre-registered exploratory tests (uncorrelated return-ablation p-values removed):

| Rank | Multiplier | Exploratory Test | Raw p-value | Holm-Adjusted p-value | Significant at alpha=0.05 |
| :---: | :---: | :--- | :---: | :---: | :---: |
| 1 | 36x | Control (f vs b): share_degree vs Tabular (ROC) | 1.3111e-08 | 4.7199e-07 | YES (p < 0.05) |
| 2 | 35x | Control (f vs b): share_degree vs Tabular (PR) | 6.0304e-07 | 2.1106e-05 | YES (p < 0.05) |
| 3 | 34x | Group Ablation: Tabular + S+B vs Tabular (ROC) | 3.7505e-06 | 1.2752e-04 | YES (p < 0.05) |
| 4 | 33x | Control (e vs b): Plain Pairwise Counts vs Tabular (PR) | 7.1733e-06 | 2.3672e-04 | YES (p < 0.05) |
| 5 | 32x | Group Ablation: Tabular + S+E vs Tabular (ROC) | 8.6438e-06 | 2.7660e-04 | YES (p < 0.05) |
| 6 | 31x | Control (e_weekly vs b): Weekly Counts vs Tabular (PR) | 9.8191e-06 | 3.0439e-04 | YES (p < 0.05) |
| 7 | 30x | Control (e_weekly vs b): Weekly Counts vs Tabular (ROC) | 3.7606e-05 | 1.1282e-03 | YES (p < 0.05) |
| 8 | 29x | Control (c vs d): Graph vs Plain Aggregates (ROC) | 4.6636e-05 | 1.3524e-03 | YES (p < 0.05) |
| 9 | 28x | Control (e vs b): Plain Pairwise Counts vs Tabular (ROC) | 5.7335e-05 | 1.6054e-03 | YES (p < 0.05) |
| 10 | 27x | Control (g vs b): share_component_size vs Tabular (PR) | 6.7769e-05 | 1.8298e-03 | YES (p < 0.05) |
| 11 | 26x | Control (g vs b): share_component_size vs Tabular (ROC) | 1.2177e-04 | 3.1661e-03 | YES (p < 0.05) |
| 12 | 25x | Group Ablation: Tabular + S vs Tabular (ROC) | 1.3257e-04 | 3.3143e-03 | YES (p < 0.05) |
| 13 | 24x | Control (c vs d): Graph vs Plain Aggregates (PR) | 1.2662e-03 | 3.0389e-02 | YES (p < 0.05) |
| 14 | 23x | Group (S vs e): Topological S vs Plain Counts (PR) | 3.3787e-03 | 7.7710e-02 | NO |
| 15 | 22x | Hybrid vs Tabular (Coherent N=5) | 3.9460e-03 | 8.6812e-02 | NO |
| 16 | 21x | Hybrid vs Tabular (Standard N=5) | 8.3154e-03 | 1.7462e-01 | NO |
| 17 | 20x | Group Ablation: Tabular + E vs Tabular (ROC) | 8.5518e-03 | 1.7462e-01 | NO |
| 18 | 19x | Per-Type Graph Lift: Coordinated Ring (ROC) | 2.1000e-02 | 3.9900e-01 | NO |
| 19 | 18x | Control (e_weekly vs S): Weekly Counts vs Topological S (PR) | 2.3099e-02 | 4.1577e-01 | NO |
| 20 | 17x | Group Ablation: Tabular + B vs Tabular (ROC) | 3.1829e-02 | 5.4110e-01 | NO |
| 21 | 16x | Tabular+Graph vs Tabular (Coherent Upper Bound N=5) | 3.4045e-02 | 5.4471e-01 | NO |
| 22 | 15x | Group (S vs e): Topological S vs Plain Counts (ROC) | 4.0649e-02 | 6.0973e-01 | NO |
| 23 | 14x | LOFO Loss: Drop buyer_pagerank | 4.5416e-02 | 6.3582e-01 | NO |
| 24 | 13x | LOFO Loss: Drop seller_buyer_concentration_hhi | 1.0023e-01 | 1.0000e+00 | NO |
| 25 | 12x | Per-Type Graph Lift: Fake Listing (ROC) | 1.8400e-01 | 1.0000e+00 | NO |
| 26 | 11x | Group Ablation: Tabular + B+E vs Tabular (ROC) | 1.8628e-01 | 1.0000e+00 | NO |
| 27 | 10x | LOFO Loss: Drop share_degree | 2.3811e-01 | 1.0000e+00 | NO |
| 28 | 9x | Control (d vs b): Plain Aggregates vs Tabular (ROC) | 2.7292e-01 | 1.0000e+00 | NO |
| 29 | 8x | Per-Type Graph Lift: Return Abuse (ROC) | 3.4200e-01 | 1.0000e+00 | NO |
| 30 | 7x | LOFO Loss: Drop buyer_seller_degree | 3.7037e-01 | 1.0000e+00 | NO |
| 31 | 6x | LOFO Loss: Drop seller_pagerank | 3.7401e-01 | 1.0000e+00 | NO |
| 32 | 5x | LOFO Loss: Drop buyer_seller_edge_weight_before | 3.8836e-01 | 1.0000e+00 | NO |
| 33 | 4x | LOFO Loss: Drop share_component_size | 4.5988e-01 | 1.0000e+00 | NO |
| 34 | 3x | Control (d vs b): Plain Aggregates vs Tabular (PR) | 4.8370e-01 | 1.0000e+00 | NO |
| 35 | 2x | Full (c vs e): Graph vs Plain Pairwise Counts (ROC) | 5.8628e-01 | 1.0000e+00 | NO |
| 36 | 1x | LOFO Loss: Drop seller_buyer_degree | 7.4334e-01 | 1.0000e+00 | NO |

## 6. Practical Significance: Fraud Value Caught (INR) & Recall at Review Budgets (20 Seeds)

In operational trust & safety operations, manual review capacity is constrained by investigator budgets (e.g. 2%, 5%, 10% of order volume).

| Review Budget | Model Variant | Fraud Recall (mean +/- std) | Paired Recall Lift | Paired Recall p-value | Fraud Value Caught (INR, mean) | Paired Value Lift (INR, mean) | Paired Value p-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2%** | Tabular Baseline (b) | 18.96% +/- 0.90% | baseline | — | INR 317,916 | baseline | — |
| | Tabular + Graph (c) | 18.90% +/- 0.85% | -0.06% | p = 3.6480e-01 | INR 317,124 | -792 | p = 9.3051e-01 |
| **5%** | Tabular Baseline (b) | 35.21% +/- 1.87% | baseline | — | INR 725,484 | baseline | — |
| | Tabular + Graph (c) | 35.30% +/- 1.80% | +0.09% | p = 4.6881e-01 | INR 732,427 | +6,943 | p = 2.5291e-01 |
| **10%** | Tabular Baseline (b) | 43.67% +/- 2.32% | baseline | — | INR 922,754 | baseline | — |
| | Tabular + Graph (c) | 44.48% +/- 2.85% | +0.81% | p = 4.2351e-02 | INR 945,039 | +22,285 | p = 7.7498e-02 |

> **Operational Conclusion on Review Budgets**: At tight operational review budgets (2% and 5%), graph features provide **no statistically significant lift** in either fraud recall (2%: p = 0.3648; 5%: p = 0.4688) or fraud monetary value caught (2%: p = 0.9305; 5%: p = 0.2529). Only at a relaxed 10% review budget does recall lift reach marginal significance (+0.81%, p = 0.0424).

## 7. Comprehensive Per-Type Fraud Breakdown & Formal Retraction

> **Formal Retraction of Earlier Per-Type Table**:
> The earlier per-type table reporting coordinated fraud ROC ~0.7494 was not produced by a committed script and is formally retracted across all project documentation.
> In this canonical 20-seed evaluation on the truncated test split (excluding the final 21 days for right-censoring), coordinated fraud achieves isolated ROC 0.6403 (b) vs 0.6872 (c) with only 0.68% recall at a 2% budget, and seller-buyer collusion operates strictly at chance (ROC 0.50).

| Fraud Type | Model Variant | Isolated ROC-AUC (mean +/- std) | Recall @ 2% Budget | Recall @ 5% Budget | Recall @ 10% Budget | Operational Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **fake_listing** | Tabular Baseline (b) | 0.8327 +/- 0.0186 | 42.46% | 54.71% | 62.97% | High detection via listings |
| | Tabular + Graph (c) | 0.8243 +/- 0.0152 | 42.68% | 53.54% | 60.53% | High detection via listings |
| **return_abuse** | Tabular Baseline (b) | 0.9098 +/- 0.0334 | 29.31% | 72.71% | 78.65% | High detection via history |
| | Tabular + Graph (c) | 0.9138 +/- 0.0298 | 28.75% | 72.70% | 79.14% | High detection via history |
| **coordinated_fraud** | Tabular Baseline (b) | 0.6403 +/- 0.0433 | 0.67% | 8.24% | 21.64% | Near random at 2% budget |
| | Tabular + Graph (c) | 0.6872 +/- 0.0636 | 0.68% | 9.73% | 26.69% | Near random at 2% budget |
| **seller_buyer_collusion** | Tabular Baseline (b) | 0.5001 +/- 0.0422 | 0.40% | 2.22% | 6.86% | Strictly at chance (ROC 0.50) |
| | Tabular + Graph (c) | 0.4971 +/- 0.0384 | 0.45% | 2.11% | 6.37% | Strictly at chance (ROC 0.50) |

## 8. Horizon Audit: Right-Censoring Exclusion & Git Timeline

- **Test-Period Mean Fraud Rate**: `8.85%`
- **Git Timeline & Dec-31 Order Verification**: The historical claim of '20 fraud orders on Dec-31' is not reproducible at any git commit. In pre-audit commits (`d0465b335c`), burst orders clamped to `SIM_END` producing 49 fraud orders on Dec 31 for Seed 42. Following the burst window bounds fix, Dec-31 contains 411 total orders and 7 fraud orders (1.70% fraud rate); this low final-days fraud-order rate is attributed to burst scheduling constraints (attacks require multiple days to unfold before SIM_END), not return-window truncation.
- **Exclusion Policy**: To eliminate right-censoring in returns and late-horizon boundary effects, all benchmark models strictly exclude the final 21 days (`order_date > 2025-12-10`).

### Truncated Test Horizon Benchmark (Final 21 Days Excluded for All Models):
To strictly eliminate right-censoring in returns and late-horizon burst pileup, the final 21 days (`order_date > 2025-12-10`) were excluded from the test split across all 20 seeds:
| Model Variant | Truncated Test ROC-AUC | Truncated Test PR-AUC | Paired Lift vs Baseline (b) | Two-Sided p-value |
| :--- | :---: | :---: | :---: | :---: |
| (a) Tabular (without device) | 0.7223 +/- 0.0161 | 0.4545 +/- 0.0220 | -0.0298 | — |
| (b) Tabular Baseline | 0.7521 +/- 0.0170 | 0.4775 +/- 0.0264 | baseline | — |
| (d) Plain Aggregates Control | 0.7487 +/- 0.0192 | 0.4736 +/- 0.0259 | -0.0034 | p = 1.0695e-01 |
| (c) Tabular + Graph Features | 0.7635 +/- 0.0223 | 0.4856 +/- 0.0310 | **+0.0114** | **p = 3.0153e-05** |

## 9. Capacity-Constrained Review-Budget Threshold Policy (5% Budget Selected on Val)

To simulate production operating conditions, threshold $\tau$ was chosen strictly on the validation split per seed to enforce a 5% manual review capacity constraint:
| Operational Metric | Tabular Baseline (b) | Tabular + Graph (c) | Paired Lift (c vs b) | 95% Confidence Interval | p-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Selected Threshold $\tau$ (on Val)** | 0.5673 +/- 0.0553 | 0.5794 +/- 0.0517 | +0.0121 | — | — |
| **Realized Test Review Volume** | 4.71% +/- 0.38% | 4.83% +/- 0.42% | +0.12% | — | — |
| **Realized Precision** | 68.59% +/- 3.60% | 67.70% +/- 4.19% | -0.89% | — | — |
| **Realized Fraud Recall** | 37.45% +/- 2.71% | 37.89% +/- 2.79% | +0.44% | [-0.0011, +0.0100] | p = 1.3267e-01 |
| **Realized Fraud Value Caught (INR)** | INR 424,218 | INR 428,059 | INR +3,842 | [-5,135, +12,819] | p = 4.1202e-01 |

> **Threshold Probability Scale Reconciliation**:
> - **Calibrated Probability Scale** ($P(\text{fraud})$): The validation 5% budget threshold is $\tau = 0.5673 \pm 0.0553$ (baseline b) and $0.5794 \pm 0.0517$ (variant c). Fixed thresholding at $\tau = 0.50$ achieves **31.83% +/- 3.69% test recall**.
> - **Raw Classifier Score Scale**: On uncalibrated XGBoost outputs, $\tau \approx 0.15$ captures a 5% budget, and thresholding at $\tau = 0.50$ achieves **40.22% +/- 3.30% test recall**.

### Threshold Evaluation Grid: Volume% and Recall% across Fixed $\tau$

Evaluated on the exact same test split for uncalibrated raw XGBoost scores vs isotonically calibrated probabilities ($P(\text{fraud})$):

#### Seed 42:
| Threshold $\tau$ | Raw Volume % | Raw Recall % | Calibrated Volume % | Calibrated Recall % | Monotonicity Check |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.050 | 76.74% | 87.94% | 16.20% | 58.34% | Non-increasing |
| 0.100 | 48.07% | 75.64% | 9.87% | 51.28% | Non-increasing |
| 0.150 | 31.16% | 66.63% | 6.25% | 44.46% | Non-increasing |
| 0.250 | 15.34% | 57.61% | 4.85% | 39.71% | Non-increasing |
| 0.350 | 9.66% | 51.04% | 4.36% | 37.27% | Non-increasing |
| 0.500 | 5.92% | 44.09% | 2.97% | 28.14% | Non-increasing |
| 0.567 | 5.08% | 40.80% | 2.90% | 27.41% | Non-increasing |
| 0.650 | 4.37% | 37.27% | 2.15% | 20.83% | Non-increasing |

#### Mean over 20 Seeds:
| Threshold $\tau$ | Raw Volume % (mean +/- std) | Raw Recall % (mean +/- std) | Calibrated Volume % (mean +/- std) | Calibrated Recall % (mean +/- std) | Monotonicity Check |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.050 | 65.30% +/- 8.32% | 83.66% +/- 4.78% | 18.30% +/- 5.51% | 55.58% +/- 5.53% | Non-increasing |
| 0.100 | 38.38% +/- 6.75% | 69.01% +/- 4.63% | 8.27% +/- 1.32% | 45.11% +/- 2.83% | Non-increasing |
| 0.150 | 24.45% +/- 4.50% | 60.11% +/- 4.11% | 5.85% +/- 0.76% | 40.66% +/- 2.59% | Non-increasing |
| 0.250 | 12.54% +/- 2.10% | 50.30% +/- 3.54% | 4.68% +/- 0.49% | 37.63% +/- 2.77% | Non-increasing |
| 0.350 | 7.98% +/- 1.14% | 44.52% +/- 2.90% | 3.95% +/- 0.41% | 34.85% +/- 3.04% | Non-increasing |
| 0.500 | 5.28% +/- 0.51% | 39.22% +/- 2.91% | 3.42% +/- 0.44% | 31.95% +/- 3.74% | Non-increasing |
| 0.567 | 4.69% +/- 0.43% | 37.58% +/- 2.95% | 3.25% +/- 0.48% | 30.91% +/- 3.85% | Non-increasing |
| 0.650 | 4.15% +/- 0.42% | 35.65% +/- 3.25% | 2.95% +/- 0.50% | 28.49% +/- 4.21% | Non-increasing |

> **Resolution of Prior Threshold Contradiction**:
> - Recall is strictly non-increasing across all $\tau$ thresholds on both raw and calibrated scales.
> - At $\tau = 0.50$, recall is **39.22% +/- 2.91%** on raw XGBoost scores and **31.95% +/- 3.74%** on calibrated probabilities (Seed 42: 44.09% raw, 28.14% calibrated).
> - Neither scale yields '<1% recall' at $\tau = 0.50$; the prior claim is formally retracted.

## 10. Fraud Ring Detection Recovery vs Baselines

> **Retraction Notice**: RETRACTION: The previously reported figures (Precision 76.2%, Recall 48.9%, F1 0.595) were not produced by a committed script and are formally retracted across all project documentation. The reproducible figures produced by scripts/run_remediation_addendum_audit.py are Precision 89.25%, Recall 8.30%, F1 0.152.

- **Unit Definition (Member-level)**: Evaluates precision, recall, and F1 over individual buyer nodes belonging to ground-truth rings
- **Unit Definition (Ring-level)**: Evaluates recovery of ground-truth ring clusters (defined as detecting >= 50% of the ring's member nodes)
- **Random Partition Baseline Definition**: Partitions the sharing graph into uniform random clusters whose sizes match the ground-truth rings.
- **Key Finding**: Naive connected components beat the TrustShield risk filter in ring recovery for return_abuse (54.09% vs 45.82%) and coordinated_fraud (60.26% vs 4.39%), because low isolated transaction risk scores in coordinated attacks prevent the average cluster risk from crossing the >= 0.50 operational threshold.

| Detector Method | Member-level Precision | Member-level Recall | Member-level F1 | Ring-level Recovery Rate |
| :--- | :---: | :---: | :---: | :---: |
| All Connected Components (size >= 2, no risk filter) | 37.40% | 36.42% | 0.3688 | 46.14% |
| Random Partition Baseline | 17.55% | 17.14% | 0.1734 | 23.16% |
| TrustShield High-Risk Filter (risk >= 0.50) | 89.25% | 8.30% | 0.1517 | 15.07% |

### Ring Recovery Breakdown by Fraud Type:
| Fraud Type | In Sharing Log | Single CC Share | Pure CC Recovery | Random Baseline | TrustShield (>=0.50) | Diagnostic Explanation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`return_abuse`** | 100% | 50.0% | 54.09% | 22.14% | 45.82% | High transaction risk scores allow high-risk threshold filter recovery |
| **`coordinated_fraud`** | 100% | 47.2% | 60.26% | 31.31% | 4.39% | Members connect in graph (60.26% CC recovery) but low transaction risk (~0.68 isolated ROC) drops TrustShield recovery to 4.39% |
| **`fake_listing`** | N/A | N/A | N/A | N/A | N/A | Not applicable (seller/listing perturbation, not buyer-sharing ring) |
| **`seller_buyer_collusion`** | N/A | N/A | N/A | N/A | N/A | Not applicable (bipartite transaction bursts with distinct pairs, not buyer-sharing ring) |

## 11. Conformal Prediction Audit & Mondrian Class-Conditional Evaluation (20 Seeds)

- **Nominal Target Error Rate**: $\alpha = 0.05$ (Nominal Target Coverage: 95.0%)
- **Mondrian Overall Test Coverage**: 95.12% +/- 0.58%
- **Mondrian Legit (Y=0) Test Coverage**: 95.30% +/- 0.58%
- **Mondrian Fraud (Y=1) Test Coverage**: 93.14% +/- 2.15%
- **Mondrian Mean Prediction Set Size**: 1.7430 +/- 0.0464 (abstains on ambiguous scores, returning {0, 1})
- **Diagnostic Characterization**: A mean prediction set size of 1.74 indicates abstention on most orders, and fraud coverage (93.14% < 95.0%) falls short of nominal coverage; this mechanism is an offline diagnostic rather than an active product feature.
- **Marginal Calibration Audit (Legacy)**: Overall 93.55%, but severe minority under-coverage on fraud (Y=1: 43.24%) with set size 1.02 (almost never abstaining).

## 12. Exploratory Diagnostic: Point-in-Time Order Counts and Generator Collusion Properties

> **Measured Topological Fact**:
> `buyer_seller_edge_weight_before` is an order-level point-in-time cumulative count (`cumcount()`), NOT a monthly snapshot. Only **1.22% +/- 0.51%** of collusion orders share a buyer-seller pair with an earlier burst order (the generator samples each pair once by construction). Consequently, collusion has no detectable signal in `buyer_seller_edge_weight_before` by construction.

## 13. What This Does NOT Show

To maintain scientific honesty and prevent over-interpretation of experimental results:

1. **Does NOT show GNN superiority over gradient boosted trees in the tested configuration (16-dim OOF GraphSAGE embeddings into XGBoost):** Integrating out-of-fold GNN embeddings into XGBoost results in net negative lift (-0.0263 ROC-AUC, p = 0.0083). Tabular trees with point-in-time graph features remain superior.
2. **Does NOT show double-digit graph lifts:** On honest point-in-time temporal holdouts, true graph lift is modest (+0.0114 ROC-AUC, +0.0081 PR-AUC). Historical reports claiming double-digit lifts were caused by the generator first_seen timestamp bug or unadjusted baselines.
3. **Does NOT show that a 0.50 threshold yields <1% recall:** Derived strictly from the threshold grid evaluation, thresholding at 0.50 yields **31.95% +/- 3.74% recall** on calibrated probabilities and **39.22% +/- 2.91% recall** on raw XGBoost scores across 20 seeds (Seed 42: 28.14% calibrated, 44.09% raw). Recall is strictly non-increasing across all thresholds and scales. Neither scale yields <1% recall.
4. **Does NOT show zero out-of-sample calibration error:** Out-of-sample test ECE is strictly non-zero (~0.021 - 0.023 calibrated, ~0.054 - 0.063 raw), even though in-sample isotonic validation achieves 0.0000.
5. **Does NOT show significant practical lift at operational review budgets:** At 2% and 5% review budgets, graph features show no statistically significant lift in recall (2%: p = 0.3648; 5%: p = 0.4688) or fraud value caught (2%: p = 0.9305; 5%: p = 0.2529).
6. **Does NOT show detection of seller-buyer collusion or low-budget coordinated rings:** Seller-buyer collusion discrimination is strictly at chance (isolated ROC 0.5001 vs 0.4971) because the synthetic generator uses each buyer-seller pair once by construction. Coordinated fraud recall at a 2% review budget is near random (0.68%).

<!-- END GENERATED -->
