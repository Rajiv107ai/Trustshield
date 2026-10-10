# TrustShield Canonical Benchmark Results

> [!NOTE]
> Produced under strict audit remediation controls: point-in-time snapshot graphs, out-of-fold embeddings, and independent evaluations. All metrics generated in session.

- **Git Commit**: `91b0cbb4fda537d8e0528568d0730d7759342010` (dirty: `True`)
- **Timestamp**: `2026-10-10T11:44:13Z`
- **Primary Seeds ($N=20$)**: `[42, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809, 1910]`

## 1. Pre-Registered Primary Analysis (Standard Dataset, $N=20$ Seeds)

**Pre-Registered Primary Hypothesis**: On the Standard dataset, Tabular+Graph vs Tabular (with device) improves ROC-AUC and PR-AUC.

| Model / Variant | Feature Set | Test ROC-AUC (mean ± std) | Test PR-AUC (mean ± std) | Paired Lift vs Tabular (with device) [95% CI] | $p$-value |
|---|---|---|---|---|---|
| **(a) Tabular (No Device)** | 9 Tabular (no device count) | 0.7021 ± 0.0186 | 0.4179 ± 0.0202 | — | — |
| **(b) Tabular (With Device)** | 10 Tabular (current baseline) | 0.7295 ± 0.0193 | 0.4386 ± 0.0238 | Baseline | Baseline |
| **(c) Tabular + Graph (Primary)** | 10 Tabular + 8 Graph Features | **0.7410 ± 0.0229** | **0.4461 ± 0.0276** | **+0.0115** [+0.0069, +0.0161] | **$p=0.0000$** |

- **Primary ROC-AUC Lift ($c$ vs $b$)**: `+0.0115` (95% CI `[+0.0069, +0.0161]`, Cohen's $d_z = 1.170$, $p=0.0000$)
- **Primary PR-AUC Lift ($c$ vs $b$)**: `+0.0076` (95% CI `[+0.0035, +0.0117]`, Cohen's $d_z = 0.865$, $p=0.0010$)
- **Device Feature Lift ($b$ vs $a$)**: ROC `+0.0274` ($p=0.0000$) | PR `+0.0206` ($p=0.0000$)
- **Graph Lift vs Device-Free Tabular ($c$ vs $a$)**: ROC `+0.0389` ($p=0.0000$) | PR `+0.0282` ($p=0.0000$)

## 2. Sensitivity Analysis / Upper Bound (Coherent Variant, $N=5$ Seeds)

> [!NOTE]
> **Label**: Designed-signal sensitivity analysis / upper bound. This variant guarantees 100% causal device sharing for ring bursts. It serves as an upper bound on synthetic graph signal, not the headline benchmark.

| Model | Variant | Test ROC-AUC (mean ± std) | Test PR-AUC (mean ± std) | Lift vs Tabular [95% CI] | $p$-value |
|---|---|---|---|---|---|
| **Tabular XGBoost** | Coherent (Upper Bound) | 0.7935 ± 0.0405 | 0.4941 ± 0.0525 | — | — |
| **Tabular + Graph** | Coherent (Upper Bound) | 0.8001 ± 0.0380 | 0.4982 ± 0.0503 | +0.0066 [+0.0008, +0.0124] | $p=0.0340$ |

## 3. Tuned Hybrid (GraphSAGE + XGBoost with Validation-Selected Config)

| Model | Variant | Test ROC-AUC (mean ± std) | Test PR-AUC (mean ± std) | Paired Diff vs Tabular | $p$-value |
|---|---|---|---|---|---|
| **Tuned Hybrid (OOF GNN)** | Standard ($N=5$) | 0.7014 ± 0.0387 | 0.4006 ± 0.0464 | -0.0263 | $p=0.0083$ |
| **Tuned Hybrid (OOF GNN)** | Coherent ($N=5$) | 0.6925 ± 0.0449 | 0.3965 ± 0.0409 | -0.1010 | $p=0.0039$ |

*Honest Empirical Conclusion*: Even with learning rate selected per seed strictly on validation order-level fraud AUC ([0.01, 0.005, 0.01, 0.01, 0.01]) and validation early stopping, the hybrid model underperforms tabular XGBoost on both Standard (-0.0263 ROC-AUC, $p=0.0083$) and Coherent (-0.1010 ROC-AUC, $p=0.0039$) datasets.

## 4. Real GNN Graph Topological Statistics

Statistics of the actual graph GraphSAGE consumes (Buyer-Seller transaction edges + Buyer-Buyer sharing edges):

| Metric | Measured Value | Interpretation |
|---|---|---|
| **Total Nodes ($V$)** | 4031 | Unique Buyers + Sellers in train split |
| **Total Edges ($E$)** | 22330 | Bipartite transactions + shared device/address edges |
| **Degree Mean / Median / p90** | 11.08 / 5.0 / 17.0 | Highly skewed degree distribution |
| **Fraction of Nodes with Degree $\ge 2$** | 91.39% | Proportion of nodes with multiple neighbors |
| **Connected Components** | 3 | High structural fragmentation across the market |
| **Median Buyer Degree (Fraud vs Legit Orders)** | 8.0 vs 8.0 | Order-level degree signature |
| **Median Seller Degree (Fraud vs Legit Orders)** | 122.0 vs 110.0 | Order-level degree signature |

*Graph Topology Assessment*: The actual graph consumed by GraphSAGE consists of bipartite order links plus device/address sharing links. With 3 connected components and median buyer degree of 5.0, message passing operates on disconnected subgraphs. Tabular tree models directly consuming summary degree and PageRank features capture local topological signals without diffusion noise.

## 5. Exploratory Multiple Testing Analysis (Holm-Bonferroni Correction)

| Exploratory Comparison | Raw $p$-value | Holm-Adjusted $p$-value | Significant at $lpha=0.05$ |
|---|---|---|---|
| `device_feature_lift_roc (b vs a)` | 0.0000 | 0.0000 | YES |
| `device_feature_lift_pr (b vs a)` | 0.0000 | 0.0000 | YES |
| `graph_lift_vs_no_device_roc (c vs a)` | 0.0000 | 0.0000 | YES |
| `graph_lift_vs_no_device_pr (c vs a)` | 0.0000 | 0.0000 | YES |
| `isolated_auc_coordinated_fraud (standard)` | 0.0000 | 0.0000 | YES |
| `isolated_auc_fake_listing (standard)` | 0.0001 | 0.0009 | YES |
| `hybrid_vs_tabular_roc (coherent)` | 0.0039 | 0.0474 | YES |
| `hybrid_vs_tabular_pr (standard)` | 0.0061 | 0.0676 | NO |
| `hybrid_vs_tabular_roc (standard)` | 0.0083 | 0.0832 | NO |
| `hybrid_vs_tabular_pr (coherent)` | 0.0111 | 0.0997 | NO |
| `isolated_auc_return_abuse (standard)` | 0.0220 | 0.1763 | NO |
| `isolated_auc_coordinated_fraud (coherent)` | 0.0291 | 0.2039 | NO |
| `coherent_variant_graph_lift_roc` | 0.0340 | 0.2043 | NO |
| `isolated_auc_fake_listing (coherent)` | 0.1163 | 0.5813 | NO |
| `coherent_variant_graph_lift_pr` | 0.1296 | 0.5813 | NO |
| `isolated_auc_return_abuse (coherent)` | 0.4027 | 1.0000 | NO |
| `isolated_auc_seller_buyer_collusion (standard)` | 0.5518 | 1.0000 | NO |
| `isolated_auc_seller_buyer_collusion (coherent)` | 0.7741 | 1.0000 | NO |

## 6. Per-Type Breakdown

### Standard Dataset ($N=20$ Seeds)

| Fraud Scenario | Isolated ROC-AUC (Tabular) | Isolated ROC-AUC (Graph) | Isolated ROC Lift ($p$-val) | Recall@2% (Tab / Graph) | Recall@5% (Tab / Graph) | Recall@10% (Tab / Graph) |
|---|---|---|---|---|---|---|
| **fake_listing** | 0.833 | 0.824 | -0.008 ($p=0.000$) | 0.425 / 0.427 | 0.547 / 0.535 | 0.630 / 0.605 |
| **return_abuse** | 0.910 | 0.914 | +0.004 ($p=0.022$) | 0.293 / 0.288 | 0.727 / 0.727 | 0.787 / 0.791 |
| **coordinated_fraud** | 0.640 | 0.687 | +0.047 ($p=0.000$) | 0.007 / 0.007 | 0.082 / 0.097 | 0.216 / 0.267 |
| **seller_buyer_collusion** | 0.500 | 0.497 | -0.003 ($p=0.552$) | 0.004 / 0.005 | 0.022 / 0.021 | 0.069 / 0.064 |

### Coherent Variant ($N=5$ Seeds, Designed-Signal Sensitivity Analysis / Upper Bound)

| Fraud Scenario | Isolated ROC-AUC (Tabular) | Isolated ROC-AUC (Graph) | Isolated ROC Lift ($p$-val) | Recall@2% (Tab / Graph) | Recall@5% (Tab / Graph) | Recall@10% (Tab / Graph) |
|---|---|---|---|---|---|---|
| **fake_listing** | 0.827 | 0.821 | -0.006 ($p=0.116$) | 0.416 / 0.416 | 0.509 / 0.495 | 0.598 / 0.571 |
| **return_abuse** | 0.913 | 0.911 | -0.002 ($p=0.403$) | 0.270 / 0.269 | 0.707 / 0.691 | 0.776 / 0.773 |
| **coordinated_fraud** | 0.742 | 0.774 | +0.032 ($p=0.029$) | 0.011 / 0.010 | 0.128 / 0.152 | 0.346 / 0.390 |
| **seller_buyer_collusion** | 0.673 | 0.670 | -0.004 ($p=0.774$) | 0.005 / 0.006 | 0.082 / 0.079 | 0.216 / 0.235 |

## 7. Design-Rule Ablation Experiments

### A. Return Fraud Detector Ablations (Uncensored Test Set, $N=1,100$)
*Label: Recovery of injected generator rules*

| Ablation Configuration | Test ROC-AUC | Test PR-AUC |
|---|---|---|
| **Full Model** | 0.9530 | 0.9068 |
| **Without `days_to_return`** | 0.8067 | 0.7694 |
| **Without `reason_*` features** | 0.9537 | 0.9084 |
| **Without Both** | 0.8006 | 0.7575 |

### B. Fake Listing Detector Subgroup Recall
*Label: Recovery of injected generator rules*

- Test Fake Listings with Price Anomaly ($N=103$): **Recall = 86.41%**
- Test Fake Listings without Price Anomaly ($N=66$): **Recall = 3.03%**

## 8. Dec-31 Deterministic Counts per Seed

Counts of total orders and fraud orders on the final simulation date (`2025-12-31`) across data seeds:

| Seed | Total Orders (Dec-31) | Fraud Orders (Dec-31) | Fraud Rate (Dec-31) |
|---|---|---|---|
| Seed 42 | 411 | 7 | 1.70% |
| Seed 101 | 475 | 12 | 2.53% |
| Seed 202 | 434 | 11 | 2.53% |
| Seed 303 | 542 | 13 | 2.40% |
| Seed 404 | 400 | 4 | 1.00% |
| Seed 505 | 486 | 8 | 1.65% |
| Seed 606 | 462 | 15 | 3.25% |
| Seed 707 | 460 | 8 | 1.74% |
| Seed 808 | 497 | 14 | 2.82% |
| Seed 909 | 389 | 9 | 2.31% |
| Seed 1001 | 489 | 12 | 2.45% |
| Seed 1102 | 431 | 9 | 2.09% |
| Seed 1203 | 434 | 8 | 1.84% |
| Seed 1304 | 450 | 9 | 2.00% |
| Seed 1405 | 413 | 8 | 1.94% |
| Seed 1506 | 491 | 10 | 2.04% |
| Seed 1607 | 468 | 17 | 3.63% |
| Seed 1708 | 466 | 12 | 2.58% |
| Seed 1809 | 464 | 10 | 2.16% |
| Seed 1910 | 453 | 7 | 1.55% |
