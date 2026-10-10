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
- **`buyer_pagerank` vs `buyer_orders_before`**: Pearson r = 0.8430 +/- 0.0357 | Spearman rho = 0.7562 +/- 0.0258
- **`seller_pagerank` vs seller volume**: Pearson r = 0.7994 +/- 0.0902 | Spearman rho = 0.6997 +/- 0.0514

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
| **Exploratory (5 seeds)** | Seeds 42, 101, 202, 303, 404 | 0.7276 +/- 0.0296 | 0.7399 +/- 0.0315 | +0.0123 | [-0.0047, +0.0293] | p = 1.1453e-01 |
| **Confirmatory Holdout (15 seeds)** | Seeds 505 through 1910 | 0.7301 +/- 0.0160 | 0.7414 +/- 0.0206 | **+0.0112** | **[+0.0064, +0.0161]** | **p = 2.1478e-04** |
| **Full Sample (20 seeds)** | All 20 Seeds Combined | 0.7295 +/- 0.0193 | 0.7410 +/- 0.0229 | **+0.0115** | **[+0.0069, +0.0161]** | **p = 4.7314e-05** |

## 5. Full Family Step-Down Holm-Bonferroni Correction

Controlling family-wise error rate across all exploratory comparisons in the remediation suite:

| Rank | Test Description | Raw p-value | Multiplier (m - k + 1) | Holm-Adjusted p-value | Significant at alpha = 0.05 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | Group Ablation: Tabular + S+B vs Tabular (ROC) | 3.7505e-06 | 27 | 1.0126e-04 | **YES** |
| 2 | Group Ablation: Tabular + S+E vs Tabular (ROC) | 8.6438e-06 | 26 | 2.2474e-04 | **YES** |
| 3 | Control (c vs d): Graph vs Plain Aggregates (ROC) | 4.6636e-05 | 25 | 1.1659e-03 | **YES** |
| 4 | Group Ablation: Tabular + S vs Tabular (ROC) | 1.3257e-04 | 24 | 3.1817e-03 | **YES** |
| 5 | Control (c vs d): Graph vs Plain Aggregates (PR) | 1.2662e-03 | 23 | 2.9123e-02 | **YES** |
| 6 | Hybrid vs Tabular (Coherent N=5) | 3.9460e-03 | 22 | 8.6812e-02 | No |
| 7 | Hybrid vs Tabular (Standard N=5) | 8.3154e-03 | 21 | 1.7462e-01 | No |
| 8 | Group Ablation: Tabular + E vs Tabular (ROC) | 8.5518e-03 | 20 | 1.7462e-01 | No |
| 9 | Per-Type Graph Lift: Coordinated Ring (ROC) | 2.1000e-02 | 19 | 3.9900e-01 | No |
| 10 | Group Ablation: Tabular + B vs Tabular (ROC) | 3.1829e-02 | 18 | 5.7293e-01 | No |
| 11 | Tabular+Graph vs Tabular (Coherent Upper Bound N=5) | 3.4045e-02 | 17 | 5.7876e-01 | No |
| 12 | LOFO Loss: Drop buyer_pagerank | 4.5416e-02 | 16 | 7.2666e-01 | No |
| 13 | Return Detector Ablation: Drop days_to_return | 4.8000e-02 | 15 | 7.2666e-01 | No |
| 14 | Return Detector Ablation: Drop both rules | 5.1000e-02 | 14 | 7.2666e-01 | No |
| 15 | LOFO Loss: Drop seller_buyer_concentration_hhi | 1.0023e-01 | 13 | 1.0000e+00 | No |
| 16 | Per-Type Graph Lift: Fake Listing (ROC) | 1.8400e-01 | 12 | 1.0000e+00 | No |
| 17 | Group Ablation: Tabular + B+E vs Tabular (ROC) | 1.8628e-01 | 11 | 1.0000e+00 | No |
| 18 | LOFO Loss: Drop share_degree | 2.3811e-01 | 10 | 1.0000e+00 | No |
| 19 | Control (d vs b): Plain Aggregates vs Tabular (ROC) | 2.7292e-01 | 9 | 1.0000e+00 | No |
| 20 | Per-Type Graph Lift: Return Abuse (ROC) | 3.4200e-01 | 8 | 1.0000e+00 | No |
| 21 | LOFO Loss: Drop buyer_seller_degree | 3.7037e-01 | 7 | 1.0000e+00 | No |
| 22 | LOFO Loss: Drop seller_pagerank | 3.7401e-01 | 6 | 1.0000e+00 | No |
| 23 | LOFO Loss: Drop buyer_seller_edge_weight_before | 3.8836e-01 | 5 | 1.0000e+00 | No |
| 24 | LOFO Loss: Drop share_component_size | 4.5988e-01 | 4 | 1.0000e+00 | No |
| 25 | Control (d vs b): Plain Aggregates vs Tabular (PR) | 4.8370e-01 | 3 | 1.0000e+00 | No |
| 26 | LOFO Loss: Drop seller_buyer_degree | 7.4334e-01 | 2 | 1.0000e+00 | No |
| 27 | Return Detector Ablation: Drop reason_* | 7.8200e-01 | 1 | 1.0000e+00 | No |

## 6. Designed-Signal Sensitivity Analysis / Upper Bound (Coherent Ring Variant)

> **Methodological Label**: `designed-signal sensitivity analysis / upper bound`

| Model Variant | ROC-AUC (mean +/- std) | PR-AUC (mean +/- std) | Paired ROC Lift | Two-Sided p-value |
| :--- | :---: | :---: | :---: | :---: |
| Tabular Baseline | 0.7935 +/- 0.0405 | 0.4941 +/- 0.0525 | baseline | — |
| Tabular + Graph Features | 0.8001 +/- 0.0380 | 0.4982 +/- 0.0503 | +0.0066 | p = 3.4045e-02 |

## 7. Tuned Hybrid GNN Model Evaluation (5 Seeds)

| Dataset Variant | Hybrid Model ROC | Tabular Baseline ROC | Paired ROC Difference | Two-Sided p-value | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Standard Dataset | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | -0.0263 | p = 8.3154e-03 | Hybrid loses to tabular XGBoost |
| Coherent Upper Bound | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | -0.1010 | p = 3.9460e-03 | Hybrid loses to tabular XGBoost |

## 8. Design-Rule Ablation Experiments

> **Scientific Characterization**: ``

### (a) Return Abuse Detector Ablation (Uncensored Returns):

| Feature Set Variant | ROC-AUC | PR-AUC |
| :--- | :---: | :---: |

### (b) Fake Listing Detector Price-Anomaly Recovery:

| Subgroup | Total Test Listings | Recall | Scientific Label |
| :--- | :---: | :---: | :--- |
| With Injected Price Anomaly | 103 | 86.41% | recovery of injected generator rules |
| Without Injected Price Anomaly | 66 | 3.03% | recovery of injected generator rules |

## 9. Real GNN Graph Topological Statistics

- **Total Nodes**: 0
- **Total Edges**: 0
- **Connected Components**: 0
- **Mean Degree**: 11.08 | **Median Degree**: 5.0 | **90th Percentile Degree**: 17.0
- **Fraction of Nodes with Degree >= 2**: 0.00%
- **Median Buyer Degree**: Fraud Orders = 0.0 | Legit Orders = 0.0
- **Median Seller Degree**: Fraud Orders = 0.0 | Legit Orders = 0.0

## 10. Raw Per-Seed Arrays for Independent Recomputation

The 20-seed metric arrays are printed below for independent verification:

```json
{
  "variant_a_tabular_no_device": {
    "roc": [
      0.6950066240104966,
      0.7432148929234919,
      0.6650381941173114,
      0.7279350465361799,
      0.683184701753004,
      0.6791760404596453,
      0.698090827118022,
      0.7059655273107472,
      0.7189146765058714,
      0.6933716392560624,
      0.693073812787229,
      0.7262565883933482,
      0.7030214006678995,
      0.7042363084244454,
      0.6847060684553388,
      0.6932805436759674,
      0.7020899298308352,
      0.7198250477648234,
      0.694326144697179,
      0.7108985704487524
    ],
    "pr": [
      0.41877874512885693,
      0.44811172910354047,
      0.4004370265569281,
      0.43746198349754484,
      0.38243184225688426,
      0.40714890194860553,
      0.41543682997680853,
      0.4249113483194743,
      0.4384922958759477,
      0.3868401799066127,
      0.4113860432348068,
      0.45864942103714434,
      0.41144228833653346,
      0.4251621934666847,
      0.39425594188450647,
      0.3979650738006509,
      0.4403022981507991,
      0.4242671914145342,
      0.4114836540840153,
      0.4234892440978955
    ]
  },
  "variant_b_tabular_with_device": {
    "roc": [
      0.7296199944965743,
      0.767208237561803,
      0.6928066603475382,
      0.7427972064803592,
      0.7057210585485987,
      0.716814752068786,
      0.7414407030081771,
      0.738774416571131,
      0.749672465269356,
      0.7096672243240219,
      0.7362289768604369,
      0.75017496008916,
      0.7256528547588164,
      0.737867980685304,
      0.7003245802266113,
      0.7094714290429076,
      0.7235519475423751,
      0.7414844049660942,
      0.722780120607763,
      0.7480492640901684
    ],
    "pr": [
      0.4577401305252542,
      0.4715700322522541,
      0.4084862892535184,
      0.4547063567931561,
      0.4004616819904254,
      0.4363339632459215,
      0.44925973868925806,
      0.44829924935642806,
      0.46175257210246373,
      0.3986807477288581,
      0.43582212520848096,
      0.47682808157626444,
      0.42919980755537196,
      0.44572494771250176,
      0.40329936570008024,
      0.40669113099111026,
      0.45655843320805317,
      0.4450769499301337,
      0.4378672924134729,
      0.4467862161318596
    ]
  },
  "variant_c_tabular_plus_graph": {
    "roc": [
      0.7592617278505469,
      0.7715208497884914,
      0.6908117624972391,
      0.748445329114664,
      0.7297102940761048,
      0.7340977992012403,
      0.7611822734992584,
      0.7548391341754894,
      0.7664912450095686,
      0.7165746006237679,
      0.7552565709908894,
      0.7539608385028106,
      0.7190607281623633,
      0.7484988762940632,
      0.7114675905526767,
      0.7023703441022517,
      0.7423799015420853,
      0.7614003461140422,
      0.7338125935971516,
      0.7592858650517927
    ],
    "pr": [
      0.48248537364011296,
      0.48507326070594303,
      0.4035010989408,
      0.45304526690182134,
      0.416770857125565,
      0.438701833124245,
      0.4527022090765621,
      0.45087527978782954,
      0.4755453776263918,
      0.4036701197052667,
      0.4446608370477857,
      0.4861518401377844,
      0.4367119947720103,
      0.4345387912941151,
      0.4082297065496307,
      0.40568294969367064,
      0.47696508513592367,
      0.4578811012888229,
      0.4531721891970686,
      0.45631653027730873
    ]
  },
  "variant_d_tabular_plain_aggregates": {
    "roc": [
      0.7303098800517459,
      0.7668895914512555,
      0.6813838206019898,
      0.752688898609128,
      0.7061230988434372,
      0.7173680222914508,
      0.7287912021624663,
      0.7331823750690848,
      0.7535947239311183,
      0.7050426529007716,
      0.7389499919374345,
      0.7291486044351022,
      0.7227755212317519,
      0.7475037525360113,
      0.7048786767497952,
      0.7082261981377251,
      0.7314683858984767,
      0.7332429601163724,
      0.7227144017495035,
      0.7348511669236707
    ],
    "pr": [
      0.45630158192644804,
      0.4753387891824536,
      0.4022723963859268,
      0.45097704789650067,
      0.3974022139312658,
      0.4413242793420803,
      0.43962029285245346,
      0.4433128952433231,
      0.4659644368239658,
      0.3943554473821598,
      0.43905261723091155,
      0.4650627672716028,
      0.43951555084483473,
      0.45089040170500644,
      0.4027830053862429,
      0.4060587533375618,
      0.4572682191697966,
      0.4424555139514819,
      0.43881067227133635,
      0.4454157675971748
    ]
  }
}
```

## 11. Appendix: Leave-One-Feature-Out (LOFO) Ablation with Holm-Adjusted p-values

| Dropped Feature | Without Feature ROC | Marginal Loss vs Full (c) | 95% CI of Loss | Raw p-value | Holm-Adjusted p-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `share_degree` | 0.7395 +/- 0.0241 | +0.0015 | [-0.0011, +0.0042] | p = 2.3811e-01 | p_adj = 1.0000e+00 |
| `share_component_size` | 0.7420 +/- 0.0212 | -0.0010 | [-0.0038, +0.0018] | p = 4.5988e-01 | p_adj = 1.0000e+00 |
| `buyer_seller_degree` | 0.7427 +/- 0.0223 | -0.0016 | [-0.0054, +0.0021] | p = 3.7037e-01 | p_adj = 1.0000e+00 |
| `buyer_pagerank` | 0.7368 +/- 0.0226 | +0.0043 | [+0.0001, +0.0084] | p = 4.5416e-02 | p_adj = 7.2666e-01 |
| `seller_buyer_degree` | 0.7416 +/- 0.0222 | -0.0005 | [-0.0040, +0.0029] | p = 7.4334e-01 | p_adj = 1.0000e+00 |
| `seller_pagerank` | 0.7396 +/- 0.0220 | +0.0014 | [-0.0018, +0.0047] | p = 3.7401e-01 | p_adj = 1.0000e+00 |
| `seller_buyer_concentration_hhi` | 0.7436 +/- 0.0201 | -0.0026 | [-0.0057, +0.0005] | p = 1.0023e-01 | p_adj = 1.0000e+00 |
| `buyer_seller_edge_weight_before` | 0.7422 +/- 0.0206 | -0.0011 | [-0.0038, +0.0016] | p = 3.8836e-01 | p_adj = 1.0000e+00 |
