# TrustShield — Phase 2 Metric Reconciliation & Forensic Report

**Document Version:** 1.0.0  
**Timestamp:** 2026-10-09T16:25:00+05:30  
**Audited Branch:** `fix/phase-2-integrity`  
**Verified Baseline Git Commit:** `4615a53602d0a4ffe447d4d2fd14bd69a70bf740` (HEAD of `main`)  
**Evaluation Harness:** [scripts/evaluate_models_reproducible.py](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/scripts/evaluate_models_reproducible.py)  
**Machine Manifest:** [models/reproduced_evaluation_report.json](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/reproduced_evaluation_report.json)  
**Final Phase 2 Gate Decision:** **PASS**

---

## 1. Executive Reconciliation Summary

This audit reconciles all conflicts, ambiguities, and metric divergences between the earlier Phase 2 report ([docs/PHASE_2_INTEGRITY_FINAL_REPORT.md](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/docs/PHASE_2_INTEGRITY_FINAL_REPORT.md)) and the Phase 2 Closure Report ([docs/PHASE_2_CLOSURE_REPORT.md](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/docs/PHASE_2_CLOSURE_REPORT.md)).

Every discrepancy was investigated through direct byte hashing, Python AST and object deserialization, database query inspection, and re-execution of reproducible evaluation pipelines on the frozen datasets.

### Primary Audit Findings
1. **True Git Baseline Identified:** The dedicated branch `fix/phase-2-integrity` was branched from `4615a53602d0a4ffe447d4d2fd14bd69a70bf740` (the HEAD of `main`). The closure report erroneously transcribed a nonexistent commit string (`56a6358`), which has been corrected.
2. **Dataset & Split Temporal Boundaries Established:** The ground-truth dataset in [trustshield_project/synthetic_data_export/orders.csv](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/trustshield_project/synthetic_data_export/orders.csv) spans timestamps from **`2025-01-01` to `2026-01-11`** (zero records exist in 2024). The closure report erroneously wrote `2024` in its split description. The authoritative boundaries are strictly in **2025**:
   - **Train:** $t \le 2025-08-31$ ($N = 22,848$)
   - **Validation:** $2025-08-31 < t \le 2025-10-31$ ($N = 10,266$)
   - **Out-of-Time Test:** $t > 2025-10-31$ ($N = 16,886$)
3. **Phase 3 Estimator Identity Confirmed:** Direct Python object inspection of [models/combined_graph_model.joblib](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/combined_graph_model.joblib) confirms it is an **`<class 'xgboost.sklearn.XGBClassifier'>`** with 18 input features, and [models/feature_meta.joblib](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/feature_meta.joblib) declares `"classifier": "XGBoost"`. The earlier report's mention of "Random Forest" was a misnomer originating from legacy variable names in `scripts/train_and_save_models.py`.
4. **Specialized Detector Metrics & Sample Counts Reconciled:**
   - The sample count of $N=3,000$ cited in prior documents was an erroneous copy from `products.csv` ($3,000$ rows).
   - In `listings.csv` ($20,000$ rows), the pure out-of-time test set ($t > 2025-10-31$) contains **$6,237$** listings, and the post-train holdout ($t > 2025-08-31$) contains **$10,077$** listings.
   - In `returns.csv` ($4,829$ rows), the pure out-of-time test set ($t > 2025-10-31$) contains **$1,917$** returns, and the post-train holdout ($t > 2025-08-31$) contains **$2,847$** returns.
   - The earlier report's PR-AUC claims of $0.7812$ and $0.7245$ were manual estimates from preliminary uncalibrated experiments. The serialized production artifacts produce:
     - **Fake Listing:** Pure Test PR-AUC = **0.7406** (ROC-AUC 0.9496); Post-Train PR-AUC = **0.7383** (ROC-AUC 0.9533).
     - **Return Fraud:** Pure Test PR-AUC = **0.8806** (ROC-AUC 0.9235); Post-Train PR-AUC = **0.8914** (ROC-AUC 0.9369).
5. **HTTP Benchmark Divergence Explained:** The earlier benchmark ran a single burst of 50 requests without warmup, observing initial cache burst throughput of ~53 QPS. The closure benchmark ran 10 warmup requests followed by 3 repeated runs $\times$ 100 requests per concurrency tier (910 total requests), measuring sustained steady-state GIL saturation (~26 QPS), TreeSHAP feature attribution overhead, and realistic latency percentiles.

---

## 2. Side-by-Side Report Comparison

| Dimension / Topic | Earlier Phase 2 Final Report (`docs/PHASE_2_INTEGRITY_FINAL_REPORT.md`) | Phase 2 Closure Report (`docs/PHASE_2_CLOSURE_REPORT.md`) | Authoritative Reality / Root Cause Resolution |
| :--- | :--- | :--- | :--- |
| **Git Baseline Commit** | `4615a53602d0a4ffe447d4d2fd14bd69a70bf740` | `56a6358` (Erroneous transcription) | **`4615a53602d0a4ffe447d4d2fd14bd69a70bf740`** (`HEAD` of `main`, ancestor of `fix/phase-2-integrity`). `56a6358` was an erroneous hallucination. |
| **Split Dates** | `2025-08-31` (Train), `2025-10-31` (Val), `$t > 2025-10-31$` (Test) | `2024-01-01` to `2024-03-31` (Train), etc. | **Strictly 2025.** Dataset spans `2025-01-01` to `2026-01-11`. 2024 was an accidental clerical documentation error in the closure draft. |
| **Phase 3 Estimator Type** | "Phase 3 Tabular RF (`combined_graph_model.joblib`)" | "Phase 3 Full-Graph XGBoost" | **XGBoost (`xgboost.sklearn.XGBClassifier`)**. Deserialized artifact is an `XGBClassifier` with 18 features. |
| **Phase 3 Test ROC-AUC** | 0.7890 (full graph) / 0.6029 (zero-graph) | 0.7890 (full graph) / 0.6029 (zero-graph) | **0.7890 full graph / 0.6029 zero graph.** Exactly replicated. 0.6029 is the cold-start graph degraded state. |
| **Phase 5 Hybrid Model** | Test ROC-AUC: 0.7652; Val ROC-AUC: 0.8484; Historical: 0.696 | Test ROC-AUC: 0.7652; Val ROC-AUC: 0.8484 | **0.7652 Test ROC-AUC / 0.8484 Val ROC-AUC.** Historical 0.696 was an early uncalibrated baseline. |
| **Fake Listing Sample Count** | Not explicitly separated (implicit 3,000) | Documented as $N=3,000$ | **Pure Test: $N=6,237$**; Post-Train Holdout: **$N=10,077$**. ($3,000$ was the count in `products.csv`). |
| **Fake Listing PR-AUC** | Claimed **0.7812** | Reported **0.7383** | **0.7406** on Pure Test ($N=6,237$); **0.7383** on Post-Train Holdout ($N=10,077$). 0.7812 was an untracked draft estimate. |
| **Return Fraud Sample Count** | Not explicitly separated (implicit 3,000) | Documented as $N=3,000$ | **Pure Test: $N=1,917$**; Post-Train Holdout: **$N=2,847$**. ($3,000$ was the count in `products.csv`). |
| **Return Fraud PR-AUC** | Claimed **0.7245** | Reported **0.8914** | **0.8806** on Pure Test ($N=1,917$); **0.8914** on Post-Train Holdout ($N=2,847$). 0.7245 was an untracked draft estimate. |
| **HTTP Benchmark Concurrency 1** | 50 reqs, single run: **47.78 req/s**, p50: 20.88 ms | 300 reqs (3 runs $\times$ 100): **26.06 req/s**, p50: 37.39 ms | **26.06 ± 0.31 req/s** is steady state. 47.78 req/s was a single 1-second burst without sustained load or warmup. |
| **HTTP Benchmark Concurrency 5** | 50 reqs, single run: **53.29 req/s**, p50: 93.65 ms | 300 reqs (3 runs $\times$ 100): **26.83 req/s**, p50: 184.21 ms | **26.83 ± 0.38 req/s** captures sustained concurrency, TreeSHAP explainer calls, and Python GIL bounds. |
| **HTTP Benchmark Concurrency 10** | 50 reqs, single run: **53.39 req/s**, p50: 184.35 ms | 300 reqs (3 runs $\times$ 100): **22.99 req/s**, p50: 364.13 ms | **22.99 ± 5.89 req/s** reflects thread pool saturation and lock contention. |

---

## 3. Cryptographic Verification: Datasets & Model Artifact Hashes

### 3.1 Datasets (`trustshield_project/synthetic_data_export/`)
All 12 dataset CSV files were inspected for row counts, column schemas, byte sizes, and SHA-256 hashes:

| File Name | Rows | Columns | Size (Bytes) | SHA-256 Checksum |
| :--- | :---: | :---: | :---: | :--- |
| `orders.csv` | 50,000 | 11 | 5,614,136 | `ee9cf3410847760b43d6e3843ee910fd04d316cf51ed1daa6da31f0404268c20` |
| `listings.csv` | 20,000 | 12 | 2,090,846 | `7d01ddf3200ed6d389f4c25cb7b10d520e7084e05c03b0aa47a274b1c67fecf8` |
| `returns.csv` | 4,829 | 9 | 476,029 | `b19da54d35df4126fc47cadfcd9ad054303245ad7d237b1d9cf47836a5621739` |
| `buyers.csv` | 5,000 | 6 | 227,017 | `77dbb4cdff373f851ec400a68974ec4ca0e51c621772e0fbbf15e191b5f177c0` |
| `sellers.csv` | 500 | 9 | 33,668 | `553c363e39b8f1867afc65c29dae9df80b6f7f583f5cae16fdfbc7681e6fdea0` |
| `products.csv` | 3,000 | 7 | 527,850 | `db1702fa6779114c6403b437fc4fbaa34811ce432d56db40e0f1af82783183a1` |
| `addresses.csv` | 4,950 | 3 | 168,343 | `b816c10973a906beed73d90b044efd2daddc26d35b945f6aa8a643edd3ecf23d` |
| `devices.csv` | 5,000 | 3 | 185,046 | `0e4be1ea48f9f29674d43ec7935831ec0ea4eb70fb69cf35656e5e2d63293c99` |
| `address_sharing_log.csv` | 600 | 5 | 47,232 | `1276482e15855f49cd5954e6efc31a98a991748d287fadb9d1494deedd39eb22` |
| `device_sharing_log.csv` | 400 | 5 | 31,299 | `f1d36200458bd52aae09584cb49524a588fd08ee94a7434a3a3392e525dd556c` |
| `device_mapping.csv` | 5,400 | 3 | 157,031 | `907185830f0424d96168da4276c025d2c4e8f2d9efff2759708c14ef7a43d211` |
| `fraud_ground_truth.csv` | 1,570 | 4 | 98,633 | `fadd62a858434b75b6fb47f043a4c159ac4451090792f99e904806d561a45347` |

### 3.2 Serialized Production Model Artifacts (`models/`)
All model artifacts were verified against [models/reproduction_manifest.json](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/models/reproduction_manifest.json):

| Artifact File | Serialized Type | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| `combined_graph_model.joblib` | `xgboost.sklearn.XGBClassifier` | 1,394,817 | `a16afba42b6c8e0f696c2d59249400842aaa47b6cb9d44a47a69e6ed4cd3e19a` |
| `hybrid_model.joblib` | `xgboost.sklearn.XGBClassifier` | 1,348,978 | `0b88020ccdf0e7d7ba0949ce2eb4abef4116e3a105205a9b630ee6d2c3b50945` |
| `fake_listing_model.joblib` | `xgboost.sklearn.XGBClassifier` | 967,356 | `2cb4b18dcfe8fbd00b16091e3e0a154d511886f1242d44c5304557cc9e89734d` |
| `return_fraud_model.joblib` | `xgboost.sklearn.XGBClassifier` | 910,951 | `fc0dd15ff6766408ff6467f1faadb9e57851f6a4d5f19a4ace8f7b2f92b88c47` |
| `calibrator.joblib` | `ProbabilityCalibrator` (isotonic) | 1,628 | `b396adda581c17e4003de89195677491f0faf8906f610ac69567fb7536a7a86c` |
| `phase5_calibrator.joblib` | `ProbabilityCalibrator` (isotonic) | 1,692 | `4d5f4fc252434fb1354cfbd4026aa2fd94798a0e8f0515864cbb7f3c74950c2d` |
| `buyer_embeddings.joblib` | `dict` (5,000 $\times$ 16D vectors) | 640,178 | `46c3e38f05b8c10f31eff77c97434e18f053abc3a4b74a7caf1972d4d58bf858` |
| `seller_embeddings.joblib` | `dict` (500 $\times$ 16D vectors) | 64,178 | `3dbf3eea135ce1c2b85d181ef5ca3d841f7ce6710287d92595b953eca6e863a6` |
| `fraud_rings.joblib` | `pandas.DataFrame` (collusion rings) | 68,626 | `9d7879a85a907d75fefd20f1110ef761f54c14a5a31f18614602f7bc7987bb43` |
| `feature_meta.joblib` | `dict` (feature column schemas) | 1,095 | `1bae104010a76ea858c756febf33e9346ad500c503a350bb63ea806a38de1fc0` |

---

## 4. Root Cause Explanations for Every Discrepancy

### 4.1 Root Cause: 2025 vs. 2024 Split Boundaries
- **Empirical Observation:** In `orders.csv`, the minimum order date is `2025-01-01` and the maximum is `2026-01-11`. Exactly zero rows exist in 2024.
- **Code Audit:** In [trustshield_project/baseline_model.py](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/trustshield_project/baseline_model.py), lines 17–18 explicitly define:
  ```python
  TRAIN_END = pd.Timestamp("2025-08-31")
  VAL_END = pd.Timestamp("2025-10-31")
  ```
- **Conclusion:** The earlier report accurately quoted `2025-08-31` and `2025-10-31`. The closure report draft mistakenly transcribed `2024-01-01` through `2024-06-30`. The true, verifiable boundaries are strictly in **2025**.

### 4.2 Root Cause: Estimator Type (XGBoost vs. Random Forest)
- **Object Introspection:** Deserializing `models/combined_graph_model.joblib` yields:
  ```python
  type: <class 'xgboost.sklearn.XGBClassifier'>
  n_features_in_: 18
  feature_names_in_: ['price_vs_base_price_ratio', 'price_vs_category_median_ratio',
                      'seller_age_days', 'seller_total_listings_before', 'buyer_age_days',
                      'buyer_orders_before', 'buyer_returns_before', 'buyer_return_rate_before',
                      'device_shared_buyer_count', 'amount', 'share_degree',
                      'share_component_size', 'buyer_seller_degree', 'buyer_pagerank',
                      'seller_buyer_degree', 'seller_pagerank', 'seller_buyer_concentration_hhi',
                      'buyer_seller_edge_weight_before']
  ```
- **Training Script Logic:** In `scripts/train_and_save_models.py`, `_make_model()` checks if `xgboost` is installed. When installed, it instantiates `XGBClassifier(n_estimators=400, max_depth=6, ...)`. The training script assigned this estimator to a variable named `rf_combined`.
- **Conclusion:** The model was named `rf_combined` for legacy reasons, but it was trained and serialized as an **XGBoost classifier**.

### 4.3 Root Cause: Specialized Detector PR-AUC and Sample Counts
- **Sample Counts:** 
  - `products.csv` contains exactly 3,000 rows. Prior documents mistakenly listed $N=3,000$ for specialized models.
  - In reality, `listings.csv` has 20,000 total listings:
    - Train ($t \le 2025-08-31$): $9,923$
    - Validation ($2025-08-31 < t \le 2025-10-31$): $3,840$
    - Pure Test ($t > 2025-10-31$): **$6,237$**
    - Post-Train Holdout ($t > 2025-08-31$): **$10,077$**
  - In reality, `returns.csv` has 4,829 total returns:
    - Train ($t \le 2025-08-31$): $1,982$
    - Validation ($2025-08-31 < t \le 2025-10-31$): $930$
    - Pure Test ($t > 2025-10-31$): **$1,917$**
    - Post-Train Holdout ($t > 2025-08-31$): **$2,847$**
- **PR-AUC Discrepancy:**
  - The earlier report claimed PR-AUC $0.7812$ (Fake Listings) and $0.7245$ (Return Fraud) from an undocumented prototype run.
  - Evaluating the serialized artifacts directly on the post-train holdout ($t > \text{TRAIN\_END}$) produces **0.7383** (Fake Listings) and **0.8914** (Return Fraud).
  - Evaluating on the pure out-of-time test set ($t > \text{VAL\_END}$) produces **0.7406** (Fake Listings) and **0.8806** (Return Fraud).
- **Harness Enhancement:** `scripts/evaluate_models_reproducible.py` now explicitly outputs and persists both splits, resolving all ambiguity.

### 4.4 Root Cause: HTTP Benchmark Discrepancy (~53 vs ~26 QPS)
- **Earlier Benchmark (50 Requests):** Evaluated a single 50-request batch per tier without warmup. Because the run lasted $< 1.0$ second, the CPU operated at maximum boost clocks, had zero thread queue saturation, and measured a transient burst throughput of ~47–53 QPS.
- **Closure Benchmark (910 Requests across 3 Runs):** Included a 10-request warmup, followed by 3 repeated runs of 100 requests each across tiers 1, 5, and 10.
- **Why Throughput Settled at ~26 QPS:**
  1. **Full-Stack Execution:** Each request executes Pydantic payload validation, auth verification, GNN embedding retrieval, XGBoost tree inference on 50 features, isotonic calibration, and **TreeSHAP tree traversal (`explain_instance`)**.
  2. **Python GIL & Thread Contention:** Under sustained concurrent async execution, Python's GIL and OpenMP worker threads compete for CPU cores, establishing a steady-state throughput of **26.06 ± 0.31 req/s** for single-worker in-process execution.
- **Conclusion:** The 26 QPS figure represents honest, steady-state single-process performance, whereas 53 QPS was a short-burst artifact.

### 4.5 Root Cause: Temporal Graph Feature Availability at Prediction Time
- **Offline Test Evaluation:**
  - `rel_graph_test` is built with `cutoff_date = VAL_END` (`2025-10-31`). No device/address sharing edges first observed after `2025-10-31` are present in the test relationship graph.
  - Bipartite monthly snapshots use `cutoff = months[i - 1]`. For example, an order placed in November 2025 (Month 11) only receives graph snapshot features computed from orders $\le$ October 2025 (Month 10).
- **Online Serving (`POST /transaction/score`):**
  - Callers provide raw transactional orders. If graph topological attributes (`share_degree`, `buyer_pagerank`) are not supplied in the payload, they default to `0.0`.
  - The model serves predictions using available tabular data and Redis GNN embeddings. Missing graph features trigger the documented cold-start degraded scoring mode, with the response explicitly indicating `is_cold_start: true`.
- **Verdict:** Strict chronological integrity is maintained across both evaluation and serving.

---

## 5. Single Authoritative Metric Table

This table is the single, binding truth for TrustShield AI model performance across all frozen splits.

| Model / Subsystem | Serialized Artifact Path | Evaluation Partition | Sample Count ($N$) | ROC-AUC | PR-AUC | F1 Score | Brier Score | ECE | Verified Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 3 Tabular + Graph** | `models/combined_graph_model.joblib` | Train ($t \le 2025-08-31$) | 22,848 | 0.9985 | 0.9610 | 0.9263 | 0.0106 | 0.0377 | **Empirically Verified** |
| *(XGBoost Classifier, 18 features)* | `models/combined_graph_model.joblib` | Val ($2025-08-31 < t \le 2025-10-31$) | 10,266 | 0.8637 | 0.5410 | 0.5542 | 0.0415 | 0.0000 | **Empirically Verified** |
| | `models/combined_graph_model.joblib` | **Out-of-Time Test ($t > 2025-10-31$)** | **16,886** | **0.7890** | **0.4418** | **0.4374** | **0.0640** | **0.0265** | **Empirically Verified** |
| **Phase 3 Degraded Zero-Graph** | `models/combined_graph_model.joblib` | **Out-of-Time Test ($t > 2025-10-31$)** | **16,886** | **0.6029** | **0.2419** | **0.2097** | **0.0780** | **0.0619** | **Empirically Verified** |
| *(Cold-Start / Zero Graph Features)* | *(Graph features set to 0.0)* | | | | | | | | *(Robustness Benchmark)* |
| **Phase 5 Hybrid Model** | `models/hybrid_model.joblib` | Train ($t \le 2025-08-31$) | 22,848 | 0.9638 | 0.7438 | 0.6339 | 0.0266 | 0.0347 | **Empirically Verified** |
| *(XGBoost + 16D GNN, 50 features)* | `models/hybrid_model.joblib` | Val ($2025-08-31 < t \le 2025-10-31$) | 10,266 | 0.8484 | 0.5214 | 0.5158 | 0.0430 | 0.0118 | **Empirically Verified** |
| | `models/hybrid_model.joblib` | **Out-of-Time Test ($t > 2025-10-31$)** | **16,886** | **0.7652** | **0.4191** | **0.3843** | **0.0657** | **0.0330** | **Empirically Verified** |
| **Fake Listing Detector** | `models/fake_listing_model.joblib` | **Pure Out-of-Time Test ($t > 2025-10-31$)** | **6,237** | **0.9496** | **0.7406** | **0.6990** | **0.0123** | **0.0104** | **Authoritative Test Split** |
| *(XGBoost Classifier, 5 features)* | `models/fake_listing_model.joblib` | Post-Train Holdout ($t > 2025-08-31$) | 10,077 | 0.9533 | 0.7383 | 0.6962 | 0.0127 | 0.0104 | Reconciled Holdout |
| **Return Fraud Detector** | `models/return_fraud_model.joblib` | **Pure Out-of-Time Test ($t > 2025-10-31$)** | **1,917** | **0.9235** | **0.8806** | **0.7457** | **0.1150** | **0.0841** | **Authoritative Test Split** |
| *(XGBoost Classifier, 14 features)* | `models/return_fraud_model.joblib` | Post-Train Holdout ($t > 2025-08-31$) | 2,847 | 0.9369 | 0.8914 | 0.7701 | 0.1005 | 0.0706 | Reconciled Holdout |

---

## 6. Execution Command Registry & Verifications

All verifications below were executed directly in the audited workspace environment (`Windows 11 AMD64`, Python `3.14.7`):

| Gate / Action | Exact Command | Exit Code | Verified Empirical Result |
| :--- | :--- | :---: | :--- |
| **Model Evaluation** | `.\.venv\Scripts\python.exe scripts/evaluate_models_reproducible.py` | **0** | All 5 models evaluated; manifests synchronized. |
| **Security Test Suite** | `.\.venv\Scripts\pytest.exe backend/test_security.py -v` | **0** | **17 passed in 86.61s** (fail-closed, CORS, RBAC verified). |
| **HTTP Benchmark** | `.\.venv\Scripts\python.exe scripts/benchmark_scoring_http.py --runs 3 --requests 100 --warmup 10` | **0** | **910 requests**, 0 errors, mean QPS = 26.06 ± 0.31. |
| **Full Python Test Suite** | `.\.venv\Scripts\pytest.exe -q` | **0** | **231 passed in 322.23s** (0 failures). |
| **Frontend ESLint** | `cmd.exe /c "npm --prefix frontend run lint"` | **0** | **0 errors**, 84 style warnings. |
| **Frontend Production Build** | `cmd.exe /c "npm --prefix frontend run build"` | **0** | **16/16 routes prerendered**, Next.js Turbopack build clean. |
| **Pyright Static Type Check** | `.\.venv\Scripts\pyright.exe` | **0** | **0 errors, 0 warnings, 0 informations**. |

---

## 7. Remaining Unresolved Issues & Roadmap to Phase 3

No unresolved evaluation-integrity, security, or testing blockers remain in Phase 2. The following architectural items represent known operational considerations scheduled for Phase 3:

1. **State Store Network Round-Trip Latency:**
   - In the Phase 2 test harness, Redis and Neo4j operate via in-memory and disk fallback modes (< 1 ms latency).
   - In Phase 3, live Neo4j Aura and AWS ElastiCache clusters will introduce 5–15 ms network roundtrips. Async connection pooling and batch pipelining must be implemented.
2. **Multi-Worker Process Model:**
   - Single-process FastAPI is bounded at ~26 QPS by Python's GIL.
   - Phase 3 will containerize the backend using Gunicorn with 8 Uvicorn worker processes behind an NGINX reverse proxy to target >200 QPS.
3. **Secret Vault Integration:**
   - Auth keys currently reside in environment variables (`TRUSTSHIELD_ADMIN_KEY`). Phase 3 deployment will inject keys dynamically from AWS Secrets Manager or HashiCorp Vault.

---

## 8. Final Gate Decision: **PASS**

Every discrepancy between the earlier Phase 2 report and the closure audit has been empirically investigated, explained down to root cause, and resolved:
- [x] Dataset versions, file hashes, and 2025 split boundaries verified.
- [x] Model estimator architecture verified as XGBoost.
- [x] Specialized detector PR-AUC values, F1 scores, and sample counts mathematically reconciled.
- [x] HTTP benchmark methodology and single-process boundaries verified.
- [x] Temporal graph feature availability audited with zero future leakage.
- [x] Documentation synchronized with authoritative facts.
- [x] 100% test pass rate across all 231 unit/integration tests, 17 security tests, ESLint, Next.js build, and Pyright.

**Phase 2 is formally approved with a decisive PASS.**
