# TrustShield Phase 3 — Stage 3.1.1 Generator Integrity Fixes Report

**Report Date:** 2026-10-09  
**Stage:** 3.1.1 — Corrective Audit & Generator Integrity Fixes  
**Auditor / Engineers:** Senior ML Research Auditor, Backend Security Architect, Data Quality Engineer  
**Git Branch:** `phase-3-data-generalization`  
**Baseline Git Commit:** `4615a53602d0a4ffe447d4d2fd14bd69a70bf740`  
**Execution Environment:** Windows Server / Python 3.14.7 Virtualenv (`.\.venv`)  
**Generated Dataset Directory:** [`data/synthetic_v2_1/`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1)  
**Machine-Readable Deliverables:**
- Dataset Manifest: [`data/synthetic_v2_1/dataset_manifest.json`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/dataset_manifest.json)
- Validation Report: [`data/synthetic_v2_1/validation_report.json`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/validation_report.json)
- Focused Unit Tests: [`trustshield_project/test_synthetic_v2_1.py`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/trustshield_project/test_synthetic_v2_1.py)

---

## 1. Executive Summary & Verification Verdict

Stage 3.1.1 has been completed with an overall verdict of **PASS**. All implementation gaps identified during the Stage 3.1 review have been systematically audited, remediated, and verified with automated test suites before any model retraining.

### Key Remediation Highlights:
1. **Category-Constrained Perturbations with Compatible Subtype Matching**:
   All 500 fake listing perturbations are strictly constrained to the **exact same category** (0 cross-category violations). Subtype matching algorithm pairs compatible product types (179 subtype matches, e.g. case-to-case, shoe-to-shoe) and same-category variants (321 matches). Observable listing evidence is modified via `displayed_product_id` and price ratios, driving continuous, overlapping multimodal cosine similarity (~0.78 genuine vs ~0.66 fake, AUC 0.75) without artificial collapse or shortcut boolean flags.
2. **Generator-Label & Persona Leakage Eliminated**:
   Internal simulation control metadata (`buyer_persona`, `seller_persona`, `compromise_date`) has been completely stripped from public operational tables [`buyers.csv`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/buyers.csv) and [`sellers.csv`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/sellers.csv). Personas are segregated into [`generator_personas.csv`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/generator_personas.csv), while fraud rings and fraud types are tracked exclusively in [`fraud_ground_truth.csv`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/fraud_ground_truth.csv).
3. **Rigorous Return-Date Semantics & Survival Right-Censoring**:
   Every return occurs strictly after its order (`days_to_return >= 1`, 0 zero-day returns, 0 negative delay returns). Orders placed near `SIM_END` (December 30/31) whose sampled turnaround delay falls into 2026 are handled using standard survival analysis **right-censoring**: they are unobserved in the 2025 window and excluded from `returns.csv`. Sampled delays are never silently shortened or clamped to zero days. High-precision nanosecond timestamp indexing eliminates sub-day order-before-listing jitter (0 violations).
4. **Prediction-Time Feature Validity & Temporal Invariance**:
   Entity table snapshot totals (`total_orders`, `total_returns`, `total_listings`, `total_orders_received`) are classified as generator metadata / simulation-end snapshots and banned from model feature matrices. Point-in-time features are strictly derived via backward `merge_asof` (events strictly before order time $T$). A temporal invariance test proves that deliberately inserting future December events causes **zero change** in historical point-in-time features computed at $T$.
5. **Observable Seller Behaviors Empirically Confirmed**:
   Disposable sellers exhibit measurable short-tenure bursts (median active listing tenure of 2.0 days $\le 7$ days from signup), whereas established sellers average 130.0 days. Takeover (ATO) sellers exhibit steady pre-compromise listing velocity in their focus category followed by an anomalous category surge post-compromise.
6. **Preservation of Artifacts & Code**:
   All Phase 2 and Phase 3 baseline artifacts, v1 datasets in `synthetic_data_export/`, benchmark v2 in `data/synthetic_v2/`, model weights in `models/`, and production backend code remain **100% untouched and preserved**.

Per mandatory constraints, **no models were retrained** and execution has halted after Stage 3.1.1 awaiting explicit review.

---

## 2. Before-and-After Remediation Matrix

| Implementation Gap / Audit Finding | Stage 3.1 Baseline State (`data/synthetic_v2/`) | Stage 3.1.1 Remediation (`data/synthetic_v2_1/`) | Before Metric | After Metric | Verification Status |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **1. Fake Listing Perturbations** | No product or image swap was performed; only price was discounted. Zero observable image mismatch. | Candidate products strictly constrained to same category & compatible subtypes (179 subtype, 321 variant). Updates `displayed_product_id`. | Cross-category: N/A<br>Observable swap: None | Cross-category: **0 violations**<br>Compatible subtypes: **179**<br>Variants: **321** | **RESOLVED**: Genuine observable multimodal signal without shortcut flags. |
| **2. Persona & Label Leakage** | `buyer_persona` leaked in `buyers.csv`; `seller_persona` and `compromise_date` leaked in `sellers.csv`. | Stripped from entity tables; segregated into `generator_personas.csv` and `fraud_ground_truth.csv`. Feature gate rejects forbidden cols. | `buyer_persona` in buyers: **YES**<br>`seller_persona` in sellers: **YES** | Leaked cols in buyers: **0**<br>Leaked cols in sellers: **0**<br>Feature gate: **PASS** | **RESOLVED**: Public schema matches production. Zero persona leakage. |
| **3. Return-Date Semantics & Zero-Day Clamping** | Clamping returns to `SIM_END` caused shortened delays. Sub-day timestamp jitter caused 146 order-before-listing errors. | Enforced `delay >= 1`. Right-censoring policy for 2026 returns (320 organic, 30 abuse). Nanosecond indexing fixed order timing. | Order < Listing: **146**<br>Zero-day policy: Silent truncation | Order < Listing: **0**<br>Zero-day returns: **0**<br>Right-censored: **350** | **RESOLVED**: True survival right-censoring; 0 timing violations. |
| **4. Prediction-Time Feature Validity** | `total_orders`, `total_returns`, `total_listings` present on entity tables without explicit prediction-time usage guardrails. | Documented as simulation-end metadata; banned from model inputs. Point-in-time features computed via backward `merge_asof`. | Temporal test: Not implemented | Temporal invariance test: **PASS** (Zero delta under future events) | **RESOLVED**: Proved historical features invariant to future events. |
| **5. Observable Seller Behaviors** | Behaviors claimed in persona labels but not verified through empirical timestamp analysis. | Disposable sellers restricted to 2–7 day listing bursts. Takeover sellers surge post-compromise in anomalous category. | Empirically verified: No | Disposable median tenure: **2.0 days**<br>Established median tenure: **130.0 days** | **RESOLVED**: Behaviors measurable from observable event logs. |
| **6. ID Label Leakage** | `RETURN_{i:06d}` implemented in v2, but validation was not comprehensive across all tables. | Comprehensive regex validation across orders, listings, returns, buyers, sellers. | Leaking IDs: 0 | Leaking IDs: **0** | **VERIFIED**: Zero label tokens in entity identifiers. |
| **7. Distribution Overlap** | Return delay overlap achieved in v2. | Maintained continuous overlapping delay distribution. Single-feature predictive power remains moderate. | KS Stat: 0.1382<br>Single AUC: 0.5620 | KS Stat: **0.1279**<br>Cohen's $d$: **-0.2706**<br>Single AUC: **0.5584** | **VERIFIED**: Delay distribution is realistic and continuous. |

---

## 3. Detailed Architectural & Methodological Fixes

### 3.1 Category-Constrained Product & Image Perturbations

In baseline v1, fake listings swapped product IDs across the entire catalog indiscriminately (e.g., swapping a 3D printer filament with a women's shoe), producing an artificial text/image cosine similarity divergence (< 0.15) that trivialized multimodal detection. In Stage 3.1, product swaps were omitted entirely.

In Generator v2.1 ([`scripts/generate_realistic_synthetic_data_v2_1.py`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/scripts/generate_realistic_synthetic_data_v2_1.py)):
- All 500 fake listings are matched against products within the **exact same category** (`prods_by_cat_df`).
- **Compatible Subtype Matching**: A title keyword extractor identifies product subtypes (e.g. `case`, `cover`, `shoe`, `cable`, `charger`, `speaker`, `headphone`, `lamp`, `pot`, `drawer`). If candidates sharing the subtype exist, one is selected at random. This simulates real-world fraudulent listings where a seller claims to sell an authentic brand-name phone case but displays an image of a generic phone case of the same type.
- **Observable Evidence**: The listing's `displayed_product_id` is updated to the candidate product, pointing to the real image file on disk in `trustshield_project/data/external/abo/images/small/`.
- **Pricing**: Applied subtle competitive discounts (80%–92% of base price) or slight scalper markups (115%–130%), overlapping natural lognormal market variance.
- **Audit Logging**: Full perturbation details are exported to [`perturbation_audit.csv`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/perturbation_audit.csv) with columns: `listing_id`, `claimed_product_id`, `category`, `perturbed_product_id`, `price_factor`, `perturbation_type`.
- **Public Schema Sanitation**: Shortcut boolean flags `price_anomaly` and `image_mismatch` are strictly prohibited from `listings.csv`.

### 3.2 Generator-Label & Persona Segregation

To prevent data leakage during supervised model training:
- Public entity tables [`buyers.csv`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/buyers.csv) and [`sellers.csv`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/sellers.csv) match production operational schemas:
  - `buyers.csv`: `buyer_id`, `signup_date`, `address_id`, `trust_score_current`, `total_orders`, `total_returns`
  - `sellers.csv`: `seller_id`, `signup_date`, `address_id`, `category_focus`, `trust_score_current`, `total_listings`, `total_orders_received`, `account_status`
- Generator control personas (`standard`, `high_return_legit`, `abuser` for buyers; `established`, `disposable`, `takeover` for sellers) are exported exclusively to [`generator_personas.csv`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/generator_personas.csv).
- Fraud rings, compromise dates, and ground-truth targets are preserved in [`fraud_ground_truth.csv`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/fraud_ground_truth.csv).
- A strict validation gate test in [`test_synthetic_v2_1.py`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/trustshield_project/test_synthetic_v2_1.py) verifies that feature builders fail if any forbidden field enters the model feature matrix.

### 3.3 Return-Date Semantics & Survival Right-Censoring

- **Delay Sampling**: Organic return delays are sampled from a mixture distribution with minimum delay of 1 day (`integers(1, 6)` or `integers(4, 22)`). Abusive returns are sampled with minimum delay of 2 days (`integers(2, 7)` or `integers(5, 19)`).
- **Survival Analysis Policy**: If `scheduled_return_date = order_date + timedelta(days=delay) > SIM_END (2025-12-31)`, the return is treated as a **right-censored observation**. In a real-world platform, an order placed on December 30 with a 7-day return turnaround is simply still in the return window as of December 31; the return has not occurred yet.
  - The return is **NOT** recorded in `returns.csv`.
  - It is **NEVER** clamped to `SIM_END`, which would have created an artificial zero-day return and distorted turnaround distributions.
  - In Generator v2.1, 320 organic returns and 30 abusive returns were right-censored.
- **December 30 and 31 Verification**:
  - Orders on December 30: 395 orders; all recorded returns occurred on December 31 (`days_to_return == 1`).
  - Orders on December 31: 1 order; exactly 0 returns recorded (100% right-censored).
  - Across all 5,537 returns: `min(days_to_return) == 1`, 0 zero-day returns, 0 negative delay returns.
- **Sub-Day Chronological Ordering**:
  - Replaced integer-day `toordinal()` search in order generation with nanosecond timestamp indexing (`datetime64[ns]`).
  - Ensured `order_date >= listing_date + 5 minutes` and `order_date >= buyer_signup_date + 1 minute`.
  - Result: Chronological violations dropped from 146 to **exactly 0**.

### 3.4 Prediction-Time Feature Validity & Temporal Invariance

- **Full-Period Aggregate Catalog**:
  - `buyer.total_orders`: Total orders across the entire 12-month simulation.
  - `buyer.total_returns`: Total returns across the entire 12-month simulation.
  - `seller.total_listings`: Total listings created across the entire 12-month simulation.
  - `seller.total_orders_received`: Total orders received across the entire 12-month simulation.
  These fields represent database snapshot totals at `SIM_END`. They are **generator metadata / database summary counters**. Using them as features for an order at time $T$ constitutes severe future event leakage.
- **Prediction-Time Features**:
  Must be derived exclusively via point-in-time backward joins:
  - `buyer_orders_before`: `_cumulative_count_asof(orders, 'buyer_id')`
  - `buyer_returns_before`: `_asof_cumulative_from_events(orders, returns, 'buyer_id', 'return_date')`
  - `seller_total_listings_before`: `_asof_cumulative_from_events(orders, listings, 'seller_id', 'listing_date')`
- **Empirical Invariance Test**:
  In [`TestPredictionTimeFeatureValidity`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/trustshield_project/test_synthetic_v2_1.py#L254), an order evaluated at $T = \text{2025-05-15}$ was subjected to deliberately injected future events in December 2025 (5 future orders, 2 future returns). The point-in-time features computed before and after future event injection showed a delta of **exactly 0.0000**.

### 3.5 Observable Seller Behaviors

- **Disposable Sellers (15% of sellers, 75 accounts)**:
  - Account registered throughout the year.
  - 100% of listings published in a rapid burst within **2 to 7 days of signup**.
  - Empirical verification: Median active listing tenure is **2.0 days** from signup (vs established sellers at **130.0 days**).
- **Takeover / Compromised Sellers (5% of sellers, 25 accounts)**:
  - Account registered early in the year, active in an established focus category (e.g. Books, Furniture).
  - Explicit `compromise_date` assigned between Month 6 and Month 10.
  - Pre-compromise listings: 100% in focus category, staggered over time.
  - Post-compromise listings: Sudden surge in a high-risk category (e.g. Electronics) published within 1 to 5 days of compromise.
  - Observable without relying on persona labels.

---

## 4. Complete Dataset Manifest & Hash Verification

All generated tables, metadata, and validation reports have been exported to [`data/synthetic_v2_1/`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1). The table below documents the exact row counts, file sizes, and SHA-256 hashes verified by [`dataset_manifest.json`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1/dataset_manifest.json).

| File Name | Row Count | File Size (Bytes) | SHA-256 Checksum | Description / Schema Role |
| :--- | :---: | :---: | :--- | :--- |
| **`addresses.csv`** | 4,950 | 168,343 | `6bd1810838f5fff64f253d9a022f3cb69e3a87a65713a699f5d032ef2e2385cc` | Physical address entities & regions |
| **`devices.csv`** | 5,000 | 185,046 | `fbff496bad859c4350faae2fc9488461a8d775f0399f73e5b4efb0efd5aef964` | Hardware device fingerprints |
| **`device_mapping.csv`** | 5,400 | 157,031 | `69e2ab78eb2d55515b2ea726d72c92049814d72083dde34a38c3c3df2560569c` | Buyer-to-device mapping |
| **`address_sharing_log.csv`** | 600 | 47,222 | `858b6c3af3dad6d49ef464c2bbf0a9e425a86527ca285a4d026e9473a6a005ce` | Shared address event log |
| **`device_sharing_log.csv`** | 400 | 31,277 | `2c2b0d691c15543d8930644a577546f9b52e2f0ee9f709e3c1f2cd994ab5cbf3` | Shared device event log |
| **`buyers.csv`** | 5,000 | 227,006 | `3113e81c0e7b4ac0d561ca90cbee541bed8290907f77aefee5badbf9943992b5` | Public buyers table (clean schema) |
| **`sellers.csv`** | 500 | 31,524 | `4368a6fd88ff4d6fab5b8dc4eb8af20a0123e9a5feb86d334f714ed608e61d3d` | Public sellers table (clean schema) |
| **`products.csv`** | 3,000 | 527,904 | `b699e4e80d59d50e48d912273b868d523d3180fcb52615ecda6898fefa02afa7` | ABO catalog with real image refs |
| **`listings.csv`** | 20,000 | 2,173,321 | `0081b562dc900c14e1b90f2310261401ab2f8f40dda4805aa72bd17959d5e619` | Public seller listings (`displayed_product_id`) |
| **`orders.csv`** | 50,000 | 6,417,053 | `325597a6e0c2afa533c01d26895457cba166cd0bc2c0f9ce4f12c43355741b8f` | Marketplace transactions |
| **`returns.csv`** | 5,537 | 622,972 | `34eb6514d2ebe4b90608f31fb09fd71c2aa1709424d3e00c22c6a38df119a14a` | Orders returned (delays $\ge 1$ day) |
| **`fraud_ground_truth.csv`** | 2,034 | 128,051 | `5b56a996287f6d6bd6f23bf2312a49914c9a84b937ad75279f579255682d6929` | Separate ground truth labels & ring IDs |
| **`generator_personas.csv`** | 5,500 | 168,122 | `79be2849c3441f3852d213cfa5cb3ff0ef6b30bd8c7574adbe9a6c4dd2891c89` | Segregated generator personas metadata |
| **`perturbation_audit.csv`** | 500 | 43,770 | `6b2010d5286e49d07980d610bdd322e75f1a01f716cd59ee343150200ddb29a8` | Internal audit log of perturbations |
| **`dataset_manifest.json`** | — | 2,587 | — | Cryptographic dataset manifest |
| **`validation_report.json`** | — | 3,427 | — | 12-point integrity check results (PASS) |

---

## 5. Commands Executed & Test Results

### 5.1 Generator Execution
```powershell
python scripts/generate_realistic_synthetic_data_v2_1.py
```
**Console Output Summary:**
```
================================================================================
TRUSTSHIELD STAGE 3.1.1 — GENERATOR INTEGRITY FIXES & V2.1 EXPORT
================================================================================
1. Generating base entities with separate persona tracking...
2. Generating catalog and listings with observable disposable & ATO seller behaviors...
3. Generating transactions and organic returns with right-censored end-of-period policy...
  Right-censored organic returns (in 2026): 320
4. Injecting realistic fraud with category-constrained perturbations & clamped bursts...
5. Running comprehensive Stage 3.1.1 validation suite...
Validation Status: PASS
  - foreign_keys: PASS (0 missing)
  - null_values: PASS (0 nulls)
  - primary_key_uniqueness: PASS (0 duplicates)
  - date_boundary_clamping: PASS (0 out-of-bounds)
  - chronological_invariants: PASS (order < listing: 0, return < order: 0, zero-day: 0)
  - id_leakage: PASS (0 leaking tokens)
  - observable_seller_behavior: PASS (disposable median: 2.0d, established median: 130.0d)
  - return_delay_overlap: PASS (legit mean: 8.78d, fraud mean: 7.29d, single AUC: 0.5584)
  - category_constrained_perturbations: PASS (500 perturbed, 0 cross-cat violations)
  - generator_metadata_segregation: PASS (0 banned columns in buyers/sellers/listings)
  - fraud_prevalence_targets: PASS (total: 50,000, fraud: 3,502, rate: 7.00%)
  - return_date_semantics_december: PASS (0 timing violations, 100% right-censored)
Exported 14 CSV files + manifest + validation report in 12.41s.
```

### 5.2 Focused Unit Test Suite (`test_synthetic_v2_1.py`)
```powershell
.\.venv\Scripts\pytest.exe trustshield_project/test_synthetic_v2_1.py -v
```
**Test Results:**
```
trustshield_project/test_synthetic_v2_1.py::TestMandatorySafeguards::test_v1_and_v2_artifacts_preserved PASSED [  5%]
trustshield_project/test_synthetic_v2_1.py::TestMandatorySafeguards::test_manifest_file_hashes_match PASSED [ 10%]
trustshield_project/test_synthetic_v2_1.py::TestMandatorySafeguards::test_validation_report_status_pass PASSED [ 15%]
trustshield_project/test_synthetic_v2_1.py::TestCategoryConstrainedPerturbations::test_all_swaps_within_exact_same_category PASSED [ 21%]
trustshield_project/test_synthetic_v2_1.py::TestCategoryConstrainedPerturbations::test_compatible_subtype_presence PASSED [ 26%]
trustshield_project/test_synthetic_v2_1.py::TestCategoryConstrainedPerturbations::test_perturbation_changes_observable_listing_evidence PASSED [ 31%]
trustshield_project/test_synthetic_v2_1.py::TestCategoryConstrainedPerturbations::test_no_shortcut_boolean_flags_in_listings PASSED [ 36%]
trustshield_project/test_synthetic_v2_1.py::TestGeneratorLabelLeakagePrevention::test_persona_segregated_from_public_entity_tables PASSED [ 42%]
trustshield_project/test_synthetic_v2_1.py::TestGeneratorLabelLeakagePrevention::test_personas_preserved_in_separate_evaluation_metadata PASSED [ 47%]
trustshield_project/test_synthetic_v2_1.py::TestGeneratorLabelLeakagePrevention::test_forbidden_fields_cannot_enter_feature_matrix PASSED [ 52%]
trustshield_project/test_synthetic_v2_1.py::TestReturnDateSemantics::test_every_return_occurs_strictly_after_order PASSED [ 57%]
trustshield_project/test_synthetic_v2_1.py::TestReturnDateSemantics::test_december_30_and_31_return_policy PASSED [ 63%]
trustshield_project/test_synthetic_v2_1.py::TestReturnDateSemantics::test_foreign_key_and_one_to_one_order_mapping PASSED [ 68%]
trustshield_project/test_synthetic_v2_1.py::TestPredictionTimeFeatureValidity::test_full_period_aggregates_classified_as_metadata PASSED [ 73%]
trustshield_project/test_synthetic_v2_1.py::TestPredictionTimeFeatureValidity::test_point_in_time_feature_invariance_under_future_events PASSED [ 78%]
trustshield_project/test_synthetic_v2_1.py::TestObservableSellerBehaviors::test_disposable_seller_listing_burst PASSED [ 84%]
trustshield_project/test_synthetic_v2_1.py::TestObservableSellerBehaviors::test_takeover_seller_post_compromise_category_surge PASSED [ 89%]
trustshield_project/test_synthetic_v2_1.py::TestDistributionOverlaps::test_return_delay_distribution_overlap PASSED [ 94%]
trustshield_project/test_synthetic_v2_1.py::TestDistributionOverlaps::test_no_id_label_leakage PASSED [100%]

============================= 19 passed in 1.06s ==============================
```

### 5.3 Regression Test Suites
```powershell
.\.venv\Scripts\pytest.exe trustshield_project/test_synthetic_v2.py -v
```
**Result:** **4 passed in 0.31s**.

```powershell
.\.venv\Scripts\pytest.exe trustshield_project/test_leakage.py -q
```
**Result:** **31 passed in 8.83s**.

**Combined Total:** **54 passed tests across all data integrity suites (100% pass rate)**.

---

## 6. Known Limitations & Scope Boundaries

1. **Multimodal Catalog Assets**:
   The local Amazon Berkeley Objects (ABO) dataset contains 2,616 real images on disk across 226 shards in `trustshield_project/data/external/abo/images/small/`. The remaining 384 products use synthetic text-based fallbacks. While multimodal embedding alignment reflects real ABO images and text cosine similarities, the benchmark does not simulate raw uploaded byte streams or OCR text manipulation on images.
2. **Entity Table Snapshot Counters**:
   Columns `total_orders`, `total_returns`, `total_listings`, `total_orders_received` exist on `buyers.csv` and `sellers.csv` as operational database metadata reflecting state at simulation end. They are strictly prohibited from candidate model feature sets at prediction time. Production feature stores must always compute point-in-time statistics using `merge_asof`.
3. **Model Retraining Deferred**:
   In strict compliance with Stage 3.1.1 instructions, no model weights in `models/` were modified, no models were retrained on `synthetic_v2_1/`, and no production API endpoints were altered.

---

## 7. Stop Condition Confirmation

Stage 3.1.1 is complete. Execution has halted. All corrected tables are safely stored in [`data/synthetic_v2_1/`](file:///c:/Users/rajiv_pis9z8x/Downloads/trustshield_full_handoff/data/synthetic_v2_1). No changes have been committed to Git or pushed to remote. Awaiting explicit user approval before proceeding to any subsequent stage.
