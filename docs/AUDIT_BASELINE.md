# TrustShield AI — Phase 0 Audit Baseline Report

**Execution Timestamp:** 2026-10-10  
**Environment:** Python 3.14.7 | pytest 9.1.1 | Windows (x86_64)  
**Auditor Role:** Senior ML Engineer (Remediation & Reproducibility)

---

## 1. Test Suite Inventory & Baseline Run

- **Total Test Functions Collected:** `520` tests across the test suite (`trustshield_project/` and `backend/`).
- **Full Test Run Result:**
  - **Passed:** `520` (100%)
  - **Failed:** `0`
  - **Warnings:** `53` (deprecation warnings regarding Python 3.14 torch.jit and shap color maps, plus sklearn version metadata notice)
  - **Runtime:** `131.72s` (0:02:11)
- **Status:** **PASS** (Zero test failures on current branch with line-ending and auth query fix).

---

## 2. File & Path Audit (Prompt "Known Missing" vs. Reality)

Rule #4 states: *"If something in this prompt contradicts what you see in the code, stop and report the discrepancy instead of guessing."*

Below is the verified inventory of the files cited in Phase 0:

| Path Cited in Prompt | Reported in Prompt | Actual Reality in Repo | Status / Action |
| :--- | :--- | :--- | :--- |
| `trustshield_project/evaluation_gate.py` | Known missing | **Exists** (384 lines) | Per instructions, obsolete cryptographic gate to be removed/de-emphasized |
| `backend/auth.py` | Known missing | **Exists** (199 lines) | Real RBAC & API Key authentication already implemented & integrated into `main.py` |
| `reports/` | Known missing | **Exists** | Present in directory structure |
| `data/synthetic_v2_1/` | Known missing | **Exists** | Present with complete generated CSV datasets and manifest |
| `models/stage34/` | Known missing | **Exists** | Present with Stage 3.4 joblib artifacts |
| `docker/Dockerfile.evaluator` | Known missing | **Exists** | Present |
| `scripts/benchmark_scoring_http.py` | Known missing | **Exists** (238 lines) | Present |
| `scripts/generate_realistic_synthetic_data_v2_1.py` | Known missing | **Exists** | Present |
| `scripts/evaluate_models_reproducible.py` | Known missing | **Exists** | Present |
| `test_stage3*.py` | Known missing | **Exists** (5 files) | Present |
| `docs/HTTP_BENCHMARK_REPORT.md` | Known missing | **Exists** | Present |
| `models/baseline_rf.joblib` | Known missing | **MISSING** | Confirmed missing; referenced in `frontend/src/app/models/page.tsx` |

---

## 3. Discrepancies, Fabrications & Misleading Claims Identified

1. **Missing Artifacts Referenced in Frontend/Docs:**
   - `models/baseline_rf.joblib`: Referenced in frontend models view (`frontend/src/app/models/page.tsx`), but file does not exist.
2. **Fabricated Metrics & Hard-Coded Frontend Math:**
   - `frontend/src/app/evaluation/page.tsx`: Contains hardcoded synthetic confusion matrix formulas: `tp = 1000 * (1 - threshold * 0.3)`, `fp = 200 * (1 - threshold * 0.8)`. These do not reflect actual model evaluation.
   - `frontend/src/app/models/page.tsx`: Contains static hard-coded ROC-AUC / PR-AUC values not loaded from any evaluation run artifact.
3. **Currency Inconsistency:**
   - Generated transaction amounts are calibrated in INR scale (mean ~1,744, max ~141,000), but UI, docs, and agent dossiers display `$` dollar symbols.
4. **Data Generator Timestamp Bug (Pandas >= 3 datetime unit):**
   - In `trustshield_project/entity_generator.py`: `assign_shared_addresses` and `assign_shared_devices` use `.asi8` on pandas datetime. In pandas 3+, datetime units default to non-nanosecond units (`s` or `us`), resulting in 1970-era dates (`1970-01-01 ...`) for 90%+ of sharing events.
5. **Address Collision Mismatch:**
   - In `entity_generator.py`: `generate_buyers` samples `address_id` with replacement from ~4950 addresses, causing ~68% of buyers to collide by chance rather than the documented 12%.
6. **Feature Leakage:**
   - `device_shared_buyer_count` in `baseline_model.py` uses all train orders across the entire train window rather than strictly prior to order time.
   - `share_degree` and `share_component_size` in `graph_features.py` use static graph snapshots across splits without strict per-order point-in-time filtering.
   - Hybrid GraphSAGE model evaluates on in-fold embeddings without out-of-fold cross-fitting.
7. **Multimodal Surrogate Mislabeling:**
   - Surrogate TF-IDF + Gaussian noise feature was called "CLIP Multimodal" in multiple docs despite lacking true visual-textual CLIP embeddings.

---

## 4. Phase 0 Conclusion

The repository foundation is functional (all 520 tests pass), but the underlying feature engineering, data generation timestamps, and metrics claims suffer from the documented scientific discrepancies. We proceed to **Phase 1** to fix the data generator timestamp bug, address collisions, and currency standardization.
