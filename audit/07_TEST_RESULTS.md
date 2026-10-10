# TrustShield AI — Automated Test Results & Quality Verification

---

## 1. Test Execution Summary

| Test Suite / Scope | Command Executed | Tests Passed | Tests Failed | Execution Time | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Complete Repository Test Suite** | `python -m pytest -q` | 513 | 6 | 160.85s | **98.8% PASS** (Failures due to CRLF vs LF) |
| **FastAPI Backend Suite** | `python -m pytest backend/ -q` | 60 | 0 | 34.57s | **100% PASS** |
| **Data Leakage & Temporal Safety** | `python -m pytest trustshield_project/test_leakage.py -q` | 31 | 0 | 12.33s | **100% PASS** |
| **Stage 3.5 Gate & Ledger Security**| `python -m pytest trustshield_project/test_stage352* ... -q` | 156 | 0 | 1.45s | **100% PASS** |
| **Stage 3.3 Evaluation & Synthetic v2**| `python -m pytest trustshield_project/test_stage33* ... -q` | 56 | 0 | 0.75s | **100% PASS** |
| **Operational E2E Smoke Suite** | `python scripts/e2e_smoke_validation.py` | 9 | 0 | 7.82s | **100% PASS** |
| **Scoring Benchmark Suite** | `python scripts/benchmark_scoring_http.py --requests 5` | N/A | N/A | 7.91s | **100% PASS** (p50: 20.2ms) |
| **Next.js Production Build** | `npm.cmd run build` | 16 routes | 0 | 6.09s | **100% PASS** |
| **Next.js ESLint** | `npm.cmd run lint` | 0 | 1 error (85 warnings)| 1.80s | **FAILED (ISSUE-04)** |
| **Pyright Static Type Check** | `python -m pyright trustshield_project backend` | 0 | 12 errors | 2.10s | **FAILED (ISSUE-05)** |

---

## 2. Analysis of the 6 Pytest Failures

All 6 test failures occurred in test cases that compute binary `hashlib.sha256` hashes over text files (`.csv`, `.json`):
1. `trustshield_project/test_stage32_audit.py::TestDatasetContract::test_manifest_hashes_match_disk`
2. `trustshield_project/test_stage32_final_gate.py::TestTaskETestSetProtection::test_test_set_unmodified_and_virgin`
3. `trustshield_project/test_stage351_environment_sealing.py::TestHoldoutProtectionGuards::test_dataset_manifest_v2_1_unmutated`
4. `trustshield_project/test_stage35_temporal_holdout_readiness.py::TestArtifactIntegrityAndSchema::test_artifact_hashes_match_records`
5. `trustshield_project/test_stage35_temporal_holdout_readiness.py::TestProtectedArtifactImmutability::test_no_changes_to_protected_artifacts`
6. `trustshield_project/test_synthetic_v2_1.py::TestMandatorySafeguards::test_manifest_file_hashes_match`

### Root Cause Proof:
The tests expect the SHA-256 hash of the files with Windows CRLF (`\r\n`) line endings. However, Git checked out the files with Unix LF (`\n`) line endings due to `.gitattributes` (`text eol=lf`). Converting the files back to CRLF in memory produces the exact expected hashes.
