
# TrustShield Training & Pipeline Execution Scripts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4+-eb5424.svg)](https://xgboost.ai)
[![FAISS](https://img.shields.io/badge/FAISS-VectorSearch-00599C.svg)](https://github.com/facebookresearch/faiss)

This directory contains standalone execution scripts for synthetic data generation, end-to-end model training, probability calibration, multimodal feature extraction, benchmark evaluation, and deployment smoke validation.

Canonical multi-step experiment workflows are located in the top-level `experiments/` directory.

---

## 📜 Script Index & Descriptions

| Script | Purpose | Output Artifacts | Primary Dependencies |
| :--- | :--- | :--- | :--- |
| **`generate_realistic_synthetic_data_v2_1.py`** | Generates realistic synthetic multi-relational e-commerce entities, transactions, device logs, and fraud ground truth under strict temporal cutoff. | `data/synthetic_data_v2_1.joblib` | `numpy`, `pandas`, `scipy` |
| **`train_and_save_models.py`** | Trains baseline tabular and graph models under chronological cutoff (`TRAIN_END`), fits isotonic calibrators, and discovers fraud rings. | `models/combined_graph_model.joblib`<br>`models/fake_listing_model.joblib`<br>`models/return_fraud_model.joblib`<br>`models/fraud_rings.joblib`<br>`models/calibrator.joblib` | `scikit-learn`, `networkx`, `joblib` |
| **`generate_phase3_golden_baseline.py`** | Generates the golden baseline JSON fixture used by regression tests to ensure probability invariance. | `backend/fixtures/phase3_golden_baseline.json` | `joblib`, `requests` |
| **`build_clip_embeddings.py`** | Offline batch generator extracting image and text CLIP embeddings from catalog items and populating the FAISS vector index cache. | `models/clip_cache/` | `transformers`, `torch`, `faiss` |
| **`e2e_smoke_validation.py`** | Holistic end-to-end validation harness verifying model loading, offline vs. online consistency, calibration, and live API endpoints across 9 operational cases. | Terminal diagnostic report (`PASS` / `FAIL`) | `requests`, `joblib`, `numpy` |
| **`smoke_test_api.py`** | Live HTTP server test verifying all 10 core FastAPI endpoints (health, ready, scoring, explain, rings, listings, returns, metrics). | Terminal diagnostic report (`PASS` / `FAIL`) | `requests` or `urllib` |
| **`benchmark_scoring_http.py`** | Live scoring latency and throughput benchmark measuring p50, p90, p95, and p99 percentiles over HTTP. | Benchmark summary metrics | `requests` or `urllib` |
| **`verify_production_invariance.py`** | Golden baseline invariance verification checking 29 schema fields across 4 core production scenarios. | Regression check report (`PASS` / `FAIL`) | `requests`, `joblib` |
| **`check_no_hardcoded_metrics.py`** | CI Guard enforcing allowlist grounding for all published metrics and statistics across markdown docs and frontend assets. | Pass / Fail CI exit code | `hashlib`, `re` |
| **`run_all_experiments.py`** | Executes the 20-seed canonical benchmark, graph ablations, and Holm-Bonferroni family-wise error rate corrections. | `results/results.json` | `numpy`, `pandas`, `scikit-learn`, `networkx` |
| **`verify_results.py`** | Verifies statistical coherence and numerical precision of audited benchmark results. | Diagnostic verification report | `json`, `numpy` |
| **`generate_results_md.py`** | Dynamically compiles markdown tables and narrative summaries in `results/RESULTS.md` from `results/results.json`. | `results/RESULTS.md` | Standard Library |
| **`evaluate_naive_connected_components_20seeds.py`** | Evaluates naive connected components on fraud rings across all 20 canonical seeds with graph cutoff strictly at `VAL_END: 2025-10-31`. | `results/ring_detection_20seeds.json` | `networkx`, `pandas` |

---

## 🚀 Execution Guide

### 1. Generating Synthetic Data
```bash
python scripts/generate_realistic_synthetic_data_v2_1.py
```
*(Or via `python experiments/01_generate_data.py`)*

### 2. Training Models & Generating Calibrators
```bash
python scripts/train_and_save_models.py
```
*(Or via `python experiments/02_train_models.py`)*

### 3. Running 20-Seed Canonical Benchmark
```bash
python scripts/run_all_experiments.py
```
*(Or via `python experiments/03_evaluate_20seeds.py`)*

### 4. Running End-to-End Smoke Validation
Validates model serialization, offline-vs-online numerical parity, and API response contracts:
```bash
python scripts/e2e_smoke_validation.py
```

### 5. Rebuilding Multimodal Embeddings Cache (Optional)
To regenerate the CLIP embeddings cache used by `/listing/analyze`:
```bash
python scripts/build_clip_embeddings.py
```
*(Or via `python experiments/04_build_clip_cache.py`)*

### 6. Running CI Allowlist Metric Grounding Check
```bash
python scripts/check_no_hardcoded_metrics.py
```

### 7. Running Live HTTP Smoke Tests
Validates all 10 core FastAPI endpoints against a running server:
```bash
python scripts/smoke_test_api.py
```

### 8. Running HTTP Latency & Throughput Benchmark
Benchmarks median, 90th, 95th, and 99th percentile response times:
```bash
python scripts/benchmark_scoring_http.py --requests 100
```

### 9. Verifying Production Invariance
Validates all 29 golden schema fields against the Phase 3 golden baseline fixture:
```bash
python scripts/verify_production_invariance.py
```

