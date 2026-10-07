# TrustShield Training & Pipeline Execution Scripts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4+-eb5424.svg)](https://xgboost.ai)
[![FAISS](https://img.shields.io/badge/FAISS-VectorSearch-00599C.svg)](https://github.com/facebookresearch/faiss)

This directory contains standalone execution scripts for end-to-end model training, probability calibration fitting, offline multimodal feature extraction, and deployment smoke validation.

---

## 📜 Script Index & Descriptions

| Script | Purpose | Output Artifacts | Primary Dependencies |
| :--- | :--- | :--- | :--- |
| **`train_and_save_models.py`** | Trains baseline tabular and graph models under strict chronological cutoff (`TRAIN_END`), fits isotonic calibrators, and discovers fraud rings. | `models/combined_graph_model.joblib`<br>`models/fake_listing_model.joblib`<br>`models/return_fraud_model.joblib`<br>`models/fraud_rings.joblib`<br>`models/calibrator.joblib` | `scikit-learn`, `networkx`, `joblib` |
| **`train_phase5.py`** | Trains the Phase 5 Hybrid XGBoost model incorporating GNN node embeddings, fits the Phase 5 isotonic probability calibrator, and computes test metrics. | `models/hybrid_model.joblib`<br>`models/phase5_calibrator.joblib`<br>`models/buyer_embeddings.joblib`<br>`models/seller_embeddings.joblib` | `xgboost`, `scikit-learn`, `torch` |
| **`reproduce_all.py`** | Single-command reproduction master runner: trains baseline + hybrid models, fits calibrators, and exports `models/reproduction_manifest.json`. | `models/reproduction_manifest.json`<br>All model artifacts | `scikit-learn`, `xgboost`, `torch`, `networkx` |
| **`e2e_smoke_validation.py`** | Holistic end-to-end validation harness verifying model loading, offline vs. online consistency, calibration, FAISS index queries, and cold start paths. | Terminal diagnostic report (`PASS` / `FAIL`) | `requests`, `joblib`, `numpy` |
| **`smoke_test_api.py`** | Live API smoke tester verifying HTTP status codes, schema contracts, `/ready` health, and latency against a running server. | HTTP test summary | `requests` |
| **`build_clip_embeddings.py`** | Offline batch generator that extracts image and text CLIP embeddings from catalog items and populates the FAISS vector index cache. | `models/clip_cache/` | `transformers`, `torch`, `faiss` |

---

## 🚀 Execution Guide

### 1. Training All Models & Generating Calibrators
Run the primary training pipeline from the repository root:
```bash
# Train Phase 1-3 baseline, graph, and fraud ring models
python scripts/train_and_save_models.py

# Train Phase 5 Hybrid XGBoost model and fit Phase 5 calibrator
python scripts/train_phase5.py
```

### 2. Running End-to-End Smoke Validation
Validates model serialization, offline-vs-online numerical parity, and API response contracts without requiring a manual browser session:
```bash
python scripts/e2e_smoke_validation.py
```

### 3. Testing a Live Running API Server
Ensure the FastAPI server is running on port 8000 (e.g. `uvicorn backend.main:app --port 8000`), then in a separate terminal execute:
```bash
python scripts/smoke_test_api.py --host http://localhost --port 8000
```

### 4. Rebuilding Multimodal Embeddings Cache (Optional)
To regenerate the CLIP embeddings cache used by `/listing/analyze`:
```bash
python scripts/build_clip_embeddings.py
```
