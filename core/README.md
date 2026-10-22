# TrustShield Core Intelligence & Research Suite

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4+-eb5424.svg)](https://xgboost.ai)
[![FAISS](https://img.shields.io/badge/FAISS-VectorSearch-00599C.svg)](https://github.com/facebookresearch/faiss)
[![Explainability](https://img.shields.io/badge/Explainability-TreeSHAP%20Exact-8A2BE2.svg)](shap_explainer.py)
[![Temporal Safety](https://img.shields.io/badge/Temporal%20Invariant-event__time%20%3C%20decision__time-blue.svg)](temporal_utils.py)

This directory contains the core machine learning models, point-in-time graph topology extractors, multimodal scoring utilities, and information-theoretic decisioning algorithms powering the **TrustShield AI** platform.

---

## 🔬 Module Architecture & Breakdown

### 1. Topology & Temporal Invariants
- **`temporal_utils.py`**:  
  Canonical temporal invariant enforcement. Implements strict historical inequality (`event_time < decision_time`), chronological DataFrame filtering, temporal graph edge slicing, and validation boundary isolation.
- **`graph_features.py`**:  
  NetworkX topological metrics (PageRank, degree centrality, connected component size, HHI concentration) computed with strict chronological cutoff boundaries (`cutoff_date <= TRAIN_END`).
- **`point_in_time_history.py`**:  
  Point-in-time causal event history preserving strictly observable entity states prior to decision timestamps.

### 2. Decisioning, Calibration & Conformal Coverage
- **`advanced_trust_engine.py`**:  
  Canonical Trust Engine combining tabular gradient-boosted models with graph topological features. Implements split conformal prediction sets (`{0}`, `{1}`, or `{0, 1}`) with $\ge 95\%$ statistical coverage guarantees ($\alpha = 0.05$).
- **`calibration.py`**:  
  Post-hoc probability calibration utilizing validation-fitted Isotonic Regression and Platt scaling. Evaluates Expected Calibration Error (ECE) and Brier Score.
- **`policy_config.py` & `policy_rules.py`**:  
  Centralized risk policy definitions, thresholds, reason codes, and forensic RAG guidelines for operational decisions (`ALLOW`, `REVIEW`, `HOLD`, `BLOCK`).

### 3. Machine Learning Models
- **`baseline_model.py`**:  
  Baseline tabular and graph model training pipelines under chronological partition (`TRAIN_END`, `VAL_END`).
- **`specialized_fraud_detectors.py`** (formerly `phase2_specialized_models.py`):  
  Specialized fraud detectors for fake listing detection and return fraud anomaly scoring.
- **`multimodal_scoring.py`**:  
  Vision-language cross-modal alignment scoring for listing risk evaluation, combining text TF-IDF surrogates and CLIP embedding cosine similarities with self-match exclusion.

### 4. Explainability & Forensic Dossiers
- **`shap_explainer.py`**:  
  Production TreeSHAP explainability engine computing exact local Shapley attributions in sub-10ms. Separates positive risk contributors from protective factors and synthesizes human-friendly feature definitions.
- **`dossier_generator.py`**:  
  Rule-based forensic investigation dossier generator producing evidence-grounded reports with an automated regex Grounding Guard verifying reported metrics against model telemetry.

### 5. Synthetic Generation & Simulation
- **`entity_generator.py`**:  
  Deterministic entity and account generation for buyers, sellers, addresses, and hardware fingerprints.
- **`order_return_generator.py`**:  
  E-commerce transaction and return lifecycle simulation under realistic arrival distributions.
- **`product_listing_generator.py`**:  
  Catalog and listing generation matching real product taxonomy and price dynamics.
- **`fraud_injection.py`**:  
  Causal fraud ring topology injection, device sharing collusion, and return abuse pattern synthesis.
- **`robustness.py`**:  
  Statistical bootstrap validation and sensitivity analysis across multi-seed evaluations.

---

## 🧪 Running the Intelligence Test Suite

```bash
# 1. Run all unit and regression tests in this directory
pytest trustshield_project/ -v

# 2. Run TreeSHAP explainability tests
pytest trustshield_project/test_shap_explainer.py -v

# 3. Run multimodal CLIP scoring tests
pytest trustshield_project/test_multimodal_clip.py -v

# 4. Run temporal leakage and data integrity guards
pytest trustshield_project/test_leakage.py trustshield_project/test_data_integrity.py -v

# 5. Run CI metric guard negative tests
pytest trustshield_project/test_ci_guard_negative.py -v
```

---

## 📚 Governance & Documentation

Authoritative system architecture, evaluation, and operational specifications are maintained in [`docs/`](../docs/README.md):
- [**Model Card**](../docs/MODEL_CARD.md)
- [**Operations & Retraining Guide**](../docs/OPERATIONS.md)
- [**Deployment Architecture**](../docs/DEPLOYMENT.md)
- [**Reliability & Resilience Guide**](../docs/RELIABILITY.md)
- [**Investigation API Guide**](../docs/INVESTIGATION_API.md)
- [**Audit Traceability Matrix**](../docs/AUDIT_TRACEABILITY.md)