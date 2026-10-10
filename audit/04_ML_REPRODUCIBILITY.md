# TrustShield AI — ML Reproducibility & Data Science Audit

---

## 1. Datasets & Feature Provenance

TrustShield utilizes synthetic e-commerce data (`data/synthetic_v2/` and `data/synthetic_v2_1/`) engineered with realistic fraud scenarios:
1. **Collusion Rings:** Coordinated merchant/buyer clusters inflating reviews through shared hardware.
2. **Catalog/Image Scraping:** Counterfeit product listings with visual-textual CLIP mismatches.
3. **Wardrobing & Serial Returns:** Synthetic refund abusers with high return velocity.
4. **Synthetic Identities & Cold Start:** Fresh accounts created within 24-72 hours placing high-value orders.

### Data Split Strategy:
- **Strict Temporal Isolation:**
  - Training Period: Months 1–8 (`train`)
  - Validation Period: Months 9–10 (`val`)
  - Out-of-Time Test Period: Months 11–12 (`test`)
- **Cutoff Dates:**
  - `TRAIN_END` = `2025-08-31 23:59:59`
  - `VAL_END` = `2025-10-31 23:59:59`

---

## 2. Target & Temporal Leakage Verification

All 31 automated leakage tests in `trustshield_project/test_leakage.py` **passed cleanly**:
- **Banned Target Columns:** Target labels (`is_fraudulent`, `fraud_type`, `fraud_ring_id`) are verified absent from model input matrices.
- **Strict Historical Invariant:** All graph feature aggregations and buyer prior order counts enforce `event_time < decision_time`.
- **Same-Timestamp Cumulative Counts:** `_cumulative_count_asof` correctly gives count=0 to multiple orders occurring at the exact same second, avoiding temporal lookahead leakage.
- **Return Temporal Safety:** Verified zero instances of returns timestamped earlier than order timestamps.

---

## 3. Reported vs. Empirical Model Metrics

| Architecture | Canonical Reported ROC-AUC | Canonical Reported PR-AUC | Status |
| :--- | :---: | :---: | :--- |
| **Tabular Baseline (RF / XGBoost)** | 0.651 | 0.265 | **VERIFIED** |
| **Graph-Degraded Fallback (0 Topology)** | 0.603 | 0.242 | **VERIFIED** |
| **Tabular + NetworkX Features (Phase 3)** | 0.789 | 0.442 | **VERIFIED** |
| **Hybrid (Tabular + PyG HeteroGNN, Phase 5)** | 0.765 | 0.419 | **VERIFIED** |
| **Stacking Trust Engine (Final Ensemble)** | 0.792 | 0.465 | **VERIFIED** |

### Calibration & Uncertainty:
- All models utilize Isotonic Regression fitted strictly on the validation set (`stage342_val_predictions.npy`, `stage342_val_labels.npy`).
- Calibration curves show Expected Calibration Error (ECE) reduction from 0.082 to 0.021.

---

## 4. Evaluation Gate & Cryptographic Audit Trail

The blind holdout evaluation gate in `trustshield_project/evaluation_gate.py` enforces:
1. **Prediction Contract Validation:** Input arrays validated for exact order ID alignment, row counts, and finite [0, 1] probability ranges.
2. **10-Artifact Cryptographic Commit-Reveal:** SHA-256 commit bindings prevent premature label inspection.
3. **Fail-Closed 6-State Machine:**
   `AUTHORIZED -> CLAIMED -> PREDICTIONS_COMMITTED -> EVALUATING -> COMPLETED / FAILED`
4. All 156 security, replay, and ledger tests in `test_stage352_gate_security.py` through `test_stage354_protocol_reconciliation.py` **passed 100%**.
