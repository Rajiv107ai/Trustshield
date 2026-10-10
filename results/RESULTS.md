# TrustShield Benchmark Results (5 Seeds, Out-of-Time Test)

> [!NOTE]
> Produced under strict audit controls: point-in-time snapshot graphs, out-of-fold embeddings, no synthetic surrogate tuning.

- **Git Commit**: `0afe768d9393ffae97e4ab3c8fe698ee08bdf044` (dirty: `True`)
- **Timestamp**: `2026-10-10T11:17:04Z`
- **Seeds**: [42, 101, 202, 303, 404]

## 1. Pre-Registered Side-by-Side Model Performance

| Model | Variant | Test ROC-AUC (mean ± std) | Test PR-AUC (mean ± std) | Lift vs Tabular (ROC-AUC) | Lift vs Tabular (PR-AUC) |
|---|---|---|---|---|---|
| **Tabular XGBoost** | Standard | 0.7276 ± 0.0296 | 0.4386 ± 0.0319 | — | — |
| **Tabular + Graph** | Standard | 0.7399 ± 0.0315 | 0.4482 ± 0.0372 | +0.0123 (p=0.1145) | +0.0096 (p=0.1627) |
| **Hybrid (Time-Aware OOF GNN)** | Standard | 0.7022 ± 0.0381 | 0.4210 ± 0.0370 | -0.0255 (p=0.1139) | -0.0176 (p=0.2425) |
| **Tabular XGBoost** | Coherent | 0.7935 ± 0.0405 | 0.4941 ± 0.0525 | — | — |
| **Tabular + Graph** | Coherent | 0.8001 ± 0.0380 | 0.4982 ± 0.0503 | +0.0066 (p=0.0340) | +0.0040 (p=0.1296) |
| **Hybrid (Time-Aware OOF GNN)** | Coherent | 0.7058 ± 0.0543 | 0.3957 ± 0.0498 | -0.0877 (p=0.0058) | -0.0984 (p=0.0049) |

## 2. Ring Coherence Diagnostics

| Metric | Standard Variant | Coherent Variant |
|---|---|---|
| **Burst Orders with Shared Device** | 77.09% ± 0.94% | 100.00% ± 0.00% |
| **Sharing Log First-Seen <= Burst Date** | 63.47% ± 3.27% | 100.00% ± 0.00% |

## 3. Recall@5% Budget per Fraud Scenario (Tabular vs Tabular + Graph)

| Fraud Scenario | Standard Tabular | Standard Graph | Coherent Tabular | Coherent Graph |
|---|---|---|---|---|
| **fake_listing** | 0.527 | 0.497 | 0.509 | 0.495 |
| **return_abuse** | 0.724 | 0.713 | 0.707 | 0.691 |
| **coordinated_fraud** | 0.087 | 0.119 | 0.128 | 0.152 |
| **seller_buyer_collusion** | 0.025 | 0.024 | 0.082 | 0.079 |

## 4. Fake Listing Detector — Multimodal Honesty (Phase 3)

> [!WARNING]
> ABO images are NOT present locally (`ABO_DATA_DIR` unset). The surrogate feature (`synthetic_mismatch_score`) uses TF-IDF + Gaussian noise (sigma=0.45) on `displayed_product_id`, which directly reflects the synthetic label swap mechanism. The surrogate score is **never** termed a "multimodal win". The **Headline** result is the honest tabular-only model.

| Evaluation Row | Feature Set | Test ROC-AUC | Test PR-AUC | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|---|---|
| **1. HEADLINE (Honest)** | Tabular Only (4 features, without surrogate) | **0.809** | **0.572** | 0.563 | 0.556 | 0.560 |
| **2. Diagnostic** | Tabular + Synthetic Mismatch Score | 0.944 | 0.770 | 0.658 | 0.751 | 0.702 |
| **3. Canary Alone** | Synthetic Mismatch Score Alone | 0.923 | 0.530 | 0.116 | 0.852 | 0.205 |

## 5. Return Fraud Detector (Right-Censoring Exclusion)

Orders placed in the final 21 days of the horizon (after 2025-12-10) are excluded from the test split to prevent right-censoring distortion.

| Model | Test Set N | Test ROC-AUC | Test PR-AUC | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 1,100 | 0.910 | 0.858 | 0.780 | 0.675 | 0.724 |
| **Random Forest** | 1,100 | 0.954 | 0.910 | 0.890 | 0.617 | 0.729 |
| **XGBoost (Headline)** | 1,100 | **0.953** | **0.906** | 0.818 | 0.696 | 0.752 |
| **XGBoost @ Cost Threshold (0.02)** | 1,100 | **0.953** | **0.906** | 0.682 | 0.953 | 0.795 |

