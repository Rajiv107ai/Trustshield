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
