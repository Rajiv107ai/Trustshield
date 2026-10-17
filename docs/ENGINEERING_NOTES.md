# TrustShield AI — Engineering & Research Notes

## 1. Graph Neural Network (GNN) Negative Result

### Summary of Experimentation
As part of the Phase 5 research architecture, an out-of-fold GraphSAGE GNN encoder was evaluated to determine whether graph representation learning could outperform gradient boosted decision trees trained on point-in-time (PIT) topological graph features.

<!-- BEGIN GENERATED -->
### Empirical Findings (from `results/results.json` and `results/RESULTS.md`)
- **Baseline Model (Phase 3):** XGBoost with 10 tabular features + 8 point-in-time NetworkX graph features (ROC-AUC: 0.7410 +/- 0.0229, PR-AUC: 0.4461 +/- 0.0276).
- **GNN Hybrid Model (Phase 5):** Integrating 16-dimensional out-of-fold GraphSAGE embeddings into XGBoost.
- **Statistical Result:** Net negative lift of -0.0263 ROC-AUC (p = 0.0083, statistically significant degradation).
<!-- END GENERATED -->

### Engineering Decision
Graph representation embeddings from shallow bipartite graphs add substantial parameter overhead, increased cold-start latency, and PyTorch/PyG dependency weight while demonstrably degrading generalization accuracy. 

Consequently, the GNN stack (`gnn_model.py`, `hetero_gnn.py`, `temporal_gnn.py`, `phase5_hybrid_model.py`) was purged from the serving layer. The authoritative production engine serves the Phase 3 combined XGBoost model exclusively.
