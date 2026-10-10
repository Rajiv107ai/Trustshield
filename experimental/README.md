# TrustShield Experimental Deep Architectures

This directory documents and archives exploratory graph neural network architectures investigated during TrustShield research.

## Architectures

### 1. Heterogeneous Graph Neural Network (`hetero_gnn.py`)
- **Framework**: PyTorch Geometric (`torch_geometric.data.HeteroData`, `HeteroConv`).
- **Entity Types**: `buyer`, `seller`, `device`, `address`.
- **Edge Types**: `uses_device`, `uses_address`, `transacts_with` (and reverse relations).
- **Status**: **Experimental Research Prototype**.
- **Empirical Assessment**: While multi-relational message passing captures multi-hop connectivity, the high degree skew and disconnected components in transaction graphs introduce diffusion noise. Standalone tabular tree ensembles consuming local topological features (degree, PageRank, connected component sizes) outperform message passing on holdout detection.

### 2. Temporal GNN with Continuous Decay (`temporal_gnn.py`)
- **Mechanism**: Learnable exponential time decay kernel ($\exp(-\lambda \Delta t)$) over historical edge sequences.
- **Status**: **Experimental Research Prototype**.
- **Empirical Assessment**: High forward pass inference latency (~135 ms vs 8 ms for tree models) without statistically significant AUC lift over point-in-time snapshot features.

## Canonical Production Pipeline
For production transaction risk scoring, TrustShield standardizes on:
- **Canonical Model**: Tabular + Historical Snapshot Graph Features (`baseline_model.py` + `graph_features.py`).
- **Classifier**: Calibrated XGBoost with Isotonic Regression.
- **Serving Engine**: `CanonicalTrustEngine` in `trustshield_project/advanced_trust_engine.py`.
