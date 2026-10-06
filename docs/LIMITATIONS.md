# TrustShield AI — System Limitations & Research Disclosure

**Document Revision:** 2026-10-07  
**Scope:** Phase 1 Scientific Assessment & Technical Limitations  

---

## 1. Synthetic Data Distribution vs. Real-World Production

1. **Synthetic Fraud Behavioral Generative Priors:**  
   The training and evaluation sets are generated using synthetic generators (`entity_generator.py`, `order_return_generator.py`, `fraud_injection.py`). While parameters mirror realistic marketplace statistical mechanics, synthetic behavioral distributions possess lower variance, fewer edge cases, and more predictable noise structures than live production traffic.
2. **Missing Label Feedback Delays:**  
   In live production fraud systems, chargebacks and fraud reports arrive with a **30 to 90 day lag** (chargeback delay). The synthetic simulator labels fraud synchronously at transaction creation. In real deployment, models must be trained with delayed feedback and unconfirmed label estimators.

---

## 2. Graph Computation & Scalability Limits

1. **In-Memory NetworkX Bottleneck:**  
   The current graph analytics engine builds homogeneous and multi-relational graphs using Python's `networkx.Graph` and `MultiDiGraph`. While optimal for sub-100k entity datasets, NetworkX stores pointers in Python heap memory, causing memory usage to scale $O(V + E)$ with significant overhead (~1–2 KB per node).
2. **Dynamic Live Graph Query Latency:**  
   Computing connected component sizes and personalized PageRank at scoring time currently relies on pre-computed snapshots. True real-time sub-10ms graph traversals at e-commerce scale require dedicated graph databases (Neo4j, Memgraph) or low-latency graph stores with localized k-hop sampling.

---

## 3. Deep GNN vs. Classical Tabular Reality

1. **GNN Over-Smoothing & Over-Squashing:**  
   Experimental evidence in TrustShield shows standalone 2-layer GCN achieving only $0.519$ ROC-AUC. Transaction topologies have massive hubs (popular merchants and delivery hubs) connected to honest buyers. Uniform message passing aggregates noise across these dense hubs, washing out subtle fraud signals.
2. **Need for Heterogeneous Edge-Type Attention:**  
   To outperform tabular boosted trees, graph architectures must use heterogeneous relation attention (e.g. RGCN or HGT) with learned edge-type filtering, separating high-risk hardware sharing from benign address sharing.

---

## 4. Cold-Start Entities & Epistemic Uncertainty

1. **Zero-History Blind Spots:**  
   When a completely new buyer registers, places an order, and uses a pristine hardware ID, both graph degree and historical velocity features evaluate to zero.
2. **Mitigation in Phase 1:**  
   TrustShield mitigates this by assigning an explicit `cold_start: True` flag, penalizing confidence by $30\%$, and falling back to listing anomaly and price deviation signals. However, cold-start fraud detection inherently carries lower precision than established account monitoring.

---

## 5. Adversarial Adaptation & Evasion

1. **Threshold Gaming:**  
   If malicious actors deduce the `ALLOW` cutoffs (e.g., maintaining transaction amounts right below median category prices or artificially spacing out transactions), they can evade fixed rules.
2. **Mitigation:**  
   The Unified Trust Engine incorporates binary Shannon entropy confidence and model disagreement tracking. Significant divergence across tabular, graph, and velocity detectors triggers `HIGH_DETECTOR_DISAGREEMENT` and manual investigation routing.

---

## 6. Phase 2 Infrastructure Prerequisites

Phase 1 established scientific correctness, temporal validity, and core algorithms. Deploying this system to enterprise payment volume requires Phase 2 infrastructure extensions:
- **Distributed Streaming:** Apache Kafka / Flink for sub-second event ingestion
- **Low-Latency Feature Store:** Redis / Feast for online historical entity aggregations
- **Graph Database:** Neo4j / AWS Neptune for distributed k-hop subgraph traversal
- **Continuous Tracking:** MLflow / DVC for automated model registry and artifact hashing in CI/CD pipelines
