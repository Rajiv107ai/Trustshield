"""Advanced Fraud Ring Intelligence & Subgraph Anomaly Discovery for TrustShield.

Goes beyond naive connected components to evaluate multi-entity collusion structures:
- Candidate community discovery on multi-relational hardware and transaction graphs
- Cluster-level topological and temporal burstiness metrics:
    - Temporal order arrival dispersion (burstiness score)
    - Hardware identifier collision density
    - Merchant-buyer concentration (Herfindahl-Hirschman Index)
    - Coordinated return velocities
- Cluster Risk Scoring:
    - Classifies clusters into 'benign_cluster', 'suspicious_cluster', 'high_risk_candidate_ring'
- Strict scientific terminology: distinguishes candidate clusters from confirmed ground-truth rings.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import numpy as np
import pandas as pd
import networkx as nx


@dataclass(frozen=True)
class CandidateRingReport:
    cluster_id: str
    member_count: int
    order_count: int
    temporal_burstiness_score: float         # [0.0, 1.0] (high = burst clustered)
    hardware_sharing_density: float          # [0.0, 1.0]
    merchant_concentration_hhi: float        # [0.0, 1.0]
    coordinated_return_rate: float           # [0.0, 1.0]
    subgraph_edge_density: float             # [0.0, 1.0]
    composite_ring_risk: float               # [0.0, 1.0]
    cluster_classification: str              # 'benign_cluster' | 'suspicious_cluster' | 'high_risk_candidate_ring'
    member_ids: List[str]


def discover_candidate_communities(
    multi_graph: nx.Graph,
    min_size: int = 3,
    max_size: int = 100,
) -> List[Set[str]]:
    """Discover candidate collusion clusters using connected components or community modularity."""
    undirected = multi_graph.to_undirected()
    components = [{str(node) for node in c} for c in nx.connected_components(undirected)]
    # Filter valid candidate sizes
    return [c for c in components if min_size <= len(c) <= max_size]


def analyze_candidate_community(
    cluster_nodes: Set[str],
    orders_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    multi_graph: nx.MultiGraph,
    cluster_id: str = "CLUSTER_001",
) -> CandidateRingReport:
    """Extract deep temporal, relational, and behavioral features for a candidate community."""
    n_members = len(cluster_nodes)

    # Filter cluster orders
    cluster_list = list(cluster_nodes)
    c_orders = orders_df[
        orders_df["buyer_id"].isin(cluster_list) | orders_df["seller_id"].isin(cluster_list)
    ]
    n_orders = len(c_orders)

    # 1. Temporal Burstiness
    if n_orders >= 3 and "order_date" in c_orders.columns:
        dates = pd.to_datetime(c_orders["order_date"]).sort_values()
        diffs = (dates.diff().dt.total_seconds() / 3600.0).dropna()
        if len(diffs) > 1 and diffs.mean() > 0:
            # Coefficient of variation of inter-arrival time
            cv = float(diffs.std() / (diffs.mean() + 1e-5))
            burstiness = float(min(1.0, cv / 3.0))
        else:
            burstiness = 0.5
    else:
        burstiness = 0.2

    # 2. Hardware / Device Sharing Density
    sub = multi_graph.subgraph(cluster_nodes)
    n_edges = sub.number_of_edges()
    dev_edges = 0
    for u, v, k, d in sub.edges(keys=True, data=True):
        if d.get("relation") == "uses_device" or "device" in str(k):
            dev_edges += 1
    hw_density = float(dev_edges / max(n_edges, 1))

    # 3. Merchant Concentration HHI
    if n_orders > 0 and "seller_id" in c_orders.columns:
        seller_shares = np.asarray(c_orders["seller_id"].value_counts(normalize=True))
        hhi = float(np.sum(seller_shares ** 2))
    else:
        hhi = 0.0

    # 4. Coordinated Return Rate
    if n_orders > 0 and "order_id" in c_orders.columns:
        order_ids = list(set(c_orders["order_id"]))
        c_returns = returns_df[returns_df["order_id"].isin(order_ids)]
        ret_rate = float(len(c_returns) / n_orders)
    else:
        ret_rate = 0.0

    # 5. Internal Subgraph Edge Density
    possible_edges = (n_members * (n_members - 1)) / 2.0
    edge_density = float(min(1.0, n_edges / max(possible_edges, 1.0)))

    # Composite Ring Risk (Heuristic weighted prior)
    risk = (
        0.30 * burstiness +
        0.30 * hw_density +
        0.20 * hhi +
        0.10 * ret_rate +
        0.10 * edge_density
    )
    risk = float(np.clip(risk, 0.0, 1.0))

    if risk >= 0.70:
        classification = "high_risk_candidate_ring"
    elif risk >= 0.40:
        classification = "suspicious_cluster"
    else:
        classification = "benign_cluster"

    return CandidateRingReport(
        cluster_id=cluster_id,
        member_count=n_members,
        order_count=n_orders,
        temporal_burstiness_score=round(burstiness, 4),
        hardware_sharing_density=round(hw_density, 4),
        merchant_concentration_hhi=round(hhi, 4),
        coordinated_return_rate=round(ret_rate, 4),
        subgraph_edge_density=round(edge_density, 4),
        composite_ring_risk=round(risk, 4),
        cluster_classification=classification,
        member_ids=sorted(list(cluster_nodes)),
    )
