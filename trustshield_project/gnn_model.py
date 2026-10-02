"""
TrustShield AI — Phase 3 (GNN upgrade)
A real GraphSAGE model via PyTorch Geometric, replacing graph_features.py's
networkx-topological-feature approximation with actual learned graph
embeddings.

CANNOT BE TESTED IN THE SANDBOX THIS PROJECT WAS BUILT IN — no torch, no
torch_geometric, no network access to install them there. This file is
written carefully against documented, stable PyTorch Geometric APIs
(SAGEConv) but has not been executed. Please run it locally and
send back the exact console output/traceback — errors here get fixed
from real feedback, not guessed at.

Install locally (CPU is fine for this graph's size):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install torch_geometric

Temporal-safety design (the tricky part, same principle as graph_features.py):
- ONE graph is built from TRAIN+VAL period orders only, plus the static
  buyer-buyer device/address-sharing edges (undated in the current schema,
  same documented limitation as elsewhere in this project).
- The encoder (GraphSAGE) and classifier head are trained/validated using
  ONLY that graph.
- For TEST predictions, node embeddings are computed by a forward pass over
  the SAME train+val graph (test-period orders are NEVER added as edges) —
  so no test-period structure ever informs an embedding used to predict a
  test-period order. This avoids the subtle leakage risk of a transductive
  GNN "seeing" the very edges (or their neighborhood effects) it's asked
  to classify.
- Known, documented simplification: a buyer/seller with zero TRAIN+VAL
  orders is still a node (so it always has an embedding), but that
  embedding carries no message-passing signal — it's effectively just a
  transform of that node's own static features. This is a real limitation
  (cold-start nodes), not hidden leakage.
"""

import numpy as np
import pandas as pd
import networkx as nx

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
except ImportError as e:
    raise SystemExit(
        "This script needs torch + torch_geometric, which aren't installed here.\n"
        "Install locally with:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "  pip install torch_geometric\n"
        f"(underlying import error: {e})"
    )

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from entity_generator import build_base_entities
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud
from baseline_model import build_features as build_tabular_features, TRAIN_END, VAL_END
from graph_features import build_relationship_graph


# ---------------------------------------------------------------------------
# Node index bookkeeping
# ---------------------------------------------------------------------------

def build_node_index(buyer_ids, seller_ids):
    buyer_idx = {b: i + 1 for i, b in enumerate(sorted(buyer_ids))}
    n_buyers = len(buyer_idx)
    seller_idx = {s: n_buyers + 1 + i for i, s in enumerate(sorted(seller_ids))}
    n_total = n_buyers + len(seller_idx) + 1
    return buyer_idx, seller_idx, n_buyers, n_total


# ---------------------------------------------------------------------------
# Node features — temporally-safe: computed only from TRAIN+VAL orders
# ---------------------------------------------------------------------------

def build_node_features(buyer_idx, seller_idx, n_total, trainval_orders, rel_graph, buyers_df, sellers_df):
    """
    Feature layout (padded to a common width across buyer/seller nodes,
    plus a node-type one-hot so the encoder can tell them apart):
      [is_buyer, is_seller, age_days, orders_or_listings_count,
       returns_or_avg_price_norm, share_degree, share_component_size]
    """
    feat_dim = 7
    X = np.zeros((n_total, feat_dim), dtype=np.float32)

    buyer_order_counts = trainval_orders.groupby("buyer_id").size()
    components = {node: comp for comp in nx.connected_components(rel_graph) for node in comp}

    buyers_df = buyers_df.set_index("buyer_id")
    for b, idx in buyer_idx.items():
        age = 0
        if b in buyers_df.index:
            age = max(0, (VAL_END - pd.to_datetime(buyers_df.loc[b, "signup_date"])).days)
        X[idx, 0] = 1.0  # is_buyer
        X[idx, 2] = age / 365.0
        X[idx, 3] = np.log1p(float(buyer_order_counts.get(b, 0)))
        X[idx, 4] = 0.0  # reserved
        if b in rel_graph:
            X[idx, 5] = rel_graph.degree[b]
            X[idx, 6] = len(components.get(b, {b}))

    seller_order_counts = trainval_orders.groupby("seller_id").size()
    seller_avg_amount = trainval_orders.groupby("seller_id")["amount"].mean()
    sellers_df = sellers_df.set_index("seller_id")
    for s, idx in seller_idx.items():
        age = 0
        if s in sellers_df.index:
            age = max(0, (VAL_END - pd.to_datetime(sellers_df.loc[s, "signup_date"])).days)
        X[idx, 1] = 1.0  # is_seller
        X[idx, 2] = age / 365.0
        X[idx, 3] = np.log1p(float(seller_order_counts.get(s, 0)))
        X[idx, 4] = np.log1p(float(seller_avg_amount.get(s, 0.0)) / 1000.0)  # rough normalization

    return torch.tensor(X, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

def build_edge_index(buyer_idx, seller_idx, trainval_orders, rel_graph):
    edges = []
    # Deduplicate edges between same buyer and seller
    unique_txns = set(zip(trainval_orders["buyer_id"], trainval_orders["seller_id"]))
    for b, s in unique_txns:
        if b in buyer_idx and s in seller_idx:
            edges.append((buyer_idx[b], seller_idx[s]))
            edges.append((seller_idx[s], buyer_idx[b]))  # undirected

    for u, v in rel_graph.edges():
        if u in buyer_idx and v in buyer_idx:
            edges.append((buyer_idx[u], buyer_idx[v]))
            edges.append((buyer_idx[v], buyer_idx[u]))

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    
    # Optional deduplication in case of multi-edges across different graphs
    edges = list(set(edges))
    
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=16):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class EdgeClassifier(nn.Module):
    def __init__(self, emb_dim, edge_feat_dim, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, buyer_emb, seller_emb, edge_feats):
        combined = torch.cat([buyer_emb, seller_emb, edge_feats], dim=1)
        return self.mlp(combined).squeeze(-1)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_gnn():
    print("Building full pipeline...")
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"])
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"], catalog["products"],
        base["address_sharing_log"], base["device_sharing_log"],
    )

    print("Building tabular features (reused as edge features)...")
    df, tabular_cols = build_tabular_features(
        result["orders"], result["listings"], result["returns"], txn["buyers"], catalog["sellers"], catalog["products"]
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["y"] = df["is_fraudulent"].astype(int)
    edge_feature_cols = ["price_vs_base_price_ratio", "amount"]  # small, deliberately — keep the GNN's embeddings doing the work

    trainval = df[df["order_date"] <= VAL_END]
    train = df[df["order_date"] <= TRAIN_END]
    val = df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)]
    test = df[df["order_date"] > VAL_END]
    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

    print("Building graph (train+val orders + static buyer-buyer sharing edges)...")
    buyer_idx, seller_idx, n_buyers, n_total = build_node_index(
        txn["buyers"]["buyer_id"], catalog["sellers"]["seller_id"]
    )
    rel_graph = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"])
    X = build_node_features(buyer_idx, seller_idx, n_total, trainval, rel_graph, txn["buyers"], catalog["sellers"])
    edge_index = build_edge_index(buyer_idx, seller_idx, trainval, rel_graph)
    print(f"  Nodes: {n_total} ({n_buyers} buyers + {n_total - n_buyers} sellers), Edges: {edge_index.shape[1]}")

    encoder = GraphSAGEEncoder(in_dim=X.shape[1])
    classifier = EdgeClassifier(emb_dim=16, edge_feat_dim=len(edge_feature_cols))
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.002)

    def make_edge_tensors(split_df):
        b_ids = split_df["buyer_id"].map(buyer_idx).fillna(0).astype(int).to_numpy()
        s_ids = split_df["seller_id"].map(seller_idx).fillna(0).astype(int).to_numpy()
        
        feat_arr = split_df[edge_feature_cols].fillna(0).copy()
        feat_arr["amount"] = np.log1p(feat_arr["amount"].clip(lower=0))
        feats = torch.tensor(feat_arr.to_numpy(), dtype=torch.float32)
        
        labels = torch.tensor(split_df["y"].to_numpy(), dtype=torch.float32)
        return torch.tensor(b_ids, dtype=torch.long), torch.tensor(s_ids, dtype=torch.long), feats, labels

    train_b, train_s, train_feats, train_y = make_edge_tensors(train)
    val_b, val_s, val_feats, val_y = make_edge_tensors(val)
    test_b, test_s, test_feats, test_y = make_edge_tensors(test)

    # Class-imbalance handling (same principle as the sklearn class_weight
    # used elsewhere): weight the minority (fraud) class in the loss.
    pos_weight = torch.tensor([(train_y == 0).sum().item() / max((train_y == 1).sum().item(), 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("\nTraining GraphSAGE encoder + edge classifier...")
    for epoch in range(1, 51):
        encoder.train(); classifier.train()
        optimizer.zero_grad()
        node_emb = encoder(X, edge_index)
        logits = classifier(node_emb[train_b], node_emb[train_s], train_feats)
        loss = loss_fn(logits, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(classifier.parameters()), 1.0)
        optimizer.step()

        if epoch % 10 == 0:
            encoder.eval(); classifier.eval()
            with torch.no_grad():
                node_emb_eval = encoder(X, edge_index)  # same train+val graph — no test edges added
                val_logits = classifier(node_emb_eval[val_b], node_emb_eval[val_s], val_feats)
                val_probs = torch.sigmoid(val_logits).numpy()
                val_auc = roc_auc_score(val_y.numpy(), val_probs)
            print(f"  epoch {epoch:3d}  train_loss={loss.item():.4f}  val_ROC-AUC={val_auc:.3f}")

    print("\nFinal evaluation (test set — embeddings computed WITHOUT any test-period edges):")
    encoder.eval(); classifier.eval()
    with torch.no_grad():
        node_emb_final = encoder(X, edge_index)
        test_logits = classifier(node_emb_final[test_b], node_emb_final[test_s], test_feats)
        test_probs = torch.sigmoid(test_logits).numpy()
    test_preds = (test_probs >= 0.5).astype(int)
    y_test_np = test_y.numpy()

    print(f"Precision: {precision_score(y_test_np, test_preds, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test_np, test_preds, zero_division=0):.3f}")
    print(f"F1:        {f1_score(y_test_np, test_preds, zero_division=0):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test_np, test_probs):.3f}")
    print(f"PR-AUC:    {average_precision_score(y_test_np, test_probs):.3f}")
    print("\nCompare these numbers against graph_features.py's tabular+graph ablation "
          "(networkx-topological-features version) to see whether learned GraphSAGE "
          "embeddings beat hand-engineered graph features on this dataset.")


if __name__ == "__main__":
    run_gnn()
