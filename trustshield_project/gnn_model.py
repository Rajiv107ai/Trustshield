"""GraphSAGE inductive graph neural network for transaction fraud detection."""

from typing import Any, cast
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from entity_generator import build_base_entities
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud
from baseline_model import build_features as build_tabular_features, TRAIN_END, VAL_END
from graph_features import build_relationship_graph


def build_node_index(buyer_ids, seller_ids):
    """Contiguous 1-indexed mapping for bipartite graph nodes."""
    buyer_idx = {b: i + 1 for i, b in enumerate(sorted(buyer_ids))}
    n_buyers = len(buyer_idx)
    seller_idx = {s: n_buyers + 1 + i for i, s in enumerate(sorted(seller_ids))}
    n_total = n_buyers + len(seller_idx) + 1
    return buyer_idx, seller_idx, n_buyers, n_total


def build_node_features(buyer_idx: dict, seller_idx: dict, n_total: int, orders_df: pd.DataFrame, 
                        rel_graph: nx.Graph, buyers_df: pd.DataFrame, sellers_df: pd.DataFrame,
                        cutoff_date=None) -> torch.Tensor:
    """Builds node feature matrix [is_buyer, is_seller, age_days, orders/listings, log_amount, degree, component_size]."""
    X = np.zeros((n_total, 7), dtype=np.float32)
    ref_date = pd.Timestamp(cutoff_date) if cutoff_date is not None else VAL_END
    buyer_order_counts = orders_df.groupby("buyer_id").size().to_dict()
    components = {node: comp for comp in nx.connected_components(rel_graph) for node in comp}

    buyers_indexed = buyers_df.set_index("buyer_id")
    for b, idx in buyer_idx.items():
        age = max(0, (ref_date - pd.to_datetime(buyers_indexed.loc[b, "signup_date"])).days) if b in buyers_indexed.index else 0
        X[idx, 0] = 1.0
        X[idx, 2] = age / 365.0
        X[idx, 3] = np.log1p(float(buyer_order_counts.get(b, 0)))
        if b in rel_graph:
            X[idx, 5] = rel_graph.degree[b]
            X[idx, 6] = len(components.get(b, {b}))

    seller_order_counts = pd.Series(orders_df.groupby("seller_id").size()).to_dict()
    seller_avg_amount = pd.Series(orders_df.groupby("seller_id")["amount"].mean()).to_dict()
    sellers_indexed = sellers_df.set_index("seller_id")
    for s, idx in seller_idx.items():
        age = max(0, (ref_date - pd.to_datetime(sellers_indexed.loc[s, "signup_date"])).days) if s in sellers_indexed.index else 0
        X[idx, 1] = 1.0
        X[idx, 2] = age / 365.0
        X[idx, 3] = np.log1p(float(seller_order_counts.get(s, 0)))
        X[idx, 4] = np.log1p(float(seller_avg_amount.get(s, 0.0)) / 1000.0)

    return torch.tensor(X, dtype=torch.float32)


def build_edge_index(buyer_idx: dict, seller_idx: dict, orders_df: pd.DataFrame, rel_graph: nx.Graph) -> torch.Tensor:
    """Constructs PyG undirected edge_index from order interactions and shared devices/addresses."""
    edges = set()
    for b, s in zip(orders_df["buyer_id"], orders_df["seller_id"]):
        if b in buyer_idx and s in seller_idx:
            u, v = buyer_idx[b], seller_idx[s]
            edges.add((u, v))
            edges.add((v, u))

    for u, v in rel_graph.edges():
        if u in buyer_idx and v in buyer_idx:
            bu, bv = buyer_idx[u], buyer_idx[v]
            edges.add((bu, bv))
            edges.add((bv, bu))

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(list(edges), dtype=torch.long).t().contiguous()


class GraphSAGEEncoder(nn.Module):
    """Two-layer GraphSAGE structural node encoder."""
    def __init__(self, in_dim: int, hidden_dim: int = 32, out_dim: int = 16):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class EdgeClassifier(nn.Module):
    """MLP classifier operating on concatenated node embeddings and edge attributes."""
    def __init__(self, emb_dim: int, edge_feat_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, buyer_emb, seller_emb, edge_feats):
        return self.mlp(torch.cat([buyer_emb, seller_emb, edge_feats], dim=1)).squeeze(-1)


def run_gnn():
    """Trains and evaluates the GraphSAGE fraud detection model."""
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"])
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"], catalog["products"],
        base["address_sharing_log"], base["device_sharing_log"],
    )

    df, _ = build_tabular_features(
        result["orders"], result["listings"], result["returns"], txn["buyers"], catalog["sellers"], catalog["products"]
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["y"] = df["is_fraudulent"].astype(int)
    edge_feature_cols = ["price_vs_base_price_ratio", "amount"]

    trainval = df[df["order_date"] <= VAL_END]
    train = df[df["order_date"] <= TRAIN_END]
    val = df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)]
    test = df[df["order_date"] > VAL_END]

    buyer_idx, seller_idx, n_buyers, n_total = build_node_index(txn["buyers"]["buyer_id"], catalog["sellers"]["seller_id"])
    rel_graph_train = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"], cutoff_date=TRAIN_END)
    rel_graph_val = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"], cutoff_date=VAL_END)

    X_train = build_node_features(buyer_idx, seller_idx, n_total, train, rel_graph_train, txn["buyers"], catalog["sellers"], cutoff_date=TRAIN_END)
    edge_index_train = build_edge_index(buyer_idx, seller_idx, train, rel_graph_train)

    X_val = build_node_features(buyer_idx, seller_idx, n_total, trainval, rel_graph_val, txn["buyers"], catalog["sellers"], cutoff_date=VAL_END)
    edge_index_val = build_edge_index(buyer_idx, seller_idx, trainval, rel_graph_val)

    encoder = GraphSAGEEncoder(in_dim=X_train.shape[1])
    classifier = EdgeClassifier(emb_dim=16, edge_feat_dim=len(edge_feature_cols))
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.002)

    def make_edge_tensors(split_df):
        b_ids = split_df["buyer_id"].map(buyer_idx).fillna(0).astype(int).to_numpy()
        s_ids = split_df["seller_id"].map(seller_idx).fillna(0).astype(int).to_numpy()
        feat_arr = split_df[edge_feature_cols].fillna(0).copy()
        feat_arr["amount"] = np.log1p(feat_arr["amount"].clip(lower=0))
        return (
            torch.tensor(b_ids, dtype=torch.long),
            torch.tensor(s_ids, dtype=torch.long),
            torch.tensor(feat_arr.to_numpy(), dtype=torch.float32),
            torch.tensor(split_df["y"].to_numpy(), dtype=torch.float32),
        )

    train_b, train_s, train_feats, train_y = make_edge_tensors(train)
    val_b, val_s, val_feats, val_y = make_edge_tensors(val)
    test_b, test_s, test_feats, test_y = make_edge_tensors(test)

    pos_weight = torch.tensor([(train_y == 0).sum().item() / max((train_y == 1).sum().item(), 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("Training GraphSAGE model...")
    for epoch in range(1, 51):
        encoder.train()
        classifier.train()
        optimizer.zero_grad()
        node_emb = encoder(X_train, edge_index_train)
        logits = classifier(node_emb[train_b], node_emb[train_s], train_feats)
        loss = loss_fn(logits, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(classifier.parameters()), 1.0)
        optimizer.step()

        if epoch % 10 == 0:
            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                node_emb_eval = encoder(X_train, edge_index_train)
                val_probs = torch.sigmoid(classifier(node_emb_eval[val_b], node_emb_eval[val_s], val_feats)).numpy()
                val_auc = roc_auc_score(val_y.numpy(), val_probs)
            print(f"  epoch {epoch:02d} | train_loss={loss.item():.4f} | val_auc={val_auc:.3f}")

    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        node_emb_final = encoder(X_val, edge_index_val)
        test_probs = torch.sigmoid(classifier(node_emb_final[test_b], node_emb_final[test_s], test_feats)).numpy()

    test_preds = (test_probs >= 0.5).astype(int)
    y_test_np = test_y.numpy()

    print(f"\nGNN Test Metrics:")
    print(f"Precision: {precision_score(y_test_np, test_preds, zero_division=cast(Any, 0)):.3f}")
    print(f"Recall:    {recall_score(y_test_np, test_preds, zero_division=cast(Any, 0)):.3f}")
    print(f"F1:        {f1_score(y_test_np, test_preds, zero_division=cast(Any, 0)):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test_np, test_probs):.3f}")
    print(f"PR-AUC:    {average_precision_score(y_test_np, test_probs):.3f}")


if __name__ == "__main__":
    run_gnn()
