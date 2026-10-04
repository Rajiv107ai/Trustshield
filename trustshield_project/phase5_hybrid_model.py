"""Hybrid fraud architecture: GraphSAGE learned node embeddings combined with XGBoost."""

from typing import Any, cast
import sys
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier

from entity_generator import build_base_entities, SIM_START
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud
from baseline_model import build_features as build_tabular_features, TRAIN_END, VAL_END
from graph_features import (
    build_relationship_graph,
    compute_relationship_features,
    build_monthly_snapshots,
    attach_snapshot_features,
    add_edge_weight_before,
)

if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")

GNN_EMB_DIM = 16
GNN_HIDDEN_DIM = 32
GNN_NODE_FEAT_DIM = 7
GNN_EDGE_FEAT_COLS = ["price_vs_base_price_ratio", "amount"]


def build_node_index(buyer_ids, seller_ids):
    """Contiguous index mapping for bipartite graph nodes."""
    buyer_idx = {b: i + 1 for i, b in enumerate(sorted(buyer_ids))}
    n_buyers = len(buyer_idx)
    seller_idx = {s: n_buyers + 1 + i for i, s in enumerate(sorted(seller_ids))}
    return buyer_idx, seller_idx, n_buyers, n_buyers + len(seller_idx) + 1


def build_node_features(buyer_idx, seller_idx, n_total, trainval_orders, rel_graph, buyers_df, sellers_df):
    """Builds node feature matrix for graph encoder."""
    X = np.zeros((n_total, GNN_NODE_FEAT_DIM), dtype=np.float32)
    buyer_order_counts = trainval_orders.groupby("buyer_id").size()
    components = {node: comp for comp in nx.connected_components(rel_graph) for node in comp}

    buyers_indexed = buyers_df.set_index("buyer_id")
    for b, idx in buyer_idx.items():
        age = max(0, (VAL_END - pd.to_datetime(buyers_indexed.loc[b, "signup_date"])).days) if b in buyers_indexed.index else 0
        X[idx, 0] = 1.0
        X[idx, 2] = age / 365.0
        X[idx, 3] = np.log1p(float(buyer_order_counts.get(b, 0)))
        if b in rel_graph:
            X[idx, 5] = rel_graph.degree[b]
            X[idx, 6] = len(components.get(b, {b}))

    seller_order_counts = trainval_orders.groupby("seller_id").size()
    seller_avg_amount = trainval_orders.groupby("seller_id")["amount"].mean()
    sellers_indexed = sellers_df.set_index("seller_id")
    for s, idx in seller_idx.items():
        age = max(0, (VAL_END - pd.to_datetime(sellers_indexed.loc[s, "signup_date"])).days) if s in sellers_indexed.index else 0
        X[idx, 1] = 1.0
        X[idx, 2] = age / 365.0
        X[idx, 3] = np.log1p(float(seller_order_counts.get(s, 0)))
        X[idx, 4] = np.log1p(float(seller_avg_amount.get(s, 0.0)) / 1000.0)

    return torch.tensor(X, dtype=torch.float32)


def build_edge_index(buyer_idx, seller_idx, trainval_orders, rel_graph):
    """Builds PyG undirected edge tensor."""
    edges = set()
    for b, s in zip(trainval_orders["buyer_id"], trainval_orders["seller_id"]):
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
    def __init__(self, in_dim=GNN_NODE_FEAT_DIM, hidden_dim=GNN_HIDDEN_DIM, out_dim=GNN_EMB_DIM):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class EdgeClassifier(nn.Module):
    def __init__(self, emb_dim=GNN_EMB_DIM, edge_feat_dim=len(GNN_EDGE_FEAT_COLS), hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, buyer_emb, seller_emb, edge_feats):
        return self.mlp(torch.cat([buyer_emb, seller_emb, edge_feats], dim=1)).squeeze(-1)


def train_sage_encoder(X, edge_index, train_b, train_s, train_feats, train_y, val_b, val_s, val_feats, val_y, epochs=50):
    """Trains GraphSAGE encoder and checkpoint selects highest validation AUC."""
    encoder = GraphSAGEEncoder()
    head = EdgeClassifier()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=0.002)
    pos_weight = torch.tensor([(train_y == 0).sum().item() / max((train_y == 1).sum().item(), 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0
    best_enc_state = None

    for epoch in range(1, epochs + 1):
        encoder.train()
        head.train()
        optimizer.zero_grad()
        node_emb = encoder(X, edge_index)
        logits = head(node_emb[train_b], node_emb[train_s], train_feats)
        loss = loss_fn(logits, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
        optimizer.step()

        if epoch % 10 == 0:
            encoder.eval()
            head.eval()
            with torch.no_grad():
                emb_eval = encoder(X, edge_index)
                val_probs = torch.sigmoid(head(emb_eval[val_b], emb_eval[val_s], val_feats)).numpy()
                val_auc = roc_auc_score(val_y.numpy(), val_probs)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_enc_state = {k: v.clone() for k, v in encoder.state_dict().items()}

    if best_enc_state is not None:
        encoder.load_state_dict(best_enc_state)
    encoder.eval()
    return encoder


def extract_embeddings(encoder, X, edge_index, buyer_idx, seller_idx):
    """Extracts node embeddings for all indexed buyers and sellers."""
    encoder.eval()
    with torch.no_grad():
        node_emb = encoder(X, edge_index).numpy()

    buyer_embs = {b: node_emb[idx] for b, idx in buyer_idx.items()}
    seller_embs = {s: node_emb[idx] for s, idx in seller_idx.items()}
    return buyer_embs, seller_embs


def attach_gnn_embeddings(df, buyer_embs, seller_embs, emb_dim=GNN_EMB_DIM):
    """Joins vector embeddings onto order rows."""
    buyer_cols = [f"gnn_buyer_emb_{i}" for i in range(emb_dim)]
    seller_cols = [f"gnn_seller_emb_{i}" for i in range(emb_dim)]

    if df.empty:
        df_out = df.copy()
        for col in buyer_cols + seller_cols:
            df_out[col] = pd.Series(dtype=np.float32)
        return df_out, buyer_cols + seller_cols

    zero = np.zeros(emb_dim, dtype=np.float32)
    buyer_arr = np.vstack([buyer_embs.get(bid, zero) for bid in df["buyer_id"]])
    seller_arr = np.vstack([seller_embs.get(sid, zero) for sid in df["seller_id"]])

    df_out = df.copy()
    df_out[buyer_cols] = buyer_arr.astype(np.float32)
    df_out[seller_cols] = seller_arr.astype(np.float32)
    return df_out, buyer_cols + seller_cols


def train_hybrid_classifier(train_df, hybrid_feature_cols):
    """Trains final XGBoost model using tabular features + graph topology + GNN embeddings."""
    X_train = train_df[hybrid_feature_cols].fillna(0.0)
    y_train = train_df["y"]
    pos = int(y_train.sum())
    neg = len(y_train) - pos

    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="aucpr", random_state=42,
        n_jobs=-1, verbosity=0,
    )
    clf.fit(X_train, y_train)
    return clf


def _print_metrics(label, y_true, y_score):
    y_pred = (y_score >= 0.5).astype(int)
    print(f"  {label:<30} P: {precision_score(y_true, y_pred, zero_division=cast(Any, 0)):.3f} | "
          f"R: {recall_score(y_true, y_pred, zero_division=cast(Any, 0)):.3f} | "
          f"F1: {f1_score(y_true, y_pred, zero_division=cast(Any, 0)):.3f} | "
          f"ROC-AUC: {roc_auc_score(y_true, y_score):.3f} | "
          f"PR-AUC: {average_precision_score(y_true, y_score):.3f}")


def run_phase5(gnn_epochs=50):
    """Full hybrid model training and validation pipeline."""
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"])
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"],
        catalog["products"], base["address_sharing_log"], base["device_sharing_log"],
    )

    df, tabular_cols = build_tabular_features(
        result["orders"], result["listings"], result["returns"],
        txn["buyers"], catalog["sellers"], catalog["products"]
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["y"] = df["is_fraudulent"].astype(int)

    rel_graph = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"])
    rel_features = compute_relationship_features(rel_graph, df["buyer_id"].unique())
    df = df.merge(rel_features, on="buyer_id", how="left")
    df[["share_degree", "share_component_size"]] = df[["share_degree", "share_component_size"]].fillna(0)

    snapshots, months = build_monthly_snapshots(
        result["orders"].assign(order_date=pd.to_datetime(result["orders"]["order_date"])),
        SIM_START,
    )
    df = attach_snapshot_features(df, snapshots, months)
    df = add_edge_weight_before(df)

    graph_cols = [
        "share_degree", "share_component_size",
        "buyer_seller_degree", "buyer_pagerank",
        "seller_buyer_degree", "seller_pagerank",
        "seller_buyer_concentration_hhi",
        "buyer_seller_edge_weight_before",
    ]
    phase3_feature_cols = tabular_cols + graph_cols

    trainval = pd.DataFrame(df[df["order_date"] <= VAL_END])
    train = pd.DataFrame(df[df["order_date"] <= TRAIN_END])
    val = pd.DataFrame(df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)])
    test = pd.DataFrame(df[df["order_date"] > VAL_END])

    buyer_idx, seller_idx, n_buyers, n_total = build_node_index(txn["buyers"]["buyer_id"], catalog["sellers"]["seller_id"])
    X_nodes = build_node_features(buyer_idx, seller_idx, n_total, trainval, rel_graph, txn["buyers"], catalog["sellers"])
    edge_index = build_edge_index(buyer_idx, seller_idx, trainval, rel_graph)

    def _edge_tensors(split_df):
        b_ids = np.clip(split_df["buyer_id"].map(buyer_idx).fillna(0).astype(int).to_numpy(), 0, n_total - 1)
        s_ids = np.clip(split_df["seller_id"].map(seller_idx).fillna(0).astype(int).to_numpy(), 0, n_total - 1)
        feat_arr = split_df[GNN_EDGE_FEAT_COLS].fillna(0).copy()
        feat_arr["amount"] = np.log1p(feat_arr["amount"].clip(lower=0))
        return (
            torch.tensor(b_ids, dtype=torch.long),
            torch.tensor(s_ids, dtype=torch.long),
            torch.tensor(feat_arr.to_numpy(), dtype=torch.float32),
            torch.tensor(split_df["y"].to_numpy(), dtype=torch.float32)
        )

    train_b, train_s, train_feats, train_y_t = _edge_tensors(train)
    val_b, val_s, val_feats, val_y_t = _edge_tensors(val)

    print("Training GraphSAGE representation encoder...")
    encoder = train_sage_encoder(
        X_nodes, edge_index, train_b, train_s, train_feats, train_y_t,
        val_b, val_s, val_feats, val_y_t, epochs=gnn_epochs
    )

    buyer_embs, seller_embs = extract_embeddings(encoder, X_nodes, edge_index, buyer_idx, seller_idx)
    df_aug, gnn_cols = attach_gnn_embeddings(df, buyer_embs, seller_embs)
    hybrid_feature_cols = phase3_feature_cols + gnn_cols

    train_aug = pd.DataFrame(df_aug[df_aug["order_date"] <= TRAIN_END])
    val_aug = pd.DataFrame(df_aug[(df_aug["order_date"] > TRAIN_END) & (df_aug["order_date"] <= VAL_END)])
    test_aug = pd.DataFrame(df_aug[df_aug["order_date"] > VAL_END])

    print(f"Training Hybrid XGBoost on {len(hybrid_feature_cols)} features...")
    hybrid_model = train_hybrid_classifier(train_aug, hybrid_feature_cols)

    # Baseline comparison (RF on Phase 3 features)
    rf_p3 = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    rf_p3.fit(pd.DataFrame(train[phase3_feature_cols]).fillna(0), train["y"])

    val_base_scores = np.asarray(rf_p3.predict_proba(pd.DataFrame(val[phase3_feature_cols]).fillna(0)))[:, 1]
    test_base_scores = np.asarray(rf_p3.predict_proba(pd.DataFrame(test[phase3_feature_cols]).fillna(0)))[:, 1]

    val_hybrid_scores = np.asarray(hybrid_model.predict_proba(pd.DataFrame(val_aug[hybrid_feature_cols]).fillna(0)))[:, 1]
    test_hybrid_scores = np.asarray(hybrid_model.predict_proba(pd.DataFrame(test_aug[hybrid_feature_cols]).fillna(0)))[:, 1]

    val_y_arr = np.asarray(val["y"])
    test_y_arr = np.asarray(test["y"])

    print("\n[Validation Metrics]")
    _print_metrics("Phase 3 RF Baseline", val_y_arr, val_base_scores)
    _print_metrics("Phase 5 Hybrid XGBoost", val_y_arr, val_hybrid_scores)

    print("\n[Test Metrics]")
    _print_metrics("Phase 3 RF Baseline", test_y_arr, test_base_scores)
    _print_metrics("Phase 5 Hybrid XGBoost", test_y_arr, test_hybrid_scores)

    val_auc = roc_auc_score(val_y_arr, val_hybrid_scores)
    test_auc = roc_auc_score(test_y_arr, test_hybrid_scores)

    phase5_meta = {
        "tabular_cols": tabular_cols,
        "graph_cols": graph_cols,
        "phase3_feature_cols": phase3_feature_cols,
        "gnn_embedding_cols": gnn_cols,
        "hybrid_feature_cols": hybrid_feature_cols,
        "emb_dim": GNN_EMB_DIM,
        "classifier": "XGBoost",
    }

    return {
        "hybrid_model": hybrid_model,
        "buyer_embeddings": buyer_embs,
        "seller_embeddings": seller_embs,
        "phase5_meta": phase5_meta,
        "val_auc": val_auc,
        "test_auc": test_auc,
    }


if __name__ == "__main__":
    out = run_phase5()
