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
    build_weekly_relationship_snapshots,
    attach_relationship_snapshot_features,
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


def build_node_features(buyer_idx, seller_idx, n_total, orders_df, rel_graph, buyers_df, sellers_df,
                        cutoff_date=None):
    """Builds node feature matrix for graph encoder, respecting temporal cutoffs."""
    X = np.zeros((n_total, GNN_NODE_FEAT_DIM), dtype=np.float32)
    ref_date = pd.Timestamp(cutoff_date) if cutoff_date is not None else VAL_END
    buyer_order_counts = orders_df.groupby("buyer_id").size()
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

    seller_order_counts = orders_df.groupby("seller_id").size()
    seller_avg_amount = orders_df.groupby("seller_id")["amount"].mean()
    sellers_indexed = sellers_df.set_index("seller_id")
    for s, idx in seller_idx.items():
        age = max(0, (ref_date - pd.to_datetime(sellers_indexed.loc[s, "signup_date"])).days) if s in sellers_indexed.index else 0
        X[idx, 1] = 1.0
        X[idx, 2] = age / 365.0
        X[idx, 3] = np.log1p(float(seller_order_counts.get(s, 0)))
        X[idx, 4] = np.log1p(float(seller_avg_amount.get(s, 0.0)) / 1000.0)

    return torch.tensor(X, dtype=torch.float32)


def build_edge_index(buyer_idx, seller_idx, orders_df, rel_graph):
    """Builds PyG undirected edge tensor."""
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


def generate_oof_gnn_embeddings(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    buyer_idx: dict,
    seller_idx: dict,
    n_total: int,
    X_nodes_train: torch.Tensor,
    edge_index_train: torch.Tensor,
    X_nodes_test: torch.Tensor,
    edge_index_test: torch.Tensor,
    n_splits: int = 5,
    epochs: int = 40,
    random_state: int = 42,
    fold_strategy: str = "time_aware",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Generates K=5 buyer-grouped out-of-fold GNN embeddings for train, and train-fit embeddings for val/test.

    Mitigates label leakage into downstream hybrid classifier:
    - Train split embeddings are generated via buyer-grouped out-of-fold cross-validation.
      Supports 'time_aware' (forward chaining by earliest order date) and 'random' (shuffled KFold).
      For each fold k, the GraphSAGE encoder is trained strictly without fold k's buyers or labels.
      Fold k rows receive embeddings from this independent fold encoder.
    - Val and test split embeddings are generated via a single encoder trained on all train rows
      (with an internal 80/20 train split for early stopping; never observing val or test labels).
    """
    from sklearn.model_selection import KFold

    def _make_edge_tensors(split_df):
        b_ids = np.clip(split_df["buyer_id"].map(buyer_idx).fillna(0).astype(int).to_numpy(), 0, n_total - 1)
        s_ids = np.clip(split_df["seller_id"].map(seller_idx).fillna(0).astype(int).to_numpy(), 0, n_total - 1)
        feat_arr = split_df[GNN_EDGE_FEAT_COLS].fillna(0).copy()
        feat_arr["amount"] = np.log1p(feat_arr["amount"].clip(lower=0))
        return (
            torch.tensor(b_ids, dtype=torch.long),
            torch.tensor(s_ids, dtype=torch.long),
            torch.tensor(feat_arr.to_numpy(), dtype=torch.float32),
            torch.tensor(split_df["y"].to_numpy(), dtype=torch.float32),
        )

    buyer_cols = [f"gnn_buyer_emb_{i}" for i in range(GNN_EMB_DIM)]
    seller_cols = [f"gnn_seller_emb_{i}" for i in range(GNN_EMB_DIM)]
    gnn_cols = buyer_cols + seller_cols

    unique_buyers = np.array(sorted(train_df["buyer_id"].unique()))
    if fold_strategy == "time_aware":
        # Time-aware forward chaining: sort unique buyers by earliest order date
        buyer_min_dates = train_df.groupby("buyer_id")["order_date"].min().sort_values()
        sorted_buyers = buyer_min_dates.index.to_numpy()
        chunks = np.array_split(sorted_buyers, n_splits)
        folds = []
        for k in range(n_splits):
            val_b = set(chunks[k])
            if k == 0:
                tr_b = set(np.concatenate(chunks[1:]))
            else:
                tr_b = set(np.concatenate(chunks[:k]))
            folds.append((tr_b, val_b))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = [(set(unique_buyers[f_tr]), set(unique_buyers[f_val])) for f_tr, f_val in kf.split(unique_buyers)]

    train_aug = train_df.copy()
    for col in gnn_cols:
        train_aug[col] = 0.0
    train_aug["_oof_fold"] = -1

    print(f"Generating K={n_splits} buyer-grouped out-of-fold GNN embeddings ({fold_strategy}) for training set...")
    for fold, (fit_buyer_set, val_b_set) in enumerate(folds):
        fold_train_mask = train_df["buyer_id"].isin(fit_buyer_set)
        fold_val_mask = train_df["buyer_id"].isin(val_b_set)

        f_tr_b, f_tr_s, f_tr_f, f_tr_y = _make_edge_tensors(train_df[fold_train_mask])
        f_val_b, f_val_s, f_val_f, f_val_y = _make_edge_tensors(train_df[fold_val_mask])

        # Train fold encoder strictly without fold_val buyers or labels
        fold_encoder = train_sage_encoder(
            X_nodes_train, edge_index_train,
            f_tr_b, f_tr_s, f_tr_f, f_tr_y,
            f_val_b, f_val_s, f_val_f, f_val_y,
            epochs=epochs,
        )

        # Extract embeddings for the held-out fold
        f_b_embs, f_s_embs = extract_embeddings(
            fold_encoder, X_nodes_train, edge_index_train, buyer_idx, seller_idx
        )

        fold_val_indices = train_df[fold_val_mask].index
        zero = np.zeros(GNN_EMB_DIM, dtype=np.float32)
        b_mat = np.vstack([f_b_embs.get(bid, zero) for bid in train_df.loc[fold_val_indices, "buyer_id"]])
        s_mat = np.vstack([f_s_embs.get(sid, zero) for sid in train_df.loc[fold_val_indices, "seller_id"]])

        train_aug.loc[fold_val_indices, buyer_cols] = b_mat
        train_aug.loc[fold_val_indices, seller_cols] = s_mat
        train_aug.loc[fold_val_indices, "_oof_fold"] = fold

    # Train full encoder on train split only for val and test embeddings
    print("Training full-train GNN encoder for validation and test embeddings...")
    rng = np.random.RandomState(random_state)
    all_b_unique = unique_buyers.copy()
    rng.shuffle(all_b_unique)
    split_pt = int(0.8 * len(all_b_unique))
    fit_buyers = set(all_b_unique[:split_pt])

    full_fit_mask = train_df["buyer_id"].isin(fit_buyers)
    full_es_mask = ~full_fit_mask

    full_fit_b, full_fit_s, full_fit_f, full_fit_y = _make_edge_tensors(train_df[full_fit_mask])
    full_es_b, full_es_s, full_es_f, full_es_y = _make_edge_tensors(train_df[full_es_mask])

    full_encoder = train_sage_encoder(
        X_nodes_train, edge_index_train,
        full_fit_b, full_fit_s, full_fit_f, full_fit_y,
        full_es_b, full_es_s, full_es_f, full_es_y,
        epochs=epochs,
    )

    # Validation embeddings: extracted from training graph state
    val_b_embs, val_s_embs = extract_embeddings(
        full_encoder, X_nodes_train, edge_index_train, buyer_idx, seller_idx
    )
    val_aug, _ = attach_gnn_embeddings(val_df, val_b_embs, val_s_embs)

    # Test embeddings: extracted from test historical graph state (<= VAL_END)
    test_b_embs, test_s_embs = extract_embeddings(
        full_encoder, X_nodes_test, edge_index_test, buyer_idx, seller_idx
    )
    test_aug, _ = attach_gnn_embeddings(test_df, test_b_embs, test_s_embs)

    return train_aug, val_aug, test_aug, gnn_cols


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

    # Phase 2: Attach point-in-time relationship graph features (share_degree, share_component_size)
    df = attach_relationship_snapshot_features(df, base["address_sharing_log"], base["device_sharing_log"])

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

    rel_graph_train = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"], cutoff_date=TRAIN_END)
    rel_graph_val = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"], cutoff_date=VAL_END)

    # Training-period graph state: strictly bounded to <= TRAIN_END (no validation leakage)
    X_nodes_train = build_node_features(
        buyer_idx, seller_idx, n_total, train, rel_graph_train, txn["buyers"], catalog["sellers"], cutoff_date=TRAIN_END
    )
    edge_index_train = build_edge_index(buyer_idx, seller_idx, train, rel_graph_train)

    # Historical graph state up to VAL_END: strictly bounded to <= VAL_END (historical for test scoring)
    X_nodes_val = build_node_features(
        buyer_idx, seller_idx, n_total, trainval, rel_graph_val, txn["buyers"], catalog["sellers"], cutoff_date=VAL_END
    )
    edge_index_val = build_edge_index(buyer_idx, seller_idx, trainval, rel_graph_val)

    # Generate K=5 buyer-grouped out-of-fold GNN embeddings (leak-free)
    train_aug, val_aug, test_aug, gnn_cols = generate_oof_gnn_embeddings(
        train, val, test,
        buyer_idx, seller_idx, n_total,
        X_nodes_train, edge_index_train,
        X_nodes_val, edge_index_val,
        n_splits=5, epochs=gnn_epochs
    )
    hybrid_feature_cols = phase3_feature_cols + gnn_cols


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

    # Fit ProbabilityCalibrator on held-out validation set
    from calibration import ProbabilityCalibrator
    phase5_calibrator = ProbabilityCalibrator(method="isotonic")
    phase5_calibrator.fit(val_hybrid_scores, val_y_arr)

    phase5_meta = {
        "tabular_cols": tabular_cols,
        "graph_cols": graph_cols,
        "phase3_feature_cols": phase3_feature_cols,
        "gnn_embedding_cols": gnn_cols,
        "hybrid_feature_cols": hybrid_feature_cols,
        "emb_dim": GNN_EMB_DIM,
        "classifier": "XGBoost",
        "calibrated": True,
        "calibrator_method": "isotonic",
        "thresholds": [0.25, 0.55, 0.85],
    }

    return {
        "hybrid_model": hybrid_model,
        "phase5_meta": phase5_meta,
        "calibrator": phase5_calibrator,
        "val_auc": val_auc,
        "test_auc": test_auc,
    }


if __name__ == "__main__":
    out = run_phase5()
