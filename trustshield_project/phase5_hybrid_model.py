"""
TrustShield AI — Phase 5: Hybrid GNN + XGBoost Architecture

Combines:
  - Phase 3 tabular + NetworkX graph topology features (18 total)
  - GraphSAGE-learned node embeddings (16-dim buyer + 16-dim seller = 32 extra)

into a single XGBoost (RF fallback) classifier — 50 features total.

Architecture rationale
----------------------
- The pure-GNN EdgeClassifier (gnn_model.py) uses only 2 edge features and
  relies entirely on graph structure.  It misses the rich tabular signal.
- The Phase 3 RF uses hand-crafted topology features (degree, PageRank) but
  cannot leverage learned neighbourhood representations.
- Hybrid: GraphSAGE encodes structural context into dense vectors; XGBoost
  classifies using BOTH tabular features AND those embeddings.

Temporal-safety design (critical — same rules as Phase 3 and gnn_model.py)
---------------------------------------------------------------------------
- GNN graph is built from TRAIN+VAL orders + static sharing edges ONLY.
  No test-period order edges ever enter the graph.
- Embeddings are computed in a single frozen forward pass over that graph.
  The resulting lookup dicts are snapshots — test orders are scored using
  embeddings computed WITHOUT their own neighbourhood information.
- The hybrid XGBoost is trained on the TRAIN split only (not train+val),
  validated on val, and evaluated finally on test.
- Cold-start nodes (buyers/sellers with no train+val orders) get zero
  embeddings — a documented limitation, not hidden leakage.

Install requirements (training only — NOT needed to serve pre-trained models):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install torch_geometric
    pip install xgboost          (recommended; RF used as fallback otherwise)
"""

import sys
import numpy as np
import pandas as pd
import networkx as nx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Optional torch import — graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

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
from utils import evaluate

# ---------------------------------------------------------------------------
# Constants — must match GraphSAGEEncoder definition below
# ---------------------------------------------------------------------------
GNN_EMB_DIM       = 16   # out_dim of GraphSAGEEncoder
GNN_HIDDEN_DIM    = 32   # hidden_dim of GraphSAGEEncoder
GNN_NODE_FEAT_DIM = 7    # width of build_node_features()
# Edge features passed to the EdgeClassifier during GNN training only.
# Kept at 2 (same as gnn_model.py) so the GNN head is identical.
GNN_EDGE_FEAT_COLS = ["price_vs_base_price_ratio", "order_amount"]


# ---------------------------------------------------------------------------
# Graph building helpers
# (Self-contained copy of gnn_model.py helpers so this module does not
# depend on gnn_model.py's module-level torch import guard.)
# ---------------------------------------------------------------------------

def build_node_index(buyer_ids, seller_ids):
    """Assign a contiguous integer index to every buyer and seller node."""
    buyer_idx  = {b: i + 1             for i, b in enumerate(sorted(buyer_ids))}
    n_buyers   = len(buyer_idx)
    seller_idx = {s: n_buyers + 1 + i  for i, s in enumerate(sorted(seller_ids))}
    n_total    = n_buyers + len(seller_idx) + 1
    return buyer_idx, seller_idx, n_buyers, n_total


def build_node_features(buyer_idx, seller_idx, n_total, trainval_orders,
                         rel_graph, buyers_df, sellers_df):
    """
    Build a float32 node feature matrix from TRAIN+VAL data only.

    Layout per node (7 dims):
        [is_buyer, is_seller, age_days_norm, order_or_listing_count,
         price_norm_or_reserved, share_degree, share_component_size]
    """
    X = np.zeros((n_total, GNN_NODE_FEAT_DIM), dtype=np.float32)
    buyer_order_counts = trainval_orders.groupby("buyer_id").size()
    components = {
        node: comp
        for comp in nx.connected_components(rel_graph)
        for node in comp
    }

    buyers_indexed = buyers_df.set_index("buyer_id")
    for b, idx in buyer_idx.items():
        age = 0
        if b in buyers_indexed.index:
            age = max(0, (VAL_END - pd.to_datetime(buyers_indexed.loc[b, "signup_date"])).days)
        X[idx, 0] = 1.0  # is_buyer
        X[idx, 2] = age / 365.0
        X[idx, 3] = np.log1p(float(buyer_order_counts.get(b, 0)))
        # X[idx, 4] reserved for buyers
        if b in rel_graph:
            X[idx, 5] = rel_graph.degree[b]
            X[idx, 6] = len(components.get(b, {b}))

    seller_order_counts = trainval_orders.groupby("seller_id").size()
    seller_avg_amount   = trainval_orders.groupby("seller_id")["amount"].mean()
    sellers_indexed = sellers_df.set_index("seller_id")
    for s, idx in seller_idx.items():
        age = 0
        if s in sellers_indexed.index:
            age = max(0, (VAL_END - pd.to_datetime(sellers_indexed.loc[s, "signup_date"])).days)
        X[idx, 1] = 1.0  # is_seller
        X[idx, 2] = age / 365.0
        X[idx, 3] = np.log1p(float(seller_order_counts.get(s, 0)))
        X[idx, 4] = np.log1p(float(seller_avg_amount.get(s, 0.0)) / 1000.0)  # rough normalisation

    return torch.tensor(X, dtype=torch.float32)


def build_edge_index(buyer_idx, seller_idx, trainval_orders, rel_graph):
    """
    Build a PyG edge_index tensor (train+val buyer-seller transaction edges
    + static device/address sharing edges).  All edges are made undirected.
    """
    edges = []
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
    
    edges = list(set(edges))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ---------------------------------------------------------------------------
# GNN model definitions (same architecture as gnn_model.py)
# Defined inside the _HAS_TORCH guard so the module can be safely imported
# even when torch is absent — only run_phase5() will fail at call time.
# ---------------------------------------------------------------------------

if _HAS_TORCH:
    class GraphSAGEEncoder(nn.Module):
        """Two-layer GraphSAGE encoder: node features → dense embeddings."""

        def __init__(self, in_dim=GNN_NODE_FEAT_DIM,
                     hidden_dim=GNN_HIDDEN_DIM, out_dim=GNN_EMB_DIM):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, out_dim)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)

    class EdgeClassifier(nn.Module):
        """
        MLP head that classifies a buyer-seller edge as fraud using the
        concatenation of buyer embedding, seller embedding, and edge features.
        Only used during GNN training; discarded after embedding extraction.
        """

        def __init__(self, emb_dim=GNN_EMB_DIM,
                     edge_feat_dim=len(GNN_EDGE_FEAT_COLS), hidden_dim=32):
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
# GNN training
# ---------------------------------------------------------------------------

def train_sage_encoder(X, edge_index,
                        train_b, train_s, train_feats, train_y,
                        val_b,   val_s,   val_feats,   val_y,
                        epochs=50):
    """
    Train a GraphSAGE encoder with an EdgeClassifier head.

    The encoder learns fraud-discriminative node representations through the
    edge-level fraud classification loss.  After training, only the encoder
    is used for embedding extraction — the head is discarded.  The best
    encoder checkpoint (by validation ROC-AUC) is restored before returning.

    Args:
        X             : node feature tensor [n_nodes, GNN_NODE_FEAT_DIM]
        edge_index    : graph connectivity  [2, n_edges] — train+val only
        train_b/s     : buyer/seller node indices for training edges [n_train]
        train_feats   : edge feature tensor for training edges [n_train, 2]
        train_y       : binary fraud labels for training edges [n_train]
        val_b/s       : buyer/seller node indices for validation edges [n_val]
        val_feats     : edge feature tensor for validation edges [n_val, 2]
        val_y         : binary fraud labels for validation edges [n_val]
        epochs        : number of training epochs (default 50)

    Returns:
        encoder (GraphSAGEEncoder) in eval() mode — best val ROC-AUC checkpoint
    """
    if not _HAS_TORCH:
        raise RuntimeError(
            "PyTorch and torch_geometric are required for Phase 5 training.\n"
            "Install:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install torch_geometric"
        )

    encoder    = GraphSAGEEncoder()
    head       = EdgeClassifier()
    optimizer  = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=0.002
    )
    pos_weight = torch.tensor([
        (train_y == 0).sum().item() / max((train_y == 1).sum().item(), 1)
    ])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"  Training GraphSAGE encoder ({epochs} epochs)...")
    best_val_auc  = 0.0
    best_enc_state = None

    for epoch in range(1, epochs + 1):
        encoder.train(); head.train()
        optimizer.zero_grad()
        node_emb = encoder(X, edge_index)
        logits   = head(node_emb[train_b], node_emb[train_s], train_feats)
        loss     = loss_fn(logits, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
        optimizer.step()

        if epoch % 10 == 0:
            encoder.eval(); head.eval()
            with torch.no_grad():
                emb_eval   = encoder(X, edge_index)
                val_logits = head(emb_eval[val_b], emb_eval[val_s], val_feats)
                val_probs  = torch.sigmoid(val_logits).numpy()
                val_auc    = roc_auc_score(val_y.numpy(), val_probs)
            print(f"    epoch {epoch:3d}  train_loss={loss.item():.4f}  "
                  f"val_ROC-AUC={val_auc:.3f}")
            if val_auc > best_val_auc:
                best_val_auc   = val_auc
                best_enc_state = {k: v.clone()
                                  for k, v in encoder.state_dict().items()}
            encoder.train(); head.train()

    # Restore best checkpoint
    if best_enc_state is not None:
        encoder.load_state_dict(best_enc_state)
    encoder.eval()
    print(f"  Best GNN val ROC-AUC: {best_val_auc:.3f}")
    return encoder


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(encoder, X, edge_index, buyer_idx, seller_idx):
    """
    Run a single frozen forward pass over the train+val graph and extract
    per-node embeddings as numpy arrays.

    Args:
        encoder    : trained GraphSAGEEncoder in eval() mode
        X          : node feature tensor [n_nodes, GNN_NODE_FEAT_DIM]
        edge_index : train+val graph connectivity [2, n_edges]
        buyer_idx  : dict {buyer_id  → int node index}
        seller_idx : dict {seller_id → int node index}

    Returns:
        buyer_embs  : dict {buyer_id  → np.array(GNN_EMB_DIM,) float32}
        seller_embs : dict {seller_id → np.array(GNN_EMB_DIM,) float32}
    """
    encoder.eval()
    with torch.no_grad():
        node_emb = encoder(X, edge_index).numpy()  # [n_nodes, GNN_EMB_DIM]

    buyer_embs  = {b: node_emb[idx] for b, idx in buyer_idx.items()}
    seller_embs = {s: node_emb[idx] for s, idx in seller_idx.items()}
    return buyer_embs, seller_embs


# ---------------------------------------------------------------------------
# Embedding attachment
# ---------------------------------------------------------------------------

def attach_gnn_embeddings(df, buyer_embs, seller_embs, emb_dim=GNN_EMB_DIM):
    """
    Append GNN node embedding columns to an order-level DataFrame.

    For each row, looks up buyer_embs[buyer_id] and seller_embs[seller_id].
    Rows whose buyer/seller ID is absent from the dicts (cold-start nodes
    not seen during train+val) receive a zero vector — documented limitation.

    New column names:
        gnn_buyer_emb_0  … gnn_buyer_emb_{emb_dim-1}
        gnn_seller_emb_0 … gnn_seller_emb_{emb_dim-1}

    Args:
        df          : DataFrame with buyer_id and seller_id columns
        buyer_embs  : dict {buyer_id  → np.array(emb_dim,)}
        seller_embs : dict {seller_id → np.array(emb_dim,)}
        emb_dim     : embedding dimension (default GNN_EMB_DIM = 16)

    Returns:
        df_out   : copy of df with 2*emb_dim new columns appended
        gnn_cols : list of the new column names (length 2*emb_dim)
    """
    buyer_col_names  = [f"gnn_buyer_emb_{i}"  for i in range(emb_dim)]
    seller_col_names = [f"gnn_seller_emb_{i}" for i in range(emb_dim)]

    if len(df) == 0:
        df_out = df.copy()
        for col in buyer_col_names + seller_col_names:
            df_out[col] = pd.Series(dtype=np.float32)
        return df_out, buyer_col_names + seller_col_names

    zero             = np.zeros(emb_dim, dtype=np.float32)
    buyer_arr  = np.vstack([buyer_embs.get(bid, zero)  for bid in df["buyer_id"]])
    seller_arr = np.vstack([seller_embs.get(sid, zero) for sid in df["seller_id"]])

    df_out = df.copy()
    df_out[buyer_col_names]  = buyer_arr.astype(np.float32)
    df_out[seller_col_names] = seller_arr.astype(np.float32)

    return df_out, buyer_col_names + seller_col_names


# ---------------------------------------------------------------------------
# Hybrid classifier
# ---------------------------------------------------------------------------

def _make_classifier(n_train_pos=None, n_train_total=None):
    """Return an XGBoost classifier if available, otherwise RandomForest."""
    if _HAS_XGBOOST:
        neg = (n_train_total or 1000) - (n_train_pos or 100)
        pos = n_train_pos or 100
        return XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg / max(pos, 1),
            eval_metric="aucpr", random_state=42,
            n_jobs=-1, verbosity=0,
        )
    return RandomForestClassifier(
        n_estimators=200, max_depth=8,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1,
    )


def train_hybrid_classifier(train_df, hybrid_feature_cols):
    """
    Train the final XGBoost/RF on TRAIN split using
    Phase-3 tabular+graph features PLUS GNN embedding columns.

    Args:
        train_df             : TRAIN split DataFrame (already has GNN cols attached)
        hybrid_feature_cols  : ordered feature column list

    Returns:
        Fitted classifier (XGBoost or RandomForest)
    """
    X_train = train_df[hybrid_feature_cols].fillna(0.0)
    y_train = train_df["y"]
    clf = _make_classifier(int(y_train.sum()), len(y_train))
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _print_metrics(label, y_true, y_score):
    y_pred = (y_score >= 0.5).astype(int)
    print(f"  {label}")
    print(f"    Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}  "
          f"Recall: {recall_score(y_true, y_pred, zero_division=0):.3f}  "
          f"F1: {f1_score(y_true, y_pred, zero_division=0):.3f}  "
          f"ROC-AUC: {roc_auc_score(y_true, y_score):.3f}  "
          f"PR-AUC: {average_precision_score(y_true, y_score):.3f}")


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------

def run_phase5(gnn_epochs=50):
    """
    End-to-end Phase 5 training orchestration.

    Steps:
      1. Data pipeline (entities → catalog → orders/returns → fraud injection)
      2. Phase 3 features: 10 tabular + 8 graph topology (18 total)
      3. Train/val/test split
      4. Build GNN graph (train+val orders + static sharing edges)
      5. Train GraphSAGE encoder (gnn_epochs)
      6. Extract frozen node embeddings → lookup dicts
      7. Attach 32 GNN embedding columns to all splits
      8. Train hybrid XGBoost/RF on TRAIN split (50 features total)
      9. Evaluate val & test; print comparison vs Phase 3 baseline

    Args:
        gnn_epochs : number of GraphSAGE training epochs (default 50)

    Returns:
        dict with keys:
            hybrid_model      — fitted XGBoost/RF
            buyer_embeddings  — {buyer_id  → np.array(16,)}
            seller_embeddings — {seller_id → np.array(16,)}
            phase5_meta       — feature metadata dict
            val_auc           — validation ROC-AUC of the hybrid model
            test_auc          — test ROC-AUC of the hybrid model
    """
    if not _HAS_TORCH:
        raise RuntimeError(
            "Phase 5 training requires PyTorch and torch_geometric.\n"
            "Install with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install torch_geometric"
        )

    # ------------------------------------------------------------------
    # 1. Data pipeline
    # ------------------------------------------------------------------
    print("[Phase 5 — 1/9] Building data pipeline...")
    base    = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn     = build_orders_and_returns(
        base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"]
    )
    result  = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"],
        catalog["products"], base["address_sharing_log"], base["device_sharing_log"],
    )

    # ------------------------------------------------------------------
    # 2. Phase 3 features: tabular + graph topology
    # ------------------------------------------------------------------
    print("[Phase 5 — 2/9] Building tabular + Phase 3 graph topology features...")
    df, tabular_cols = build_tabular_features(
        result["orders"], result["listings"], result["returns"],
        txn["buyers"], catalog["sellers"], catalog["products"]
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["y"] = df["is_fraudulent"].astype(int)

    rel_graph    = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"])
    rel_features = compute_relationship_features(rel_graph, df["buyer_id"].unique())
    df = df.merge(rel_features, on="buyer_id", how="left")
    df[["share_degree", "share_component_size"]] = (
        df[["share_degree", "share_component_size"]].fillna(0)
    )

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
    phase3_feature_cols = tabular_cols + graph_cols  # 10 + 8 = 18

    # ------------------------------------------------------------------
    # 3. Train/val/test split
    # ------------------------------------------------------------------
    print("[Phase 5 — 3/9] Splitting data...")
    trainval = df[df["order_date"] <= VAL_END]
    train    = df[df["order_date"] <= TRAIN_END]
    val      = df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)]
    test     = df[df["order_date"] > VAL_END]
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}  "
          f"(train fraud: {train['y'].mean():.2%})")

    # ------------------------------------------------------------------
    # 4. Build GNN components (train+val graph only — temporal safety)
    # ------------------------------------------------------------------
    print("[Phase 5 — 4/9] Building GNN graph (train+val orders + static sharing edges)...")
    buyer_idx, seller_idx, n_buyers, n_total = build_node_index(
        txn["buyers"]["buyer_id"], catalog["sellers"]["seller_id"]
    )
    X_nodes    = build_node_features(
        buyer_idx, seller_idx, n_total, trainval, rel_graph,
        txn["buyers"], catalog["sellers"]
    )
    edge_index = build_edge_index(buyer_idx, seller_idx, trainval, rel_graph)
    print(f"  Nodes: {n_total:,}  ({n_buyers:,} buyers + {n_total - n_buyers:,} sellers)  "
          f"Edges: {edge_index.shape[1]:,}")

    def _edge_tensors(split_df):
        """Map buyer_id / seller_id strings to node indices; clamp unmapped to 0."""
        b_ids = np.clip(
            split_df["buyer_id"].map(buyer_idx).fillna(0).astype(int).to_numpy(),
            0, n_total - 1
        )
        s_ids = np.clip(
            split_df["seller_id"].map(seller_idx).fillna(0).astype(int).to_numpy(),
            0, n_total - 1
        )
        
        feat_arr = split_df[GNN_EDGE_FEAT_COLS].fillna(0).copy()
        feat_arr["order_amount"] = np.log1p(feat_arr["order_amount"].clip(lower=0))
        feats = torch.tensor(feat_arr.to_numpy(), dtype=torch.float32)
        
        labels = torch.tensor(split_df["y"].to_numpy(), dtype=torch.float32)
        return (torch.tensor(b_ids, dtype=torch.long),
                torch.tensor(s_ids, dtype=torch.long),
                feats, labels)

    train_b, train_s, train_feats, train_y_t = _edge_tensors(train)
    val_b,   val_s,   val_feats,   val_y_t   = _edge_tensors(val)

    # ------------------------------------------------------------------
    # 5. Train GNN encoder
    # ------------------------------------------------------------------
    print("[Phase 5 — 5/9] Training GraphSAGE encoder...")
    encoder = train_sage_encoder(
        X_nodes, edge_index,
        train_b, train_s, train_feats, train_y_t,
        val_b,   val_s,   val_feats,   val_y_t,
        epochs=gnn_epochs,
    )

    # ------------------------------------------------------------------
    # 6. Extract embeddings (frozen; train+val graph — no test-period edges)
    # ------------------------------------------------------------------
    print("[Phase 5 — 6/9] Extracting node embeddings...")
    buyer_embs, seller_embs = extract_embeddings(
        encoder, X_nodes, edge_index, buyer_idx, seller_idx
    )
    print(f"  {len(buyer_embs):,} buyer embeddings  +  "
          f"{len(seller_embs):,} seller embeddings  ({GNN_EMB_DIM}-dim each)")

    # ------------------------------------------------------------------
    # 7. Attach embeddings to ALL splits
    # ------------------------------------------------------------------
    print("[Phase 5 — 7/9] Attaching GNN embeddings to feature DataFrame...")
    df_aug, gnn_cols = attach_gnn_embeddings(df, buyer_embs, seller_embs)
    hybrid_feature_cols = phase3_feature_cols + gnn_cols  # 18 + 32 = 50

    train_aug = df_aug[df_aug["order_date"] <= TRAIN_END]
    val_aug   = df_aug[(df_aug["order_date"] > TRAIN_END) & (df_aug["order_date"] <= VAL_END)]
    test_aug  = df_aug[df_aug["order_date"] > VAL_END]

    # ------------------------------------------------------------------
    # 8. Train hybrid classifier (TRAIN split only)
    # ------------------------------------------------------------------
    clf_name = "XGBoost" if _HAS_XGBOOST else "RandomForest"
    print(f"[Phase 5 — 8/9] Training hybrid {clf_name} "
          f"({len(hybrid_feature_cols)} features = "
          f"{len(phase3_feature_cols)} Phase-3 + {len(gnn_cols)} GNN)...")
    hybrid_model = train_hybrid_classifier(train_aug, hybrid_feature_cols)

    # ------------------------------------------------------------------
    # 9. Evaluate — Hybrid vs Phase 3 baseline comparison
    # ------------------------------------------------------------------
    print("[Phase 5 — 9/9] Evaluating...")

    # Phase 3 RF baseline (re-trained here for a fair apples-to-apples
    # comparison — same train/val/test split, RF so XGBoost availability
    # doesn't skew the gap).
    rf_p3 = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    rf_p3.fit(train[phase3_feature_cols].fillna(0), train["y"])

    val_base_scores  = rf_p3.predict_proba(val[phase3_feature_cols].fillna(0))[:, 1]
    test_base_scores = rf_p3.predict_proba(test[phase3_feature_cols].fillna(0))[:, 1]

    val_hybrid_scores  = hybrid_model.predict_proba(
        val_aug[hybrid_feature_cols].fillna(0)
    )[:, 1]
    test_hybrid_scores = hybrid_model.predict_proba(
        test_aug[hybrid_feature_cols].fillna(0)
    )[:, 1]

    print("\n" + "=" * 68)
    print("PHASE 5 RESULTS — Hybrid GNN+XGBoost vs Phase 3 RF Baseline")
    print("=" * 68)
    print("\n[Validation Set]")
    _print_metrics("Phase 3 RF Baseline  (18 features)", val["y"].to_numpy(), val_base_scores)
    _print_metrics(f"Phase 5 Hybrid {clf_name:8s} (50 features)", val["y"].to_numpy(), val_hybrid_scores)
    print("\n[Test Set]")
    _print_metrics("Phase 3 RF Baseline  (18 features)", test["y"].to_numpy(), test_base_scores)
    _print_metrics(f"Phase 5 Hybrid {clf_name:8s} (50 features)", test["y"].to_numpy(), test_hybrid_scores)

    # Cold-start report
    all_buyers  = set(df["buyer_id"].unique())
    all_sellers = set(df["seller_id"].unique())
    cold_buyers  = len(all_buyers  - set(buyer_embs))
    cold_sellers = len(all_sellers - set(seller_embs))
    print(f"\n[Cold-start]  Buyers with zero embedding: {cold_buyers}  "
          f"Sellers with zero embedding: {cold_sellers}")
    print("  (Nodes not in train+val graph receive zero vectors — "
          "scored using tabular features only.)")

    val_auc  = roc_auc_score(val["y"].to_numpy(),  val_hybrid_scores)
    test_auc = roc_auc_score(test["y"].to_numpy(), test_hybrid_scores)

    phase5_meta = {
        "tabular_cols":        tabular_cols,
        "graph_cols":          graph_cols,
        "phase3_feature_cols": phase3_feature_cols,
        "gnn_embedding_cols":  gnn_cols,
        "hybrid_feature_cols": hybrid_feature_cols,
        "emb_dim":             GNN_EMB_DIM,
        "classifier":          clf_name,
    }

    return {
        "hybrid_model":      hybrid_model,
        "buyer_embeddings":  buyer_embs,
        "seller_embeddings": seller_embs,
        "phase5_meta":       phase5_meta,
        "val_auc":           val_auc,
        "test_auc":          test_auc,
    }


if __name__ == "__main__":
    out = run_phase5()
    print(f"\nDone.  Val ROC-AUC={out['val_auc']:.3f}  "
          f"Test ROC-AUC={out['test_auc']:.3f}")
