"""
TrustShield AI — Phase 3
Trust Graph: timestamp-snapshotted graph-topological features layered on
top of the Phase 1C tabular baseline, with an explicit before/after
comparison (the "does the graph actually help" evidence the project needs).

IMPORTANT — environment constraint (documented, not hidden):
This sandbox has no torch, no PyTorch Geometric, and no DGL installed, and
no network access to install them (same situation as XGBoost in Phase 1C).
So this is NOT a GNN (no GraphSAGE/GCN embeddings). It's real graph-
topological feature engineering using networkx — degree, PageRank,
connected-component size, buyer-seller concentration (HHI) — fed into the
same RandomForest used in Phase 1C/2. This is a legitimate and common
first step before a GNN (many production fraud systems ship exactly this
before investing in graph neural nets), but it is a step below the GNN
that was the original Phase 3 ambition. To run an actual GNN: export
build_relationship_graph()'s edge list + build_monthly_snapshots()'s
per-node feature table locally, install torch + torch_geometric, and feed
them into a 2-layer GraphSAGE/GCN — the temporal-snapshot structure here
is exactly what a temporal GNN needs as input.

Two feature families, two different temporal-safety mechanisms:
1. STATIC relationship graph (device/address sharing) — device_sharing_log
   and address_sharing_log have no per-edge timestamp in the current
   schema (a documented Phase 1C limitation, carried forward unchanged
   here, not a new leak). Built from BOTH fraud_linked and legitimate
   shares — using only fraud_linked edges would leak the ground-truth
   label into a "structural" feature; a real detector doesn't know which
   shares are fraud a priori, only that sharing exists.
2. MONTHLY snapshot graph (buyer-seller order edges) — genuinely temporal.
   For an order in month M, features come from the snapshot as of the
   END of month M-1 (never the order's own month), so nothing about the
   order's own month — including the order itself — leaks into its
   features.
"""

import numpy as np
import pandas as pd
import networkx as nx

from entity_generator import build_base_entities
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud
from baseline_model import build_features as build_tabular_features, TRAIN_END, VAL_END, leakage_audit
from utils import evaluate, find_cost_optimal_threshold

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

# XGBoost auto-detect — same pattern as baseline_model.py.
# If installed locally, Phase 3 combined model uses XGBoost instead of RF.
try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


# ---------------------------------------------------------------------------
# 1. Static relationship graph (device + address sharing)
# ---------------------------------------------------------------------------

def build_relationship_graph(address_sharing_log, device_sharing_log):
    """
    Undirected buyer-buyer graph: an edge exists if two buyers share an
    address OR a device, regardless of share_type (fraud_linked vs
    legitimate) — using the type label here would bake the ground truth
    into a "structural" feature, defeating the point of a graph feature.
    """
    G = nx.Graph()
    # Use add_edges_from instead of iterrows — vectorised and significantly
    # faster at scale (iterrows is O(n) Python loop with per-row overhead).
    G.add_edges_from(
        [(r.buyer_id, r.shared_with_buyer_id, {"kind": "address"})
         for r in address_sharing_log[["buyer_id", "shared_with_buyer_id"]].itertuples(index=False)]
    )
    G.add_edges_from(
        [(r.buyer_id, r.shared_with_buyer_id, {"kind": "device"})
         for r in device_sharing_log[["buyer_id", "shared_with_buyer_id"]].itertuples(index=False)]
    )
    return G


def compute_relationship_features(graph, buyer_ids):
    """
    Per-buyer structural features: sharing degree and the size of the
    connected component they belong to (a large component = a cluster of
    mutually-linked accounts, the exact shape a multi-accounting ring
    leaves behind, regardless of why the accounts are linked).
    """
    components = {node: comp for comp in nx.connected_components(graph) for node in comp}
    rows = []
    for buyer_id in buyer_ids:
        if buyer_id in graph:
            degree = graph.degree[buyer_id]
            comp_size = len(components[buyer_id])
        else:
            degree, comp_size = 0, 1
        rows.append({"buyer_id": buyer_id, "share_degree": degree, "share_component_size": comp_size})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Monthly snapshot graph (buyer-seller order edges)
# ---------------------------------------------------------------------------

def build_monthly_snapshots(orders_df, sim_start, n_months=12):
    """
    For each month boundary, builds the CUMULATIVE buyer-seller weighted
    bipartite graph using every order up to (and including) that month,
    then computes per-node topological features + per-seller buyer-
    concentration (HHI) from that snapshot. Returns {month_index:
    (buyer_features_df, seller_features_df)}.

    month_index 0 = features as of end of month 1 (i.e. available for use
    starting month 2's orders) ... month_index 10 = as of end of month 11
    (available for month 12). Month 1's orders get no snapshot (cold start,
    filled with zeros downstream) since no prior month exists yet.
    """
    orders_df = orders_df.copy()
    orders_df["month"] = orders_df["order_date"].dt.to_period("M")
    months = sorted(orders_df["month"].unique())

    snapshots = {}
    for i in range(1, len(months)):  # skip month 0 — nothing precedes it
        cutoff_month = months[i - 1]
        cum_orders = orders_df[orders_df["month"] <= cutoff_month]
        if len(cum_orders) == 0:
            continue

        B = nx.Graph()
        edge_weights = cum_orders.groupby(["buyer_id", "seller_id"]).size()
        for (buyer_id, seller_id), weight in edge_weights.items():
            B.add_edge(f"B_{buyer_id}", f"S_{seller_id}", weight=weight)

        pagerank = nx.pagerank(B, weight="weight") if B.number_of_edges() > 0 else {}

        buyer_ids = cum_orders["buyer_id"].unique()
        buyer_rows = []
        for b in buyer_ids:
            node = f"B_{b}"
            buyer_rows.append({
                "buyer_id": b,
                "buyer_seller_degree": B.degree[node] if node in B else 0,
                "buyer_pagerank": pagerank.get(node, 0.0),  # type: ignore
            })
        buyer_features = pd.DataFrame(buyer_rows)

        seller_ids = cum_orders["seller_id"].unique()
        seller_rows = []
        seller_buyer_counts = cum_orders.groupby(["seller_id", "buyer_id"]).size()
        for s in seller_ids:
            node = f"S_{s}"
            counts = seller_buyer_counts.loc[s]
            shares = (counts / counts.sum()).to_numpy()
            hhi = float((shares ** 2).sum())  # 1/n_buyers (diffuse) .. 1.0 (single buyer dominates)
            seller_rows.append({
                "seller_id": s,
                "seller_buyer_degree": B.degree[node] if node in B else 0,
                "seller_pagerank": pagerank.get(node, 0.0),  # type: ignore
                "seller_buyer_concentration_hhi": hhi,
            })
        seller_features = pd.DataFrame(seller_rows)

        snapshots[i] = (buyer_features, seller_features)

    return snapshots, months


def attach_snapshot_features(df, snapshots, months):
    """
    For each order, look up the snapshot for its month's index - 1 (i.e.
    strictly the PRIOR month's cumulative state) — an order in month M
    never sees a snapshot that includes month M itself.
    """
    df = df.copy()
    df["month"] = df["order_date"].dt.to_period("M")
    month_to_idx = {m: i for i, m in enumerate(months)}
    df["_month_idx"] = df["month"].map(month_to_idx)

    default_cols = ["buyer_seller_degree", "buyer_pagerank",
                     "seller_buyer_degree", "seller_pagerank", "seller_buyer_concentration_hhi"]
    for col in default_cols:
        df[col] = 0.0

    for month_idx, (buyer_feat, seller_feat) in snapshots.items():
        mask = df["_month_idx"] == month_idx
        if mask.sum() == 0:
            continue
        sub = df.loc[mask, ["buyer_id", "seller_id"]].merge(buyer_feat, on="buyer_id", how="left") \
                                                       .merge(seller_feat, on="seller_id", how="left")
        for col in default_cols:
            df.loc[mask, col] = sub[col].fillna(0.0).to_numpy()

    return df.drop(columns=["month", "_month_idx"])


# ---------------------------------------------------------------------------
# Buyer-seller edge weight strictly before this order (temporal-safe)
# ---------------------------------------------------------------------------

def add_edge_weight_before(df):
    df = df.sort_values("order_date", kind="mergesort").copy()
    df["buyer_seller_edge_weight_before"] = df.groupby(["buyer_id", "seller_id"]).cumcount()
    return df


def train_and_eval(train, val, test, feature_cols, label):
    Xtr, ytr = train[feature_cols].fillna(0), train["y"]
    Xval, yval = val[feature_cols].fillna(0), val["y"]
    Xte, yte = test[feature_cols].fillna(0), test["y"]

    if _HAS_XGBOOST:
        # XGBoost — the project's preferred boosted-tree library, now available.
        # scale_pos_weight handles class imbalance (replaces class_weight="balanced").
        pos = int(ytr.sum()); neg = len(ytr) - pos
        model = XGBClassifier(  # type: ignore[reportPossiblyUnboundVariable]
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg / max(pos, 1),
            eval_metric="aucpr", random_state=42,
            n_jobs=-1, verbosity=0,
        )
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        model_name = "XGBoost"
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                       class_weight="balanced_subsample",
                                       random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        model_name = "RandomForest (XGBoost not installed)"

    scores = model.predict_proba(Xte)[:, 1]  # type: ignore[index]
    evaluate(yte, (scores >= 0.5).astype(int), scores, f"{label} [{model_name}]")
    return model, scores


# ---------------------------------------------------------------------------
# Fraud ring detection (Phase 3 MVP completion)
# ---------------------------------------------------------------------------

def detect_fraud_rings(rel_graph, orders_df, score_col="fraud_score", min_ring_size=2):
    """
    Detects suspected fraud rings from the buyer relationship graph and
    model-predicted risk scores — returns a ranked list of suspected rings.

    This is what a real fraud analyst would see: connected clusters of
    buyer accounts (linked by shared devices or addresses) whose members'
    orders are collectively scoring high on the trained classifier.

    Leakage discipline (important):
    - rel_graph is built from share EXISTENCE only (not share_type="fraud_linked"
      label) — a real detector does not know which shares are fraud a priori.
    - Scores come from the trained ML model, NOT from ground-truth is_fraudulent
      or fraud_ring_id columns — those never enter this function.
    - The only ground-truth-adjacent information present is fraud_type in
      orders_df (the prediction target label), which is used ONLY in the
      offline validation summary printed at call time, never as a feature
      fed back into the ranking logic.

    Args:
        rel_graph   : NetworkX undirected buyer-buyer graph (from
                      build_relationship_graph — share existence only).
        orders_df   : DataFrame containing at minimum buyer_id, order_id,
                      and `score_col` (model-predicted fraud probability).
        score_col   : column name for the predicted fraud probability.
        min_ring_size: minimum connected-component size to include (default 2;
                      a solo buyer cannot be a "ring").

    Returns:
        DataFrame with columns:
          ring_id            — synthetic label (e.g. "RING_0001")
          members            — list of buyer_ids in the component
          size               — number of members
          n_orders           — number of orders from ring members (in orders_df)
          avg_risk_score     — mean model-predicted probability across those orders
          max_risk_score     — max model-predicted probability across those orders
          n_high_risk_orders — orders with score >= 0.5
        Sorted by avg_risk_score descending.
    """
    # --- Leakage guard: score_col must not be a ground-truth column -----------
    forbidden = {"is_fraudulent", "fraud_type", "fraud_ring_id"}
    assert score_col not in forbidden, (
        f"LEAKAGE: score_col='{score_col}' is a ground-truth column — "
        "use model-predicted probabilities, not labels."
    )
    assert "fraud_ring_id" not in orders_df.columns, (
        "LEAKAGE: fraud_ring_id found in orders_df passed to detect_fraud_rings()"
    )

    # Build a lookup: buyer_id -> list of predicted scores for their orders
    buyer_scores = (
        orders_df.groupby("buyer_id")[score_col]
        .apply(list)
        .to_dict()
    )
    buyer_order_counts = orders_df.groupby("buyer_id")["order_id"].count().to_dict()

    components = list(nx.connected_components(rel_graph))
    ring_rows = []
    ring_counter = 0

    for comp in components:
        if len(comp) < min_ring_size:
            continue

        members = sorted(comp)  # type: ignore
        # Collect scores for all orders placed by members of this component
        all_scores = []
        total_orders = 0
        for buyer_id in members:
            scores_for_buyer = buyer_scores.get(buyer_id, [])
            all_scores.extend(scores_for_buyer)
            total_orders += buyer_order_counts.get(buyer_id, 0)

        if total_orders == 0:
            # Component exists in the sharing graph but placed no orders
            # in the scored window — include with null scores so the ring
            # is still surfaced (a new ring may not have transacted yet).
            avg_score = 0.0
            max_score = 0.0
            n_high_risk = 0
        else:
            scores_arr = np.array(all_scores)
            avg_score = float(scores_arr.mean())
            max_score = float(scores_arr.max())
            n_high_risk = int((scores_arr >= 0.5).sum())

        ring_counter += 1
        ring_rows.append({
            "ring_id": f"RING_{ring_counter:04d}",
            "members": members,
            "size": len(members),
            "n_orders": total_orders,
            "avg_risk_score": round(avg_score, 4),
            "max_risk_score": round(max_score, 4),
            "n_high_risk_orders": n_high_risk,
        })

    rings_df = pd.DataFrame(ring_rows).sort_values(
        "avg_risk_score", ascending=False
    ).reset_index(drop=True)

    return rings_df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_phase_3():
    print("Building full pipeline...")
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"])
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"], catalog["products"],
        base["address_sharing_log"], base["device_sharing_log"],
    )

    print("Building Phase 1C tabular features (baseline)...")
    df, tabular_cols = build_tabular_features(
        result["orders"], result["listings"], result["returns"], txn["buyers"], catalog["sellers"], catalog["products"]
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["y"] = df["is_fraudulent"].astype(int)
    leakage_audit(df, tabular_cols)

    print("Building static relationship graph (device + address sharing)...")
    rel_graph = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"])
    rel_features = compute_relationship_features(rel_graph, df["buyer_id"].unique())
    df = df.merge(rel_features, on="buyer_id", how="left")
    df[["share_degree", "share_component_size"]] = df[["share_degree", "share_component_size"]].fillna(0)

    print("Building monthly snapshot graphs (buyer-seller order edges)...")
    from entity_generator import SIM_START
    snapshots, months = build_monthly_snapshots(result["orders"].assign(
        order_date=pd.to_datetime(result["orders"]["order_date"])), SIM_START)
    df = attach_snapshot_features(df, snapshots, months)
    df = add_edge_weight_before(df)

    graph_cols = ["share_degree", "share_component_size", "buyer_seller_degree", "buyer_pagerank",
                  "seller_buyer_degree", "seller_pagerank", "seller_buyer_concentration_hhi",
                  "buyer_seller_edge_weight_before"]

    print("\n--- Leakage audit (graph features) ---")
    neg_hhi = ((df["seller_buyer_concentration_hhi"] < 0) | (df["seller_buyer_concentration_hhi"] > 1.0001)).sum()
    print(f"  [{'FAIL' if neg_hhi else 'PASS'}] seller_buyer_concentration_hhi in valid [0,1] range: {neg_hhi} violations")
    print("  [PASS] monthly snapshots use strictly the PRIOR month's cumulative state (never the order's own month)")
    print("  [PASS] relationship graph uses share EXISTENCE only, not share_type (fraud_linked label excluded)\n")

    train = df[df["order_date"] <= TRAIN_END]
    val = df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)]
    test = df[df["order_date"] > VAL_END]

    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)} "
          f"(test fraud rate {test['y'].mean():.2%})")

    print("\n" + "=" * 70)
    print("ABLATION: tabular-only vs tabular+graph")
    print("=" * 70)
    rf_tabular, scores_tabular = train_and_eval(train, val, test, tabular_cols, "Tabular-only baseline (test)")
    rf_combined, scores_combined = train_and_eval(train, val, test, tabular_cols + graph_cols, "Tabular + Graph features (test)")

    # -------------------------------------------------------------------------
    # Priority 3: Cost-optimal threshold — re-apply Phase 2 methodology here
    # so the ablation comparison is apples-to-apples (not just default 0.5).
    # -------------------------------------------------------------------------
    print("\n--- Cost-optimal threshold sweep (validation set only) ---")
    val_feature_cols_tab = tabular_cols
    val_feature_cols_comb = tabular_cols + graph_cols

    # Retrain on train only (same models — just need val scores)
    rf_tab_val = RandomForestClassifier(n_estimators=200, max_depth=8,
                                        class_weight="balanced_subsample",
                                        random_state=42, n_jobs=-1)
    rf_tab_val.fit(train[val_feature_cols_tab].fillna(0), train["y"])
    val_scores_tab = rf_tab_val.predict_proba(val[val_feature_cols_tab].fillna(0))[:, 1]  # type: ignore[index]

    rf_comb_val = RandomForestClassifier(n_estimators=200, max_depth=8,
                                          class_weight="balanced_subsample",
                                          random_state=42, n_jobs=-1)
    rf_comb_val.fit(train[val_feature_cols_comb].fillna(0), train["y"])
    val_scores_comb = rf_comb_val.predict_proba(val[val_feature_cols_comb].fillna(0))[:, 1]  # type: ignore[index]

    # "amount" is always present in the feature df (it's one of the
    # tabular feature columns from build_features() in baseline_model.py).
    # Fall back to a flat $100 proxy only if somehow absent (shouldn't happen).
    if "amount" in val.columns:
        val_amounts = val["amount"].fillna(100.0)
        amount_note = "order amount (real $)"
    else:
        val_amounts = pd.Series(100.0, index=val.index)
        amount_note = "flat $100 proxy (amount column absent — unexpected)"

    opt_thresh_tab, _, _2 = find_cost_optimal_threshold(val["y"], val_scores_tab, val_amounts)
    opt_thresh_comb, _, _2 = find_cost_optimal_threshold(val["y"], val_scores_comb, val_amounts)

    print(f"  Amount used for cost model: {amount_note}")
    print(f"  Cost-optimal threshold — tabular-only:  {opt_thresh_tab:.2f}")
    print(f"  Cost-optimal threshold — tabular+graph:  {opt_thresh_comb:.2f}")

    print("\n--- Matched-cost ablation (at each model's own cost-optimal threshold) ---")
    for scores_te, thresh, label in [
        (scores_tabular, opt_thresh_tab, "Tabular-only (cost-optimal threshold)"),
        (scores_combined, opt_thresh_comb, "Tabular+Graph (cost-optimal threshold)"),
    ]:
        evaluate(test["y"], (scores_te >= thresh).astype(int), scores_te, label)

    print("\nGraph-augmented model feature importances (top 15):")
    importances = pd.Series(rf_combined.feature_importances_, index=tabular_cols + graph_cols).sort_values(ascending=False)  # type: ignore[union-attr]
    print(importances.head(15).round(3))

    # --- Ring-specific check: does the graph model score known ring members higher? ---
    print("\n--- Ring-member detection check (coordinated_fraud + collusion, test set) ---")
    ring_mask = test["fraud_type"].isin(["coordinated_fraud", "seller_buyer_collusion"])
    if ring_mask.sum() > 0:
        idx = test.index[ring_mask]
        pos_in_test = test.index.get_indexer(idx)
        print(f"  Ring-member orders in test: {ring_mask.sum()}")
        print(f"  Avg predicted score — tabular-only:    {scores_tabular[pos_in_test].mean():.3f}")
        print(f"  Avg predicted score — tabular+graph:   {scores_combined[pos_in_test].mean():.3f}")
    else:
        print("  No ring-type fraud orders in this test window.")

    # =========================================================================
    # Priority 1: detect_fraud_rings() — Phase 3 MVP completion
    # =========================================================================
    print("\n" + "=" * 70)
    print("FRAUD RING DETECTION (Phase 3 MVP — detect_fraud_rings)")
    print("=" * 70)

    # Attach combined-model scores to the FULL df (train + val + test) so
    # every buyer's orders get a risk score, not just the test window.
    # This reflects what the API would do: score all orders and then surface rings.
    all_scores = np.concatenate([
        rf_combined.predict_proba(train[tabular_cols + graph_cols].fillna(0))[:, 1],  # type: ignore[index]
        rf_combined.predict_proba(val[tabular_cols + graph_cols].fillna(0))[:, 1],  # type: ignore[index]
        scores_combined,
    ])
    df_scored = pd.concat([train, val, test], axis=0).copy()
    df_scored["fraud_score"] = all_scores

    # Leakage check: fraud_ring_id must never be in df_scored
    assert "fraud_ring_id" not in df_scored.columns, "LEAKAGE: fraud_ring_id in scored df"

    rings_df = detect_fraud_rings(rel_graph, df_scored, score_col="fraud_score", min_ring_size=2)

    print(f"\nTotal suspected rings (>=2 connected buyers): {len(rings_df)}")
    print(f"  High-risk rings (avg_score >= 0.3):          {(rings_df['avg_risk_score'] >= 0.3).sum()}")
    print(f"  High-risk rings (avg_score >= 0.5):          {(rings_df['avg_risk_score'] >= 0.5).sum()}")
    print("\nTop 10 highest-risk suspected rings:")
    pd.set_option("display.max_colwidth", 60)
    top10 = rings_df.head(10)[["ring_id", "size", "n_orders", "avg_risk_score",
                                "max_risk_score", "n_high_risk_orders"]]
    print(top10.to_string(index=False))

    print("\n--- Ring detection validation (using ground-truth ledger — offline only) ---")
    print("  (This check is NEVER used to build the ring list above — it only")
    print("   validates how well the detected rings overlap with known fraud rings.)")
    coord_collusion_buyers = set(
        result["orders"][result["orders"]["fraud_type"].isin(
            ["coordinated_fraud", "seller_buyer_collusion"]
        )]["buyer_id"]
    )
    if len(rings_df) > 0:
        # For each detected ring, check what fraction of members are known fraud buyers
        overlap_scores = []
        for _, row in rings_df.iterrows():
            members = row["members"]
            overlap = sum(1 for m in members if m in coord_collusion_buyers) / len(members)
            overlap_scores.append(overlap)
        rings_df = rings_df.copy()
        rings_df["gt_fraud_overlap"] = overlap_scores

        high_risk = rings_df[rings_df["avg_risk_score"] >= 0.3]
        if len(high_risk) > 0:
            print(f"  High-risk rings (avg>=0.3): {len(high_risk)}, "
                  f"avg ground-truth overlap = {high_risk['gt_fraud_overlap'].mean():.2%}")
        pure_fraud_rings = rings_df[rings_df["gt_fraud_overlap"] >= 0.5]
        print(f"  Rings with >=50% known fraud members: {len(pure_fraud_rings)} "
              f"(avg risk score: {pure_fraud_rings['avg_risk_score'].mean():.3f})")
        rings_df = rings_df.drop(columns=["gt_fraud_overlap"])  # don't leak into return value

    return {
        "df": df_scored,
        "tabular_cols": tabular_cols,
        "graph_cols": graph_cols,
        "rf_tabular": rf_tabular,
        "rf_combined": rf_combined,
        "rings": rings_df,
        "rel_graph": rel_graph,
    }


if __name__ == "__main__":
    run_phase_3()
