"""Graph-topological feature extraction and fraud ring detection using NetworkX."""

import numpy as np
import pandas as pd
import networkx as nx
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from entity_generator import build_base_entities, SIM_START
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud
from baseline_model import build_features as build_tabular_features, TRAIN_END, VAL_END, leakage_audit
from utils import evaluate, find_cost_optimal_threshold


def build_relationship_graph(address_sharing_log: pd.DataFrame, device_sharing_log: pd.DataFrame) -> nx.Graph:
    """Builds undirected buyer-buyer graph from shared physical addresses and device IDs."""
    G = nx.Graph()
    G.add_edges_from(
        [(b1, b2, {"kind": "address"})
         for b1, b2 in zip(address_sharing_log["buyer_id"], address_sharing_log["shared_with_buyer_id"])]
    )
    G.add_edges_from(
        [(b1, b2, {"kind": "device"})
         for b1, b2 in zip(device_sharing_log["buyer_id"], device_sharing_log["shared_with_buyer_id"])]
    )
    return G


def compute_relationship_features(graph: nx.Graph, buyer_ids) -> pd.DataFrame:
    """Extracts degree centrality and connected component size per buyer."""
    components = {node: comp for comp in nx.connected_components(graph) for node in comp}
    rows = [
        {
            "buyer_id": b,
            "share_degree": graph.degree[b] if b in graph else 0,
            "share_component_size": len(components[b]) if b in graph else 1,
        }
        for b in buyer_ids
    ]
    return pd.DataFrame(rows)


def build_monthly_snapshots(orders_df: pd.DataFrame, sim_start, n_months: int = 12):
    """Builds monthly bipartite transaction graphs and computes PageRank & HHI concentration."""
    orders = orders_df.copy()
    orders["month"] = pd.PeriodIndex(orders["order_date"], freq="M")
    months = sorted(orders["month"].unique())

    snapshots = {}
    for i in range(1, len(months)):
        cutoff = months[i - 1]
        cum_orders = pd.DataFrame(orders[orders["month"] <= cutoff])
        if cum_orders.empty:
            continue

        B = nx.Graph()
        edge_df = cum_orders.groupby(["buyer_id", "seller_id"], as_index=False).size()
        for b, s, weight in zip(edge_df["buyer_id"], edge_df["seller_id"], edge_df["size"]):
            B.add_edge(f"B_{b}", f"S_{s}", weight=weight)

        pagerank = nx.pagerank(B, weight="weight") if B.number_of_edges() > 0 else {}

        buyer_ids = list(dict.fromkeys(cum_orders["buyer_id"]))
        buyer_rows = [
            {
                "buyer_id": b,
                "buyer_seller_degree": B.degree[f"B_{b}"] if f"B_{b}" in B else 0,
                "buyer_pagerank": pagerank.get(f"B_{b}", 0.0),
            }
            for b in buyer_ids
        ]

        seller_ids = list(dict.fromkeys(cum_orders["seller_id"]))
        seller_buyer_counts = cum_orders.groupby(["seller_id", "buyer_id"]).size()
        seller_rows = []
        for s in seller_ids:
            counts = seller_buyer_counts.loc[s]
            shares = (counts / counts.sum()).to_numpy()
            seller_rows.append({
                "seller_id": s,
                "seller_buyer_degree": B.degree[f"S_{s}"] if f"S_{s}" in B else 0,
                "seller_pagerank": pagerank.get(f"S_{s}", 0.0),
                "seller_buyer_concentration_hhi": float((shares ** 2).sum()),
            })

        snapshots[i] = (pd.DataFrame(buyer_rows), pd.DataFrame(seller_rows))

    return snapshots, months


def attach_snapshot_features(df: pd.DataFrame, snapshots: dict, months: list) -> pd.DataFrame:
    """Attaches prior-month graph snapshot features to orders."""
    df = df.copy()
    df["month"] = pd.PeriodIndex(df["order_date"], freq="M")
    month_to_idx = {m: i for i, m in enumerate(months)}
    df["_month_idx"] = df["month"].map(month_to_idx.get)

    feature_cols = [
        "buyer_seller_degree", "buyer_pagerank",
        "seller_buyer_degree", "seller_pagerank", "seller_buyer_concentration_hhi"
    ]
    for col in feature_cols:
        df[col] = 0.0

    for month_idx, (buyer_feat, seller_feat) in snapshots.items():
        mask = df["_month_idx"] == month_idx
        if not mask.any():
            continue
        sub = df.loc[mask, ["buyer_id", "seller_id"]].merge(buyer_feat, on="buyer_id", how="left") \
                                                       .merge(seller_feat, on="seller_id", how="left")
        for col in feature_cols:
            df.loc[mask, col] = sub[col].fillna(0.0).to_numpy()

    return df.drop(columns=["month", "_month_idx"])


def add_edge_weight_before(df: pd.DataFrame) -> pd.DataFrame:
    """Computes prior transaction count between specific buyer-seller pairs."""
    df = df.sort_values("order_date", kind="mergesort").copy()
    df["buyer_seller_edge_weight_before"] = df.groupby(["buyer_id", "seller_id"]).cumcount()
    return df


def train_and_eval(train, val, test, feature_cols, label: str):
    """Trains an XGBoost model on train, checks early stopping on val, and evaluates on test."""
    Xtr = pd.DataFrame(train[feature_cols]).fillna(0)
    ytr = train["y"]
    Xval = pd.DataFrame(val[feature_cols]).fillna(0)
    yval = val["y"]
    Xte = pd.DataFrame(test[feature_cols]).fillna(0)
    yte = test["y"]

    pos = int(float(ytr.sum()))
    neg = len(ytr) - pos

    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="aucpr", random_state=42,
        n_jobs=-1, verbosity=0,
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)

    scores = np.asarray(model.predict_proba(Xte))[:, 1]
    evaluate(yte, (scores >= 0.5).astype(int), scores, f"{label} [XGBoost]")
    return model, scores


def detect_fraud_rings(rel_graph: nx.Graph, orders_df: pd.DataFrame, score_col: str = "fraud_score", min_ring_size: int = 2) -> pd.DataFrame:
    """Surfaces suspicious clusters of accounts sharing infrastructure scored by predicted risk."""
    forbidden = {"is_fraudulent", "fraud_type", "fraud_ring_id"}
    assert score_col not in forbidden, f"Leakage: ground-truth column {score_col} used in ring detection"
    assert "fraud_ring_id" not in orders_df.columns, "Leakage: fraud_ring_id present in orders_df"

    buyer_scores = orders_df.groupby("buyer_id")[score_col].apply(list).to_dict()
    buyer_order_counts = orders_df.groupby("buyer_id")["order_id"].count().to_dict()

    ring_rows = []
    ring_counter = 0

    for comp in nx.connected_components(rel_graph):
        if len(comp) < min_ring_size:
            continue

        members = sorted(comp)
        all_scores = [s for b in members for s in buyer_scores.get(b, [])]
        total_orders = sum(buyer_order_counts.get(b, 0) for b in members)

        if total_orders == 0:
            avg_score, max_score, n_high_risk = 0.0, 0.0, 0
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

    return pd.DataFrame(ring_rows).sort_values("avg_risk_score", ascending=False).reset_index(drop=True)


def run_phase_3():
    """Runs full graph-topological feature engineering, model training, and ring detection."""
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"])
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"], catalog["products"],
        base["address_sharing_log"], base["device_sharing_log"],
    )

    df, tabular_cols = build_tabular_features(
        result["orders"], result["listings"], result["returns"], txn["buyers"], catalog["sellers"], catalog["products"]
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["y"] = df["is_fraudulent"].astype(int)
    leakage_audit(df, tabular_cols)

    rel_graph = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"])
    rel_features = compute_relationship_features(rel_graph, df["buyer_id"].unique())
    df = df.merge(rel_features, on="buyer_id", how="left")
    df[["share_degree", "share_component_size"]] = df[["share_degree", "share_component_size"]].fillna(0)

    snapshots, months = build_monthly_snapshots(result["orders"].assign(order_date=pd.to_datetime(result["orders"]["order_date"])), SIM_START)
    df = attach_snapshot_features(df, snapshots, months)
    df = add_edge_weight_before(df)

    graph_cols = [
        "share_degree", "share_component_size", "buyer_seller_degree", "buyer_pagerank",
        "seller_buyer_degree", "seller_pagerank", "seller_buyer_concentration_hhi",
        "buyer_seller_edge_weight_before"
    ]

    train = pd.DataFrame(df[df["order_date"] <= TRAIN_END])
    val = pd.DataFrame(df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)])
    test = pd.DataFrame(df[df["order_date"] > VAL_END])

    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)} (test fraud rate {test['y'].mean():.2%})")

    rf_tabular, scores_tabular = train_and_eval(train, val, test, tabular_cols, "Tabular-only baseline (test)")
    rf_combined, scores_combined = train_and_eval(train, val, test, tabular_cols + graph_cols, "Tabular + Graph features (test)")

    # Cost-optimal threshold evaluation
    val_amounts = val["amount"].fillna(100.0) if "amount" in val.columns else pd.Series(100.0, index=val.index)
    rf_tab_val = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    rf_tab_val.fit(pd.DataFrame(train[tabular_cols]).fillna(0), train["y"])
    val_scores_tab = np.asarray(rf_tab_val.predict_proba(pd.DataFrame(val[tabular_cols]).fillna(0)))[:, 1]

    rf_comb_val = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    rf_comb_val.fit(pd.DataFrame(train[tabular_cols + graph_cols]).fillna(0), train["y"])
    val_scores_comb = np.asarray(rf_comb_val.predict_proba(pd.DataFrame(val[tabular_cols + graph_cols]).fillna(0)))[:, 1]

    opt_thresh_tab, _, _ = find_cost_optimal_threshold(val["y"], val_scores_tab, val_amounts)
    opt_thresh_comb, _, _ = find_cost_optimal_threshold(val["y"], val_scores_comb, val_amounts)

    print(f"Cost-optimal threshold — tabular-only: {opt_thresh_tab:.2f} | tabular+graph: {opt_thresh_comb:.2f}")
    evaluate(test["y"], (scores_tabular >= opt_thresh_tab).astype(int), scores_tabular, "Tabular-only @ optimal threshold")
    evaluate(test["y"], (scores_combined >= opt_thresh_comb).astype(int), scores_combined, "Tabular+Graph @ optimal threshold")

    # Score full dataset and surface rings
    df_scored = pd.concat([train, val, test], axis=0, ignore_index=True)
    features_full = df_scored[tabular_cols + graph_cols].fillna(0)
    df_scored = df_scored.assign(
        fraud_score=np.asarray(rf_combined.predict_proba(features_full))[:, 1]
    )

    rings_df = detect_fraud_rings(rel_graph, df_scored, score_col="fraud_score", min_ring_size=2)
    print(f"\nDetected {len(rings_df)} suspicious account clusters (rings). Top 5:")
    print(rings_df.head(5)[["ring_id", "size", "n_orders", "avg_risk_score", "max_risk_score"]].to_string(index=False))

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
