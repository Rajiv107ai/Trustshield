"""5-Seed Pipeline Benchmark: Tabular vs Tabular+Graph vs Hybrid (Time-Aware OOF GNN).

Evaluates test ROC-AUC and PR-AUC across 5 distinct data generation seeds:
[42, 101, 202, 303, 404].
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

from entity_generator import generate_full_pipeline, SIM_START
from baseline_model import build_features as build_tabular_features, TRAIN_END, VAL_END
from graph_features import (
    build_relationship_graph,
    attach_relationship_snapshot_features,
    build_monthly_snapshots,
    attach_snapshot_features,
    add_edge_weight_before,
)
from phase5_hybrid_model import (
    build_node_index,
    build_node_features,
    build_edge_index,
    generate_oof_gnn_embeddings,
)

if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")


GRAPH_COLS = [
    "share_degree",
    "share_component_size",
    "buyer_seller_degree",
    "buyer_pagerank",
    "seller_buyer_degree",
    "seller_pagerank",
    "seller_buyer_concentration_hhi",
    "buyer_seller_edge_weight_before",
]


def train_xgb(X_tr, y_tr, random_state=42):
    pos = int(y_tr.sum())
    neg = len(y_tr) - pos
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X_tr.fillna(0.0), y_tr)
    return clf


def run_pipeline_for_seed(seed: int, gnn_epochs: int = 35):
    print(f"\n=======================================================")
    print(f" Running Pipeline on Data Seed: {seed}")
    print(f"=======================================================")
    pipe = generate_full_pipeline(seed=seed)
    base = pipe["base"]
    catalog = pipe["catalog"]
    txn = pipe["txn"]
    result = pipe["result"]

    df, tabular_cols = build_tabular_features(
        result["orders"], result["listings"], result["returns"],
        txn["buyers"], catalog["sellers"], catalog["products"]
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["y"] = df["is_fraudulent"].astype(int)

    # Attach graph features
    df = attach_relationship_snapshot_features(df, base["address_sharing_log"], base["device_sharing_log"])
    snapshots, months = build_monthly_snapshots(
        result["orders"].assign(order_date=pd.to_datetime(result["orders"]["order_date"])),
        SIM_START,
    )
    df = attach_snapshot_features(df, snapshots, months)
    df = add_edge_weight_before(df)

    tab_graph_cols = tabular_cols + GRAPH_COLS

    train = pd.DataFrame(df[df["order_date"] <= TRAIN_END])
    val = pd.DataFrame(df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)])
    test = pd.DataFrame(df[df["order_date"] > VAL_END])
    trainval = pd.DataFrame(df[df["order_date"] <= VAL_END])

    y_train = train["y"]
    y_test = test["y"].to_numpy()

    # 1. Tabular Model
    print("Training Tabular XGBoost...")
    tab_clf = train_xgb(train[tabular_cols], y_train)
    tab_scores = tab_clf.predict_proba(test[tabular_cols].fillna(0.0))[:, 1]
    tab_roc = float(roc_auc_score(y_test, tab_scores))
    tab_pr = float(average_precision_score(y_test, tab_scores))
    print(f"  Tabular       -> ROC-AUC: {tab_roc:.4f} | PR-AUC: {tab_pr:.4f}")

    # 2. Tabular + Graph Model
    print("Training Tabular + Graph XGBoost...")
    tg_clf = train_xgb(train[tab_graph_cols], y_train)
    tg_scores = tg_clf.predict_proba(test[tab_graph_cols].fillna(0.0))[:, 1]
    tg_roc = float(roc_auc_score(y_test, tg_scores))
    tg_pr = float(average_precision_score(y_test, tg_scores))
    lift_roc = tg_roc - tab_roc
    lift_pr = tg_pr - tab_pr
    print(f"  Tabular+Graph -> ROC-AUC: {tg_roc:.4f} | PR-AUC: {tg_pr:.4f} | Lift ROC: {lift_roc:+.4f} | Lift PR: {lift_pr:+.4f}")

    # 3. Hybrid Model with Time-Aware OOF GNN Embeddings
    print("Generating Time-Aware OOF GNN Embeddings...")
    buyer_idx, seller_idx, n_buyers, n_total = build_node_index(txn["buyers"]["buyer_id"], catalog["sellers"]["seller_id"])
    rel_graph_train = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"], cutoff_date=TRAIN_END)
    rel_graph_val = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"], cutoff_date=VAL_END)

    X_nodes_train = build_node_features(buyer_idx, seller_idx, n_total, train, rel_graph_train, txn["buyers"], catalog["sellers"], cutoff_date=TRAIN_END)
    edge_index_train = build_edge_index(buyer_idx, seller_idx, train, rel_graph_train)
    X_nodes_val = build_node_features(buyer_idx, seller_idx, n_total, trainval, rel_graph_val, txn["buyers"], catalog["sellers"], cutoff_date=VAL_END)
    edge_index_val = build_edge_index(buyer_idx, seller_idx, trainval, rel_graph_val)

    train_aug, val_aug, test_aug, gnn_cols = generate_oof_gnn_embeddings(
        train, val, test,
        buyer_idx, seller_idx, n_total,
        X_nodes_train, edge_index_train,
        X_nodes_val, edge_index_val,
        n_splits=5, epochs=gnn_epochs,
        fold_strategy="time_aware",
        random_state=seed,
    )
    hybrid_cols = tab_graph_cols + gnn_cols

    print("Training Hybrid XGBoost...")
    hyb_clf = train_xgb(train_aug[hybrid_cols], y_train)
    hyb_scores = hyb_clf.predict_proba(test_aug[hybrid_cols].fillna(0.0))[:, 1]
    hyb_roc = float(roc_auc_score(y_test, hyb_scores))
    hyb_pr = float(average_precision_score(y_test, hyb_scores))
    print(f"  Hybrid        -> ROC-AUC: {hyb_roc:.4f} | PR-AUC: {hyb_pr:.4f}")

    return {
        "seed": seed,
        "tab_roc": tab_roc,
        "tab_pr": tab_pr,
        "tg_roc": tg_roc,
        "tg_pr": tg_pr,
        "lift_roc": lift_roc,
        "lift_pr": lift_pr,
        "hyb_roc": hyb_roc,
        "hyb_pr": hyb_pr,
    }


def main():
    seeds = [42, 101, 202, 303, 404]
    results = []
    for s in seeds:
        res = run_pipeline_for_seed(s, gnn_epochs=35)
        results.append(res)

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("FINAL 5-SEED BENCHMARK SUMMARY")
    print("=" * 80)
    print(res_df.to_string(index=False))

    print("\nAGGREGATED PERFORMANCE (Mean +/- Std):")
    print(f"Tabular:        ROC-AUC = {res_df['tab_roc'].mean():.4f} +/- {res_df['tab_roc'].std():.4f} | PR-AUC = {res_df['tab_pr'].mean():.4f} +/- {res_df['tab_pr'].std():.4f}")
    print(f"Tabular+Graph:  ROC-AUC = {res_df['tg_roc'].mean():.4f} +/- {res_df['tg_roc'].std():.4f} | PR-AUC = {res_df['tg_pr'].mean():.4f} +/- {res_df['tg_pr'].std():.4f}")
    print(f"Hybrid (OOF):   ROC-AUC = {res_df['hyb_roc'].mean():.4f} +/- {res_df['hyb_roc'].std():.4f} | PR-AUC = {res_df['hyb_pr'].mean():.4f} +/- {res_df['hyb_pr'].std():.4f}")
    print(f"Graph Lift:     ROC-AUC Lift = {res_df['lift_roc'].mean():+.4f} +/- {res_df['lift_roc'].std():.4f} | PR-AUC Lift = {res_df['lift_pr'].mean():+.4f} +/- {res_df['lift_pr'].std():.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
