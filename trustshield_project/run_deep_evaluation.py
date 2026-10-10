"""Deep Empirical Evaluation Script for TrustShield Audit Remediation.

Complete suite covering:
1. Ring Coherence Diagnostic & Benchmark (ring_coherent=False vs ring_coherent=True across 5 seeds).
2. End-of-Horizon Audit (weekly fraud rates, last 8 weeks, 2025-12-31 fraud count).
3. GNN Convergence & Hyperparameter Sweep on Seed 42 (loss curve, val AUC curve, train AUC, lr sweep {0.002, 0.005, 0.01}, early stopping).
4. Graph Topology Statistics per Seed (nodes, edges, components, density).
5. Comprehensive Statistical Analysis (Cohen's d, paired t-tests, 95% paired CI).
6. Production of results/results.json and results/RESULTS.md with full provenance.
"""

from __future__ import annotations
import os
import sys
import json
import time
import hashlib
import subprocess
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from xgboost import XGBClassifier
import networkx as nx

# Ensure trustshield_project is in python path
_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    extract_embeddings,
    attach_gnn_embeddings,
    GraphSAGEEncoder,
    EdgeClassifier,
    GNN_EMB_DIM,
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

FRAUD_TYPES = [
    "fake_listing",
    "return_abuse",
    "coordinated_fraud",
    "seller_buyer_collusion",
]


def compute_dataset_hash(pipe: dict) -> str:
    """Computes composite SHA-256 hash of all dataset tables."""
    hasher = hashlib.sha256()
    for key in ["orders", "listings", "returns"]:
        df = pipe["result"][key]
        hasher.update(df.to_csv(index=False).encode("utf-8"))
    for key in ["buyers", "sellers", "products"]:
        df = pipe["txn"].get(key, pipe["catalog"].get(key, pipe["base"].get(key)))
        if df is not None:
            hasher.update(df.to_csv(index=False).encode("utf-8"))
    for key in ["address_sharing_log", "device_sharing_log"]:
        df = pipe["base"][key]
        hasher.update(df.to_csv(index=False).encode("utf-8"))
    return hasher.hexdigest()


def paired_stats_calc(baseline_vals, new_vals):
    """Calculates mean diff, std diff, 95% paired CI, Cohen's dz, and paired t-test p-value."""
    diffs = np.array(new_vals) - np.array(baseline_vals)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_d = std_d / np.sqrt(n) if n > 1 else 0.0
    t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 1.96
    ci_lower = mean_d - t_crit * se_d
    ci_upper = mean_d + t_crit * se_d
    cohens_d = (mean_d / std_d) if std_d > 1e-9 else 0.0
    t_res = stats.ttest_rel(new_vals, baseline_vals) if (std_d > 1e-9 and n > 1) else None
    p_val = float(t_res.pvalue) if t_res is not None and not np.isnan(t_res.pvalue) else 1.0
    return {
        "mean_diff": mean_d,
        "std_diff": std_d,
        "ci_95": [float(ci_lower), float(ci_upper)],
        "cohens_d": float(cohens_d),
        "p_value": p_val,
        "paired_diffs": [float(d) for d in diffs],
    }


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


def compute_recall_at_budget(y_test, scores, fraud_types_test, budget_pct=0.05):
    k = max(1, int(len(scores) * budget_pct))
    top_indices = np.argsort(scores)[::-1][:k]
    top_set = set(top_indices)

    recalls = {}
    for ft in FRAUD_TYPES:
        target_idx = np.where((y_test == 1) & (fraud_types_test == ft))[0]
        n_ft = len(target_idx)
        if n_ft == 0:
            recalls[ft] = 0.0
        else:
            caught = len(top_set.intersection(target_idx))
            recalls[ft] = caught / n_ft
    return recalls


def compute_ring_diagnostics(orders_df, dev_log_df):
    """Computes ring coherence metrics (a) and (b)."""
    coord_orders = orders_df[orders_df["fraud_type"] == "coordinated_fraud"]
    if len(coord_orders) == 0:
        return {"shared_device_fraction": 0.0, "causal_first_seen_fraction": 0.0}

    shared_devs = set(dev_log_df["shared_device_id"])
    frac_shared_dev = float(coord_orders["device_id"].isin(shared_devs).mean())

    dev_log = dev_log_df.copy()
    dev_log["first_seen_date"] = pd.to_datetime(dev_log["first_seen_date"])
    earliest_order = coord_orders.groupby("buyer_id")["order_date"].min()

    all_shares = pd.concat([
        dev_log[["buyer_id", "first_seen_date"]],
        dev_log[["shared_with_buyer_id", "first_seen_date"]].rename(columns={"shared_with_buyer_id": "buyer_id"}),
    ])
    min_shared = all_shares.groupby("buyer_id")["first_seen_date"].min()
    
    # Filter for buyers present in both
    common_buyers = earliest_order.index.intersection(min_shared.index)
    if len(common_buyers) == 0:
        frac_causal = 0.0
    else:
        frac_causal = float((min_shared.loc[common_buyers] <= earliest_order.loc[common_buyers]).mean())

    return {
        "shared_device_fraction": frac_shared_dev,
        "causal_first_seen_fraction": frac_causal,
    }


def compute_graph_topology_stats(orders_df, base_dict):
    """Computes topological summary of relationship & transaction graphs."""
    rel_graph = build_relationship_graph(
        base_dict["address_sharing_log"], base_dict["device_sharing_log"], cutoff_date=TRAIN_END
    )
    n_nodes = rel_graph.number_of_nodes()
    n_edges = rel_graph.number_of_edges()
    n_components = nx.number_connected_components(rel_graph)
    density = nx.density(rel_graph)
    return {
        "relationship_graph_train": {
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "num_connected_components": n_components,
            "density": float(density),
        }
    }


def run_gnn_convergence_experiment(seed42_ctx):
    """Runs lr sweep {0.002, 0.005, 0.01} with early stopping on validation link AUC."""
    print("\n" + "=" * 80)
    print("RUNNING GNN CONVERGENCE & LEARNING RATE SWEEP (DATA SEED 42)")
    print("=" * 80)

    ctx = seed42_ctx
    learning_rates = [0.002, 0.005, 0.01]
    sweep_results = {}

    def _make_tensors(split_df):
        b_ids = np.clip(split_df["buyer_id"].map(ctx["buyer_idx"]).fillna(0).astype(int).to_numpy(), 0, ctx["n_total"] - 1)
        s_ids = np.clip(split_df["seller_id"].map(ctx["seller_idx"]).fillna(0).astype(int).to_numpy(), 0, ctx["n_total"] - 1)
        feat_arr = split_df[["price_vs_base_price_ratio", "amount"]].fillna(0).copy()
        feat_arr["amount"] = np.log1p(feat_arr["amount"].clip(lower=0))
        return (
            torch.tensor(b_ids, dtype=torch.long),
            torch.tensor(s_ids, dtype=torch.long),
            torch.tensor(feat_arr.to_numpy(), dtype=torch.float32),
            torch.tensor(split_df["y"].to_numpy(), dtype=torch.float32),
        )

    tr_b, tr_s, tr_f, tr_y = _make_tensors(ctx["train"])
    val_b, val_s, val_f, val_y = _make_tensors(ctx["val"])

    pos_weight = torch.tensor([(tr_y == 0).sum().item() / max((tr_y == 1).sum().item(), 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_lr = None
    overall_best_val_auc = 0.0

    for lr in learning_rates:
        print(f"\n--- Testing Learning Rate: {lr} (Max 500 epochs, Early Stopping Patience=50) ---")
        torch.manual_seed(42)
        encoder = GraphSAGEEncoder()
        head = EdgeClassifier()
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)

        loss_curve = []
        train_auc_curve = []
        val_auc_curve = []
        best_val_auc = 0.0
        epochs_no_improve = 0
        best_epoch = 0

        for epoch in range(1, 501):
            encoder.train()
            head.train()
            optimizer.zero_grad()
            node_emb = encoder(ctx["X_nodes_train"], ctx["edge_index_train"])
            logits = head(node_emb[tr_b], node_emb[tr_s], tr_f)
            loss = loss_fn(logits, tr_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()

            if epoch % 10 == 0:
                encoder.eval()
                head.eval()
                with torch.no_grad():
                    tr_probs = torch.sigmoid(logits).detach().numpy()
                    tr_auc = float(roc_auc_score(tr_y.numpy(), tr_probs))
                    emb_val = encoder(ctx["X_nodes_val"], ctx["edge_index_val"])
                    val_probs = torch.sigmoid(head(emb_val[val_b], emb_val[val_s], val_f)).numpy()
                    val_auc = float(roc_auc_score(val_y.numpy(), val_probs))

                current_loss = float(loss.item())
                loss_curve.append({"epoch": epoch, "loss": current_loss})
                train_auc_curve.append({"epoch": epoch, "train_link_auc": tr_auc})
                val_auc_curve.append({"epoch": epoch, "val_link_auc": val_auc})

                if epoch <= 100 or epoch % 50 == 0:
                    print(f"  Epoch {epoch:3d}: Loss = {current_loss:.4f} | Train Link AUC = {tr_auc:.4f} | Val Link AUC = {val_auc:.4f}")

                if val_auc > best_val_auc + 1e-4:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 10

                if epochs_no_improve >= 50:
                    print(f"  Early stopping triggered at epoch {epoch} (Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch})")
                    break

        sweep_results[str(lr)] = {
            "learning_rate": lr,
            "best_val_link_auc": best_val_auc,
            "best_epoch": best_epoch,
            "loss_curve": loss_curve,
            "train_auc_curve": train_auc_curve,
            "val_auc_curve": val_auc_curve,
        }

        if best_val_auc > overall_best_val_auc:
            overall_best_val_auc = best_val_auc
            best_lr = lr

    print(f"\nOptimal Learning Rate selected on validation: {best_lr} (Best Val AUC: {overall_best_val_auc:.4f})")
    return {
        "best_lr": best_lr,
        "best_val_auc": overall_best_val_auc,
        "lr_sweep": sweep_results,
    }


def run_full_suite():
    print("=" * 80)
    print("TRUSTSHIELD AI — DEEP EMPIRICAL EVALUATION ACROSS 5 SEEDS")
    print("Pre-Registered Comparison: ring_coherent=False vs ring_coherent=True")
    print("=" * 80)

    seeds = [42, 101, 202, 303, 404]
    variants = [False, True]
    all_runs = {False: [], True: []}
    seed_hashes = {False: {}, True: {}}
    graph_stats_per_seed = {}
    weekly_fraud_rates_record = []
    seed42_context = None

    for seed in seeds:
        print(f"\n==================== PROCESSING DATA SEED: {seed} ====================")
        for ring_coherent in variants:
            var_name = "COHERENT" if ring_coherent else "STANDARD (INCOHERENT)"
            print(f"\n>>> Running Seed {seed} | Variant: {var_name}")
            t0 = time.time()
            pipe = generate_full_pipeline(seed=seed, ring_coherent=ring_coherent)
            base = pipe["base"]
            catalog = pipe["catalog"]
            txn = pipe["txn"]
            result = pipe["result"]

            dataset_hash = compute_dataset_hash(pipe)
            seed_hashes[ring_coherent][seed] = dataset_hash

            df, tabular_cols = build_tabular_features(
                result["orders"], result["listings"], result["returns"],
                txn["buyers"], catalog["sellers"], catalog["products"]
            )
            df["order_date"] = pd.to_datetime(df["order_date"])
            df["y"] = df["is_fraudulent"].astype(int)

            # Record graph stats for Seed 42
            if seed == 42 and not ring_coherent:
                graph_stats_per_seed[seed] = compute_graph_topology_stats(result["orders"], base)

            # Record ring diagnostics
            ring_diags = compute_ring_diagnostics(result["orders"], base["device_sharing_log"])

            # Attach point-in-time graph features
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
            fraud_types_test = test["fraud_type"].fillna("legitimate").to_numpy()

            # Record end of horizon stats for seed 42
            if seed == 42 and ring_coherent:
                orders_all = result["orders"].copy()
                orders_all["order_date"] = pd.to_datetime(orders_all["order_date"])
                orders_all["week"] = orders_all["order_date"].dt.isocalendar().week
                weekly = orders_all.groupby("week")["is_fraudulent"].agg(["count", "sum"])
                weekly["fraud_rate"] = weekly["sum"] / weekly["count"]
                test_orders = orders_all[orders_all["order_date"] > VAL_END]
                test_mean_rate = float(test_orders["is_fraudulent"].mean())
                dec31_count = int((orders_all["order_date"] == "2025-12-31").sum())
                dec31_fraud = int(((orders_all["order_date"] == "2025-12-31") & orders_all["is_fraudulent"]).sum())

                weekly_fraud_rates_record = {
                    "test_period_mean_rate": test_mean_rate,
                    "dec_31_total_orders": dec31_count,
                    "dec_31_fraud_orders": dec31_fraud,
                    "last_8_weeks": [
                        {"week": int(w), "total": int(r["count"]), "fraud": int(r["sum"]), "rate": float(r["fraud_rate"])}
                        for w, r in weekly.tail(8).iterrows()
                    ]
                }

            # 1. Tabular Model
            tab_clf = train_xgb(train[tabular_cols], y_train)
            tab_scores = tab_clf.predict_proba(test[tabular_cols].fillna(0.0))[:, 1]
            tab_roc = float(roc_auc_score(y_test, tab_scores))
            tab_pr = float(average_precision_score(y_test, tab_scores))
            tab_recalls = compute_recall_at_budget(y_test, tab_scores, fraud_types_test)

            # 2. Tabular + Graph Model
            tg_clf = train_xgb(train[tab_graph_cols], y_train)
            tg_scores = tg_clf.predict_proba(test[tab_graph_cols].fillna(0.0))[:, 1]
            tg_roc = float(roc_auc_score(y_test, tg_scores))
            tg_pr = float(average_precision_score(y_test, tg_scores))
            tg_recalls = compute_recall_at_budget(y_test, tg_scores, fraud_types_test)

            lift_roc = tg_roc - tab_roc
            lift_pr = tg_pr - tab_pr

            # GNN Graph Setup
            buyer_idx, seller_idx, n_buyers, n_total = build_node_index(txn["buyers"]["buyer_id"], catalog["sellers"]["seller_id"])
            rel_graph_train = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"], cutoff_date=TRAIN_END)
            rel_graph_val = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"], cutoff_date=VAL_END)

            X_nodes_train = build_node_features(buyer_idx, seller_idx, n_total, train, rel_graph_train, txn["buyers"], catalog["sellers"], cutoff_date=TRAIN_END)
            edge_index_train = build_edge_index(buyer_idx, seller_idx, train, rel_graph_train)
            X_nodes_val = build_node_features(buyer_idx, seller_idx, n_total, trainval, rel_graph_val, txn["buyers"], catalog["sellers"], cutoff_date=VAL_END)
            edge_index_val = build_edge_index(buyer_idx, seller_idx, trainval, rel_graph_val)

            # 3. Time-Aware OOF GNN
            tr_ta, v_ta, te_ta, gnn_cols = generate_oof_gnn_embeddings(
                train, val, test, buyer_idx, seller_idx, n_total,
                X_nodes_train, edge_index_train, X_nodes_val, edge_index_val,
                n_splits=5, epochs=30, random_state=seed, fold_strategy="time_aware"
            )
            hyb_ta_clf = train_xgb(tr_ta[tab_graph_cols + gnn_cols], y_train)
            hyb_ta_scores = hyb_ta_clf.predict_proba(te_ta[tab_graph_cols + gnn_cols].fillna(0.0))[:, 1]
            hyb_ta_roc = float(roc_auc_score(y_test, hyb_ta_scores))
            hyb_ta_pr = float(average_precision_score(y_test, hyb_ta_scores))

            # 4. Random Shuffled OOF GNN
            tr_rnd, v_rnd, te_rnd, _ = generate_oof_gnn_embeddings(
                train, val, test, buyer_idx, seller_idx, n_total,
                X_nodes_train, edge_index_train, X_nodes_val, edge_index_val,
                n_splits=5, epochs=30, random_state=seed, fold_strategy="random"
            )
            hyb_rnd_clf = train_xgb(tr_rnd[tab_graph_cols + gnn_cols], y_train)
            hyb_rnd_scores = hyb_rnd_clf.predict_proba(te_rnd[tab_graph_cols + gnn_cols].fillna(0.0))[:, 1]
            hyb_rnd_roc = float(roc_auc_score(y_test, hyb_rnd_scores))
            hyb_rnd_pr = float(average_precision_score(y_test, hyb_rnd_scores))

            elapsed = time.time() - t0
            print(f"  [Diagnostics] Shared Dev: {ring_diags['shared_device_fraction']:.2%} | Causal First-Seen: {ring_diags['causal_first_seen_fraction']:.2%}")
            print(f"  Tabular:       ROC={tab_roc:.4f} | PR={tab_pr:.4f}")
            print(f"  Tabular+Graph: ROC={tg_roc:.4f} | PR={tg_pr:.4f} | Lift ROC={lift_roc:+.4f} | PR={lift_pr:+.4f}")
            print(f"  Hybrid (Time): ROC={hyb_ta_roc:.4f} | PR={hyb_ta_pr:.4f}")
            print(f"  Completed in {elapsed:.1f}s")

            entry = {
                "seed": seed,
                "dataset_hash": dataset_hash,
                "ring_coherent": ring_coherent,
                "tab_roc": tab_roc,
                "tab_pr": tab_pr,
                "tg_roc": tg_roc,
                "tg_pr": tg_pr,
                "lift_roc": lift_roc,
                "lift_pr": lift_pr,
                "hyb_time_roc": hyb_ta_roc,
                "hyb_time_pr": hyb_ta_pr,
                "hyb_rnd_roc": hyb_rnd_roc,
                "hyb_rnd_pr": hyb_rnd_pr,
                "tab_recalls_at_5pct": tab_recalls,
                "tg_recalls_at_5pct": tg_recalls,
                "diagnostics": ring_diags,
            }
            all_runs[ring_coherent].append(entry)

            if seed == 42 and not ring_coherent:
                seed42_context = {
                    "train": train, "val": val, "test": test, "trainval": trainval,
                    "y_train": y_train, "y_test": y_test,
                    "tab_graph_cols": tab_graph_cols,
                    "buyer_idx": buyer_idx, "seller_idx": seller_idx, "n_total": n_total,
                    "X_nodes_train": X_nodes_train, "edge_index_train": edge_index_train,
                    "X_nodes_val": X_nodes_val, "edge_index_val": edge_index_val,
                }

    # =========================================================================
    # GNN CONVERGENCE & LR SWEEP ON SEED 42
    # =========================================================================
    gnn_convergence_report = run_gnn_convergence_experiment(seed42_context)

    # =========================================================================
    # STATISTICAL COMPARISONS
    # =========================================================================
    def _agg_variant(runs_list):
        df_r = pd.DataFrame(runs_list)
        tg_vs_tab_roc = paired_stats_calc(df_r["tab_roc"], df_r["tg_roc"])
        tg_vs_tab_pr = paired_stats_calc(df_r["tab_pr"], df_r["tg_pr"])
        hyb_vs_tab_roc = paired_stats_calc(df_r["tab_roc"], df_r["hyb_time_roc"])
        hyb_vs_tab_pr = paired_stats_calc(df_r["tab_pr"], df_r["hyb_time_pr"])
        time_vs_rnd_roc = paired_stats_calc(df_r["hyb_rnd_roc"], df_r["hyb_time_roc"])
        time_vs_rnd_pr = paired_stats_calc(df_r["hyb_rnd_pr"], df_r["hyb_time_pr"])

        recall_analysis = {}
        for ft in FRAUD_TYPES:
            tab_recs = [r["tab_recalls_at_5pct"][ft] for r in runs_list]
            tg_recs = [r["tg_recalls_at_5pct"][ft] for r in runs_list]
            recall_analysis[ft] = {
                "tab_mean": float(np.mean(tab_recs)),
                "tab_std": float(np.std(tab_recs, ddof=1)),
                "tg_mean": float(np.mean(tg_recs)),
                "tg_std": float(np.std(tg_recs, ddof=1)),
                "paired_stats": paired_stats_calc(tab_recs, tg_recs),
            }

        diag_shared = [r["diagnostics"]["shared_device_fraction"] for r in runs_list]
        diag_causal = [r["diagnostics"]["causal_first_seen_fraction"] for r in runs_list]

        return {
            "summary_metrics": {
                "tabular": {
                    "roc_mean": float(df_r["tab_roc"].mean()),
                    "roc_std": float(df_r["tab_roc"].std(ddof=1)),
                    "pr_mean": float(df_r["tab_pr"].mean()),
                    "pr_std": float(df_r["tab_pr"].std(ddof=1)),
                },
                "tabular_plus_graph": {
                    "roc_mean": float(df_r["tg_roc"].mean()),
                    "roc_std": float(df_r["tg_roc"].std(ddof=1)),
                    "pr_mean": float(df_r["tg_pr"].mean()),
                    "pr_std": float(df_r["tg_pr"].std(ddof=1)),
                },
                "hybrid_time_aware": {
                    "roc_mean": float(df_r["hyb_time_roc"].mean()),
                    "roc_std": float(df_r["hyb_time_roc"].std(ddof=1)),
                    "pr_mean": float(df_r["hyb_time_pr"].mean()),
                    "pr_std": float(df_r["hyb_time_pr"].std(ddof=1)),
                },
                "hybrid_random_kfold": {
                    "roc_mean": float(df_r["hyb_rnd_roc"].mean()),
                    "roc_std": float(df_r["hyb_rnd_roc"].std(ddof=1)),
                    "pr_mean": float(df_r["hyb_rnd_pr"].mean()),
                    "pr_std": float(df_r["hyb_rnd_pr"].std(ddof=1)),
                },
            },
            "diagnostics": {
                "shared_device_fraction_mean": float(np.mean(diag_shared)),
                "shared_device_fraction_std": float(np.std(diag_shared, ddof=1)),
                "causal_first_seen_fraction_mean": float(np.mean(diag_causal)),
                "causal_first_seen_fraction_std": float(np.std(diag_causal, ddof=1)),
            },
            "paired_comparisons": {
                "tabular_plus_graph_vs_tabular": {
                    "roc": tg_vs_tab_roc,
                    "pr": tg_vs_tab_pr,
                },
                "hybrid_vs_tabular": {
                    "roc": hyb_vs_tab_roc,
                    "pr": hyb_vs_tab_pr,
                },
                "time_aware_vs_random_kfold": {
                    "roc": time_vs_rnd_roc,
                    "pr": time_vs_rnd_pr,
                },
            },
            "recall_at_5pct_decomposition": recall_analysis,
            "per_seed_runs": runs_list,
        }

    standard_summary = _agg_variant(all_runs[False])
    coherent_summary = _agg_variant(all_runs[True])

    # Paired test between Coherent and Standard variants across seeds
    coherent_vs_standard = {
        "tg_roc": paired_stats_calc(
            [r["tg_roc"] for r in all_runs[False]],
            [r["tg_roc"] for r in all_runs[True]]
        ),
        "tg_pr": paired_stats_calc(
            [r["tg_pr"] for r in all_runs[False]],
            [r["tg_pr"] for r in all_runs[True]]
        ),
        "coord_fraud_recall_at_5pct": paired_stats_calc(
            [r["tg_recalls_at_5pct"]["coordinated_fraud"] for r in all_runs[False]],
            [r["tg_recalls_at_5pct"]["coordinated_fraud"] for r in all_runs[True]]
        ),
    }

    # Git metadata & dirty flag
    try:
        commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit_sha = "unknown"

    try:
        git_status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        is_dirty = len(git_status) > 0
    except Exception:
        is_dirty = True

    import xgboost
    import sklearn

    output_payload = {
        "metadata": {
            "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": commit_sha,
            "dirty": is_dirty,
            "seeds": seeds,
            "dataset_hashes": {
                "standard": seed_hashes[False],
                "coherent": seed_hashes[True],
            },
            "library_versions": {
                "python": sys.version.split()[0],
                "xgboost": xgboost.__version__,
                "torch": torch.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "networkx": nx.__version__,
            },
        },
        "side_by_side_results": {
            "standard_variant": standard_summary,
            "coherent_variant": coherent_summary,
            "coherent_vs_standard_paired": coherent_vs_standard,
        },
        "end_of_horizon_audit": weekly_fraud_rates_record,
        "gnn_convergence_seed42": gnn_convergence_report,
        "graph_topology_stats_train": graph_stats_per_seed,
    }

    os.makedirs(os.path.join(_ROOT_DIR, "results"), exist_ok=True)
    json_path = os.path.join(_ROOT_DIR, "results", "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved comprehensive benchmark payload to: {json_path}")

    # Generate Markdown Summary
    md_content = f"""# TrustShield Benchmark Results (5 Seeds, Out-of-Time Test)

> [!NOTE]
> Produced under strict audit controls: point-in-time snapshot graphs, out-of-fold embeddings, no synthetic surrogate tuning.

- **Git Commit**: `{commit_sha}` (dirty: `{is_dirty}`)
- **Timestamp**: `{output_payload['metadata']['execution_timestamp']}`
- **Seeds**: {seeds}

## 1. Pre-Registered Side-by-Side Model Performance

| Model | Variant | Test ROC-AUC (mean ± std) | Test PR-AUC (mean ± std) | Lift vs Tabular (ROC-AUC) | Lift vs Tabular (PR-AUC) |
|---|---|---|---|---|---|
| **Tabular XGBoost** | Standard | {standard_summary['summary_metrics']['tabular']['roc_mean']:.4f} ± {standard_summary['summary_metrics']['tabular']['roc_std']:.4f} | {standard_summary['summary_metrics']['tabular']['pr_mean']:.4f} ± {standard_summary['summary_metrics']['tabular']['pr_std']:.4f} | — | — |
| **Tabular + Graph** | Standard | {standard_summary['summary_metrics']['tabular_plus_graph']['roc_mean']:.4f} ± {standard_summary['summary_metrics']['tabular_plus_graph']['roc_std']:.4f} | {standard_summary['summary_metrics']['tabular_plus_graph']['pr_mean']:.4f} ± {standard_summary['summary_metrics']['tabular_plus_graph']['pr_std']:.4f} | {standard_summary['paired_comparisons']['tabular_plus_graph_vs_tabular']['roc']['mean_diff']:+.4f} (p={standard_summary['paired_comparisons']['tabular_plus_graph_vs_tabular']['roc']['p_value']:.4f}) | {standard_summary['paired_comparisons']['tabular_plus_graph_vs_tabular']['pr']['mean_diff']:+.4f} (p={standard_summary['paired_comparisons']['tabular_plus_graph_vs_tabular']['pr']['p_value']:.4f}) |
| **Hybrid (Time-Aware OOF GNN)** | Standard | {standard_summary['summary_metrics']['hybrid_time_aware']['roc_mean']:.4f} ± {standard_summary['summary_metrics']['hybrid_time_aware']['roc_std']:.4f} | {standard_summary['summary_metrics']['hybrid_time_aware']['pr_mean']:.4f} ± {standard_summary['summary_metrics']['hybrid_time_aware']['pr_std']:.4f} | {standard_summary['paired_comparisons']['hybrid_vs_tabular']['roc']['mean_diff']:+.4f} (p={standard_summary['paired_comparisons']['hybrid_vs_tabular']['roc']['p_value']:.4f}) | {standard_summary['paired_comparisons']['hybrid_vs_tabular']['pr']['mean_diff']:+.4f} (p={standard_summary['paired_comparisons']['hybrid_vs_tabular']['pr']['p_value']:.4f}) |
| **Tabular XGBoost** | Coherent | {coherent_summary['summary_metrics']['tabular']['roc_mean']:.4f} ± {coherent_summary['summary_metrics']['tabular']['roc_std']:.4f} | {coherent_summary['summary_metrics']['tabular']['pr_mean']:.4f} ± {coherent_summary['summary_metrics']['tabular']['pr_std']:.4f} | — | — |
| **Tabular + Graph** | Coherent | {coherent_summary['summary_metrics']['tabular_plus_graph']['roc_mean']:.4f} ± {coherent_summary['summary_metrics']['tabular_plus_graph']['roc_std']:.4f} | {coherent_summary['summary_metrics']['tabular_plus_graph']['pr_mean']:.4f} ± {coherent_summary['summary_metrics']['tabular_plus_graph']['pr_std']:.4f} | {coherent_summary['paired_comparisons']['tabular_plus_graph_vs_tabular']['roc']['mean_diff']:+.4f} (p={coherent_summary['paired_comparisons']['tabular_plus_graph_vs_tabular']['roc']['p_value']:.4f}) | {coherent_summary['paired_comparisons']['tabular_plus_graph_vs_tabular']['pr']['mean_diff']:+.4f} (p={coherent_summary['paired_comparisons']['tabular_plus_graph_vs_tabular']['pr']['p_value']:.4f}) |
| **Hybrid (Time-Aware OOF GNN)** | Coherent | {coherent_summary['summary_metrics']['hybrid_time_aware']['roc_mean']:.4f} ± {coherent_summary['summary_metrics']['hybrid_time_aware']['roc_std']:.4f} | {coherent_summary['summary_metrics']['hybrid_time_aware']['pr_mean']:.4f} ± {coherent_summary['summary_metrics']['hybrid_time_aware']['pr_std']:.4f} | {coherent_summary['paired_comparisons']['hybrid_vs_tabular']['roc']['mean_diff']:+.4f} (p={coherent_summary['paired_comparisons']['hybrid_vs_tabular']['roc']['p_value']:.4f}) | {coherent_summary['paired_comparisons']['hybrid_vs_tabular']['pr']['mean_diff']:+.4f} (p={coherent_summary['paired_comparisons']['hybrid_vs_tabular']['pr']['p_value']:.4f}) |

## 2. Ring Coherence Diagnostics

| Metric | Standard Variant | Coherent Variant |
|---|---|---|
| **Burst Orders with Shared Device** | {standard_summary['diagnostics']['shared_device_fraction_mean']:.2%} ± {standard_summary['diagnostics']['shared_device_fraction_std']:.2%} | {coherent_summary['diagnostics']['shared_device_fraction_mean']:.2%} ± {coherent_summary['diagnostics']['shared_device_fraction_std']:.2%} |
| **Sharing Log First-Seen <= Burst Date** | {standard_summary['diagnostics']['causal_first_seen_fraction_mean']:.2%} ± {standard_summary['diagnostics']['causal_first_seen_fraction_std']:.2%} | {coherent_summary['diagnostics']['causal_first_seen_fraction_mean']:.2%} ± {coherent_summary['diagnostics']['causal_first_seen_fraction_std']:.2%} |

## 3. Recall@5% Budget per Fraud Scenario (Tabular vs Tabular + Graph)

| Fraud Scenario | Standard Tabular | Standard Graph | Coherent Tabular | Coherent Graph |
|---|---|---|---|---|
| **fake_listing** | {standard_summary['recall_at_5pct_decomposition']['fake_listing']['tab_mean']:.3f} | {standard_summary['recall_at_5pct_decomposition']['fake_listing']['tg_mean']:.3f} | {coherent_summary['recall_at_5pct_decomposition']['fake_listing']['tab_mean']:.3f} | {coherent_summary['recall_at_5pct_decomposition']['fake_listing']['tg_mean']:.3f} |
| **return_abuse** | {standard_summary['recall_at_5pct_decomposition']['return_abuse']['tab_mean']:.3f} | {standard_summary['recall_at_5pct_decomposition']['return_abuse']['tg_mean']:.3f} | {coherent_summary['recall_at_5pct_decomposition']['return_abuse']['tab_mean']:.3f} | {coherent_summary['recall_at_5pct_decomposition']['return_abuse']['tg_mean']:.3f} |
| **coordinated_fraud** | {standard_summary['recall_at_5pct_decomposition']['coordinated_fraud']['tab_mean']:.3f} | {standard_summary['recall_at_5pct_decomposition']['coordinated_fraud']['tg_mean']:.3f} | {coherent_summary['recall_at_5pct_decomposition']['coordinated_fraud']['tab_mean']:.3f} | {coherent_summary['recall_at_5pct_decomposition']['coordinated_fraud']['tg_mean']:.3f} |
| **seller_buyer_collusion** | {standard_summary['recall_at_5pct_decomposition']['seller_buyer_collusion']['tab_mean']:.3f} | {standard_summary['recall_at_5pct_decomposition']['seller_buyer_collusion']['tg_mean']:.3f} | {coherent_summary['recall_at_5pct_decomposition']['seller_buyer_collusion']['tab_mean']:.3f} | {coherent_summary['recall_at_5pct_decomposition']['seller_buyer_collusion']['tg_mean']:.3f} |
"""
    md_path = os.path.join(_ROOT_DIR, "results", "RESULTS.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved human-readable report to: {md_path}")
    print("=" * 80)
    print("EXECUTION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_full_suite()
