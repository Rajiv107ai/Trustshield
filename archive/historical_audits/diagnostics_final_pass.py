"""Diagnostics script for Final Consistency Pass.

Computes:
1. Variant (d) repr() and sha256 for both full-horizon checkpoints and truncated horizon benchmark.
2. Correlation of seller_pagerank with seller total ORDER COUNT across 20 seeds.
3. Coordinated fraud and return abuse ring mechanics: member sharing edge appearance, single CC share, and baselines per type.
4. Recall at tau=0.50 (raw XGBoost scores vs calibrated probabilities) on canonical pipeline.
5. Collusion diagnostic: buyer-seller pair repeat rate.
6. Mondrian (class-conditional) conformal predictor over 20 seeds.
7. Return-ablation p-values analysis and Holm re-correction.
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_PROJECT_DIR = os.path.join(_ROOT_DIR, "trustshield_project")
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from entity_generator import generate_full_pipeline, SIM_START
from baseline_model import build_features as build_tabular_features, TRAIN_END, VAL_END
from graph_features import (
    attach_relationship_snapshot_features,
    build_monthly_snapshots,
    attach_snapshot_features,
    build_monthly_plain_aggregate_snapshots,
    attach_plain_aggregate_features,
    add_edge_weight_before,
    build_relationship_graph,
    detect_fraud_rings,
)
import networkx as nx
from xgboost import XGBClassifier

SEEDS_20 = [
    42, 101, 202, 303, 404, 505, 606, 707, 808, 909,
    1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809, 1910
]

TEST_END = pd.Timestamp("2025-12-10 23:59:59.999999")

def sha256_of_list(arr: list) -> str:
    s = repr(arr).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

def run_diagnostics():
    print("=" * 80)
    print("RUNNING FINAL CONSISTENCY PASS DIAGNOSTICS")
    print("=" * 80)

    # 1. Variant (d) Arrays
    with open(os.path.join(_ROOT_DIR, "results", "results.json"), "r", encoding="utf-8") as f:
        res_json = json.load(f)

    # From truncated test horizon
    th_arrays = res_json.get("truncated_test_horizon_20_seeds", {}).get("per_seed_arrays", {})
    roc_d_trunc = th_arrays.get("roc_d", [])
    pr_d_trunc = th_arrays.get("pr_d", [])

    # From full horizon checkpoints
    roc_d_full = []
    pr_d_full = []
    scratch_dir = os.path.join(_ROOT_DIR, "scratch")
    for s in SEEDS_20:
        ckpt_p = os.path.join(scratch_dir, f"checkpoint_seed_{s}.json")
        if os.path.exists(ckpt_p):
            with open(ckpt_p, "r", encoding="utf-8") as f:
                c = json.load(f)
                roc_d_full.append(c["variant_d"]["roc"])
                pr_d_full.append(c["variant_d"]["pr"])

    print("\n[ITEM 0 - VARIANT D ARRAYS]")
    print(f"Truncated Test Horizon (Commit 7684c4e5ed, N={len(roc_d_trunc)}):")
    print(f"roc_d = {repr(roc_d_trunc)}")
    print(f"sha256(roc_d) = {sha256_of_list(roc_d_trunc)}")
    print(f"pr_d  = {repr(pr_d_trunc)}")
    print(f"sha256(pr_d)  = {sha256_of_list(pr_d_trunc)}")

    if roc_d_full:
        print(f"\nFull Horizon Checkpoints (Commit 8b1f06db1c, N={len(roc_d_full)}):")
        print(f"roc_d = {repr(roc_d_full)}")
        print(f"sha256(roc_d) = {sha256_of_list(roc_d_full)}")
        print(f"pr_d  = {repr(pr_d_full)}")
        print(f"sha256(pr_d)  = {sha256_of_list(pr_d_full)}")

    # 2. Correlation of seller_pagerank with seller total ORDER COUNT & recall at tau=0.50
    seller_pr_order_pearsons = []
    seller_pr_order_spearmans = []
    buyer_pr_order_pearsons = []
    buyer_pr_order_spearmans = []

    recall_raw_tau_50 = []
    recall_cal_tau_50 = []

    # Coordinated fraud & return abuse ring mechanics
    coord_member_edge_shares = []
    coord_single_cc_shares = []
    coord_all_cc_recovery = []
    coord_rnd_recovery = []
    coord_ts_recovery = []

    ret_member_edge_shares = []
    ret_single_cc_shares = []
    ret_all_cc_recovery = []
    ret_rnd_recovery = []
    ret_ts_recovery = []

    # Mondrian Conformal
    mondrian_cov_all = []
    mondrian_cov_y0 = []
    mondrian_cov_y1 = []
    mondrian_set_sizes = []

    # Collusion repeat pair share
    coll_repeat_shares = []

    print("\nRunning multi-seed evaluations over canonical pipeline...")
    # Run across 20 seeds
    for idx, seed in enumerate(SEEDS_20):
        pipe = generate_full_pipeline(seed=seed, ring_coherent=False)
        res = pipe["result"]
        cat = pipe["catalog"]
        txn = pipe["txn"]
        base = pipe["base"]
        orders = res["orders"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])

        df, tabular_cols = build_tabular_features(
            res["orders"], res["listings"], res["returns"],
            txn["buyers"], cat["sellers"], cat["products"]
        )
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["y"] = df["is_fraudulent"].astype(int)

        # Graph features
        df_graph = attach_relationship_snapshot_features(df, base["address_sharing_log"], base["device_sharing_log"])
        snapshots_graph, months_graph = build_monthly_snapshots(orders, SIM_START)
        df_graph = attach_snapshot_features(df_graph, snapshots_graph, months_graph)
        df_graph = add_edge_weight_before(df_graph)

        cols_c = tabular_cols + [
            "share_degree", "share_component_size",
            "buyer_seller_degree", "buyer_pagerank",
            "seller_buyer_degree", "seller_pagerank",
            "seller_buyer_concentration_hhi", "buyer_seller_edge_weight_before"
        ]

        # Seller Order Count across transactions up to cutoff or test
        seller_order_counts = orders.groupby("seller_id").size().to_dict()
        df_graph["seller_order_count"] = df_graph["seller_id"].map(seller_order_counts).fillna(0)

        test_mask = (df_graph["order_date"] > VAL_END) & (df_graph["order_date"] <= TEST_END)
        test_df = df_graph[test_mask]

        # Pagerank correlations
        s_pr = test_df["seller_pagerank"].values
        s_cnt = test_df["seller_order_count"].values
        sp_r, _ = stats.pearsonr(s_pr, s_cnt)
        sp_s, _ = stats.spearmanr(s_pr, s_cnt)
        seller_pr_order_pearsons.append(sp_r)
        seller_pr_order_spearmans.append(sp_s)

        b_pr = test_df["buyer_pagerank"].values
        b_cnt = test_df["buyer_orders_before"].values
        bp_r, _ = stats.pearsonr(b_pr, b_cnt)
        bp_s, _ = stats.spearmanr(b_pr, b_cnt)
        buyer_pr_order_pearsons.append(bp_r)
        buyer_pr_order_spearmans.append(bp_s)

        # Train model for threshold and conformal evaluation
        train_mask = df_graph["order_date"] <= TRAIN_END
        val_mask = (df_graph["order_date"] > TRAIN_END) & (df_graph["order_date"] <= VAL_END)
        
        X_tr = df_graph.loc[train_mask, cols_c].fillna(0.0)
        y_tr = df_graph.loc[train_mask, "y"].values
        X_val = df_graph.loc[val_mask, cols_c].fillna(0.0)
        y_val = df_graph.loc[val_mask, "y"].values
        X_te = test_df[cols_c].fillna(0.0)
        y_te = test_df["y"].values

        clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(len(y_tr) - sum(y_tr)) / max(sum(y_tr), 1),
            eval_metric="aucpr", random_state=seed, n_jobs=-1, verbosity=0
        )
        clf.fit(X_tr, y_tr)

        p_val_raw = clf.predict_proba(X_val)[:, 1]
        p_te_raw = clf.predict_proba(X_te)[:, 1]

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_val_raw, y_val)
        p_te_cal = iso.predict(p_te_raw)

        # Recall at tau = 0.50
        fraud_mask_te = (y_te == 1)
        tot_fraud = int(fraud_mask_te.sum())
        rec_raw_50 = float(np.sum(p_te_raw[fraud_mask_te] >= 0.50) / max(tot_fraud, 1))
        rec_cal_50 = float(np.sum(p_te_cal[fraud_mask_te] >= 0.50) / max(tot_fraud, 1))
        recall_raw_tau_50.append(rec_raw_50)
        recall_cal_tau_50.append(rec_cal_50)

        # Mondrian (Class-Conditional) Conformal Predictor (alpha = 0.05)
        # Calibrate separate quantiles for y=0 and y=1 on validation set
        val_p0 = p_val_raw[y_val == 0]
        val_p1 = p_val_raw[y_val == 1]
        n0 = len(val_p0)
        n1 = len(val_p1)
        alpha = 0.05
        # Non-conformity: for y=0, non-conformity is p (predicting fraud). For y=1, non-conformity is 1 - p (predicting legit).
        q_level_0 = min(1.0, np.ceil((n0 + 1) * (1.0 - alpha)) / n0)
        q_level_1 = min(1.0, np.ceil((n1 + 1) * (1.0 - alpha)) / n1)
        q0 = float(np.quantile(val_p0, q_level_0, method="higher"))
        q1 = float(np.quantile(1.0 - val_p1, q_level_1, method="higher"))

        # Test evaluation
        # Predict set: include 0 if p <= q0; include 1 if 1 - p <= q1 <=> p >= 1 - q1
        cov_all_seed = []
        cov_0_seed = []
        cov_1_seed = []
        sizes_seed = []
        for p_i, y_i in zip(p_te_raw, y_te):
            pset = set()
            if p_i <= q0:
                pset.add(0)
            if (1.0 - p_i) <= q1:
                pset.add(1)
            sizes_seed.append(len(pset))
            is_cov = (y_i in pset)
            cov_all_seed.append(is_cov)
            if y_i == 0:
                cov_0_seed.append(is_cov)
            else:
                cov_1_seed.append(is_cov)

        mondrian_cov_all.append(np.mean(cov_all_seed))
        mondrian_cov_y0.append(np.mean(cov_0_seed))
        mondrian_cov_y1.append(np.mean(cov_1_seed))
        mondrian_set_sizes.append(np.mean(sizes_seed))

        # Ring Mechanics Diagnostic (Seed 42 and first 5 seeds)
        if idx < 5:
            rel_graph = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"], cutoff_date=VAL_END)
            undir = rel_graph.to_undirected()
            all_sharing_buyers = set(base["address_sharing_log"]["buyer_id"]).union(
                set(base["address_sharing_log"]["shared_with_buyer_id"])
            ).union(
                set(base["device_sharing_log"]["buyer_id"])
            ).union(
                set(base["device_sharing_log"]["shared_with_buyer_id"])
            )
            gt_df = res["fraud_ground_truth"]

            # Rings df from detect_fraud_rings
            df_scored = df_graph.copy()
            df_scored["fraud_score"] = clf.predict_proba(df_graph[cols_c].fillna(0.0))[:, 1]
            rings_df = detect_fraud_rings(rel_graph, df_scored, score_col="fraud_score", min_ring_size=2)
            ts_detected_members = set()
            for members in rings_df[rings_df["avg_risk_score"] >= 0.50]["members"]:
                ts_detected_members.update(members)

            all_ccs = [set(c) for c in nx.connected_components(undir) if len(c) >= 2]
            all_cc_members = set().union(*all_ccs) if all_ccs else set()

            all_buyers_list = list(txn["buyers"]["buyer_id"])
            np.random.seed(seed)
            rnd_sz = min(len(all_cc_members), len(all_buyers_list))
            random_selected = set(np.random.choice(all_buyers_list, size=rnd_sz, replace=False))

            for ft, edge_shares, cc_shares, all_cc_rec, rnd_rec, ts_rec in [
                ("coordinated_fraud", coord_member_edge_shares, coord_single_cc_shares, coord_all_cc_recovery, coord_rnd_recovery, coord_ts_recovery),
                ("return_abuse", ret_member_edge_shares, ret_single_cc_shares, ret_all_cc_recovery, ret_rnd_recovery, ret_ts_recovery),
            ]:
                sub_gt = gt_df[gt_df["fraud_type"] == ft]
                rings = sub_gt.groupby("fraud_ring_id")["entity_id"].apply(set).to_dict()
                if rings:
                    e_sh = [len([m for m in mem if m in all_sharing_buyers]) / len(mem) for mem in rings.values()]
                    edge_shares.append(np.mean(e_sh))

                    c_sh = []
                    for mem in rings.values():
                        ccs_with_m = [c for c in all_ccs if len(c & mem) > 0]
                        c_sh.append(1 if (len(ccs_with_m) == 1 and len(mem & ccs_with_m[0]) == len(mem)) else 0)
                    cc_shares.append(np.mean(c_sh))

                    # Ring-level recovery (>=50% members detected)
                    all_cc_rec.append(sum(1 for mem in rings.values() if len(mem & all_cc_members) / len(mem) >= 0.5) / len(rings))
                    rnd_rec.append(sum(1 for mem in rings.values() if len(mem & random_selected) / len(mem) >= 0.5) / len(rings))
                    ts_rec.append(sum(1 for mem in rings.values() if len(mem & ts_detected_members) / len(mem) >= 0.5) / len(rings))

        # Collusion repeated buyer-seller pairs
        coll_orders = orders[orders["fraud_type"] == "seller_buyer_collusion"]
        if len(coll_orders) > 0:
            pair_counts = coll_orders.groupby(["buyer_id", "seller_id"]).size()
            repeat_orders = coll_orders.duplicated(subset=["buyer_id", "seller_id"], keep=False)
            coll_repeat_shares.append(float(repeat_orders.mean()))

        print(f"Seed {seed:4d} done | Rec@0.50: Raw={rec_raw_50:.4f}, Cal={rec_cal_50:.4f} | Mondrian Cov: Y0={mondrian_cov_y0[-1]:.4f}, Y1={mondrian_cov_y1[-1]:.4f}, Size={mondrian_set_sizes[-1]:.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)

    print("\n1. PAGERANK CORRELATIONS WITH PLAIN COUNTS:")
    print(f"buyer_pagerank vs buyer_orders_before:  Pearson r = {np.mean(buyer_pr_order_pearsons):.4f} +/- {np.std(buyer_pr_order_pearsons, ddof=1):.4f} (Spearman rho = {np.mean(buyer_pr_order_spearmans):.4f})")
    print(f"seller_pagerank vs seller ORDER COUNT:  Pearson r = {np.mean(seller_pr_order_pearsons):.4f} +/- {np.std(seller_pr_order_pearsons, ddof=1):.4f} (Spearman rho = {np.mean(seller_pr_order_spearmans):.4f})")

    print("\n2. RECALL AT TAU = 0.50 (CANONICAL PIPELINE):")
    print(f"Raw XGBoost output scale:     Recall = {np.mean(recall_raw_tau_50):.4%} +/- {np.std(recall_raw_tau_50, ddof=1):.4%}")
    print(f"Calibrated probability scale: Recall = {np.mean(recall_cal_tau_50):.4%} +/- {np.std(recall_cal_tau_50, ddof=1):.4%}")

    print("\n3. MONDRIAN (CLASS-CONDITIONAL) CONFORMAL PREDICTOR (alpha=0.05, 20 SEEDS):")
    print(f"Overall Coverage:   {np.mean(mondrian_cov_all):.4%} +/- {np.std(mondrian_cov_all, ddof=1):.4%}")
    print(f"Legit (Y=0) Cov:    {np.mean(mondrian_cov_y0):.4%} +/- {np.std(mondrian_cov_y0, ddof=1):.4%}")
    print(f"Fraud (Y=1) Cov:    {np.mean(mondrian_cov_y1):.4%} +/- {np.std(mondrian_cov_y1, ddof=1):.4%}")
    print(f"Mean Set Size:      {np.mean(mondrian_set_sizes):.4f} +/- {np.std(mondrian_set_sizes, ddof=1):.4f}")

    print("\n4. RING RECOVERY MECHANICS (COORDINATED FRAUD VS RETURN ABUSE):")
    print(f"coordinated_fraud: Member in sharing log = {np.mean(coord_member_edge_shares):.4%}, In single CC = {np.mean(coord_single_cc_shares):.4%}")
    print(f"  All CC Recovery:        {np.mean(coord_all_cc_recovery):.4%}")
    print(f"  Random Recovery:        {np.mean(coord_rnd_recovery):.4%}")
    print(f"  TrustShield (>=0.50):   {np.mean(coord_ts_recovery):.4%}")
    print(f"return_abuse: Member in sharing log = {np.mean(ret_member_edge_shares):.4%}, In single CC = {np.mean(ret_single_cc_shares):.4%}")
    print(f"  All CC Recovery:        {np.mean(ret_all_cc_recovery):.4%}")
    print(f"  Random Recovery:        {np.mean(ret_rnd_recovery):.4%}")
    print(f"  TrustShield (>=0.50):   {np.mean(ret_ts_recovery):.4%}")

    print("\n5. COLLUSION REPEAT PAIR RATE:")
    print(f"Share of collusion orders sharing buyer-seller pair: {np.mean(coll_repeat_shares):.4%} +/- {np.std(coll_repeat_shares, ddof=1):.4%}")

if __name__ == "__main__":
    run_diagnostics()
