"""TrustShield Remediation Addendum Comprehensive Benchmark & Audit Harness.

Executes and verifies:
1. HORIZON FIX: Exclude final 21 days from the test split for ALL models (order_date <= 2025-12-10).
2. 20-SEED BENCHMARK on truncated test split: (a), (b), (c), (d), with paired lifts and t-tests.
3. PRACTICAL SIGNIFICANCE: 2%, 5%, 10% review budget: Fraud value caught (INR) & Recall for (b) vs (c).
4. ECE & BRIER SCORE: Fit isotonic on validation per seed; report test ECE and Brier before & after calibration for (b) and (c).
5. THRESHOLD POLICY: Choose tau per seed on validation only for 5% budget; report realized test volume, precision, recall, value caught.
6. RING DETECTION: Ring-level vs member-level evaluation against (1) all connected components >= 2 (no filter), (2) random cluster baseline. Breakdown by fraud type.
7. CONFORMAL PREDICTOR: Fit on validation; evaluate empirical test coverage and mean set size per seed and per class.
8. COLLUSION BURST DIAGNOSTIC: Share of collusion orders with edge_weight_before > 0, and share sharing buyer-seller pair with earlier burst order.
"""

from __future__ import annotations
import os
import sys
import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
import networkx as nx

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_PROJECT_DIR = os.path.join(_ROOT_DIR, "trustshield_project")
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from entity_generator import generate_full_pipeline, SIM_START, SIM_END
from baseline_model import build_features as build_tabular_features, TRAIN_END, VAL_END
from graph_features import (
    attach_relationship_snapshot_features,
    build_monthly_snapshots,
    attach_snapshot_features,
    add_edge_weight_before,
    detect_fraud_rings,
    build_relationship_graph,
)
from advanced_trust_engine import ConformalPredictor

if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")

SEEDS_20 = [
    42, 101, 202, 303, 404, 505, 606, 707, 808, 909,
    1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809, 1910
]

ALL_GRAPH_COLS = [
    "share_degree", "share_component_size",
    "buyer_seller_degree", "buyer_pagerank",
    "seller_buyer_degree", "seller_pagerank",
    "seller_buyer_concentration_hhi",
    "buyer_seller_edge_weight_before",
]

CONTROL_D_COLS = [
    "buyer_prev_month_order_count",
    "buyer_distinct_sellers_prior",
    "seller_prev_month_order_count",
    "seller_distinct_buyers_prior",
    "seller_top_buyer_share_prior",
    "buyer_seller_edge_weight_before",
]

FRAUD_TYPES = [
    "fake_listing",
    "return_abuse",
    "coordinated_fraud",
    "seller_buyer_collusion",
]

# Exclude final 21 days from simulation end
TEST_END = pd.to_datetime(SIM_END) - pd.Timedelta(days=21)


def pr_auc_score(y_true, y_score):
    p, r, _ = precision_recall_curve(y_true, y_score)
    return float(auc(r, p))


def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        b_low, b_high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= b_low) & (y_prob < b_high if i < n_bins - 1 else y_prob <= b_high)
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def attach_control_features(df_input: pd.DataFrame, orders_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    ord_df = orders_input.copy()
    ord_df["order_date"] = pd.to_datetime(ord_df["order_date"])
    ord_df["year_month"] = ord_df["order_date"].dt.to_period("M")

    b_m_counts = ord_df.groupby(["buyer_id", "year_month"]).size().unstack(fill_value=0)
    s_m_counts = ord_df.groupby(["seller_id", "year_month"]).size().unstack(fill_value=0)

    order_dates = pd.to_datetime(df["order_date"])
    order_ym = order_dates.dt.to_period("M")
    prev_ym = order_ym - 1

    b_prev = [b_m_counts.loc[b, ym] if (b in b_m_counts.index and ym in b_m_counts.columns) else 0
              for b, ym in zip(df["buyer_id"], prev_ym)]
    s_prev = [s_m_counts.loc[s, ym] if (s in s_m_counts.index and ym in s_m_counts.columns) else 0
              for s, ym in zip(df["seller_id"], prev_ym)]
    df["buyer_prev_month_order_count"] = b_prev
    df["seller_prev_month_order_count"] = s_prev

    ord_sorted = ord_df.sort_values("order_date").reset_index(drop=True)
    b_sellers_map = {}
    s_buyers_map = {}
    b_distinct = []
    s_distinct = []
    s_top_share = []

    ord_records = ord_sorted[["buyer_id", "seller_id", "order_date"]].to_dict("records")
    ptr = 0
    n_ord = len(ord_records)

    eval_records = df[["buyer_id", "seller_id", "order_date"]].reset_index().to_dict("records")
    eval_sorted = sorted(eval_records, key=lambda x: x["order_date"])
    res_b_dist = {}
    res_s_dist = {}
    res_s_top = {}

    for ev in eval_sorted:
        ev_dt = ev["order_date"]
        while ptr < n_ord and ord_records[ptr]["order_date"] < ev_dt:
            rec = ord_records[ptr]
            b_id, s_id = rec["buyer_id"], rec["seller_id"]
            if b_id not in b_sellers_map:
                b_sellers_map[b_id] = set()
            b_sellers_map[b_id].add(s_id)
            if s_id not in s_buyers_map:
                s_buyers_map[s_id] = {}
            s_buyers_map[s_id][b_id] = s_buyers_map[s_id].get(b_id, 0) + 1
            ptr += 1

        idx = ev["index"]
        b_curr, s_curr = ev["buyer_id"], ev["seller_id"]
        res_b_dist[idx] = len(b_sellers_map.get(b_curr, set()))
        s_dict = s_buyers_map.get(s_curr, {})
        res_s_dist[idx] = len(s_dict)
        if s_dict:
            tot = sum(s_dict.values())
            res_s_top[idx] = max(s_dict.values()) / max(tot, 1)
        else:
            res_s_top[idx] = 0.0

    df["buyer_distinct_sellers_prior"] = [res_b_dist[i] for i in df.index]
    df["seller_distinct_buyers_prior"] = [res_s_dist[i] for i in df.index]
    df["seller_top_buyer_share_prior"] = [res_s_top[i] for i in df.index]
    return df


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


def paired_summary(a_arr, b_arr):
    diff = np.asarray(a_arr) - np.asarray(b_arr)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    n = len(diff)
    se = std_diff / np.sqrt(n)
    ci = [float(mean_diff - 1.96 * se), float(mean_diff + 1.96 * se)]
    t_stat, p_val = stats.ttest_rel(a_arr, b_arr)
    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_95": ci,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
    }


def main():
    print("=" * 80)
    print("TRUSTSHIELD REMEDIATION ADDENDUM AUDIT (20 SEEDS)")
    print(f"Test Split Horizon: {VAL_END.strftime('%Y-%m-%d')} to {TEST_END.strftime('%Y-%m-%d')} (final 21 days excluded)")
    print("=" * 80)

    # Tracking arrays across 20 seeds
    roc_a, roc_b, roc_c, roc_d = [], [], [], []
    pr_a, pr_b, pr_c, pr_d = [], [], [], []

    # Review budgets (2%, 5%, 10%)
    budgets = [0.02, 0.05, 0.10]
    val_caught_b = {b: [] for b in budgets}
    val_caught_c = {b: [] for b in budgets}
    rec_b = {b: [] for b in budgets}
    rec_c = {b: [] for b in budgets}

    # Per-type metrics
    per_type_roc_b = {ft: [] for ft in FRAUD_TYPES}
    per_type_roc_c = {ft: [] for ft in FRAUD_TYPES}
    per_type_rec_b = {ft: {b: [] for b in budgets} for ft in FRAUD_TYPES}
    per_type_rec_c = {ft: {b: [] for b in budgets} for ft in FRAUD_TYPES}

    # Calibration & Brier
    ece_b_raw, ece_c_raw = [], []
    ece_b_cal, ece_c_cal = [], []
    brier_b_raw, brier_c_raw = [], []
    brier_b_cal, brier_c_cal = [], []

    # Threshold policy on val for 5% budget
    tau_b_list, tau_c_list = [], []
    realized_vol_b, realized_vol_c = [], []
    realized_prec_b, realized_prec_c = [], []
    realized_rec_b, realized_rec_c = [], []
    realized_val_b, realized_val_c = [], []

    # Ring detection metrics
    # Baselines: (1) CC >= 2 no filter, (2) random cluster baseline, (3) Risk >= 0.50
    ring_res = {"member_level": {"all_cc": [], "random": [], "trustshield": []},
                "ring_level": {"all_cc": [], "random": [], "trustshield": []},
                "by_type": {ft: [] for ft in FRAUD_TYPES}}

    # Conformal metrics
    conf_coverage_all, conf_coverage_y0, conf_coverage_y1 = [], [], []
    conf_set_sizes = []

    # Collusion diagnostic metrics
    coll_edge_pos_ratio = []
    coll_share_prior_burst_ratio = []

    os.makedirs(os.path.join(_ROOT_DIR, "scratch"), exist_ok=True)

    for idx, seed in enumerate(SEEDS_20, 1):
        ckpt_path = os.path.join(_ROOT_DIR, "scratch", f"addendum_ckpt_seed_{seed}.json")
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            roc_a.append(c_data["roc_a"])
            roc_b.append(c_data["roc_b"])
            roc_c.append(c_data["roc_c"])
            roc_d.append(c_data["roc_d"])
            pr_a.append(c_data["pr_a"])
            pr_b.append(c_data["pr_b"])
            pr_c.append(c_data["pr_c"])
            pr_d.append(c_data["pr_d"])
            for b in budgets:
                val_caught_b[b].append(c_data["val_b"][str(b)])
                val_caught_c[b].append(c_data["val_c"][str(b)])
                rec_b[b].append(c_data["rec_b"][str(b)])
                rec_c[b].append(c_data["rec_c"][str(b)])
                for ft in FRAUD_TYPES:
                    per_type_rec_b[ft][b].append(c_data["pt_rec_b"][ft][str(b)])
                    per_type_rec_c[ft][b].append(c_data["pt_rec_c"][ft][str(b)])
            for ft in FRAUD_TYPES:
                per_type_roc_b[ft].append(c_data["pt_roc_b"][ft])
                per_type_roc_c[ft].append(c_data["pt_roc_c"][ft])
            ece_b_raw.append(c_data["ece_b_raw"])
            ece_c_raw.append(c_data["ece_c_raw"])
            ece_b_cal.append(c_data["ece_b_cal"])
            ece_c_cal.append(c_data["ece_c_cal"])
            brier_b_raw.append(c_data["brier_b_raw"])
            brier_c_raw.append(c_data["brier_c_raw"])
            brier_b_cal.append(c_data["brier_b_cal"])
            brier_c_cal.append(c_data["brier_c_cal"])
            tau_b_list.append(c_data["tau_b"])
            tau_c_list.append(c_data["tau_c"])
            realized_vol_b.append(c_data["vol_b"])
            realized_vol_c.append(c_data["vol_c"])
            realized_prec_b.append(c_data["prec_b"])
            realized_prec_c.append(c_data["prec_c"])
            realized_rec_b.append(c_data["rec_tp_b"])
            realized_rec_c.append(c_data["rec_tp_c"])
            realized_val_b.append(c_data["val_tp_b"])
            realized_val_c.append(c_data["val_tp_c"])
            conf_coverage_all.append(c_data["conf_all"])
            conf_coverage_y0.append(c_data["conf_y0"])
            conf_coverage_y1.append(c_data["conf_y1"])
            conf_set_sizes.append(c_data["conf_size"])
            if "ring" in c_data:
                ring_res["member_level"]["all_cc"].append(c_data["ring"]["mem_all"])
                ring_res["member_level"]["random"].append(c_data["ring"]["mem_rnd"])
                ring_res["member_level"]["trustshield"].append(c_data["ring"]["mem_ts"])
                ring_res["ring_level"]["all_cc"].append(c_data["ring"]["r_all"])
                ring_res["ring_level"]["random"].append(c_data["ring"]["r_rnd"])
                ring_res["ring_level"]["trustshield"].append(c_data["ring"]["r_ts"])
                for ft in FRAUD_TYPES:
                    ring_res["by_type"][ft].append(c_data["ring"]["by_type"][ft])
            coll_edge_pos_ratio.append(c_data["coll_edge"])
            coll_share_prior_burst_ratio.append(c_data["coll_burst"])
            print(f"[{idx:02d}/20] Seed {seed} (cached) | Truncated Test ROC (b): {roc_b[-1]:.4f}, (c): {roc_c[-1]:.4f}")
            continue

        t0 = time.perf_counter()
        print(f"[{idx:02d}/20] Seed {seed}...", end="", flush=True)

        pipe = generate_full_pipeline(seed=seed, ring_coherent=False)
        res = pipe["result"]
        cat = pipe["catalog"]
        txn = pipe["txn"]
        base = pipe["base"]

        orders = res["orders"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        orders["amount"] = orders["amount"].fillna(100.0)

        df, tabular_cols = build_tabular_features(
            res["orders"], res["listings"], res["returns"],
            txn["buyers"], cat["sellers"], cat["products"]
        )
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["y"] = df["is_fraudulent"].astype(int)
        df["amount"] = orders["amount"]

        # Variant a columns: tabular without device_shared_buyer_count
        cols_a = [c for c in tabular_cols if c != "device_shared_buyer_count"]
        cols_b = list(tabular_cols)

        # Graph features (Variant c)
        df = attach_relationship_snapshot_features(df, base["address_sharing_log"], base["device_sharing_log"])
        snapshots, months = build_monthly_snapshots(orders, SIM_START)
        df = attach_snapshot_features(df, snapshots, months)
        df = add_edge_weight_before(df)
        cols_c = cols_b + ALL_GRAPH_COLS

        # Control features (Variant d)
        df = attach_control_features(df, orders)
        cols_d = cols_b + CONTROL_D_COLS

        # Splits
        train_mask = df["order_date"] <= TRAIN_END
        val_mask = (df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)
        # TRUNCATED TEST SPLIT: exclude final 21 days
        test_mask = (df["order_date"] > VAL_END) & (df["order_date"] <= TEST_END)

        train = df[train_mask]
        val = df[val_mask]
        test = df[test_mask]

        y_tr = train["y"]
        y_val = val["y"].to_numpy()
        y_test = test["y"].to_numpy()
        amounts_test = test["amount"].to_numpy()
        fraud_types_test = test["fraud_type"].fillna("legitimate").to_numpy()

        # Train models
        m_a = train_xgb(train[cols_a], y_tr, random_state=seed)
        m_b = train_xgb(train[cols_b], y_tr, random_state=seed)
        m_c = train_xgb(train[cols_c], y_tr, random_state=seed)
        m_d = train_xgb(train[cols_d], y_tr, random_state=seed)

        # Score val & test
        p_val_b = m_b.predict_proba(val[cols_b].fillna(0.0))[:, 1]
        p_val_c = m_c.predict_proba(val[cols_c].fillna(0.0))[:, 1]

        p_test_a = m_a.predict_proba(test[cols_a].fillna(0.0))[:, 1]
        p_test_b = m_b.predict_proba(test[cols_b].fillna(0.0))[:, 1]
        p_test_c = m_c.predict_proba(test[cols_c].fillna(0.0))[:, 1]
        p_test_d = m_d.predict_proba(test[cols_d].fillna(0.0))[:, 1]

        # 1. Headline metrics on truncated test split
        roc_a.append(roc_auc_score(y_test, p_test_a))
        roc_b.append(roc_auc_score(y_test, p_test_b))
        roc_c.append(roc_auc_score(y_test, p_test_c))
        roc_d.append(roc_auc_score(y_test, p_test_d))

        pr_a.append(pr_auc_score(y_test, p_test_a))
        pr_b.append(pr_auc_score(y_test, p_test_b))
        pr_c.append(pr_auc_score(y_test, p_test_c))
        pr_d.append(pr_auc_score(y_test, p_test_d))

        # 2. Practical significance at budgets
        n_test = len(y_test)
        tot_fraud_idx = np.where(y_test == 1)[0]
        tot_fraud_cnt = len(tot_fraud_idx)
        idx_b_sort = np.argsort(p_test_b)[::-1]
        idx_c_sort = np.argsort(p_test_c)[::-1]

        for b in budgets:
            k = max(1, int(n_test * b))
            top_b = set(idx_b_sort[:k])
            top_c = set(idx_c_sort[:k])

            caught_b = [i for i in top_b if y_test[i] == 1]
            caught_c = [i for i in top_c if y_test[i] == 1]

            val_caught_b[b].append(float(np.sum(amounts_test[caught_b])))
            val_caught_c[b].append(float(np.sum(amounts_test[caught_c])))
            rec_b[b].append(float(len(caught_b) / max(tot_fraud_cnt, 1)))
            rec_c[b].append(float(len(caught_c) / max(tot_fraud_cnt, 1)))

            # Per-type recall@K
            for ft in FRAUD_TYPES:
                ft_idx = np.where((y_test == 1) & (fraud_types_test == ft))[0]
                if len(ft_idx) > 0:
                    per_type_rec_b[ft][b].append(len(top_b.intersection(ft_idx)) / len(ft_idx))
                    per_type_rec_c[ft][b].append(len(top_c.intersection(ft_idx)) / len(ft_idx))
                else:
                    per_type_rec_b[ft][b].append(0.0)
                    per_type_rec_c[ft][b].append(0.0)

        # Per-type isolated ROC
        legit_mask = (y_test == 0)
        for ft in FRAUD_TYPES:
            ft_m = (y_test == 1) & (fraud_types_test == ft)
            sub_m = legit_mask | ft_m
            if ft_m.sum() > 0:
                per_type_roc_b[ft].append(float(roc_auc_score(y_test[sub_m], p_test_b[sub_m])))
                per_type_roc_c[ft].append(float(roc_auc_score(y_test[sub_m], p_test_c[sub_m])))
            else:
                per_type_roc_b[ft].append(0.5)
                per_type_roc_c[ft].append(0.5)

        # 3. Calibration ECE & Brier before and after Isotonic
        iso_b = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_val_b, y_val)
        iso_c = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_val_c, y_val)

        p_test_b_cal = iso_b.predict(p_test_b)
        p_test_c_cal = iso_c.predict(p_test_c)

        ece_b_raw.append(compute_ece(y_test, p_test_b))
        ece_c_raw.append(compute_ece(y_test, p_test_c))
        ece_b_cal.append(compute_ece(y_test, p_test_b_cal))
        ece_c_cal.append(compute_ece(y_test, p_test_c_cal))

        brier_b_raw.append(float(brier_score_loss(y_test, p_test_b)))
        brier_c_raw.append(float(brier_score_loss(y_test, p_test_c)))
        brier_b_cal.append(float(brier_score_loss(y_test, p_test_b_cal)))
        brier_c_cal.append(float(brier_score_loss(y_test, p_test_c_cal)))

        # 4. Threshold policy on validation only (5% review budget)
        # Find highest tau on val such that fraction flagged <= 0.05
        q95_b = np.quantile(p_val_b, 0.95)
        q95_c = np.quantile(p_val_c, 0.95)
        tau_b_list.append(float(q95_b))
        tau_c_list.append(float(q95_c))

        flag_b = (p_test_b >= q95_b)
        flag_c = (p_test_c >= q95_c)

        realized_vol_b.append(float(np.mean(flag_b)))
        realized_vol_c.append(float(np.mean(flag_c)))

        tp_b = int(np.sum((y_test == 1) & flag_b))
        tp_c = int(np.sum((y_test == 1) & flag_c))
        realized_prec_b.append(float(tp_b / max(np.sum(flag_b), 1)))
        realized_prec_c.append(float(tp_c / max(np.sum(flag_c), 1)))
        realized_rec_b.append(float(tp_b / max(tot_fraud_cnt, 1)))
        realized_rec_c.append(float(tp_c / max(tot_fraud_cnt, 1)))

        realized_val_b.append(float(np.sum(amounts_test[(y_test == 1) & flag_b])))
        realized_val_c.append(float(np.sum(amounts_test[(y_test == 1) & flag_c])))

        # 5. Conformal evaluation (alpha = 0.05)
        conf = ConformalPredictor(alpha=0.05).calibrate(p_val_c, y_val)
        covered = []
        covered_0 = []
        covered_1 = []
        set_sizes = []
        for p_i, y_i in zip(p_test_c, y_test):
            pred_s = conf.predict_set(p_i)
            set_sizes.append(len(pred_s))
            is_cov = (y_i in pred_s)
            covered.append(is_cov)
            if y_i == 0:
                covered_0.append(is_cov)
            else:
                covered_1.append(is_cov)

        conf_coverage_all.append(float(np.mean(covered)))
        conf_coverage_y0.append(float(np.mean(covered_0)))
        conf_coverage_y1.append(float(np.mean(covered_1)))
        conf_set_sizes.append(float(np.mean(set_sizes)))

        # 6. Ring detection evaluation
        if idx <= 5:  # First 5 seeds
            rel_graph = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"], cutoff_date=VAL_END)
            undirected = rel_graph.to_undirected()
            all_ccs = [set(c) for c in nx.connected_components(undirected) if len(c) >= 2]

            gt_df = res["fraud_ground_truth"]
            gt_buyers_df = gt_df[gt_df["entity_type"] == "buyer"] if "entity_type" in gt_df.columns else gt_df
            gt_id_col = "entity_id" if "entity_id" in gt_buyers_df.columns else "buyer_id"
            gt_buyers = set(gt_buyers_df[gt_id_col].unique())
            gt_rings = {}
            if "fraud_ring_id" in gt_buyers_df.columns:
                for rid, grp in gt_buyers_df.groupby("fraud_ring_id"):
                    gt_rings[rid] = set(grp[gt_id_col].unique())

            # Baseline 1: All CCs >= 2
            detected_all_cc_members = set().union(*all_ccs) if all_ccs else set()
            prec_all = len(detected_all_cc_members & gt_buyers) / max(len(detected_all_cc_members), 1)
            rec_all = len(detected_all_cc_members & gt_buyers) / max(len(gt_buyers), 1)
            ring_res["member_level"]["all_cc"].append({"p": prec_all, "r": rec_all, "f1": 2*prec_all*rec_all/max(prec_all+rec_all, 1e-6)})

            # Baseline 2: Random cluster baseline
            all_buyers_list = list(txn["buyers"]["buyer_id"])
            np.random.seed(seed)
            rnd_sz = min(len(detected_all_cc_members), len(all_buyers_list))
            random_selected = set(np.random.choice(all_buyers_list, size=rnd_sz, replace=False))
            prec_rnd = len(random_selected & gt_buyers) / max(len(random_selected), 1)
            rec_rnd = len(random_selected & gt_buyers) / max(len(gt_buyers), 1)
            ring_res["member_level"]["random"].append({"p": prec_rnd, "r": rec_rnd, "f1": 2*prec_rnd*rec_rnd/max(prec_rnd+rec_rnd, 1e-6)})

            # TrustShield: CCs >= 2 with risk >= 0.50
            df_scored = df.copy()
            df_scored["fraud_score"] = m_c.predict_proba(df[cols_c].fillna(0.0))[:, 1]
            rings_df = detect_fraud_rings(rel_graph, df_scored, score_col="fraud_score", min_ring_size=2)
            ts_detected_members = set()
            for members in rings_df[rings_df["avg_risk_score"] >= 0.5]["members"]:
                ts_detected_members.update(members)
            prec_ts = len(ts_detected_members & gt_buyers) / max(len(ts_detected_members), 1)
            rec_ts = len(ts_detected_members & gt_buyers) / max(len(gt_buyers), 1)
            ring_res["member_level"]["trustshield"].append({"p": prec_ts, "r": rec_ts, "f1": 2*prec_ts*rec_ts/max(prec_ts+rec_ts, 1e-6)})

            # Ring-level recovery: A ground-truth ring is recovered if >= 50% members detected
            if gt_rings:
                rec_r_all = sum(1 for r_nodes in gt_rings.values() if len(r_nodes & detected_all_cc_members) / len(r_nodes) >= 0.5) / len(gt_rings)
                rec_r_rnd = sum(1 for r_nodes in gt_rings.values() if len(r_nodes & random_selected) / len(r_nodes) >= 0.5) / len(gt_rings)
                rec_r_ts = sum(1 for r_nodes in gt_rings.values() if len(r_nodes & ts_detected_members) / len(r_nodes) >= 0.5) / len(gt_rings)
                ring_res["ring_level"]["all_cc"].append(rec_r_all)
                ring_res["ring_level"]["random"].append(rec_r_rnd)
                ring_res["ring_level"]["trustshield"].append(rec_r_ts)

                # Breakdown by fraud type
                for ft in FRAUD_TYPES:
                    ft_m = (gt_df["fraud_type"] == ft) & (gt_df["entity_type"] == "buyer") if "entity_type" in gt_df.columns else (gt_df["fraud_type"] == ft)
                    ft_rings = gt_df[ft_m]["fraud_ring_id"].unique()
                    if len(ft_rings) > 0:
                        rec_ft = sum(1 for rid in ft_rings if rid in gt_rings and len(gt_rings[rid] & ts_detected_members)/len(gt_rings[rid]) >= 0.5) / len(ft_rings)
                        ring_res["by_type"][ft].append(rec_ft)
                    else:
                        ring_res["by_type"][ft].append(0.0)
            else:
                ring_res["ring_level"]["all_cc"].append(0.0)
                ring_res["ring_level"]["random"].append(0.0)
                ring_res["ring_level"]["trustshield"].append(0.0)
                for ft in FRAUD_TYPES:
                    ring_res["by_type"][ft].append(0.0)

        # 7. Collusion Burst Diagnostic
        coll_orders = orders[orders["fraud_type"] == "seller_buyer_collusion"]
        if len(coll_orders) > 0:
            df_coll = df.loc[coll_orders.index]
            pos_ratio = float(np.mean(df_coll["buyer_seller_edge_weight_before"] > 0))
            coll_edge_pos_ratio.append(pos_ratio)

            # Check if order shares buyer-seller pair with an earlier burst order
            coll_sorted = coll_orders.sort_values("order_date")
            seen_pairs = set()
            share_earlier = []
            for _, r in coll_sorted.iterrows():
                pair = (r["buyer_id"], r["seller_id"])
                if pair in seen_pairs:
                    share_earlier.append(1)
                else:
                    share_earlier.append(0)
                seen_pairs.add(pair)
            coll_share_prior_burst_ratio.append(float(np.mean(share_earlier)))
        else:
            coll_edge_pos_ratio.append(0.0)
            coll_share_prior_burst_ratio.append(0.0)

        # Save checkpoint
        ckpt_payload = {
            "seed": seed,
            "roc_a": roc_a[-1], "roc_b": roc_b[-1], "roc_c": roc_c[-1], "roc_d": roc_d[-1],
            "pr_a": pr_a[-1], "pr_b": pr_b[-1], "pr_c": pr_c[-1], "pr_d": pr_d[-1],
            "val_b": {str(b): val_caught_b[b][-1] for b in budgets},
            "val_c": {str(b): val_caught_c[b][-1] for b in budgets},
            "rec_b": {str(b): rec_b[b][-1] for b in budgets},
            "rec_c": {str(b): rec_c[b][-1] for b in budgets},
            "pt_rec_b": {ft: {str(b): per_type_rec_b[ft][b][-1] for b in budgets} for ft in FRAUD_TYPES},
            "pt_rec_c": {ft: {str(b): per_type_rec_c[ft][b][-1] for b in budgets} for ft in FRAUD_TYPES},
            "pt_roc_b": {ft: per_type_roc_b[ft][-1] for ft in FRAUD_TYPES},
            "pt_roc_c": {ft: per_type_roc_c[ft][-1] for ft in FRAUD_TYPES},
            "ece_b_raw": ece_b_raw[-1], "ece_c_raw": ece_c_raw[-1],
            "ece_b_cal": ece_b_cal[-1], "ece_c_cal": ece_c_cal[-1],
            "brier_b_raw": brier_b_raw[-1], "brier_c_raw": brier_c_raw[-1],
            "brier_b_cal": brier_b_cal[-1], "brier_c_cal": brier_c_cal[-1],
            "tau_b": tau_b_list[-1], "tau_c": tau_c_list[-1],
            "vol_b": realized_vol_b[-1], "vol_c": realized_vol_c[-1],
            "prec_b": realized_prec_b[-1], "prec_c": realized_prec_c[-1],
            "rec_tp_b": realized_rec_b[-1], "rec_tp_c": realized_rec_c[-1],
            "val_tp_b": realized_val_b[-1], "val_tp_c": realized_val_c[-1],
            "conf_all": conf_coverage_all[-1], "conf_y0": conf_coverage_y0[-1],
            "conf_y1": conf_coverage_y1[-1], "conf_size": conf_set_sizes[-1],
            "coll_edge": coll_edge_pos_ratio[-1],
            "coll_burst": coll_share_prior_burst_ratio[-1],
        }
        if idx <= 5:
            ckpt_payload["ring"] = {
                "mem_all": ring_res["member_level"]["all_cc"][-1],
                "mem_rnd": ring_res["member_level"]["random"][-1],
                "mem_ts": ring_res["member_level"]["trustshield"][-1],
                "r_all": ring_res["ring_level"]["all_cc"][-1],
                "r_rnd": ring_res["ring_level"]["random"][-1],
                "r_ts": ring_res["ring_level"]["trustshield"][-1],
                "by_type": {ft: ring_res["by_type"][ft][-1] for ft in FRAUD_TYPES},
            }
        with open(ckpt_path, "w", encoding="utf-8") as f_ckpt:
            json.dump(ckpt_payload, f_ckpt)

        el = time.perf_counter() - t0
        print(f" done in {el:.1f}s | Truncated Test ROC (b): {roc_b[-1]:.4f}, (c): {roc_c[-1]:.4f} (lift: {roc_c[-1]-roc_b[-1]:+.4f})")

    # Output compilation
    print("\n" + "=" * 80)
    print("CANONICAL 20-SEED BENCHMARK (TRUNCATED TEST: FINAL 21 DAYS EXCLUDED)")
    print("=" * 80)
    print(f"(a) Tabular (no dev):  ROC = {np.mean(roc_a):.4f} +/- {np.std(roc_a):.4f} | PR = {np.mean(pr_a):.4f} +/- {np.std(pr_a):.4f}")
    print(f"(b) Tabular (baseline):ROC = {np.mean(roc_b):.4f} +/- {np.std(roc_b):.4f} | PR = {np.mean(pr_b):.4f} +/- {np.std(pr_b):.4f}")
    print(f"(d) Plain Aggregates:  ROC = {np.mean(roc_d):.4f} +/- {np.std(roc_d):.4f} | PR = {np.mean(pr_d):.4f} +/- {np.std(pr_d):.4f}")
    print(f"(c) Tabular + Graph:   ROC = {np.mean(roc_c):.4f} +/- {np.std(roc_c):.4f} | PR = {np.mean(pr_c):.4f} +/- {np.std(pr_c):.4f}")

    lift_c_vs_b = paired_summary(roc_c, roc_b)
    lift_c_vs_b_pr = paired_summary(pr_c, pr_b)
    lift_d_vs_b = paired_summary(roc_d, roc_b)
    lift_c_vs_d = paired_summary(roc_c, roc_d)

    print(f"\nPaired Lift (c vs b): ROC = {lift_c_vs_b['mean_diff']:+.4f} (p = {lift_c_vs_b['p_value']:.4e}) | PR = {lift_c_vs_b_pr['mean_diff']:+.4f} (p = {lift_c_vs_b_pr['p_value']:.4e})")
    print(f"Paired Lift (d vs b): ROC = {lift_d_vs_b['mean_diff']:+.4f} (p = {lift_d_vs_b['p_value']:.4e})")
    print(f"Paired Lift (c vs d): ROC = {lift_c_vs_d['mean_diff']:+.4f} (p = {lift_c_vs_d['p_value']:.4e})")

    # Serialize complete audited results
    results_path = os.path.join(_ROOT_DIR, "results", "results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        full_res = json.load(f)

    full_res["truncated_test_horizon_20_seeds"] = {
        "test_horizon": f"{VAL_END.strftime('%Y-%m-%d')} to {TEST_END.strftime('%Y-%m-%d')} (final 21 days excluded)",
        "models": {
            "variant_a_tabular_no_device": {"roc": {"mean": float(np.mean(roc_a)), "std": float(np.std(roc_a))}, "pr": {"mean": float(np.mean(pr_a)), "std": float(np.std(pr_a))}},
            "variant_b_tabular_with_device": {"roc": {"mean": float(np.mean(roc_b)), "std": float(np.std(roc_b))}, "pr": {"mean": float(np.mean(pr_b)), "std": float(np.std(pr_b))}},
            "variant_d_plain_aggregates": {"roc": {"mean": float(np.mean(roc_d)), "std": float(np.std(roc_d))}, "pr": {"mean": float(np.mean(pr_d)), "std": float(np.std(pr_d))}},
            "variant_c_tabular_plus_graph": {"roc": {"mean": float(np.mean(roc_c)), "std": float(np.std(roc_c))}, "pr": {"mean": float(np.mean(pr_c)), "std": float(np.std(pr_c))}},
        },
        "paired_lifts": {
            "c_vs_b_roc": lift_c_vs_b,
            "c_vs_b_pr": lift_c_vs_b_pr,
            "d_vs_b_roc": lift_d_vs_b,
            "c_vs_d_roc": lift_c_vs_d,
        },
        "per_seed_arrays": {
            "roc_a": [float(x) for x in roc_a],
            "roc_b": [float(x) for x in roc_b],
            "roc_c": [float(x) for x in roc_c],
            "roc_d": [float(x) for x in roc_d],
            "pr_a": [float(x) for x in pr_a],
            "pr_b": [float(x) for x in pr_b],
            "pr_c": [float(x) for x in pr_c],
            "pr_d": [float(x) for x in pr_d],
        }
    }

    # Calibration ECE and Brier
    full_res["calibration_audit_20_seeds"] = {
        "tabular_b": {
            "raw_ece": {"mean": float(np.mean(ece_b_raw)), "std": float(np.std(ece_b_raw))},
            "calibrated_ece": {"mean": float(np.mean(ece_b_cal)), "std": float(np.std(ece_b_cal))},
            "raw_brier": {"mean": float(np.mean(brier_b_raw)), "std": float(np.std(brier_b_raw))},
            "calibrated_brier": {"mean": float(np.mean(brier_b_cal)), "std": float(np.std(brier_b_cal))},
        },
        "tabular_plus_graph_c": {
            "raw_ece": {"mean": float(np.mean(ece_c_raw)), "std": float(np.std(ece_c_raw))},
            "calibrated_ece": {"mean": float(np.mean(ece_c_cal)), "std": float(np.std(ece_c_cal))},
            "raw_brier": {"mean": float(np.mean(brier_c_raw)), "std": float(np.std(brier_c_raw))},
            "calibrated_brier": {"mean": float(np.mean(brier_c_cal)), "std": float(np.std(brier_c_cal))},
        },
        "paired_calibrated_ece_c_vs_b": paired_summary(ece_c_cal, ece_b_cal),
        "paired_calibrated_brier_c_vs_b": paired_summary(brier_c_cal, brier_b_cal),
    }

    # Threshold Policy on Validation (5% budget)
    full_res["threshold_policy_5pct_budget_20_seeds"] = {
        "tau_selected_on_val": {
            "b": {"mean": float(np.mean(tau_b_list)), "std": float(np.std(tau_b_list))},
            "c": {"mean": float(np.mean(tau_c_list)), "std": float(np.std(tau_c_list))},
        },
        "realized_test_volume": {
            "b": {"mean": float(np.mean(realized_vol_b)), "std": float(np.std(realized_vol_b))},
            "c": {"mean": float(np.mean(realized_vol_c)), "std": float(np.std(realized_vol_c))},
        },
        "realized_precision": {
            "b": {"mean": float(np.mean(realized_prec_b)), "std": float(np.std(realized_prec_b))},
            "c": {"mean": float(np.mean(realized_prec_c)), "std": float(np.std(realized_prec_c))},
        },
        "realized_recall": {
            "b": {"mean": float(np.mean(realized_rec_b)), "std": float(np.std(realized_rec_b))},
            "c": {"mean": float(np.mean(realized_rec_c)), "std": float(np.std(realized_rec_c))},
        },
        "realized_fraud_value_caught_inr": {
            "b": {"mean": float(np.mean(realized_val_b)), "std": float(np.std(realized_val_b))},
            "c": {"mean": float(np.mean(realized_val_c)), "std": float(np.std(realized_val_c))},
        },
        "paired_realized_recall_lift": paired_summary(realized_rec_c, realized_rec_b),
        "paired_realized_value_lift_inr": paired_summary(realized_val_c, realized_val_b),
    }

    # Ring detection
    full_res["ring_detection_audit"] = {
        "unit_definitions": {
            "member_level": "Evaluates precision, recall, and F1 over individual buyer nodes belonging to ground-truth rings",
            "ring_level": "Evaluates recovery of ground-truth ring clusters (defined as detecting >= 50% of the ring's member nodes)"
        },
        "baselines_comparison": {
            "member_level_f1": {
                "all_cc_size_ge_2_no_filter": float(np.mean([x["f1"] for x in ring_res["member_level"]["all_cc"]])),
                "random_cluster_baseline": float(np.mean([x["f1"] for x in ring_res["member_level"]["random"]])),
                "trustshield_risk_ge_0_50": float(np.mean([x["f1"] for x in ring_res["member_level"]["trustshield"]])),
            },
            "member_level_precision": {
                "all_cc_size_ge_2_no_filter": float(np.mean([x["p"] for x in ring_res["member_level"]["all_cc"]])),
                "random_cluster_baseline": float(np.mean([x["p"] for x in ring_res["member_level"]["random"]])),
                "trustshield_risk_ge_0_50": float(np.mean([x["p"] for x in ring_res["member_level"]["trustshield"]])),
            },
            "member_level_recall": {
                "all_cc_size_ge_2_no_filter": float(np.mean([x["r"] for x in ring_res["member_level"]["all_cc"]])),
                "random_cluster_baseline": float(np.mean([x["r"] for x in ring_res["member_level"]["random"]])),
                "trustshield_risk_ge_0_50": float(np.mean([x["r"] for x in ring_res["member_level"]["trustshield"]])),
            },
            "ring_level_recovery_rate": {
                "all_cc_size_ge_2_no_filter": float(np.mean(ring_res["ring_level"]["all_cc"])),
                "random_cluster_baseline": float(np.mean(ring_res["ring_level"]["random"])),
                "trustshield_risk_ge_0_50": float(np.mean(ring_res["ring_level"]["trustshield"])),
            }
        },
        "by_fraud_type_recovery": {
            ft: float(np.mean(ring_res["by_type"][ft])) if ring_res["by_type"][ft] else 0.0
            for ft in FRAUD_TYPES
        },
        "scientific_note": "Device and address sharing graphs were generated by the synthetic injection engine; this test evaluates generator recovery mechanics."
    }

    # Conformal coverage
    full_res["conformal_prediction_audit_20_seeds"] = {
        "target_error_rate_alpha": 0.05,
        "nominal_guarantee": "95% (finite-sample distribution-free bound on calibration cohort)",
        "empirical_test_coverage_overall": {"mean": float(np.mean(conf_coverage_all)), "std": float(np.std(conf_coverage_all))},
        "empirical_test_coverage_y0_legit": {"mean": float(np.mean(conf_coverage_y0)), "std": float(np.std(conf_coverage_y0))},
        "empirical_test_coverage_y1_fraud": {"mean": float(np.mean(conf_coverage_y1)), "std": float(np.std(conf_coverage_y1))},
        "empirical_test_mean_prediction_set_size": {"mean": float(np.mean(conf_set_sizes)), "std": float(np.std(conf_set_sizes))},
        "serving_status": "Serving currently uses heuristic score thresholds (<0.20, >0.70) without calling .calibrate(). The 95% mathematical coverage guarantee only applies when .calibrate() is executed with validation scores."
    }

    # Collusion Burst Diagnostic
    full_res["seller_buyer_collusion_diagnostic_exploratory"] = {
        "share_orders_with_edge_weight_before_gt_0": {"mean": float(np.mean(coll_edge_pos_ratio)), "std": float(np.std(coll_edge_pos_ratio))},
        "share_orders_sharing_pair_with_earlier_burst": {"mean": float(np.mean(coll_share_prior_burst_ratio)), "std": float(np.std(coll_share_prior_burst_ratio))},
        "topological_blindspot_explanation": "Collusion orders occur within short 5-14 day bursts. Monthly-lagged bipartite snapshots only refresh once every calendar month; thus, orders within a burst occurring between snapshot cadences have edge_weight_before = 0 and cannot see same-month prior orders in the snapshot graph."
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_res, f, indent=2)

    print(f"\nAudit completed! Persisted updated canonical metrics to {results_path}")


if __name__ == "__main__":
    main()
