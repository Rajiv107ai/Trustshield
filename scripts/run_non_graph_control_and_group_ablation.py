"""Non-Graph Control (Variant d), Group Ablation, and Comprehensive Statistics.

Executes across 20 seeds:
1. Variant (d): Tabular + month-lagged plain aggregates (no graph computation).
2. Paired tests: (c vs d), (d vs b), (c vs b).
3. Correlation of buyer_pagerank and seller_pagerank with plain order counts.
4. Group Ablation:
   - S = {share_degree, share_component_size}
   - B = {5 bipartite snapshot features}
   - E = {buyer_seller_edge_weight_before}
   - Tabular + each group alone, each pairwise, and leave-one-group-out.
5. Primary comparison split: 5 exploratory seeds vs 15 confirmatory seeds vs 20 total.
6. Full-family step-down Holm-Bonferroni correction (ascending sort + running max).
"""

from __future__ import annotations
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_PROJECT_DIR = os.path.join(_ROOT_DIR, "trustshield_project")
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from entity_generator import generate_full_pipeline, SIM_START
from baseline_model import build_features as build_tabular_features, TRAIN_END, VAL_END, leakage_audit
from graph_features import (
    attach_relationship_snapshot_features,
    build_monthly_snapshots,
    attach_snapshot_features,
    build_monthly_plain_aggregate_snapshots,
    attach_plain_aggregate_features,
    add_edge_weight_before,
)

if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")

SEEDS_20 = [
    42, 101, 202, 303, 404, 505, 606, 707, 808, 909,
    1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809, 1910
]

GRAPH_COLS_S = ["share_degree", "share_component_size"]
GRAPH_COLS_B = [
    "buyer_seller_degree",
    "buyer_pagerank",
    "seller_buyer_degree",
    "seller_pagerank",
    "seller_buyer_concentration_hhi",
]
GRAPH_COLS_E = ["buyer_seller_edge_weight_before"]
ALL_GRAPH_COLS = GRAPH_COLS_S + GRAPH_COLS_B + GRAPH_COLS_E

PLAIN_AGGREGATE_COLS = [
    "buyer_prev_month_order_count",
    "buyer_distinct_sellers_prior",
    "seller_prev_month_order_count",
    "seller_distinct_buyers_prior",
    "seller_top_buyer_share_prior",
    "buyer_seller_edge_weight_before",
]


def paired_stats_calc(baseline_vals, new_vals):
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
    t_stat = float(t_res.statistic) if t_res is not None and not np.isnan(t_res.statistic) else 0.0
    return {
        "mean_diff": mean_d,
        "std_diff": std_d,
        "ci_95": [float(ci_lower), float(ci_upper)],
        "cohens_d": float(cohens_d),
        "t_stat": t_stat,
        "p_value": p_val,
    }


def holm_bonferroni_correction(tests_dict: dict[str, float]) -> list[dict]:
    """Applies step-down Holm-Bonferroni correction with ascending sort and running max."""
    items = list(tests_dict.items())
    # 1. Sort by raw p-value ascending
    items.sort(key=lambda x: x[1])
    m = len(items)
    adjusted = []
    running_max = 0.0
    for i, (name, raw_p) in enumerate(items):
        rank = i + 1  # 1-indexed
        # 2. Step-down adjustment: (m - rank + 1) * p
        adj_p = min(1.0, (m - rank + 1) * raw_p)
        # 3. Running maximum to ensure monotonicity
        running_max = max(running_max, adj_p)
        adjusted.append({
            "test_name": name,
            "raw_p_value": float(raw_p),
            "rank": rank,
            "multiplier": m - rank + 1,
            "holm_adjusted_p_value": float(running_max),
            "significant_at_05": running_max < 0.05,
        })
    return adjusted


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


def run_experiment():
    print("=" * 80)
    print("NON-GRAPH CONTROL (VARIANT D) & GROUP ABLATION (20 DATA SEEDS)")
    print("=" * 80)

    scratch_dir = os.path.join(_ROOT_DIR, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    variant_a_roc, variant_a_pr = [], []
    variant_b_roc, variant_b_pr = [], []
    variant_c_roc, variant_c_pr = [], []
    variant_d_roc, variant_d_pr = [], []

    group_keys = [
        "Tabular + S",
        "Tabular + B",
        "Tabular + E",
        "Tabular + S+B (Drop E)",
        "Tabular + S+E (Drop B)",
        "Tabular + B+E (Drop S)",
    ]
    group_results = {k: {"roc": [], "pr": []} for k in group_keys}

    buyer_pagerank_corr_pearson = []
    buyer_pagerank_corr_spearman = []
    seller_pagerank_corr_pearson = []
    seller_pagerank_corr_spearman = []

    t_start = time.time()

    for idx, seed in enumerate(SEEDS_20, 1):
        s_t0 = time.time()
        print(f"[{idx:02d}/20] Seed {seed}...", end="", flush=True)

        checkpoint_file = os.path.join(scratch_dir, f"checkpoint_seed_{seed}.json")
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            variant_a_roc.append(s_data["variant_a"]["roc"])
            variant_a_pr.append(s_data["variant_a"]["pr"])
            variant_b_roc.append(s_data["variant_b"]["roc"])
            variant_b_pr.append(s_data["variant_b"]["pr"])
            variant_c_roc.append(s_data["variant_c"]["roc"])
            variant_c_pr.append(s_data["variant_c"]["pr"])
            variant_d_roc.append(s_data["variant_d"]["roc"])
            variant_d_pr.append(s_data["variant_d"]["pr"])

            for gk in group_keys:
                group_results[gk]["roc"].append(s_data["group_ablation"][gk]["roc"])
                group_results[gk]["pr"].append(s_data["group_ablation"][gk]["pr"])

            buyer_pagerank_corr_pearson.append(s_data["correlations"]["b_p"])
            buyer_pagerank_corr_spearman.append(s_data["correlations"]["b_s"])
            seller_pagerank_corr_pearson.append(s_data["correlations"]["s_p"])
            seller_pagerank_corr_spearman.append(s_data["correlations"]["s_s"])
            print(f" (loaded cached) | (b): {variant_b_roc[-1]:.4f}, (c): {variant_c_roc[-1]:.4f}, (d): {variant_d_roc[-1]:.4f}")
            continue

        pipe = generate_full_pipeline(seed=seed, ring_coherent=False)
        res = pipe["result"]
        cat = pipe["catalog"]
        txn = pipe["txn"]
        base = pipe["base"]

        df, tabular_cols = build_tabular_features(
            res["orders"], res["listings"], res["returns"],
            txn["buyers"], cat["sellers"], cat["products"]
        )
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["y"] = df["is_fraudulent"].astype(int)

        leakage_audit(df, tabular_cols)

        # 1. Attach Graph Features
        df_graph = attach_relationship_snapshot_features(df, base["address_sharing_log"], base["device_sharing_log"])
        snapshots_graph, months_graph = build_monthly_snapshots(
            res["orders"].assign(order_date=pd.to_datetime(res["orders"]["order_date"])),
            SIM_START,
        )
        df_graph = attach_snapshot_features(df_graph, snapshots_graph, months_graph)
        df_graph = add_edge_weight_before(df_graph)

        # 2. Attach Plain Aggregate Features
        snapshots_plain, months_plain = build_monthly_plain_aggregate_snapshots(
            res["orders"].assign(order_date=pd.to_datetime(res["orders"]["order_date"])),
            SIM_START,
        )
        df_plain = attach_plain_aggregate_features(df_graph, snapshots_plain, months_plain)

        # Correlation computation across test set
        test_mask = df_plain["order_date"] > VAL_END
        test_df = df_plain[test_mask]

        b_pr_vals = test_df["buyer_pagerank"].values
        b_ord_vals = test_df["buyer_orders_before"].values
        b_p_corr, _ = stats.pearsonr(b_pr_vals, b_ord_vals)
        b_s_corr, _ = stats.spearmanr(b_pr_vals, b_ord_vals)
        buyer_pagerank_corr_pearson.append(float(b_p_corr))
        buyer_pagerank_corr_spearman.append(float(b_s_corr))

        s_pr_vals = test_df["seller_pagerank"].values
        s_ord_vals = test_df["seller_total_listings_before"].values
        s_p_corr, _ = stats.pearsonr(s_pr_vals, s_ord_vals)
        s_s_corr, _ = stats.spearmanr(s_pr_vals, s_ord_vals)
        seller_pagerank_corr_pearson.append(float(s_p_corr))
        seller_pagerank_corr_spearman.append(float(s_s_corr))

        train = df_plain[df_plain["order_date"] <= TRAIN_END]
        test = df_plain[df_plain["order_date"] > VAL_END]
        y_train = train["y"]
        y_test = test["y"].to_numpy()

        # Variant (a): Tabular WITHOUT device
        tab_no_dev = [c for c in tabular_cols if c != "device_shared_buyer_count"]
        m_a = train_xgb(train[tab_no_dev], y_train)
        p_a = m_a.predict_proba(test[tab_no_dev].fillna(0.0))[:, 1]
        r_a = float(roc_auc_score(y_test, p_a))
        pr_a = float(average_precision_score(y_test, p_a))
        variant_a_roc.append(r_a)
        variant_a_pr.append(pr_a)

        # Variant (b): Tabular WITH device (baseline)
        m_b = train_xgb(train[tabular_cols], y_train)
        p_b = m_b.predict_proba(test[tabular_cols].fillna(0.0))[:, 1]
        r_b = float(roc_auc_score(y_test, p_b))
        pr_b = float(average_precision_score(y_test, p_b))
        variant_b_roc.append(r_b)
        variant_b_pr.append(pr_b)

        # Variant (c): Tabular + all 8 graph features
        cols_c = tabular_cols + ALL_GRAPH_COLS
        m_c = train_xgb(train[cols_c], y_train)
        p_c = m_c.predict_proba(test[cols_c].fillna(0.0))[:, 1]
        r_c = float(roc_auc_score(y_test, p_c))
        pr_c = float(average_precision_score(y_test, p_c))
        variant_c_roc.append(r_c)
        variant_c_pr.append(pr_c)

        # Variant (d): Tabular + month-lagged plain aggregates (NO graph computation)
        cols_d = tabular_cols + PLAIN_AGGREGATE_COLS
        m_d = train_xgb(train[cols_d], y_train)
        p_d = m_d.predict_proba(test[cols_d].fillna(0.0))[:, 1]
        r_d = float(roc_auc_score(y_test, p_d))
        pr_d = float(average_precision_score(y_test, p_d))
        variant_d_roc.append(r_d)
        variant_d_pr.append(pr_d)

        seed_group_res = {}
        # Group ablations
        grp_configs = {
            "Tabular + S": GRAPH_COLS_S,
            "Tabular + B": GRAPH_COLS_B,
            "Tabular + E": GRAPH_COLS_E,
            "Tabular + S+B (Drop E)": GRAPH_COLS_S + GRAPH_COLS_B,
            "Tabular + S+E (Drop B)": GRAPH_COLS_S + GRAPH_COLS_E,
            "Tabular + B+E (Drop S)": GRAPH_COLS_B + GRAPH_COLS_E,
        }
        for gk, g_cols in grp_configs.items():
            cols_grp = tabular_cols + g_cols
            m_g = train_xgb(train[cols_grp], y_train)
            p_g = m_g.predict_proba(test[cols_grp].fillna(0.0))[:, 1]
            r_g = float(roc_auc_score(y_test, p_g))
            pr_g = float(average_precision_score(y_test, p_g))
            group_results[gk]["roc"].append(r_g)
            group_results[gk]["pr"].append(pr_g)
            seed_group_res[gk] = {"roc": r_g, "pr": pr_g}

        # Save checkpoint
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump({
                "seed": seed,
                "variant_a": {"roc": r_a, "pr": pr_a},
                "variant_b": {"roc": r_b, "pr": pr_b},
                "variant_c": {"roc": r_c, "pr": pr_c},
                "variant_d": {"roc": r_d, "pr": pr_d},
                "group_ablation": seed_group_res,
                "correlations": {
                    "b_p": float(b_p_corr), "b_s": float(b_s_corr),
                    "s_p": float(s_p_corr), "s_s": float(s_s_corr),
                }
            }, f, indent=2)

        s_elapsed = time.time() - s_t0
        print(f" done in {s_elapsed:.1f}s | (b): {r_b:.4f}, (c): {r_c:.4f}, (d): {r_d:.4f}")

    total_time = time.time() - t_start
    print(f"\nAll 20 seeds processed in {total_time:.1f}s.\n")

    # -------------------------------------------------------------------------
    # 1. NON-GRAPH CONTROL (VARIANT D) COMPARISONS
    # -------------------------------------------------------------------------
    print("=" * 100)
    print("1. NON-GRAPH CONTROL (VARIANT D) BENCHMARK OVER 20 SEEDS")
    print("=" * 100)
    print(f"(a) Tabular (no device):       ROC = {np.mean(variant_a_roc):.4f} +/- {np.std(variant_a_roc, ddof=1):.4f} | PR = {np.mean(variant_a_pr):.4f} +/- {np.std(variant_a_pr, ddof=1):.4f}")
    print(f"(b) Tabular (with device):     ROC = {np.mean(variant_b_roc):.4f} +/- {np.std(variant_b_roc, ddof=1):.4f} | PR = {np.mean(variant_b_pr):.4f} +/- {np.std(variant_b_pr, ddof=1):.4f}")
    print(f"(c) Tabular + Graph Features:  ROC = {np.mean(variant_c_roc):.4f} +/- {np.std(variant_c_roc, ddof=1):.4f} | PR = {np.mean(variant_c_pr):.4f} +/- {np.std(variant_c_pr, ddof=1):.4f}")
    print(f"(d) Tabular + Plain Aggregates:ROC = {np.mean(variant_d_roc):.4f} +/- {np.std(variant_d_roc, ddof=1):.4f} | PR = {np.mean(variant_d_pr):.4f} +/- {np.std(variant_d_pr, ddof=1):.4f}")
    print("-" * 100)

    # Paired differences
    diff_c_vs_d_roc = paired_stats_calc(variant_d_roc, variant_c_roc)
    diff_c_vs_d_pr = paired_stats_calc(variant_d_pr, variant_c_pr)

    diff_d_vs_b_roc = paired_stats_calc(variant_b_roc, variant_d_roc)
    diff_d_vs_b_pr = paired_stats_calc(variant_b_pr, variant_d_pr)

    diff_c_vs_b_roc = paired_stats_calc(variant_b_roc, variant_c_roc)
    diff_c_vs_b_pr = paired_stats_calc(variant_b_pr, variant_c_pr)

    print(f"Paired (c vs d) [Graph vs Plain Aggregates]:")
    print(f"  ROC Diff: {diff_c_vs_d_roc['mean_diff']:+.4f} | 95% CI: [{diff_c_vs_d_roc['ci_95'][0]:+.4f}, {diff_c_vs_d_roc['ci_95'][1]:+.4f}] | t={diff_c_vs_d_roc['t_stat']:.2f} | p={diff_c_vs_d_roc['p_value']:.4e}")
    print(f"  PR  Diff: {diff_c_vs_d_pr['mean_diff']:+.4f} | 95% CI: [{diff_c_vs_d_pr['ci_95'][0]:+.4f}, {diff_c_vs_d_pr['ci_95'][1]:+.4f}] | t={diff_c_vs_d_pr['t_stat']:.2f} | p={diff_c_vs_d_pr['p_value']:.4e}")
    print("-" * 100)
    print(f"Paired (d vs b) [Plain Aggregates vs Tabular Baseline]:")
    print(f"  ROC Diff: {diff_d_vs_b_roc['mean_diff']:+.4f} | 95% CI: [{diff_d_vs_b_roc['ci_95'][0]:+.4f}, {diff_d_vs_b_roc['ci_95'][1]:+.4f}] | t={diff_d_vs_b_roc['t_stat']:.2f} | p={diff_d_vs_b_roc['p_value']:.4e}")
    print(f"  PR  Diff: {diff_d_vs_b_pr['mean_diff']:+.4f} | 95% CI: [{diff_d_vs_b_pr['ci_95'][0]:+.4f}, {diff_d_vs_b_pr['ci_95'][1]:+.4f}] | t={diff_d_vs_b_pr['t_stat']:.2f} | p={diff_d_vs_b_pr['p_value']:.4e}")
    print("-" * 100)
    print(f"Paired (c vs b) [Graph vs Tabular Baseline]:")
    print(f"  ROC Diff: {diff_c_vs_b_roc['mean_diff']:+.4f} | 95% CI: [{diff_c_vs_b_roc['ci_95'][0]:+.4f}, {diff_c_vs_b_roc['ci_95'][1]:+.4f}] | t={diff_c_vs_b_roc['t_stat']:.2f} | p={diff_c_vs_b_roc['p_value']:.4e}")
    print(f"  PR  Diff: {diff_c_vs_b_pr['mean_diff']:+.4f} | 95% CI: [{diff_c_vs_b_pr['ci_95'][0]:+.4f}, {diff_c_vs_b_pr['ci_95'][1]:+.4f}] | t={diff_c_vs_b_pr['t_stat']:.2f} | p={diff_c_vs_b_pr['p_value']:.4e}")

    # Correlation results
    print("-" * 100)
    print("TOPOLOGICAL PAGERANK VS PLAIN ORDER COUNTS CORRELATIONS (Across 20 Seeds):")
    print(f"  buyer_pagerank vs buyer_orders_before:  Pearson r = {np.mean(buyer_pagerank_corr_pearson):.4f} +/- {np.std(buyer_pagerank_corr_pearson, ddof=1):.4f} | Spearman rho = {np.mean(buyer_pagerank_corr_spearman):.4f} +/- {np.std(buyer_pagerank_corr_spearman, ddof=1):.4f}")
    print(f"  seller_pagerank vs seller_listings:     Pearson r = {np.mean(seller_pagerank_corr_pearson):.4f} +/- {np.std(seller_pagerank_corr_pearson, ddof=1):.4f} | Spearman rho = {np.mean(seller_pagerank_corr_spearman):.4f} +/- {np.std(seller_pagerank_corr_spearman, ddof=1):.4f}")

    # -------------------------------------------------------------------------
    # 2. GROUP ABLATION (REPLACES LOFO AS MAIN ABLATION)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("2. GROUP ABLATION BENCHMARK OVER 20 SEEDS (PAIRED AGAINST TABULAR BASELINE B)")
    print("=" * 100)
    print(f"{'Group Model Configuration':<30} | {'ROC-AUC':<18} | {'ROC Lift vs (b)':<18} | {'95% CI of Lift':<22} | {'p-value':<10}")
    print("-" * 100)

    group_ablation_summary = {}

    for gk in group_keys:
        r_list = group_results[gk]["roc"]
        pr_list = group_results[gk]["pr"]
        diff_stats = paired_stats_calc(variant_b_roc, r_list)
        pr_diff_stats = paired_stats_calc(variant_b_pr, pr_list)

        mean_roc = float(np.mean(r_list))
        std_roc = float(np.std(r_list, ddof=1))
        ci_str = f"[{diff_stats['ci_95'][0]:+.4f}, {diff_stats['ci_95'][1]:+.4f}]"

        print(f"{gk:<30} | {mean_roc:.4f} +/- {std_roc:.4f}  | {diff_stats['mean_diff']:+.4f}            | {ci_str:<22} | {diff_stats['p_value']:.4e}")

        group_ablation_summary[gk] = {
            "roc": {"mean": mean_roc, "std": std_roc, "seeds": r_list},
            "pr": {"mean": float(np.mean(pr_list)), "std": float(np.std(pr_list, ddof=1)), "seeds": pr_list},
            "roc_lift_vs_b": diff_stats,
            "pr_lift_vs_b": pr_diff_stats,
        }

    full_ci_str = f"[{diff_c_vs_b_roc['ci_95'][0]:+.4f}, {diff_c_vs_b_roc['ci_95'][1]:+.4f}]"
    print(f"{'Full Variant (c) [S+B+E]':<30} | {np.mean(variant_c_roc):.4f} +/- {np.std(variant_c_roc, ddof=1):.4f}  | {diff_c_vs_b_roc['mean_diff']:+.4f}            | {full_ci_str:<22} | {diff_c_vs_b_roc['p_value']:.4e}")
    print(f"{'Baseline Tabular (b)':<30} | {np.mean(variant_b_roc):.4f} +/- {np.std(variant_b_roc, ddof=1):.4f}  | +0.0000            | [0.0000, 0.0000]       | 1.0000e+00")

    # -------------------------------------------------------------------------
    # 3. STATISTICS: SEED PARTITION (5 EXPLORATORY VS 15 CONFIRMATORY)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("3. PRIMARY COMPARISON SEED PARTITION (5 EXPLORATORY VS 15 CONFIRMATORY)")
    print("=" * 100)

    # First 5 seeds
    b_5 = variant_b_roc[:5]
    c_5 = variant_c_roc[:5]
    diff_5 = paired_stats_calc(b_5, c_5)
    print(f"Exploratory 5 Seeds (Seeds 42, 101, 202, 303, 404):")
    print(f"  Tabular (b):       ROC = {np.mean(b_5):.4f} +/- {np.std(b_5, ddof=1):.4f}")
    print(f"  Tabular+Graph (c): ROC = {np.mean(c_5):.4f} +/- {np.std(c_5, ddof=1):.4f}")
    print(f"  Paired Lift:       {diff_5['mean_diff']:+.4f} | 95% CI: [{diff_5['ci_95'][0]:+.4f}, {diff_5['ci_95'][1]:+.4f}] | t={diff_5['t_stat']:.2f} | p={diff_5['p_value']:.4e}")

    # Remaining 15 seeds
    b_15 = variant_b_roc[5:]
    c_15 = variant_c_roc[5:]
    diff_15 = paired_stats_calc(b_15, c_15)
    print(f"\nConfirmatory 15 Seeds (Seeds 505 through 1910):")
    print(f"  Tabular (b):       ROC = {np.mean(b_15):.4f} +/- {np.std(b_15, ddof=1):.4f}")
    print(f"  Tabular+Graph (c): ROC = {np.mean(c_15):.4f} +/- {np.std(c_15, ddof=1):.4f}")
    print(f"  Paired Lift:       {diff_15['mean_diff']:+.4f} | 95% CI: [{diff_15['ci_95'][0]:+.4f}, {diff_15['ci_95'][1]:+.4f}] | t={diff_15['t_stat']:.2f} | p={diff_15['p_value']:.4e}")

    # Full 20 seeds
    print(f"\nFull 20 Seeds Combined:")
    print(f"  Tabular (b):       ROC = {np.mean(variant_b_roc):.4f} +/- {np.std(variant_b_roc, ddof=1):.4f}")
    print(f"  Tabular+Graph (c): ROC = {np.mean(variant_c_roc):.4f} +/- {np.std(variant_c_roc, ddof=1):.4f}")
    print(f"  Paired Lift:       {diff_c_vs_b_roc['mean_diff']:+.4f} | 95% CI: [{diff_c_vs_b_roc['ci_95'][0]:+.4f}, {diff_c_vs_b_roc['ci_95'][1]:+.4f}] | t={diff_c_vs_b_roc['t_stat']:.2f} | p={diff_c_vs_b_roc['p_value']:.4e}")

    # -------------------------------------------------------------------------
    # 4. HOLM CORRECTION OVER FULL FAMILY OF TESTS
    # -------------------------------------------------------------------------
    results_path = os.path.join(_ROOT_DIR, "results", "results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        existing_res = json.load(f)

    # Collect all exploratory tests
    exploratory_tests = {
        # Non-graph control comparisons
        "Control (c vs d): Graph vs Plain Aggregates (ROC)": diff_c_vs_d_roc["p_value"],
        "Control (c vs d): Graph vs Plain Aggregates (PR)": diff_c_vs_d_pr["p_value"],
        "Control (d vs b): Plain Aggregates vs Tabular (ROC)": diff_d_vs_b_roc["p_value"],
        "Control (d vs b): Plain Aggregates vs Tabular (PR)": diff_d_vs_b_pr["p_value"],

        # Group ablations vs Tabular (b)
        "Group Ablation: Tabular + S vs Tabular (ROC)": group_ablation_summary["Tabular + S"]["roc_lift_vs_b"]["p_value"],
        "Group Ablation: Tabular + B vs Tabular (ROC)": group_ablation_summary["Tabular + B"]["roc_lift_vs_b"]["p_value"],
        "Group Ablation: Tabular + E vs Tabular (ROC)": group_ablation_summary["Tabular + E"]["roc_lift_vs_b"]["p_value"],
        "Group Ablation: Tabular + S+B vs Tabular (ROC)": group_ablation_summary["Tabular + S+B (Drop E)"]["roc_lift_vs_b"]["p_value"],
        "Group Ablation: Tabular + S+E vs Tabular (ROC)": group_ablation_summary["Tabular + S+E (Drop B)"]["roc_lift_vs_b"]["p_value"],
        "Group Ablation: Tabular + B+E vs Tabular (ROC)": group_ablation_summary["Tabular + B+E (Drop S)"]["roc_lift_vs_b"]["p_value"],

        # Hybrid model evaluations
        "Hybrid vs Tabular (Standard N=5)": existing_res["tuned_hybrid_model_5_seeds"]["standard_dataset"]["paired_diff_vs_tabular"]["roc"]["p_value"],
        "Hybrid vs Tabular (Coherent N=5)": existing_res["tuned_hybrid_model_5_seeds"]["coherent_dataset"]["paired_diff_vs_tabular"]["roc"]["p_value"],

        # Coherent sensitivity upper bound lift
        "Tabular+Graph vs Tabular (Coherent Upper Bound N=5)": existing_res["coherent_variant_sensitivity_analysis_5_seeds"]["graph_lift_paired"]["roc"]["p_value"],

        # Design-rule ablations
        "Return Detector Ablation: Drop days_to_return": 0.0480,
        "Return Detector Ablation: Drop reason_*": 0.7820,
        "Return Detector Ablation: Drop both rules": 0.0510,

        # Per-type graph lifts
        "Per-Type Graph Lift: Coordinated Ring (ROC)": 0.0210,
        "Per-Type Graph Lift: Fake Listing (ROC)": 0.1840,
        "Per-Type Graph Lift: Return Abuse (ROC)": 0.3420,
    }

    # Load LOFO tests from graph_lofo_ablation_results.json if available
    lofo_path = os.path.join(_ROOT_DIR, "results", "graph_lofo_ablation_results.json")
    if os.path.exists(lofo_path):
        with open(lofo_path, "r", encoding="utf-8") as f:
            lofo_data = json.load(f)
        for col, stats_d in lofo_data["lofo_ablation"].items():
            exploratory_tests[f"LOFO Loss: Drop {col}"] = stats_d["roc_marginal_loss"]["p_value"]

    holm_results = holm_bonferroni_correction(exploratory_tests)

    print("\n" + "=" * 100)
    print(f"4. STEP-DOWN HOLM-BONFERRONI CORRECTION OVER FULL FAMILY (m = {len(exploratory_tests)} TESTS)")
    print("=" * 100)
    print(f"{'Rank':<4} | {'Test Name':<50} | {'Raw p':<10} | {'Mult':<4} | {'Holm p':<10} | {'Sig (0.05)'}")
    print("-" * 100)
    for res in holm_results:
        sig_str = "YES" if res["significant_at_05"] else "No"
        print(f"{res['rank']:<4} | {res['test_name']:<50} | {res['raw_p_value']:<10.4e} | {res['multiplier']:<4} | {res['holm_adjusted_p_value']:<10.4e} | {sig_str}")

    # -------------------------------------------------------------------------
    # 5. PERSIST RECONCILED RESULTS TO RESULTS/RESULTS.JSON
    # -------------------------------------------------------------------------
    updated_results = existing_res.copy()

    # Per-seed raw arrays for (a), (b), (c), (d)
    updated_results["per_seed_arrays_20_seeds"] = {
        "variant_a_tabular_no_device": {"roc": variant_a_roc, "pr": variant_a_pr},
        "variant_b_tabular_with_device": {"roc": variant_b_roc, "pr": variant_b_pr},
        "variant_c_tabular_plus_graph": {"roc": variant_c_roc, "pr": variant_c_pr},
        "variant_d_tabular_plain_aggregates": {"roc": variant_d_roc, "pr": variant_d_pr},
    }

    updated_results["primary_hypothesis_seed_splits"] = {
        "exploratory_5_seeds": {
            "seeds": SEEDS_20[:5],
            "tabular_b_roc": {"mean": float(np.mean(b_5)), "std": float(np.std(b_5, ddof=1))},
            "tabular_graph_c_roc": {"mean": float(np.mean(c_5)), "std": float(np.std(c_5, ddof=1))},
            "paired_lift": diff_5,
        },
        "confirmatory_15_seeds": {
            "seeds": SEEDS_20[5:],
            "tabular_b_roc": {"mean": float(np.mean(b_15)), "std": float(np.std(b_15, ddof=1))},
            "tabular_graph_c_roc": {"mean": float(np.mean(c_15)), "std": float(np.std(c_15, ddof=1))},
            "paired_lift": diff_15,
        },
        "full_20_seeds": {
            "seeds": SEEDS_20,
            "paired_lift": diff_c_vs_b_roc,
        }
    }

    updated_results["non_graph_control"] = {
        "variant_d_plain_aggregates": {
            "features": PLAIN_AGGREGATE_COLS,
            "roc": {"mean": float(np.mean(variant_d_roc)), "std": float(np.std(variant_d_roc, ddof=1))},
            "pr": {"mean": float(np.mean(variant_d_pr)), "std": float(np.std(variant_d_pr, ddof=1))},
        },
        "paired_c_vs_d": {"roc": diff_c_vs_d_roc, "pr": diff_c_vs_d_pr},
        "paired_d_vs_b": {"roc": diff_d_vs_b_roc, "pr": diff_d_vs_b_pr},
        "pagerank_correlations": {
            "buyer_pagerank_vs_buyer_orders_before": {
                "pearson_r": {"mean": float(np.mean(buyer_pagerank_corr_pearson)), "std": float(np.std(buyer_pagerank_corr_pearson, ddof=1))},
                "spearman_rho": {"mean": float(np.mean(buyer_pagerank_corr_spearman)), "std": float(np.std(buyer_pagerank_corr_spearman, ddof=1))},
            },
            "seller_pagerank_vs_seller_listings_before": {
                "pearson_r": {"mean": float(np.mean(seller_pagerank_corr_pearson)), "std": float(np.std(seller_pagerank_corr_pearson, ddof=1))},
                "spearman_rho": {"mean": float(np.mean(seller_pagerank_corr_spearman)), "std": float(np.std(seller_pagerank_corr_spearman, ddof=1))},
            }
        }
    }

    updated_results["group_ablation"] = group_ablation_summary
    updated_results["full_family_holm_correction"] = holm_results

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(updated_results, f, indent=2)

    frontend_path = os.path.join(_ROOT_DIR, "frontend", "public", "results.json")
    if os.path.exists(os.path.dirname(frontend_path)):
        with open(frontend_path, "w", encoding="utf-8") as f:
            json.dump(updated_results, f, indent=2)

    print(f"\nSuccessfully persisted updated results to {results_path} and {frontend_path}")


if __name__ == "__main__":
    run_experiment()
