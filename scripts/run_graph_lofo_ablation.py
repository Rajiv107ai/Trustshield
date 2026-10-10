"""Graph Feature Leakage Audit & Leave-One-Feature-Out (LOFO) Ablation across 20 Seeds.

Executes:
1. Audit verification of all features in graph_cols.
2. 20-seed LOFO ablation for each of the 8 graph features.
3. Paired statistics (mean difference, 95% CI, p-value) against full variant (c).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.join(BASE_DIR, "trustshield_project")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from entity_generator import generate_full_pipeline, SIM_START, SIM_END
from baseline_model import build_features as build_tabular_features, TRAIN_END, VAL_END, leakage_audit
from graph_features import (
    attach_relationship_snapshot_features,
    build_monthly_snapshots,
    attach_snapshot_features,
    add_edge_weight_before,
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

SEEDS_20 = [
    42, 101, 202, 303, 404, 505, 606, 707, 808, 909,
    1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809, 1910
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


def run_lofo_ablation():
    print("=" * 80)
    print("LEAVE-ONE-FEATURE-OUT (LOFO) ABLATION OVER 20 SEEDS (STANDARD DATASET)")
    print("=" * 80)

    # Containers for results across 20 seeds
    # full: variant (c)
    full_roc = []
    full_pr = []

    # LOFO: without each feature
    lofo_roc = {col: [] for col in GRAPH_COLS}
    lofo_pr = {col: [] for col in GRAPH_COLS}

    # Tabular baseline for reference (variant b)
    tab_roc = []
    tab_pr = []

    t_start = time.time()

    for idx, seed in enumerate(SEEDS_20, 1):
        s_t0 = time.time()
        print(f"[{idx:02d}/20] Seed {seed}...", end="", flush=True)

        pipe = generate_full_pipeline(seed=seed, ring_coherent=False)
        result = pipe["result"]
        catalog = pipe["catalog"]
        txn = pipe["txn"]
        base = pipe["base"]

        df, tabular_cols = build_tabular_features(
            result["orders"], result["listings"], result["returns"],
            txn["buyers"], catalog["sellers"], catalog["products"]
        )
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["y"] = df["is_fraudulent"].astype(int)

        # Audit
        leakage_audit(df, tabular_cols)

        # Attach graph features
        df = attach_relationship_snapshot_features(df, base["address_sharing_log"], base["device_sharing_log"])
        snapshots, months = build_monthly_snapshots(
            result["orders"].assign(order_date=pd.to_datetime(result["orders"]["order_date"])),
            SIM_START,
        )
        df = attach_snapshot_features(df, snapshots, months)
        df = add_edge_weight_before(df)

        train = df[df["order_date"] <= TRAIN_END]
        test = df[df["order_date"] > VAL_END]
        y_train = train["y"]
        y_test = test["y"].to_numpy()

        # Tabular baseline (variant b)
        m_tab = train_xgb(train[tabular_cols], y_train, random_state=42)
        p_tab = m_tab.predict_proba(test[tabular_cols].fillna(0.0))[:, 1]
        r_tab = float(roc_auc_score(y_test, p_tab))
        pr_tab_val = float(average_precision_score(y_test, p_tab))
        tab_roc.append(r_tab)
        tab_pr.append(pr_tab_val)

        # Full Variant (c): all tabular + all 8 graph cols
        all_cols = tabular_cols + GRAPH_COLS
        m_full = train_xgb(train[all_cols], y_train, random_state=42)
        p_full = m_full.predict_proba(test[all_cols].fillna(0.0))[:, 1]
        r_full = float(roc_auc_score(y_test, p_full))
        pr_full_val = float(average_precision_score(y_test, p_full))
        full_roc.append(r_full)
        full_pr.append(pr_full_val)

        # Leave-one-feature-out evaluations
        for col_to_drop in GRAPH_COLS:
            sub_graph_cols = [c for c in GRAPH_COLS if c != col_to_drop]
            eval_cols = tabular_cols + sub_graph_cols
            m_lofo = train_xgb(train[eval_cols], y_train, random_state=42)
            p_lofo = m_lofo.predict_proba(test[eval_cols].fillna(0.0))[:, 1]
            r_lofo = float(roc_auc_score(y_test, p_lofo))
            pr_lofo_val = float(average_precision_score(y_test, p_lofo))
            lofo_roc[col_to_drop].append(r_lofo)
            lofo_pr[col_to_drop].append(pr_lofo_val)

        elapsed = time.time() - s_t0
        print(f" done in {elapsed:.1f}s | Full ROC: {r_full:.4f}, PR: {pr_full_val:.4f}")

    total_time = time.time() - t_start
    print(f"\nAll 20 seeds completed in {total_time:.1f}s.\n")

    # Summary table
    print("=" * 100)
    print("LEAVE-ONE-FEATURE-OUT (LOFO) ABLATION RESULTS OVER 20 SEEDS")
    print("=" * 100)
    print(f"Full Variant (c) [8 graph cols]: ROC = {np.mean(full_roc):.4f} +/- {np.std(full_roc, ddof=1):.4f} | PR = {np.mean(full_pr):.4f} +/- {np.std(full_pr, ddof=1):.4f}")
    print(f"Tabular Baseline (b) [0 graph cols]: ROC = {np.mean(tab_roc):.4f} +/- {np.std(tab_roc, ddof=1):.4f} | PR = {np.mean(tab_pr):.4f} +/- {np.std(tab_pr, ddof=1):.4f}")
    print("-" * 100)
    print(f"{'Dropped Graph Feature':<35} | {'ROC (Without)':<18} | {'ROC Drop (Loss)':<18} | {'95% CI of Loss':<22} | {'p-value':<10}")
    print("-" * 100)

    ablation_summary = {}

    for col in GRAPH_COLS:
        wo_roc = lofo_roc[col]
        wo_pr = lofo_pr[col]

        # Marginal contribution of feature: Full - Without
        # Positive diff means dropping the feature hurt performance (i.e. feature is valuable)
        roc_loss_stats = paired_stats_calc(wo_roc, full_roc)
        pr_loss_stats = paired_stats_calc(wo_pr, full_pr)

        mean_wo = float(np.mean(wo_roc))
        std_wo = float(np.std(wo_roc, ddof=1))

        mean_loss = roc_loss_stats["mean_diff"]
        ci_str = f"[{roc_loss_stats['ci_95'][0]:+.4f}, {roc_loss_stats['ci_95'][1]:+.4f}]"
        p_val = roc_loss_stats["p_value"]

        print(f"{col:<35} | {mean_wo:.4f} +/- {std_wo:.4f}  | {mean_loss:+.4f}            | {ci_str:<22} | {p_val:.4e}")

        ablation_summary[col] = {
            "without_feature_roc": {"mean": mean_wo, "std": std_wo},
            "roc_marginal_loss": roc_loss_stats,
            "pr_marginal_loss": pr_loss_stats,
        }

    # Save results to json
    out_path = os.path.join(BASE_DIR, "results", "graph_lofo_ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_seeds": 20,
            "seeds": SEEDS_20,
            "full_variant_c": {
                "roc": {"mean": float(np.mean(full_roc)), "std": float(np.std(full_roc, ddof=1)), "seeds": full_roc},
                "pr": {"mean": float(np.mean(full_pr)), "std": float(np.std(full_pr, ddof=1)), "seeds": full_pr},
            },
            "tabular_baseline_b": {
                "roc": {"mean": float(np.mean(tab_roc)), "std": float(np.std(tab_roc, ddof=1)), "seeds": tab_roc},
                "pr": {"mean": float(np.mean(tab_pr)), "std": float(np.std(tab_pr, ddof=1)), "seeds": tab_pr},
            },
            "lofo_ablation": ablation_summary,
        }, f, indent=2)

    print(f"\nSaved LOFO ablation summary to {out_path}")
    print("=" * 100)


if __name__ == "__main__":
    run_lofo_ablation()
