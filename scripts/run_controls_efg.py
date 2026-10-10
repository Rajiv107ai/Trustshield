"""Empirical evaluation of Controls (e), (f), (g) across 20 seeds.

Controls defined:
(e) tabular + plain pairwise counts, no graph code:
    n_buyers_sharing_device_prior, n_buyers_sharing_address_prior
    (distinct other buyers whose sharing-log first_seen < order_date, split by edge kind)
(f) tabular + share_degree only
(g) tabular + share_component_size only

Also evaluates:
(b) baseline tabular (with device)
(c) full tabular + all 8 graph features
Tabular + S (share_degree + share_component_size)

Runs across 20 seeds with identical temporal split rules.
Reports paired differences vs (b) and (c vs e): mean diff, 95% CI, p-value.
Answers: whether the gain from S is more than 'how many accounts share my device/address'.
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
from baseline_model import build_features as build_tabular_features, TRAIN_END, VAL_END
from graph_features import (
    attach_relationship_snapshot_features,
    build_weekly_relationship_snapshots,
    build_monthly_snapshots,
    attach_snapshot_features,
    add_edge_weight_before,
)

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


def build_weekly_pairwise_sharing_snapshots(
    address_sharing_log: pd.DataFrame,
    device_sharing_log: pd.DataFrame,
    min_date: pd.Timestamp = SIM_START,
    max_date: pd.Timestamp = pd.Timestamp("2025-12-31"),
    freq: str = "W-MON",
) -> pd.DataFrame:
    """Builds weekly snapshots of plain pairwise sharing counts with NO graph code."""
    dev_pairs = pd.concat([
        device_sharing_log[["buyer_id", "shared_with_buyer_id", "first_seen_date"]],
        device_sharing_log[["shared_with_buyer_id", "buyer_id", "first_seen_date"]].rename(
            columns={"shared_with_buyer_id": "buyer_id", "buyer_id": "shared_with_buyer_id"}
        )
    ]).drop_duplicates()

    addr_pairs = pd.concat([
        address_sharing_log[["buyer_id", "shared_with_buyer_id", "first_seen_date"]],
        address_sharing_log[["shared_with_buyer_id", "buyer_id", "first_seen_date"]].rename(
            columns={"shared_with_buyer_id": "buyer_id", "buyer_id": "shared_with_buyer_id"}
        )
    ]).drop_duplicates()

    dev_pairs["first_seen_date"] = pd.to_datetime(dev_pairs["first_seen_date"])
    addr_pairs["first_seen_date"] = pd.to_datetime(addr_pairs["first_seen_date"])

    snapshots = pd.date_range(start=min_date, end=max_date, freq=freq)
    rows = []

    for cutoff in snapshots:
        sub_dev = dev_pairs[dev_pairs["first_seen_date"] < cutoff]
        sub_addr = addr_pairs[addr_pairs["first_seen_date"] < cutoff]

        dev_counts = sub_dev.groupby("buyer_id")["shared_with_buyer_id"].nunique().to_dict()
        addr_counts = sub_addr.groupby("buyer_id")["shared_with_buyer_id"].nunique().to_dict()

        all_buyers = set(dev_counts.keys()) | set(addr_counts.keys())
        for b in all_buyers:
            rows.append({
                "snapshot_date": cutoff,
                "buyer_id": b,
                "n_buyers_sharing_device_prior": float(dev_counts.get(b, 0.0)),
                "n_buyers_sharing_address_prior": float(addr_counts.get(b, 0.0)),
            })

    if not rows:
        return pd.DataFrame(columns=["snapshot_date", "buyer_id", "n_buyers_sharing_device_prior", "n_buyers_sharing_address_prior"])

    snap_df = pd.DataFrame(rows)
    snap_df["snapshot_date"] = pd.to_datetime(snap_df["snapshot_date"]).dt.as_unit("ns")
    return snap_df.sort_values("snapshot_date").reset_index(drop=True)


def attach_pairwise_sharing_snapshot_features(orders_df, addr_log, dev_log, sim_start=SIM_START):
    df = orders_df.copy()
    if df.empty or "order_date" not in df.columns or "buyer_id" not in df.columns:
        df["n_buyers_sharing_device_prior"] = 0.0
        df["n_buyers_sharing_address_prior"] = 0.0
        return df

    min_date = pd.to_datetime(df["order_date"]).min()
    max_date = pd.to_datetime(df["order_date"]).max()
    snap_min = min(pd.Timestamp(sim_start), pd.Timestamp(min_date))

    snap_df = build_weekly_pairwise_sharing_snapshots(
        addr_log, dev_log, min_date=snap_min, max_date=max_date
    )

    if snap_df.empty:
        df["n_buyers_sharing_device_prior"] = 0.0
        df["n_buyers_sharing_address_prior"] = 0.0
        return df

    orders_sub = df[["buyer_id", "order_date"]].copy()
    orders_sub["_orig_idx"] = orders_sub.index
    orders_sub["order_date"] = pd.to_datetime(orders_sub["order_date"]).dt.as_unit("ns")

    orders_sub = orders_sub.sort_values("order_date").reset_index(drop=True)
    snap_df = snap_df.sort_values("snapshot_date").reset_index(drop=True)

    merged = pd.merge_asof(
        orders_sub,
        snap_df,
        by="buyer_id",
        left_on="order_date",
        right_on="snapshot_date",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.sort_values("_orig_idx").reset_index(drop=True)

    df["n_buyers_sharing_device_prior"] = merged["n_buyers_sharing_device_prior"].fillna(0.0).to_numpy()
    df["n_buyers_sharing_address_prior"] = merged["n_buyers_sharing_address_prior"].fillna(0.0).to_numpy()
    return df


def paired_summary(a, b):
    diff = np.array(a) - np.array(b)
    n = len(diff)
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se_d = std_d / np.sqrt(n) if n > 1 else 0.0
    t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 1.96
    ci = [mean_d - t_crit * se_d, mean_d + t_crit * se_d]
    cohens_d = (mean_d / std_d) if std_d > 1e-9 else 0.0
    t_res = stats.ttest_rel(a, b) if (std_d > 1e-9 and n > 1) else None
    p_val = float(t_res.pvalue) if t_res is not None and not np.isnan(t_res.pvalue) else 1.0
    t_stat = float(t_res.statistic) if t_res is not None and not np.isnan(t_res.statistic) else 0.0
    return {
        "mean_diff": mean_d,
        "std_diff": std_d,
        "ci_95": ci,
        "cohens_d": cohens_d,
        "t_stat": t_stat,
        "p_value": p_val,
    }


def train_xgb(X_tr, y_tr, random_state=42):
    pos = int(y_tr.sum())
    neg = len(y_tr) - pos
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=neg / max(pos, 1),
        eval_metric="aucpr", random_state=random_state, n_jobs=-1, verbosity=0
    )
    clf.fit(X_tr.fillna(0.0), y_tr)
    return clf


def run_controls_experiment():
    print("=" * 80)
    print("CONTROLS (e), (f), (g) EVALUATION ACROSS 20 SEEDS")
    print("=" * 80)

    # Models tracked:
    # (b): tabular baseline (with device)
    # (c): full tabular + graph (all 8 graph features)
    # (e): tabular + n_buyers_sharing_device_prior + n_buyers_sharing_address_prior
    # (f): tabular + share_degree only
    # (g): tabular + share_component_size only
    # S:   tabular + share_degree + share_component_size

    metrics = {
        "roc_b": [], "pr_b": [],
        "roc_c": [], "pr_c": [],
        "roc_e": [], "pr_e": [],
        "roc_f": [], "pr_f": [],
        "roc_g": [], "pr_g": [],
        "roc_s": [], "pr_s": [],
    }

    t0 = time.perf_counter()

    for idx, seed in enumerate(SEEDS_20):
        s_t0 = time.perf_counter()
        pipe = generate_full_pipeline(seed=seed, ring_coherent=False)
        res = pipe["result"]
        cat = pipe["catalog"]
        txn = pipe["txn"]
        base = pipe["base"]

        orders = res["orders"].copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"])

        # 1. Base tabular features
        df, tabular_cols = build_tabular_features(
            res["orders"], res["listings"], res["returns"],
            txn["buyers"], cat["sellers"], cat["products"]
        )
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["y"] = df["is_fraudulent"].astype(int)

        # 2. Attach pairwise counts for Control (e)
        df = attach_pairwise_sharing_snapshot_features(
            df, base["address_sharing_log"], base["device_sharing_log"]
        )
        cols_e = tabular_cols + ["n_buyers_sharing_device_prior", "n_buyers_sharing_address_prior"]

        # 3. Attach relationship features for (f), (g), S, and (c)
        df = attach_relationship_snapshot_features(
            df, base["address_sharing_log"], base["device_sharing_log"]
        )
        cols_f = tabular_cols + ["share_degree"]
        cols_g = tabular_cols + ["share_component_size"]
        cols_s = tabular_cols + ["share_degree", "share_component_size"]

        # 4. Attach remaining graph features for (c)
        snapshots, months = build_monthly_snapshots(
            orders.assign(order_date=pd.to_datetime(orders["order_date"])),
            SIM_START,
        )
        df = attach_snapshot_features(df, snapshots, months)
        df = add_edge_weight_before(df)
        cols_c = tabular_cols + ALL_GRAPH_COLS

        # Split: train <= 2025-08-31; test: 2025-11-01 to 2025-12-10 (truncated test horizon)
        train = df[df["order_date"] <= TRAIN_END]
        test = df[(df["order_date"] > VAL_END) & (df["order_date"] <= pd.Timestamp("2025-12-10"))]

        y_train = train["y"]
        y_test = test["y"].to_numpy()

        # Fit models
        m_b = train_xgb(train[tabular_cols], y_train)
        m_c = train_xgb(train[cols_c], y_train)
        m_e = train_xgb(train[cols_e], y_train)
        m_f = train_xgb(train[cols_f], y_train)
        m_g = train_xgb(train[cols_g], y_train)
        m_s = train_xgb(train[cols_s], y_train)

        # Predict
        p_b = m_b.predict_proba(test[tabular_cols].fillna(0.0))[:, 1]
        p_c = m_c.predict_proba(test[cols_c].fillna(0.0))[:, 1]
        p_e = m_e.predict_proba(test[cols_e].fillna(0.0))[:, 1]
        p_f = m_f.predict_proba(test[cols_f].fillna(0.0))[:, 1]
        p_g = m_g.predict_proba(test[cols_g].fillna(0.0))[:, 1]
        p_s = m_s.predict_proba(test[cols_s].fillna(0.0))[:, 1]

        # Record metrics
        metrics["roc_b"].append(float(roc_auc_score(y_test, p_b)))
        metrics["pr_b"].append(float(average_precision_score(y_test, p_b)))
        metrics["roc_c"].append(float(roc_auc_score(y_test, p_c)))
        metrics["pr_c"].append(float(average_precision_score(y_test, p_c)))
        metrics["roc_e"].append(float(roc_auc_score(y_test, p_e)))
        metrics["pr_e"].append(float(average_precision_score(y_test, p_e)))
        metrics["roc_f"].append(float(roc_auc_score(y_test, p_f)))
        metrics["pr_f"].append(float(average_precision_score(y_test, p_f)))
        metrics["roc_g"].append(float(roc_auc_score(y_test, p_g)))
        metrics["pr_g"].append(float(average_precision_score(y_test, p_g)))
        metrics["roc_s"].append(float(roc_auc_score(y_test, p_s)))
        metrics["pr_s"].append(float(average_precision_score(y_test, p_s)))

        el = time.perf_counter() - s_t0
        print(f"Seed {seed:4d} [{idx+1:2d}/20] ({el:.1f}s) | ROC (b): {metrics['roc_b'][-1]:.4f}, (e): {metrics['roc_e'][-1]:.4f}, (f): {metrics['roc_f'][-1]:.4f}, (g): {metrics['roc_g'][-1]:.4f}, (s): {metrics['roc_s'][-1]:.4f}, (c): {metrics['roc_c'][-1]:.4f}")

    tot_el = time.perf_counter() - t0
    print(f"\nAll 20 seeds completed in {tot_el:.1f}s.")

    # Summarize models
    print("\n" + "=" * 80)
    print("SUMMARY METRICS OVER 20 SEEDS (TRUNCATED TEST HORIZON)")
    print("=" * 80)
    models_summary = {
        "(b) Baseline Tabular": ("roc_b", "pr_b"),
        "(e) Tabular + Plain Pairwise Counts": ("roc_e", "pr_e"),
        "(f) Tabular + share_degree only": ("roc_f", "pr_f"),
        "(g) Tabular + share_component_size only": ("roc_g", "pr_g"),
        "(S) Tabular + S (degree + component size)": ("roc_s", "pr_s"),
        "(c) Full Tabular + Graph (8 features)": ("roc_c", "pr_c"),
    }
    for label, (rk, pk) in models_summary.items():
        mr = np.mean(metrics[rk])
        sr = np.std(metrics[rk], ddof=1)
        mp = np.mean(metrics[pk])
        sp = np.std(metrics[pk], ddof=1)
        print(f"{label:<45} | ROC = {mr:.4f} +/- {sr:.4f} | PR = {mp:.4f} +/- {sp:.4f}")

    # Paired comparisons
    print("\n" + "=" * 80)
    print("PAIRED COMPARISONS (vs Baseline b and c vs e)")
    print("=" * 80)

    paired_tests = [
        ("Control (e) vs Baseline (b)", metrics["roc_e"], metrics["roc_b"], metrics["pr_e"], metrics["pr_b"]),
        ("Control (f) vs Baseline (b)", metrics["roc_f"], metrics["roc_b"], metrics["pr_f"], metrics["pr_b"]),
        ("Control (g) vs Baseline (b)", metrics["roc_g"], metrics["roc_b"], metrics["pr_g"], metrics["pr_b"]),
        ("Group (S) vs Baseline (b)", metrics["roc_s"], metrics["roc_b"], metrics["pr_s"], metrics["pr_b"]),
        ("Full (c) vs Control (e)", metrics["roc_c"], metrics["roc_e"], metrics["pr_c"], metrics["pr_e"]),
        ("Full (c) vs Baseline (b)", metrics["roc_c"], metrics["roc_b"], metrics["pr_c"], metrics["pr_b"]),
        ("Group (S) vs Control (e)", metrics["roc_s"], metrics["roc_e"], metrics["pr_s"], metrics["pr_e"]),
    ]

    paired_results = {}
    for name, r_new, r_base, p_new, p_base in paired_tests:
        r_sum = paired_summary(r_new, r_base)
        p_sum = paired_summary(p_new, p_base)
        paired_results[name] = {"roc": r_sum, "pr": p_sum}
        print(f"{name}:")
        print(f"  ROC Lift: {r_sum['mean_diff']:+.4f} | 95% CI: [{r_sum['ci_95'][0]:+.4f}, {r_sum['ci_95'][1]:+.4f}] | t = {r_sum['t_stat']:.2f} | p = {r_sum['p_value']:.4e}")
        print(f"  PR  Lift: {p_sum['mean_diff']:+.4f} | 95% CI: [{p_sum['ci_95'][0]:+.4f}, {p_sum['ci_95'][1]:+.4f}] | t = {p_sum['t_stat']:.2f} | p = {p_sum['p_value']:.4e}")

    # Save to json
    out_obj = {
        "seeds": SEEDS_20,
        "metrics_per_seed": metrics,
        "models_summary": {
            label: {
                "roc_mean": float(np.mean(metrics[rk])), "roc_std": float(np.std(metrics[rk], ddof=1)),
                "pr_mean": float(np.mean(metrics[pk])), "pr_std": float(np.std(metrics[pk], ddof=1)),
            }
            for label, (rk, pk) in models_summary.items()
        },
        "paired_comparisons": paired_results,
    }
    with open(os.path.join(_ROOT_DIR, "results", "controls_efg_results.json"), "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)
    print("\nResults saved to results/controls_efg_results.json")


if __name__ == "__main__":
    run_controls_experiment()
