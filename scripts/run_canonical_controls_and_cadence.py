"""Canonical 20-seed evaluation of Controls (e), (e_weekly), (f), (g), and (S) on the Truncated Horizon.

Runs across the 20 canonical seeds [42, 101, 202, ..., 1910] using the canonical
truncated test horizon (VAL_END: 2025-10-31 to TEST_END: 2025-12-10).

Models evaluated:
  (b) Tabular baseline (with device)
  (c) Full Tabular + 8 Graph features
  (e) Tabular + Order-level PIT pairwise sharing counts (device & address)
  (e_weekly) Tabular + Weekly snapshot pairwise sharing counts (matching S cadence)
  (f) Tabular + share_degree only
  (g) Tabular + share_component_size only
  (S) Tabular + share_degree + share_component_size

Answers:
  1. Is the gain from S more than 'how many accounts share my device/address'?
  2. Is the advantage of plain counts over S due to cadence (PIT vs weekly snapshot)?
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

from entity_generator import generate_full_pipeline, SIM_START, SIM_END
from baseline_model import build_features as build_tabular_features, TRAIN_END, VAL_END
from graph_features import (
    attach_relationship_snapshot_features,
    build_monthly_snapshots,
    attach_snapshot_features,
    add_edge_weight_before,
)

SEEDS_20 = [
    42, 101, 202, 303, 404, 505, 606, 707, 808, 909,
    1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809, 1910
]

TEST_END = pd.to_datetime(SIM_END) - pd.Timedelta(days=21)  # 2025-12-10

ALL_GRAPH_COLS = [
    "share_degree", "share_component_size",
    "buyer_seller_degree", "buyer_pagerank",
    "seller_buyer_degree", "seller_pagerank",
    "seller_buyer_concentration_hhi",
    "buyer_seller_edge_weight_before",
]


def attach_point_in_time_pairwise_counts(orders_df: pd.DataFrame, addr_log: pd.DataFrame, dev_log: pd.DataFrame) -> pd.DataFrame:
    """Computes order-level point-in-time distinct shared buyer counts strictly before order_date."""
    df = orders_df.copy()
    for log_df, col_name in [(dev_log, "n_buyers_sharing_device_prior"), (addr_log, "n_buyers_sharing_address_prior")]:
        pairs = pd.concat([
            log_df[["buyer_id", "shared_with_buyer_id", "first_seen_date"]],
            log_df[["shared_with_buyer_id", "buyer_id", "first_seen_date"]].rename(
                columns={"shared_with_buyer_id": "buyer_id", "buyer_id": "shared_with_buyer_id"}
            )
        ]).drop_duplicates()
        pairs["first_seen_date"] = pd.to_datetime(pairs["first_seen_date"]).dt.as_unit("ns")

        earliest = pairs.groupby(["buyer_id", "shared_with_buyer_id"], as_index=False)["first_seen_date"].min()
        earliest = earliest.sort_values("first_seen_date").reset_index(drop=True)
        earliest["_cum_buyers"] = earliest.groupby("buyer_id").cumcount() + 1

        ord_sub = df[["buyer_id", "order_date"]].copy()
        ord_sub["_orig_idx"] = ord_sub.index
        ord_sub["order_date"] = pd.to_datetime(ord_sub["order_date"]).dt.as_unit("ns")
        ord_sub = ord_sub.sort_values("order_date").reset_index(drop=True)

        merged = pd.merge_asof(
            ord_sub,
            earliest[["buyer_id", "first_seen_date", "_cum_buyers"]],
            by="buyer_id",
            left_on="order_date",
            right_on="first_seen_date",
            direction="backward",
            allow_exact_matches=False,
        ).sort_values("_orig_idx").reset_index(drop=True)

        df[col_name] = merged["_cum_buyers"].fillna(0.0).to_numpy()

    return df


def attach_weekly_snapshot_pairwise_counts(orders_df: pd.DataFrame, addr_log: pd.DataFrame, dev_log: pd.DataFrame, min_date: pd.Timestamp = SIM_START, max_date: pd.Timestamp = pd.Timestamp("2025-12-31")) -> pd.DataFrame:
    """Computes weekly snapshot distinct shared buyer counts matching S snapshot schedule."""
    df = orders_df.copy()
    dev_pairs = pd.concat([
        dev_log[["buyer_id", "shared_with_buyer_id", "first_seen_date"]],
        dev_log[["shared_with_buyer_id", "buyer_id", "first_seen_date"]].rename(
            columns={"shared_with_buyer_id": "buyer_id", "buyer_id": "shared_with_buyer_id"}
        )
    ]).drop_duplicates()
    addr_pairs = pd.concat([
        addr_log[["buyer_id", "shared_with_buyer_id", "first_seen_date"]],
        addr_log[["shared_with_buyer_id", "buyer_id", "first_seen_date"]].rename(
            columns={"shared_with_buyer_id": "buyer_id", "buyer_id": "shared_with_buyer_id"}
        )
    ]).drop_duplicates()

    dev_pairs["first_seen_date"] = pd.to_datetime(dev_pairs["first_seen_date"]).dt.as_unit("ns")
    addr_pairs["first_seen_date"] = pd.to_datetime(addr_pairs["first_seen_date"]).dt.as_unit("ns")

    snapshots = pd.date_range(start=min_date, end=max_date, freq="W-MON")
    rows = []
    for cutoff in snapshots:
        c_ns = pd.to_datetime(cutoff).as_unit("ns")
        sub_dev = dev_pairs[dev_pairs["first_seen_date"] < c_ns]
        sub_addr = addr_pairs[addr_pairs["first_seen_date"] < c_ns]
        d_cnts = sub_dev.groupby("buyer_id")["shared_with_buyer_id"].nunique().to_dict()
        a_cnts = sub_addr.groupby("buyer_id")["shared_with_buyer_id"].nunique().to_dict()
        for b in set(d_cnts.keys()) | set(a_cnts.keys()):
            rows.append({
                "snapshot_date": c_ns,
                "buyer_id": b,
                "n_buyers_sharing_device_weekly": float(d_cnts.get(b, 0.0)),
                "n_buyers_sharing_address_weekly": float(a_cnts.get(b, 0.0)),
            })

    snap_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["snapshot_date", "buyer_id", "n_buyers_sharing_device_weekly", "n_buyers_sharing_address_weekly"])
    if snap_df.empty:
        df["n_buyers_sharing_device_weekly"] = 0.0
        df["n_buyers_sharing_address_weekly"] = 0.0
        return df

    snap_df["snapshot_date"] = pd.to_datetime(snap_df["snapshot_date"]).dt.as_unit("ns")
    snap_df = snap_df.sort_values("snapshot_date").reset_index(drop=True)

    ord_sub = df[["buyer_id", "order_date"]].copy()
    ord_sub["_orig_idx"] = ord_sub.index
    ord_sub["order_date"] = pd.to_datetime(ord_sub["order_date"]).dt.as_unit("ns")
    ord_sub = ord_sub.sort_values("order_date").reset_index(drop=True)

    merged = pd.merge_asof(
        ord_sub, snap_df,
        by="buyer_id", left_on="order_date", right_on="snapshot_date",
        direction="backward", allow_exact_matches=False,
    ).sort_values("_orig_idx").reset_index(drop=True)

    df["n_buyers_sharing_device_weekly"] = merged["n_buyers_sharing_device_weekly"].fillna(0.0).to_numpy()
    df["n_buyers_sharing_address_weekly"] = merged["n_buyers_sharing_address_weekly"].fillna(0.0).to_numpy()
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


def main():
    print("=" * 80)
    print("CANONICAL 20-SEED CONTROLS & CADENCE EVALUATION (TRUNCATED HORIZON)")
    print(f"Horizon: {VAL_END.strftime('%Y-%m-%d')} to {TEST_END.strftime('%Y-%m-%d')} (final 21 days excluded)")
    print(f"Seeds: {SEEDS_20}")
    print("=" * 80)

    arrays = {
        "roc_b": [], "pr_b": [],
        "roc_c": [], "pr_c": [],
        "roc_e": [], "pr_e": [],
        "roc_e_weekly": [], "pr_e_weekly": [],
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

        df, tabular_cols = build_tabular_features(
            res["orders"], res["listings"], res["returns"],
            txn["buyers"], cat["sellers"], cat["products"]
        )
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["y"] = df["is_fraudulent"].astype(int)

        # 1. Point-in-time pairwise counts for (e)
        df = attach_point_in_time_pairwise_counts(df, base["address_sharing_log"], base["device_sharing_log"])
        cols_e = tabular_cols + ["n_buyers_sharing_device_prior", "n_buyers_sharing_address_prior"]

        # 2. Weekly snapshot pairwise counts for (e_weekly)
        df = attach_weekly_snapshot_pairwise_counts(df, base["address_sharing_log"], base["device_sharing_log"])
        cols_e_weekly = tabular_cols + ["n_buyers_sharing_device_weekly", "n_buyers_sharing_address_weekly"]

        # 3. Relationship features for (f), (g), (S), (c)
        df = attach_relationship_snapshot_features(df, base["address_sharing_log"], base["device_sharing_log"])
        cols_f = tabular_cols + ["share_degree"]
        cols_g = tabular_cols + ["share_component_size"]
        cols_s = tabular_cols + ["share_degree", "share_component_size"]

        # 4. Remaining graph features for (c)
        snapshots, months = build_monthly_snapshots(orders, SIM_START)
        df = attach_snapshot_features(df, snapshots, months)
        df = add_edge_weight_before(df)
        cols_c = tabular_cols + ALL_GRAPH_COLS

        # Split: Truncated test split
        train = df[df["order_date"] <= TRAIN_END]
        test = df[(df["order_date"] > VAL_END) & (df["order_date"] <= TEST_END)]

        y_tr = train["y"]
        y_test = test["y"].to_numpy()

        # Fit models with random_state=seed matching canonical benchmark
        m_b = train_xgb(train[tabular_cols], y_tr, random_state=seed)
        m_c = train_xgb(train[cols_c], y_tr, random_state=seed)
        m_e = train_xgb(train[cols_e], y_tr, random_state=seed)
        m_ew = train_xgb(train[cols_e_weekly], y_tr, random_state=seed)
        m_f = train_xgb(train[cols_f], y_tr, random_state=seed)
        m_g = train_xgb(train[cols_g], y_tr, random_state=seed)
        m_s = train_xgb(train[cols_s], y_tr, random_state=seed)

        p_b = m_b.predict_proba(test[tabular_cols].fillna(0.0))[:, 1]
        p_c = m_c.predict_proba(test[cols_c].fillna(0.0))[:, 1]
        p_e = m_e.predict_proba(test[cols_e].fillna(0.0))[:, 1]
        p_ew = m_ew.predict_proba(test[cols_e_weekly].fillna(0.0))[:, 1]
        p_f = m_f.predict_proba(test[cols_f].fillna(0.0))[:, 1]
        p_g = m_g.predict_proba(test[cols_g].fillna(0.0))[:, 1]
        p_s = m_s.predict_proba(test[cols_s].fillna(0.0))[:, 1]

        arrays["roc_b"].append(float(roc_auc_score(y_test, p_b)))
        arrays["pr_b"].append(float(average_precision_score(y_test, p_b)))
        arrays["roc_c"].append(float(roc_auc_score(y_test, p_c)))
        arrays["pr_c"].append(float(average_precision_score(y_test, p_c)))
        arrays["roc_e"].append(float(roc_auc_score(y_test, p_e)))
        arrays["pr_e"].append(float(average_precision_score(y_test, p_e)))
        arrays["roc_e_weekly"].append(float(roc_auc_score(y_test, p_ew)))
        arrays["pr_e_weekly"].append(float(average_precision_score(y_test, p_ew)))
        arrays["roc_f"].append(float(roc_auc_score(y_test, p_f)))
        arrays["pr_f"].append(float(average_precision_score(y_test, p_f)))
        arrays["roc_g"].append(float(roc_auc_score(y_test, p_g)))
        arrays["pr_g"].append(float(average_precision_score(y_test, p_g)))
        arrays["roc_s"].append(float(roc_auc_score(y_test, p_s)))
        arrays["pr_s"].append(float(average_precision_score(y_test, p_s)))

        el = time.perf_counter() - s_t0
        print(f"Seed {seed:4d} [{idx+1:2d}/20] ({el:.1f}s) | ROC: b={arrays['roc_b'][-1]:.4f}, e={arrays['roc_e'][-1]:.4f}, e_w={arrays['roc_e_weekly'][-1]:.4f}, S={arrays['roc_s'][-1]:.4f}, c={arrays['roc_c'][-1]:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("MODEL SUMMARY ON TRUNCATED TEST HORIZON (N=20)")
    print("=" * 80)
    models = {
        "(b) Baseline Tabular": ("roc_b", "pr_b"),
        "(e) Tabular + PIT Pairwise Counts": ("roc_e", "pr_e"),
        "(e_weekly) Tabular + Weekly Pairwise Counts": ("roc_e_weekly", "pr_e_weekly"),
        "(f) Tabular + share_degree only": ("roc_f", "pr_f"),
        "(g) Tabular + share_component_size only": ("roc_g", "pr_g"),
        "(S) Tabular + S (degree + CC size)": ("roc_s", "pr_s"),
        "(c) Full Tabular + Graph (8 features)": ("roc_c", "pr_c"),
    }
    summary_dict = {}
    for name, (r_key, p_key) in models.items():
        r_arr, p_arr = arrays[r_key], arrays[p_key]
        summary_dict[name] = {
            "roc_mean": float(np.mean(r_arr)), "roc_std": float(np.std(r_arr, ddof=1)),
            "pr_mean": float(np.mean(p_arr)), "pr_std": float(np.std(p_arr, ddof=1)),
        }
        print(f"{name:<45} | ROC: {np.mean(r_arr):.6f} +/- {np.std(r_arr, ddof=1):.6f} | PR: {np.mean(p_arr):.6f} +/- {np.std(p_arr, ddof=1):.6f}")

    print("\n" + "=" * 80)
    print("PAIRED COMPARISONS (N=20 SEEDS)")
    print("=" * 80)
    paired_comps = {
        "Control (e) vs Baseline (b)": (arrays["roc_e"], arrays["roc_b"], arrays["pr_e"], arrays["pr_b"]),
        "Control (e_weekly) vs Baseline (b)": (arrays["roc_e_weekly"], arrays["roc_b"], arrays["pr_e_weekly"], arrays["pr_b"]),
        "Control (f) vs Baseline (b)": (arrays["roc_f"], arrays["roc_b"], arrays["pr_f"], arrays["pr_b"]),
        "Control (g) vs Baseline (b)": (arrays["roc_g"], arrays["roc_b"], arrays["pr_g"], arrays["pr_b"]),
        "Group (S) vs Baseline (b)": (arrays["roc_s"], arrays["roc_b"], arrays["pr_s"], arrays["pr_b"]),
        "Full (c) vs Baseline (b)": (arrays["roc_c"], arrays["roc_b"], arrays["pr_c"], arrays["pr_b"]),
        "Full (c) vs Control (e)": (arrays["roc_c"], arrays["roc_e"], arrays["pr_c"], arrays["pr_e"]),
        "Group (S) vs Control (e)": (arrays["roc_s"], arrays["roc_e"], arrays["pr_s"], arrays["pr_e"]),
        "Control (e_weekly) vs Group (S)": (arrays["roc_e_weekly"], arrays["roc_s"], arrays["pr_e_weekly"], arrays["pr_s"]),
        "Control (e_weekly) vs Control (e)": (arrays["roc_e_weekly"], arrays["roc_e"], arrays["pr_e_weekly"], arrays["pr_e"]),
    }
    paired_dict = {}
    for comp_name, (ra, rb, pa, pb) in paired_comps.items():
        res_r = paired_summary(ra, rb)
        res_p = paired_summary(pa, pb)
        paired_dict[comp_name] = {"roc": res_r, "pr": res_p}
        print(f"{comp_name}:")
        print(f"  ROC: diff={res_r['mean_diff']:+.6f}, 95% CI=[{res_r['ci_95'][0]:+.6f}, {res_r['ci_95'][1]:+.6f}], t={res_r['t_stat']:+.4f}, p={res_r['p_value']:.4e}")
        print(f"  PR:  diff={res_p['mean_diff']:+.6f}, 95% CI=[{res_p['ci_95'][0]:+.6f}, {res_p['ci_95'][1]:+.6f}], t={res_p['t_stat']:+.4f}, p={res_p['p_value']:.4e}")

    # Serialize to results/controls_efg_results.json
    out_payload = {
        "seeds": SEEDS_20,
        "test_horizon": f"{VAL_END.strftime('%Y-%m-%d')} to {TEST_END.strftime('%Y-%m-%d')} (final 21 days excluded)",
        "per_seed_arrays": arrays,
        "models_summary": summary_dict,
        "paired_comparisons": paired_dict,
    }
    out_path = os.path.join(_ROOT_DIR, "results", "controls_efg_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved updated controls and cadence benchmark to: {out_path}")

    # Also integrate into results/results.json
    rj_path = os.path.join(_ROOT_DIR, "results", "results.json")
    with open(rj_path, "r", encoding="utf-8") as f:
        rj = json.load(f)
    rj["controls_efg_cadence_truncated_20_seeds"] = out_payload
    with open(rj_path, "w", encoding="utf-8") as f:
        json.dump(rj, f, indent=2)
    print(f"Integrated controls_efg_cadence_truncated_20_seeds into: {rj_path}")

    # Copy to frontend/public/results.json to keep CI guard synchronized
    pub_path = os.path.join(_ROOT_DIR, "frontend", "public", "results.json")
    with open(pub_path, "w", encoding="utf-8") as f:
        json.dump(rj, f, indent=2)
    print(f"Synchronized with frontend/public/results.json")


if __name__ == "__main__":
    main()
