"""Threshold grid diagnosis for Seed 42 and Mean over 20 Seeds.

Computes volume% and recall% as a function of tau in:
{0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.567, 0.65}
for both raw XGBoost scores and calibrated probabilities (Isotonic).
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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
    build_monthly_snapshots,
    attach_snapshot_features,
    add_edge_weight_before,
)

SEEDS_20 = [
    42, 101, 202, 303, 404, 505, 606, 707, 808, 909,
    1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809, 1910
]

TAUS = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.567, 0.65]


def run_threshold_grid():
    print("=" * 80)
    print("THRESHOLD EVALUATION GRID: RAW VS CALIBRATED SCORES")
    print("=" * 80)

    # To store per-seed results
    # Model (b): tabular baseline
    raw_vol_b_all = {t: [] for t in TAUS}
    raw_rec_b_all = {t: [] for t in TAUS}
    cal_vol_b_all = {t: [] for t in TAUS}
    cal_rec_b_all = {t: [] for t in TAUS}

    seed_42_results = {}

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

        train = df[df["order_date"] <= TRAIN_END]
        val = df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)]
        # Canonical test split excluding final 21 days
        test = df[(df["order_date"] > VAL_END) & (df["order_date"] <= pd.Timestamp("2025-12-10"))]

        y_train = train["y"]
        y_val = val["y"].to_numpy()
        y_test = test["y"].to_numpy()

        pos = int(y_train.sum())
        neg = len(y_train) - pos

        clf_b = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=neg / max(pos, 1),
            eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0
        )
        clf_b.fit(train[tabular_cols].fillna(0.0), y_train)

        p_val_raw_b = clf_b.predict_proba(val[tabular_cols].fillna(0.0))[:, 1]
        p_test_raw_b = clf_b.predict_proba(test[tabular_cols].fillna(0.0))[:, 1]

        iso_b = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_val_raw_b, y_val)
        p_test_cal_b = iso_b.predict(p_test_raw_b)

        n_test = len(y_test)
        n_fraud = int(y_test.sum())

        s_raw_vol = {}
        s_raw_rec = {}
        s_cal_vol = {}
        s_cal_rec = {}

        for t in TAUS:
            v_raw = float(np.mean(p_test_raw_b >= t))
            r_raw = float(np.sum((y_test == 1) & (p_test_raw_b >= t)) / max(n_fraud, 1))
            v_cal = float(np.mean(p_test_cal_b >= t))
            r_cal = float(np.sum((y_test == 1) & (p_test_cal_b >= t)) / max(n_fraud, 1))

            raw_vol_b_all[t].append(v_raw)
            raw_rec_b_all[t].append(r_raw)
            cal_vol_b_all[t].append(v_cal)
            cal_rec_b_all[t].append(r_cal)

            s_raw_vol[t] = v_raw
            s_raw_rec[t] = r_raw
            s_cal_vol[t] = v_cal
            s_cal_rec[t] = r_cal

        if seed == 42:
            seed_42_results = {
                "raw_vol": s_raw_vol, "raw_rec": s_raw_rec,
                "cal_vol": s_cal_vol, "cal_rec": s_cal_rec
            }
        print(f"Seed {seed:4d} done | Test orders: {n_test}, fraud: {n_fraud}")

    print("\n" + "=" * 80)
    print("SEED 42 THRESHOLD GRID TABLE")
    print("=" * 80)
    print(f"{'tau':>8} | {'Raw Vol%':>10} | {'Raw Rec%':>10} | {'Cal Vol%':>10} | {'Cal Rec%':>10}")
    print("-" * 58)
    for t in TAUS:
        print(f"{t:8.3f} | {seed_42_results['raw_vol'][t]*100:9.2f}% | {seed_42_results['raw_rec'][t]*100:9.2f}% | {seed_42_results['cal_vol'][t]*100:9.2f}% | {seed_42_results['cal_rec'][t]*100:9.2f}%")

    # Check monotonicity for seed 42
    rec_raw_s42 = [seed_42_results['raw_rec'][t] for t in TAUS]
    rec_cal_s42 = [seed_42_results['cal_rec'][t] for t in TAUS]
    assert all(rec_raw_s42[i] >= rec_raw_s42[i+1] for i in range(len(rec_raw_s42)-1)), "Raw recall not non-increasing!"
    assert all(rec_cal_s42[i] >= rec_cal_s42[i+1] for i in range(len(rec_cal_s42)-1)), "Calibrated recall not non-increasing!"

    print("\n" + "=" * 80)
    print("20-SEED MEAN THRESHOLD GRID TABLE (N=20)")
    print("=" * 80)
    print(f"{'tau':>8} | {'Raw Vol%':>16} | {'Raw Rec%':>16} | {'Cal Vol%':>16} | {'Cal Rec%':>16}")
    print("-" * 80)
    mean_raw_vol = {}
    mean_raw_rec = {}
    mean_cal_vol = {}
    mean_cal_rec = {}

    for t in TAUS:
        mv_raw = float(np.mean(raw_vol_b_all[t]))
        mr_raw = float(np.mean(raw_rec_b_all[t]))
        mv_cal = float(np.mean(cal_vol_b_all[t]))
        mr_cal = float(np.mean(cal_rec_b_all[t]))
        sv_raw = float(np.std(raw_vol_b_all[t]))
        sr_raw = float(np.std(raw_rec_b_all[t]))
        sv_cal = float(np.std(cal_vol_b_all[t]))
        sr_cal = float(np.std(cal_rec_b_all[t]))

        mean_raw_vol[t] = mv_raw
        mean_raw_rec[t] = mr_raw
        mean_cal_vol[t] = mv_cal
        mean_cal_rec[t] = mr_cal

        print(f"{t:8.3f} | {mv_raw*100:6.2f}% +/- {sv_raw*100:4.2f}% | {mr_raw*100:6.2f}% +/- {sr_raw*100:4.2f}% | {mv_cal*100:6.2f}% +/- {sv_cal*100:4.2f}% | {mr_cal*100:6.2f}% +/- {sr_cal*100:4.2f}%")

    # Check monotonicity for 20-seed means
    m_rec_raw = [mean_raw_rec[t] for t in TAUS]
    m_rec_cal = [mean_cal_rec[t] for t in TAUS]
    assert all(m_rec_raw[i] >= m_rec_raw[i+1] for i in range(len(m_rec_raw)-1)), "Mean raw recall not non-increasing!"
    assert all(m_rec_cal[i] >= m_rec_cal[i+1] for i in range(len(m_rec_cal)-1)), "Mean calibrated recall not non-increasing!"
    print("\nMonotonicity verified: Recall is strictly non-increasing in tau across both scales.")

    # Save to json
    out_obj = {
        "taus": TAUS,
        "seed_42": {
            "raw_volume": {str(t): seed_42_results['raw_vol'][t] for t in TAUS},
            "raw_recall": {str(t): seed_42_results['raw_rec'][t] for t in TAUS},
            "calibrated_volume": {str(t): seed_42_results['cal_vol'][t] for t in TAUS},
            "calibrated_recall": {str(t): seed_42_results['cal_rec'][t] for t in TAUS},
        },
        "mean_20_seeds": {
            "raw_volume": {str(t): {"mean": float(np.mean(raw_vol_b_all[t])), "std": float(np.std(raw_vol_b_all[t]))} for t in TAUS},
            "raw_recall": {str(t): {"mean": float(np.mean(raw_rec_b_all[t])), "std": float(np.std(raw_rec_b_all[t]))} for t in TAUS},
            "calibrated_volume": {str(t): {"mean": float(np.mean(cal_vol_b_all[t])), "std": float(np.std(cal_vol_b_all[t]))} for t in TAUS},
            "calibrated_recall": {str(t): {"mean": float(np.mean(cal_rec_b_all[t])), "std": float(np.std(cal_rec_b_all[t]))} for t in TAUS},
        }
    }
    with open(os.path.join(_ROOT_DIR, "results", "threshold_grid_results.json"), "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)
    print("\nResults saved to results/threshold_grid_results.json")


if __name__ == "__main__":
    run_threshold_grid()
