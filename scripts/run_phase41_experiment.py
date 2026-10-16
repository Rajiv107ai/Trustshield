"""TrustShield Phase 4.1 — Causal Fraud Model Improvement Experiment.

Executes a focused, reproducible, leak-free model improvement experiment
comparing the Phase 3 production baseline against a Candidate model equipped
with causal pairwise-history and temporal velocity features.

Saves all experimental artifacts to: models/experiments/phase41_causal/
Zero production files or models are overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_MODELS_DIR = os.path.join(_ROOT_DIR, "models")
_EXP_DIR = os.path.join(_MODELS_DIR, "experiments", "phase41_causal")
_DATA_DIR = os.path.join(_ROOT_DIR, "data", "synthetic_v2_1")

for p in [_ROOT_DIR, os.path.join(_ROOT_DIR, "trustshield_project")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from baseline_model import build_features
from calibration import ProbabilityCalibrator, calculate_expected_calibration_error
from graph_features import (
    SIM_START,
    add_edge_weight_before,
    attach_snapshot_features,
    build_monthly_snapshots,
    build_relationship_graph,
    compute_relationship_features,
)


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_causal_velocity_and_windows(
    df: pd.DataFrame, group_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes strictly prior count and rolling window counts excluding same-timestamp orders."""
    n = len(df)
    prior_counts = np.zeros(n, dtype=np.float32)
    win_7d = np.zeros(n, dtype=np.float32)
    win_30d = np.zeros(n, dtype=np.float32)
    days_since_prev = np.full(n, 999.0, dtype=np.float32)

    sec_7d = 7 * 86400.0
    sec_30d = 30 * 86400.0

    groups = df.groupby(group_col).groups
    ts_all = df["_ts"].values

    for key, idxs in groups.items():
        sub_ts = ts_all[idxs]
        m = len(sub_ts)
        if m == 1:
            continue
        # searchsorted with side='left' strictly counts strictly earlier timestamps (t_prior < t_target)
        left_pos = np.searchsorted(sub_ts, sub_ts, side="left")
        prior_counts[idxs] = left_pos.astype(np.float32)

        p_7d = np.searchsorted(sub_ts, sub_ts - sec_7d, side="left")
        win_7d[idxs] = (left_pos - p_7d).astype(np.float32)

        p_30d = np.searchsorted(sub_ts, sub_ts - sec_30d, side="left")
        win_30d[idxs] = (left_pos - p_30d).astype(np.float32)

        has_prev = left_pos > 0
        valid_idxs = np.where(has_prev)[0]
        if len(valid_idxs) > 0:
            prev_indices = left_pos[valid_idxs] - 1
            time_diff = (sub_ts[valid_idxs] - sub_ts[prev_indices]) / 86400.0
            days_since_prev[idxs[valid_idxs]] = time_diff.astype(np.float32)

    return prior_counts, win_7d, win_30d, days_since_prev


def evaluate_binary_predictions_full(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    pr = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(calculate_expected_calibration_error(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "fpr": round(fpr, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    print("=" * 80)
    print("TRUSTSHIELD PHASE 4.1 — CAUSAL FRAUD MODEL IMPROVEMENT EXPERIMENT")
    print("=" * 80)

    os.makedirs(_EXP_DIR, exist_ok=True)

    # 1. Load Data
    print("\n[Step 1] Loading marketplace datasets...")
    orders_df = pd.read_csv(os.path.join(_DATA_DIR, "orders.csv"), parse_dates=["order_date"])
    listings_df = pd.read_csv(os.path.join(_DATA_DIR, "listings.csv"), parse_dates=["listing_date"])
    returns_df = pd.read_csv(os.path.join(_DATA_DIR, "returns.csv"), parse_dates=["return_date"])
    buyers_df = pd.read_csv(os.path.join(_DATA_DIR, "buyers.csv"), parse_dates=["signup_date"])
    sellers_df = pd.read_csv(os.path.join(_DATA_DIR, "sellers.csv"), parse_dates=["signup_date"])
    products_df = pd.read_csv(os.path.join(_DATA_DIR, "products.csv"))
    addr_log = pd.read_csv(os.path.join(_DATA_DIR, "address_sharing_log.csv"))
    dev_log = pd.read_csv(os.path.join(_DATA_DIR, "device_sharing_log.csv"))

    # Record data hashes
    data_hashes = {
        "orders_csv": sha256_file(os.path.join(_DATA_DIR, "orders.csv")),
        "listings_csv": sha256_file(os.path.join(_DATA_DIR, "listings.csv")),
        "returns_csv": sha256_file(os.path.join(_DATA_DIR, "returns.csv")),
        "buyers_csv": sha256_file(os.path.join(_DATA_DIR, "buyers.csv")),
        "sellers_csv": sha256_file(os.path.join(_DATA_DIR, "sellers.csv")),
    }

    # 2. Build Base Features
    print("[Step 1] Constructing baseline features via backward as-of merges...")
    df, tabular_cols = build_features(
        orders_df, listings_df, returns_df, buyers_df, sellers_df, products_df
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df.sort_values(by="order_date", kind="mergesort").reset_index(drop=True)

    # Baseline Graph features (leak-free training cutoff)
    rel_graph = build_relationship_graph(addr_log, dev_log, cutoff_date=pd.Timestamp("2025-08-31 23:59:59"))
    rel_feat = compute_relationship_features(rel_graph, df["buyer_id"].unique())
    df = df.merge(rel_feat, on="buyer_id", how="left")
    df["share_degree"] = df["share_degree"].fillna(0.0)
    df["share_component_size"] = df["share_component_size"].fillna(1.0)

    orders_with_dt = orders_df.assign(order_date=pd.to_datetime(orders_df["order_date"]))
    snapshots, months = build_monthly_snapshots(orders_with_dt, SIM_START)
    df = attach_snapshot_features(df, snapshots, months)
    df = add_edge_weight_before(df)

    # 3. Load Production Baseline Artifacts
    p3_meta = joblib.load(os.path.join(_MODELS_DIR, "feature_meta.joblib"))
    baseline_features = p3_meta["all_feature_cols"]
    baseline_model = joblib.load(os.path.join(_MODELS_DIR, "combined_graph_model.joblib"))
    baseline_calibrator = joblib.load(os.path.join(_MODELS_DIR, "calibrator.joblib"))

    baseline_artifact_hashes = {
        "combined_graph_model": sha256_file(os.path.join(_MODELS_DIR, "combined_graph_model.joblib")),
        "calibrator": sha256_file(os.path.join(_MODELS_DIR, "calibrator.joblib")),
        "feature_meta": sha256_file(os.path.join(_MODELS_DIR, "feature_meta.joblib")),
    }

    # 4. Construct Isolated Causal Pairwise & Velocity Features
    print("[Step 2] Constructing causal pairwise-history and velocity features...")
    df["_ts"] = df["order_date"].astype("int64") // 10**9
    df["_pair_id"] = df["buyer_id"] + "__" + df["seller_id"]

    # Pair causal features
    p_count, p_7d, p_30d, p_dt = compute_causal_velocity_and_windows(df, "_pair_id")
    df["pair_orders_last_7d"] = p_7d
    df["pair_orders_last_30d"] = p_30d
    df["pair_days_since_prior"] = p_dt
    df["pair_is_cold_start"] = (p_count == 0.0).astype(np.float32)

    # Pair Amount Ratio
    df["_pair_prev_amount_sum"] = df.groupby("_pair_id")["amount"].cumsum() - df["amount"]
    df["pair_prior_amount_mean"] = np.where(
        p_count > 0,
        df["_pair_prev_amount_sum"] / np.maximum(p_count, 1.0),
        0.0
    ).astype(np.float32)
    df["amount_vs_pair_prior_mean"] = np.where(
        p_count > 0,
        df["amount"] / np.maximum(df["pair_prior_amount_mean"], 1.0),
        1.0
    ).astype(np.float32)

    # Pair Prior Returns (causal strictly before order_date)
    pair_ret_sorted = returns_df[["buyer_id", "seller_id", "return_date"]].copy()
    pair_ret_sorted["_ret_ts"] = pd.to_datetime(pair_ret_sorted["return_date"]).astype("int64") // 10**9
    pair_ret_sorted["_pair_id"] = pair_ret_sorted["buyer_id"] + "__" + pair_ret_sorted["seller_id"]
    pair_ret_sorted = pair_ret_sorted.sort_values(by="_ret_ts", kind="mergesort")
    pair_ret_groups = pair_ret_sorted.groupby("_pair_id").groups

    pair_prior_returns = np.zeros(len(df), dtype=np.float32)
    for p_id, o_idxs in df.groupby("_pair_id").groups.items():
        if p_id in pair_ret_groups:
            r_ts = pair_ret_sorted.loc[pair_ret_groups[p_id], "_ret_ts"].values
            o_ts = df.loc[o_idxs, "_ts"].values
            r_counts = np.searchsorted(r_ts, o_ts, side="left")
            pair_prior_returns[o_idxs] = r_counts.astype(np.float32)
    df["pair_returns_before"] = pair_prior_returns

    # Buyer Activity & Velocity
    b_count, b_7d, b_30d, b_dt = compute_causal_velocity_and_windows(df, "buyer_id")
    df["buyer_orders_last_7d"] = b_7d
    df["buyer_orders_last_30d"] = b_30d
    df["buyer_days_since_prior"] = b_dt
    df["buyer_pair_order_share"] = p_count / np.maximum(b_count, 1.0)

    # Seller Activity & Velocity
    s_count, s_7d, s_30d, s_dt = compute_causal_velocity_and_windows(df, "seller_id")
    df["seller_prior_order_count"] = s_count
    df["seller_orders_last_7d"] = s_7d
    df["seller_orders_last_30d"] = s_30d
    df["seller_days_since_prior"] = s_dt
    df["seller_pair_order_share"] = p_count / np.maximum(s_count, 1.0)

    # Seller Returns Rate Causal
    ret_sorted = returns_df[["seller_id", "return_date"]].copy()
    ret_sorted["_ret_ts"] = pd.to_datetime(ret_sorted["return_date"]).astype("int64") // 10**9
    ret_sorted = ret_sorted.sort_values(by="_ret_ts", kind="mergesort")
    ret_groups = ret_sorted.groupby("seller_id").groups

    seller_prior_returns = np.zeros(len(df), dtype=np.float32)
    for s_id, o_idxs in df.groupby("seller_id").groups.items():
        if s_id in ret_groups:
            r_ts = ret_sorted.loc[ret_groups[s_id], "_ret_ts"].values
            o_ts = df.loc[o_idxs, "_ts"].values
            r_counts = np.searchsorted(r_ts, o_ts, side="left")
            seller_prior_returns[o_idxs] = r_counts.astype(np.float32)
    df["seller_returns_before"] = seller_prior_returns
    df["seller_return_rate_before"] = df["seller_returns_before"] / np.maximum(s_count, 1.0)

    # Device Velocity
    d_count, d_7d, d_30d, d_dt = compute_causal_velocity_and_windows(df, "device_id")
    df["device_orders_last_7d"] = d_7d

    # Candidate feature list (36 features)
    candidate_features = baseline_features + [
        "pair_orders_last_7d",
        "pair_orders_last_30d",
        "pair_days_since_prior",
        "pair_prior_amount_mean",
        "amount_vs_pair_prior_mean",
        "pair_returns_before",
        "pair_is_cold_start",
        "buyer_pair_order_share",
        "seller_pair_order_share",
        "buyer_orders_last_7d",
        "buyer_orders_last_30d",
        "buyer_days_since_prior",
        "seller_prior_order_count",
        "seller_orders_last_7d",
        "seller_orders_last_30d",
        "seller_days_since_prior",
        "seller_returns_before",
        "seller_return_rate_before",
        "device_orders_last_7d",
    ]

    print(f"Candidate feature count: {len(candidate_features)}")

    # 5. Cohort Slices
    train_mask = df["order_date"] <= "2025-08-31 23:59:59"
    val_mask = (df["order_date"] > "2025-08-31 23:59:59") & (df["order_date"] <= "2025-10-31 23:59:59")
    test_mask = df["order_date"] > "2025-10-31 23:59:59"

    y_train = df.loc[train_mask, "is_fraudulent"].values.astype(int)
    y_val = df.loc[val_mask, "is_fraudulent"].values.astype(int)
    y_test = df.loc[test_mask, "is_fraudulent"].values.astype(int)

    # 6. Fit Candidate Model on Training Set Only
    print("\n[Step 2] Training Candidate Model on Train cohort (N=16,952)...")
    spw = float((len(y_train) - y_train.sum()) / y_train.sum())

    cand_model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        reg_lambda=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    X_train_cand = df.loc[train_mask, candidate_features].fillna(0.0).values
    cand_model.fit(X_train_cand, y_train)

    # Fit Candidate Probability Calibrator on Validation Set
    X_val_cand = df.loc[val_mask, candidate_features].fillna(0.0).values
    cand_val_raw = cand_model.predict_proba(X_val_cand)[:, 1]

    cand_calibrator = ProbabilityCalibrator(method="isotonic")
    cand_calibrator.fit(cand_val_raw, y_val)
    cand_val_cal = cand_calibrator.predict_proba(cand_val_raw)

    # 7. Baseline Validation Evaluation
    X_val_base = df.loc[val_mask, baseline_features].fillna(0.0).values
    base_val_raw = baseline_model.predict_proba(X_val_base)[:, 1]
    base_val_cal = baseline_calibrator.predict_proba(base_val_raw)

    # Optimal Decision Threshold on Validation (Max F1)
    thresholds = np.linspace(0.05, 0.95, 91)
    base_f1s = [f1_score(y_val, (base_val_cal >= t).astype(int), zero_division=0) for t in thresholds]
    cand_f1s = [f1_score(y_val, (cand_val_cal >= t).astype(int), zero_division=0) for t in thresholds]
    opt_t_base = float(thresholds[np.argmax(base_f1s)])
    opt_t_cand = float(thresholds[np.argmax(cand_f1s)])

    print("\n" + "=" * 60)
    print("STEP 3: VALIDATION COMPARISON & DIAGNOSTICS")
    print("=" * 60)

    val_base_std = evaluate_binary_predictions_full(y_val, base_val_cal, threshold=0.5)
    val_cand_std = evaluate_binary_predictions_full(y_val, cand_val_cal, threshold=0.5)

    val_base_opt = evaluate_binary_predictions_full(y_val, base_val_cal, threshold=opt_t_base)
    val_cand_opt = evaluate_binary_predictions_full(y_val, cand_val_cal, threshold=opt_t_cand)

    print(f"Optimal F1 threshold on Validation: Baseline={opt_t_base:.2f}, Candidate={opt_t_cand:.2f}")

    # Sub-Period Performance (September vs October)
    sep_mask_val = (df.loc[val_mask, "order_date"] <= "2025-09-30 23:59:59").values
    oct_mask_val = (df.loc[val_mask, "order_date"] > "2025-09-30 23:59:59").values

    val_sep_base = evaluate_binary_predictions_full(y_val[sep_mask_val], base_val_cal[sep_mask_val], threshold=opt_t_base)
    val_sep_cand = evaluate_binary_predictions_full(y_val[sep_mask_val], cand_val_cal[sep_mask_val], threshold=opt_t_cand)

    val_oct_base = evaluate_binary_predictions_full(y_val[oct_mask_val], base_val_cal[oct_mask_val], threshold=opt_t_base)
    val_oct_cand = evaluate_binary_predictions_full(y_val[oct_mask_val], cand_val_cal[oct_mask_val], threshold=opt_t_cand)

    # Cold-Start Pairs vs Known Pairs
    cold_mask_val = (df.loc[val_mask, "pair_is_cold_start"] == 1.0).values
    known_mask_val = ~cold_mask_val

    val_cold_base = evaluate_binary_predictions_full(y_val[cold_mask_val], base_val_cal[cold_mask_val], threshold=opt_t_base)
    val_cold_cand = evaluate_binary_predictions_full(y_val[cold_mask_val], cand_val_cal[cold_mask_val], threshold=opt_t_cand)

    val_known_base = evaluate_binary_predictions_full(y_val[known_mask_val], base_val_cal[known_mask_val], threshold=opt_t_base)
    val_known_cand = evaluate_binary_predictions_full(y_val[known_mask_val], cand_val_cal[known_mask_val], threshold=opt_t_cand)

    # 8. FREEZE EXPERIMENT & EVALUATE TEST COHORT (Months 11-12)
    print("\n" + "=" * 60)
    print("STEP 3: FROZEN TEST COHORT EVALUATION (N=20,919)")
    print("=" * 60)

    # Build relationship graph strictly at VAL_END for test features
    rel_graph_test = build_relationship_graph(addr_log, dev_log, cutoff_date=pd.Timestamp("2025-10-31 23:59:59"))
    rel_feat_test = compute_relationship_features(rel_graph_test, df.loc[test_mask, "buyer_id"].unique())

    df_test = df.loc[test_mask].drop(columns=["share_degree", "share_component_size"]).merge(
        rel_feat_test, on="buyer_id", how="left"
    )
    df_test["share_degree"] = df_test["share_degree"].fillna(0.0)
    df_test["share_component_size"] = df_test["share_component_size"].fillna(1.0)

    X_test_base = df_test[baseline_features].fillna(0.0).values
    X_test_cand = df_test[candidate_features].fillna(0.0).values

    base_test_raw = baseline_model.predict_proba(X_test_base)[:, 1]
    base_test_cal = baseline_calibrator.predict_proba(base_test_raw)

    cand_test_raw = cand_model.predict_proba(X_test_cand)[:, 1]
    cand_test_cal = cand_calibrator.predict_proba(cand_test_raw)

    test_base_std = evaluate_binary_predictions_full(y_test, base_test_cal, threshold=0.5)
    test_cand_std = evaluate_binary_predictions_full(y_test, cand_test_cal, threshold=0.5)

    test_base_opt = evaluate_binary_predictions_full(y_test, base_test_cal, threshold=opt_t_base)
    test_cand_opt = evaluate_binary_predictions_full(y_test, cand_test_cal, threshold=opt_t_cand)

    # Multi-seed test evaluation for candidate model
    test_rocs = []
    test_prs = []
    for s in [42, 101, 2024, 7, 999]:
        m = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            reg_lambda=2.0,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            random_state=s,
            n_jobs=-1,
        )
        m.fit(X_train_cand, y_train)
        p = m.predict_proba(X_test_cand)[:, 1]
        test_rocs.append(float(roc_auc_score(y_test, p)))
        test_prs.append(float(average_precision_score(y_test, p)))

    # Save Experimental Artifacts
    print("\n[Step 5] Saving experimental artifacts to:", _EXP_DIR)
    cand_model_path = os.path.join(_EXP_DIR, "candidate_model.joblib")
    cand_calibrator_path = os.path.join(_EXP_DIR, "candidate_calibrator.joblib")
    cand_meta_path = os.path.join(_EXP_DIR, "candidate_feature_meta.json")
    results_path = os.path.join(_EXP_DIR, "experiment_results.json")

    joblib.dump(cand_model, cand_model_path)
    joblib.dump(cand_calibrator, cand_calibrator_path)

    meta_payload = {
        "candidate_feature_cols": candidate_features,
        "n_features": len(candidate_features),
        "hyperparameters": cand_model.get_params(),
        "optimal_threshold_validation": opt_t_cand,
        "trained_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(cand_meta_path, "w") as f:
        json.dump(meta_payload, f, indent=2, default=str)

    results_payload = {
        "experiment_name": "Phase 4.1 Causal Pairwise & Velocity Experiment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_hashes": data_hashes,
        "baseline_artifact_hashes": baseline_artifact_hashes,
        "candidate_artifact_hashes": {
            "candidate_model": sha256_file(cand_model_path),
            "candidate_calibrator": sha256_file(cand_calibrator_path),
            "candidate_feature_meta": sha256_file(cand_meta_path),
        },
        "validation_results": {
            "baseline_std_thresh_05": val_base_std,
            "candidate_std_thresh_05": val_cand_std,
            "baseline_opt_thresh": val_base_opt,
            "candidate_opt_thresh": val_cand_opt,
            "by_time_period": {
                "september": {"baseline": val_sep_base, "candidate": val_sep_cand},
                "october": {"baseline": val_oct_base, "candidate": val_oct_cand},
            },
            "by_cohort_segment": {
                "cold_start_pairs": {"baseline": val_cold_base, "candidate": val_cold_cand},
                "known_pairs": {"baseline": val_known_base, "candidate": val_known_cand},
            },
        },
        "test_holdout_results": {
            "baseline_std_thresh_05": test_base_std,
            "candidate_std_thresh_05": test_cand_std,
            "baseline_opt_thresh": test_base_opt,
            "candidate_opt_thresh": test_cand_opt,
            "multi_seed_uncertainty_5seeds": {
                "test_roc_auc_mean": round(float(np.mean(test_rocs)), 4),
                "test_roc_auc_std": round(float(np.std(test_rocs)), 4),
                "test_pr_auc_mean": round(float(np.mean(test_prs)), 4),
                "test_pr_auc_std": round(float(np.std(test_prs)), 4),
            },
        },
        "feature_importances": {
            f: round(float(imp), 4)
            for f, imp in pd.Series(cand_model.feature_importances_, index=candidate_features)
            .sort_values(ascending=False)
            .items()
        },
    }

    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    print("\n--- EXPERIMENT SUMMARY RESULTS JSON ---")
    print(json.dumps(results_payload, indent=2))


if __name__ == "__main__":
    main()
