"""
TrustShield Phase 3 — Synthetic Data Quality, Shortcut Learning & Generalization Audit.

Runs comprehensive empirical experiments investigating:
1. Feature and label distribution shortcuts across synthetic datasets.
2. Single-feature and feature-group ablations using frozen models and retrained baselines.
3. Unseen-entity (unseen buyers and sellers) generalization.
4. Temporal slice breakdown (Months 9-12).
5. Fraud subtype breakdown (fake_listing, return_abuse, coordinated_fraud, collusion).
6. Multimodal and graph feature degradation.

Persists results to reports/phase3_experiments_results.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Path setups
_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_PROJECT_DIR = os.path.join(_ROOT_DIR, "trustshield_project")
_MODELS_DIR = os.path.join(_ROOT_DIR, "models")
_REPORTS_DIR = os.path.join(_ROOT_DIR, "reports")
os.makedirs(_REPORTS_DIR, exist_ok=True)

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from baseline_model import TRAIN_END, VAL_END, build_features, leakage_audit
from calibration import calculate_expected_calibration_error
from graph_features import (
    SIM_START,
    add_edge_weight_before,
    attach_snapshot_features,
    build_monthly_snapshots,
    build_relationship_graph,
    compute_relationship_features,
)
from phase2_specialized_models import (
    build_listing_features,
    build_return_features,
)


def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute binary classification metrics with zero divisions handled."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        roc_auc = 0.5
        pr_auc = float(n_pos / max(len(y_true), 1))

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(calculate_expected_calibration_error(y_true, y_prob))

    return {
        "n_samples": int(len(y_true)),
        "n_pos": n_pos,
        "prevalence": round(float(n_pos / max(len(y_true), 1)), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
    }


def find_val_threshold(y_val: np.ndarray, prob_val: np.ndarray) -> float:
    """Find threshold maximizing F1 on validation split."""
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob_val >= t).astype(int)
        score = f1_score(y_val, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return round(best_t, 2)


# ===========================================================================
# 1. Feature Distribution & Statistical Separation Audit
# ===========================================================================

def audit_feature_distributions(df: pd.DataFrame, feature_cols: List[str], label_col: str = "y") -> List[Dict[str, Any]]:
    """Computes distribution stats and single-feature predictive power."""
    results = []
    y = df[label_col].to_numpy()
    pos_mask = (y == 1)
    neg_mask = (y == 0)

    for col in feature_cols:
        vals = df[col].fillna(0).to_numpy()
        pos_vals = vals[pos_mask]
        neg_vals = vals[neg_mask]

        # Single feature ROC-AUC
        # Check both positive and negative correlation
        try:
            auc_raw = roc_auc_score(y, vals)
            single_auc = max(auc_raw, 1.0 - auc_raw)
        except Exception:
            single_auc = 0.5

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(pos_vals, neg_vals)
        # Cohen's d effect size
        std_pooled = np.sqrt((np.var(pos_vals) + np.var(neg_vals)) / 2.0)
        cohens_d = float((np.mean(pos_vals) - np.mean(neg_vals)) / (std_pooled + 1e-9))

        results.append({
            "feature": col,
            "legit_mean": round(float(np.mean(neg_vals)), 4),
            "legit_std": round(float(np.std(neg_vals)), 4),
            "legit_median": round(float(np.median(neg_vals)), 4),
            "fraud_mean": round(float(np.mean(pos_vals)), 4),
            "fraud_std": round(float(np.std(pos_vals)), 4),
            "fraud_median": round(float(np.median(pos_vals)), 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": float(ks_pval),
            "cohens_d": round(cohens_d, 4),
            "single_feature_auc": round(float(single_auc), 4),
        })
    return results


# ===========================================================================
# 2. Fake Listing Detector Experiments
# ===========================================================================

def run_fake_listing_audit(listings_df, sellers_df, products_df) -> Dict[str, Any]:
    print("\n--- Running Fake Listing Detector Audit ---")
    df, feature_cols = build_listing_features(listings_df, sellers_df, products_df)

    train = df[df["listing_date"] <= TRAIN_END]
    val = df[(df["listing_date"] > TRAIN_END) & (df["listing_date"] <= VAL_END)]
    test = df[df["listing_date"] > VAL_END]

    dist_stats = audit_feature_distributions(train, feature_cols, label_col="y")

    # Load frozen model
    frozen_model = joblib.load(os.path.join(_MODELS_DIR, "fake_listing_model.joblib"))

    X_train = train[feature_cols].fillna(0)
    y_train = train["y"].to_numpy()
    X_val = val[feature_cols].fillna(0)
    y_val = val["y"].to_numpy()
    X_test = test[feature_cols].fillna(0)
    y_test = test["y"].to_numpy()

    # Baseline frozen test evaluation
    val_probs = frozen_model.predict_proba(X_val)[:, 1]
    best_th = find_val_threshold(y_val, val_probs)
    test_probs = frozen_model.predict_proba(X_test)[:, 1]

    baseline_metrics = calc_metrics(y_test, test_probs, threshold=0.50)
    baseline_metrics_th = calc_metrics(y_test, test_probs, threshold=best_th)

    # Feature masking ablations on test set (mean imputation from train)
    ablations: Dict[str, Any] = {}
    train_means = X_train.mean()

    for col in feature_cols:
        X_test_ablated = X_test.copy()
        X_test_ablated[col] = train_means[col]
        probs_ablated = frozen_model.predict_proba(X_test_ablated)[:, 1]
        m = calc_metrics(y_test, probs_ablated, threshold=0.50)
        ablations[f"without_{col}"] = {
            "metrics": m,
            "delta_roc_auc": round(m["roc_auc"] - baseline_metrics["roc_auc"], 4),
            "delta_pr_auc": round(m["pr_auc"] - baseline_metrics["pr_auc"], 4),
        }

    # Model trained with ONLY multimodal similarity
    xgb_mm = XGBClassifier(n_estimators=100, max_depth=3, eval_metric="aucpr", random_state=42)
    xgb_mm.fit(train[["multimodal_similarity_score"]], y_train)
    probs_mm = xgb_mm.predict_proba(test[["multimodal_similarity_score"]])[:, 1]
    ablations["only_multimodal_similarity"] = calc_metrics(y_test, probs_mm)

    # Model trained with ONLY price anomaly features
    price_cols = ["price_vs_base_price_ratio", "price_vs_category_median_ratio"]
    xgb_price = XGBClassifier(n_estimators=100, max_depth=3, eval_metric="aucpr", random_state=42)
    xgb_price.fit(train[price_cols], y_train)
    probs_price = xgb_price.predict_proba(test[price_cols])[:, 1]
    ablations["only_price_anomaly_features"] = calc_metrics(y_test, probs_price)

    # Model trained with ONLY seller history features (no price, no multimodal)
    seller_cols = ["seller_age_days_at_listing", "seller_listings_before"]
    xgb_seller = XGBClassifier(n_estimators=100, max_depth=3, eval_metric="aucpr", random_state=42)
    xgb_seller.fit(train[seller_cols], y_train)
    probs_seller = xgb_seller.predict_proba(test[seller_cols])[:, 1]
    ablations["only_seller_history_features"] = calc_metrics(y_test, probs_seller)

    # Generalization: Unseen Sellers vs Seen Sellers in Test Set
    train_sellers = set(train["seller_id"].unique())
    test_unseen_seller_mask = ~test["seller_id"].isin(train_sellers)
    test_seen_seller_mask = test["seller_id"].isin(train_sellers)

    unseen_metrics = calc_metrics(
        y_test[test_unseen_seller_mask],
        test_probs[test_unseen_seller_mask],
        threshold=0.50
    ) if test_unseen_seller_mask.sum() > 0 else None

    seen_metrics = calc_metrics(
        y_test[test_seen_seller_mask],
        test_probs[test_seen_seller_mask],
        threshold=0.50
    )

    return {
        "dataset_counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "test_unseen_sellers": int(test_unseen_seller_mask.sum()),
            "test_seen_sellers": int(test_seen_seller_mask.sum()),
        },
        "feature_distribution_stats": dist_stats,
        "baseline_metrics_th_0.50": baseline_metrics,
        "baseline_metrics_val_optimal_th": {
            "optimal_threshold": best_th,
            "metrics": baseline_metrics_th,
        },
        "ablations": ablations,
        "generalization": {
            "unseen_sellers": unseen_metrics,
            "seen_sellers": seen_metrics,
        },
    }


# ===========================================================================
# 3. Return Fraud Detector Experiments
# ===========================================================================

def run_return_fraud_audit(returns_df, orders_df, buyers_df, sellers_df) -> Dict[str, Any]:
    print("\n--- Running Return Fraud Detector Audit ---")
    returns_df = returns_df.copy()
    orders_df = orders_df.copy()
    returns_df["return_date"] = pd.to_datetime(returns_df["return_date"])
    orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])

    df, feature_cols = build_return_features(returns_df, orders_df, buyers_df, sellers_df)

    train = df[df["return_date"] <= TRAIN_END]
    val = df[(df["return_date"] > TRAIN_END) & (df["return_date"] <= VAL_END)]
    test = df[df["return_date"] > VAL_END]

    dist_stats = audit_feature_distributions(train, feature_cols, label_col="y")

    frozen_model = joblib.load(os.path.join(_MODELS_DIR, "return_fraud_model.joblib"))

    X_train = train[feature_cols].fillna(0)
    y_train = train["y"].to_numpy()
    X_val = val[feature_cols].fillna(0)
    y_val = val["y"].to_numpy()
    X_test = test[feature_cols].fillna(0)
    y_test = test["y"].to_numpy()

    val_probs = frozen_model.predict_proba(X_val)[:, 1]
    best_th = find_val_threshold(y_val, val_probs)
    test_probs = frozen_model.predict_proba(X_test)[:, 1]

    baseline_metrics = calc_metrics(y_test, test_probs, threshold=0.50)
    baseline_metrics_th = calc_metrics(y_test, test_probs, threshold=best_th)

    ablations: Dict[str, Any] = {}
    train_means = X_train.mean()

    # Individual feature masking
    suspicious_cols = [
        "days_to_return", "buyer_return_rate_before", "buyer_prior_returns",
        "seller_return_rate_before", "order_amount"
    ]
    for col in suspicious_cols:
        X_test_ablated = X_test.copy()
        X_test_ablated[col] = train_means[col]
        probs_ablated = frozen_model.predict_proba(X_test_ablated)[:, 1]
        m = calc_metrics(y_test, probs_ablated, threshold=0.50)
        ablations[f"without_{col}"] = {
            "metrics": m,
            "delta_roc_auc": round(m["roc_auc"] - baseline_metrics["roc_auc"], 4),
            "delta_pr_auc": round(m["pr_auc"] - baseline_metrics["pr_auc"], 4),
        }

    # Group ablation: without both days_to_return and buyer_return_rate_before
    X_test_no_shortcuts = X_test.copy()
    X_test_no_shortcuts["days_to_return"] = train_means["days_to_return"]
    X_test_no_shortcuts["buyer_return_rate_before"] = train_means["buyer_return_rate_before"]
    probs_no_shortcuts = frozen_model.predict_proba(X_test_no_shortcuts)[:, 1]
    m_no_sc = calc_metrics(y_test, probs_no_shortcuts, threshold=0.50)
    ablations["without_days_to_return_and_return_rate"] = {
        "metrics": m_no_sc,
        "delta_roc_auc": round(m_no_sc["roc_auc"] - baseline_metrics["roc_auc"], 4),
        "delta_pr_auc": round(m_no_sc["pr_auc"] - baseline_metrics["pr_auc"], 4),
    }

    # Model trained with ONLY days_to_return + buyer_return_rate_before
    sc_cols = ["days_to_return", "buyer_return_rate_before"]
    xgb_sc = XGBClassifier(n_estimators=100, max_depth=3, eval_metric="aucpr", random_state=42)
    xgb_sc.fit(train[sc_cols], y_train)
    probs_sc = xgb_sc.predict_proba(test[sc_cols])[:, 1]
    ablations["only_shortcut_features_days_and_rate"] = calc_metrics(y_test, probs_sc)

    # Model trained with ONLY plausible behavioral features (ages, amount, orders before, reasons)
    behavioral_cols = [
        "buyer_age_days_at_return", "seller_age_days_at_return", "order_amount",
        "buyer_orders_before_return", "seller_orders_before_return",
        "reason_defective", "reason_changed_mind", "reason_size_issue", "reason_wrong_item_received"
    ]
    xgb_beh = XGBClassifier(n_estimators=100, max_depth=3, eval_metric="aucpr", random_state=42)
    xgb_beh.fit(train[behavioral_cols], y_train)
    probs_beh = xgb_beh.predict_proba(test[behavioral_cols])[:, 1]
    ablations["only_plausible_behavioral_features"] = calc_metrics(y_test, probs_beh)

    # Generalization: Unseen Buyers vs Seen Buyers
    train_buyers = set(train["buyer_id"].unique())
    test_unseen_buyer_mask = ~test["buyer_id"].isin(train_buyers)
    test_seen_buyer_mask = test["buyer_id"].isin(train_buyers)

    unseen_b_metrics = calc_metrics(
        y_test[test_unseen_buyer_mask],
        test_probs[test_unseen_buyer_mask],
        threshold=0.50
    ) if test_unseen_buyer_mask.sum() > 0 else None

    seen_b_metrics = calc_metrics(
        y_test[test_seen_buyer_mask],
        test_probs[test_seen_buyer_mask],
        threshold=0.50
    )

    return {
        "dataset_counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "test_unseen_buyers": int(test_unseen_buyer_mask.sum()),
            "test_seen_buyers": int(test_seen_buyer_mask.sum()),
        },
        "feature_distribution_stats": dist_stats,
        "baseline_metrics_th_0.50": baseline_metrics,
        "baseline_metrics_val_optimal_th": {
            "optimal_threshold": best_th,
            "metrics": baseline_metrics_th,
        },
        "ablations": ablations,
        "generalization": {
            "unseen_buyers": unseen_b_metrics,
            "seen_buyers": seen_b_metrics,
        },
    }


# ===========================================================================
# 4. Phase 3 & Phase 5 Transaction Fraud Model Experiments
# ===========================================================================

def run_transaction_fraud_audit(orders_df, listings_df, returns_df, buyers_df, sellers_df, products_df, addr_log, dev_log) -> Dict[str, Any]:
    print("\n--- Running Phase 3 & 5 Transaction Fraud Model Audit ---")
    df, tabular_cols = build_features(orders_df, listings_df, returns_df, buyers_df, sellers_df, products_df)
    df["order_date"] = pd.to_datetime(df["order_date"])

    train_mask = df["order_date"] <= TRAIN_END
    val_mask = (df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)
    test_mask = df["order_date"] > VAL_END

    # Graph construction
    rel_graph_train = build_relationship_graph(addr_log, dev_log, cutoff_date=TRAIN_END)
    rel_graph_val = build_relationship_graph(addr_log, dev_log, cutoff_date=TRAIN_END)
    rel_graph_test = build_relationship_graph(addr_log, dev_log, cutoff_date=VAL_END)

    rel_feat_train = compute_relationship_features(rel_graph_train, df.loc[train_mask, "buyer_id"].unique())
    rel_feat_val = compute_relationship_features(rel_graph_val, df.loc[val_mask, "buyer_id"].unique())
    rel_feat_test = compute_relationship_features(rel_graph_test, df.loc[test_mask, "buyer_id"].unique())

    df["share_degree"] = 0.0
    df["share_component_size"] = 1.0

    tmp_train = df.loc[train_mask, ["buyer_id"]].merge(rel_feat_train, on="buyer_id", how="left")
    df.loc[train_mask, "share_degree"] = tmp_train["share_degree"].fillna(0.0).values
    df.loc[train_mask, "share_component_size"] = tmp_train["share_component_size"].fillna(1.0).values

    tmp_val = df.loc[val_mask, ["buyer_id"]].merge(rel_feat_val, on="buyer_id", how="left")
    df.loc[val_mask, "share_degree"] = tmp_val["share_degree"].fillna(0.0).values
    df.loc[val_mask, "share_component_size"] = tmp_val["share_component_size"].fillna(1.0).values

    tmp_test = df.loc[test_mask, ["buyer_id"]].merge(rel_feat_test, on="buyer_id", how="left")
    df.loc[test_mask, "share_degree"] = tmp_test["share_degree"].fillna(0.0).values
    df.loc[test_mask, "share_component_size"] = tmp_test["share_component_size"].fillna(1.0).values

    orders_with_dt = orders_df.assign(order_date=pd.to_datetime(orders_df["order_date"]))
    snapshots, months = build_monthly_snapshots(orders_with_dt, SIM_START)
    df = attach_snapshot_features(df, snapshots, months)
    df = add_edge_weight_before(df)

    p3_meta = joblib.load(os.path.join(_MODELS_DIR, "feature_meta.joblib"))
    p3_model = joblib.load(os.path.join(_MODELS_DIR, "combined_graph_model.joblib"))
    p3_calibrator = joblib.load(os.path.join(_MODELS_DIR, "calibrator.joblib"))

    p3_cols = p3_meta["all_feature_cols"]
    tabular_cols = p3_meta["tabular_cols"]
    graph_cols = p3_meta["graph_cols"]

    df["y"] = df["is_fraudulent"].astype(int)
    dist_stats = audit_feature_distributions(df.loc[train_mask], p3_cols, label_col="y")

    # Splits
    X_train = df.loc[train_mask, p3_cols].fillna(0.0)
    y_train = df.loc[train_mask, "y"].to_numpy()
    X_val = df.loc[val_mask, p3_cols].fillna(0.0)
    y_val = df.loc[val_mask, "y"].to_numpy()
    X_test = df.loc[test_mask, p3_cols].fillna(0.0)
    y_test = df.loc[test_mask, "y"].to_numpy()

    # Baseline Phase 3
    val_raw = p3_model.predict_proba(X_val)[:, 1]
    val_cal = p3_calibrator.predict_proba(val_raw)
    best_th = find_val_threshold(y_val, val_cal)

    test_raw = p3_model.predict_proba(X_test)[:, 1]
    test_cal = p3_calibrator.predict_proba(test_raw)

    baseline_metrics = calc_metrics(y_test, test_cal, threshold=0.50)
    baseline_metrics_th = calc_metrics(y_test, test_cal, threshold=best_th)

    train_means = X_train.mean()

    # Controlled Ablations
    ablations: Dict[str, Any] = {}

    # 1. Removal of individual suspicious features
    for col in ["device_shared_buyer_count", "buyer_return_rate_before", "price_vs_base_price_ratio", "buyer_seller_edge_weight_before"]:
        X_abl = X_test.copy()
        X_abl[col] = train_means[col]
        probs = p3_calibrator.predict_proba(p3_model.predict_proba(X_abl)[:, 1])
        m = calc_metrics(y_test, probs, threshold=0.50)
        ablations[f"without_{col}"] = {
            "metrics": m,
            "delta_roc_auc": round(m["roc_auc"] - baseline_metrics["roc_auc"], 4),
            "delta_pr_auc": round(m["pr_auc"] - baseline_metrics["pr_auc"], 4),
        }

    # 2. Removal of feature groups
    # Group A: Without return features
    X_no_ret = X_test.copy()
    for col in ["buyer_returns_before", "buyer_return_rate_before"]:
        X_no_ret[col] = train_means[col]
    probs = p3_calibrator.predict_proba(p3_model.predict_proba(X_no_ret)[:, 1])
    m_no_ret = calc_metrics(y_test, probs, threshold=0.50)
    ablations["without_return_features"] = {
        "metrics": m_no_ret,
        "delta_roc_auc": round(m_no_ret["roc_auc"] - baseline_metrics["roc_auc"], 4),
        "delta_pr_auc": round(m_no_ret["pr_auc"] - baseline_metrics["pr_auc"], 4),
    }

    # Group B: Without price features
    X_no_price = X_test.copy()
    for col in ["price_vs_base_price_ratio", "price_vs_category_median_ratio"]:
        X_no_price[col] = train_means[col]
    probs = p3_calibrator.predict_proba(p3_model.predict_proba(X_no_price)[:, 1])
    m_no_price = calc_metrics(y_test, probs, threshold=0.50)
    ablations["without_price_anomaly_features"] = {
        "metrics": m_no_price,
        "delta_roc_auc": round(m_no_price["roc_auc"] - baseline_metrics["roc_auc"], 4),
        "delta_pr_auc": round(m_no_price["pr_auc"] - baseline_metrics["pr_auc"], 4),
    }

    # Group C: Without all graph features (degraded mode)
    X_no_graph = X_test.copy()
    for col in graph_cols:
        X_no_graph[col] = 0.0
    probs_no_graph = p3_calibrator.predict_proba(p3_model.predict_proba(X_no_graph)[:, 1])
    m_no_graph = calc_metrics(y_test, probs_no_graph, threshold=0.50)
    ablations["without_all_graph_features_zero_graph"] = {
        "metrics": m_no_graph,
        "delta_roc_auc": round(m_no_graph["roc_auc"] - baseline_metrics["roc_auc"], 4),
        "delta_pr_auc": round(m_no_graph["pr_auc"] - baseline_metrics["pr_auc"], 4),
    }

    # Simple Baseline with ONLY Suspicious Artifacts (device share, return rate, price ratio)
    suspicious_cols = ["device_shared_buyer_count", "buyer_return_rate_before", "price_vs_base_price_ratio"]
    xgb_suspicious = XGBClassifier(n_estimators=100, max_depth=4, eval_metric="aucpr", random_state=42)
    xgb_suspicious.fit(X_train[suspicious_cols], y_train)
    probs_susp = xgb_suspicious.predict_proba(X_test[suspicious_cols])[:, 1]
    ablations["only_suspicious_features_device_return_price"] = calc_metrics(y_test, probs_susp)

    # Simple Baseline with ONLY Plausible Behavioral Features (age, order velocity, amount, listings before)
    behavioral_cols = ["seller_age_days", "seller_total_listings_before", "buyer_age_days", "buyer_orders_before", "amount"]
    xgb_behavioral = XGBClassifier(n_estimators=100, max_depth=4, eval_metric="aucpr", random_state=42)
    xgb_behavioral.fit(X_train[behavioral_cols], y_train)
    probs_beh = xgb_behavioral.predict_proba(X_test[behavioral_cols])[:, 1]
    ablations["only_plausible_behavioral_features"] = calc_metrics(y_test, probs_beh)

    # 5. Generalization Breakdown: Unseen Entities on Test Set
    train_sellers = set(df.loc[train_mask, "seller_id"].unique())
    train_buyers = set(df.loc[train_mask, "buyer_id"].unique())

    test_sub = df.loc[test_mask].copy()
    test_sub["unseen_seller"] = ~test_sub["seller_id"].isin(train_sellers)
    test_sub["unseen_buyer"] = ~test_sub["buyer_id"].isin(train_buyers)

    gen_unseen_seller = calc_metrics(
        y_test[test_sub["unseen_seller"]], test_cal[test_sub["unseen_seller"]], threshold=0.50
    ) if test_sub["unseen_seller"].sum() > 0 else None

    gen_seen_seller = calc_metrics(
        y_test[~test_sub["unseen_seller"]], test_cal[~test_sub["unseen_seller"]], threshold=0.50
    )

    gen_unseen_buyer = calc_metrics(
        y_test[test_sub["unseen_buyer"]], test_cal[test_sub["unseen_buyer"]], threshold=0.50
    ) if test_sub["unseen_buyer"].sum() > 0 else None

    gen_seen_buyer = calc_metrics(
        y_test[~test_sub["unseen_buyer"]], test_cal[~test_sub["unseen_buyer"]], threshold=0.50
    )

    # 6. Fraud Subtype Performance Breakdown on Test Set
    subtype_breakdown: Dict[str, Any] = {}
    test_fraud_types = test_sub["fraud_type"].fillna("legitimate")

    # Compare each fraud subtype against all legitimate test transactions
    legit_mask = (y_test == 0)
    for ftype in ["fake_listing", "return_abuse", "coordinated_fraud", "seller_buyer_collusion"]:
        ftype_mask = (test_fraud_types == ftype)
        eval_mask = ftype_mask | legit_mask
        sub_y = y_test[eval_mask]
        sub_cal = test_cal[eval_mask]
        m_subtype = calc_metrics(sub_y, sub_cal, threshold=0.50)
        subtype_breakdown[ftype] = m_subtype

    # 7. Temporal Slice Breakdown across Test Months
    temporal_breakdown: Dict[str, Any] = {}
    test_sub["month"] = test_sub["order_date"].dt.month
    for m_num in sorted(test_sub["month"].unique()):
        m_mask = (test_sub["month"] == m_num)
        temporal_breakdown[f"month_{m_num:02d}"] = calc_metrics(
            y_test[m_mask], test_cal[m_mask], threshold=0.50
        )

    return {
        "dataset_counts": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
            "test_unseen_sellers": int(test_sub["unseen_seller"].sum()),
            "test_seen_sellers": int((~test_sub["unseen_seller"]).sum()),
            "test_unseen_buyers": int(test_sub["unseen_buyer"].sum()),
            "test_seen_buyers": int((~test_sub["unseen_buyer"]).sum()),
        },
        "feature_distribution_stats": dist_stats,
        "baseline_metrics_th_0.50": baseline_metrics,
        "baseline_metrics_val_optimal_th": {
            "optimal_threshold": best_th,
            "metrics": baseline_metrics_th,
        },
        "ablations": ablations,
        "generalization": {
            "unseen_sellers": gen_unseen_seller,
            "seen_sellers": gen_seen_seller,
            "unseen_buyers": gen_unseen_buyer,
            "seen_buyers": gen_seen_buyer,
        },
        "fraud_subtype_breakdown": subtype_breakdown,
        "temporal_breakdown": temporal_breakdown,
    }


def main():
    print("=" * 80)
    print("TRUSTSHIELD PHASE 3 — CONTROLLED AUDIT EXPERIMENTS")
    print("=" * 80)

    export_dir = os.path.join(_PROJECT_DIR, "synthetic_data_export")
    orders_df = pd.read_csv(os.path.join(export_dir, "orders.csv"), parse_dates=["order_date"])
    listings_df = pd.read_csv(os.path.join(export_dir, "listings.csv"), parse_dates=["listing_date"])
    returns_df = pd.read_csv(os.path.join(export_dir, "returns.csv"), parse_dates=["return_date"])
    buyers_df = pd.read_csv(os.path.join(export_dir, "buyers.csv"), parse_dates=["signup_date"])
    sellers_df = pd.read_csv(os.path.join(export_dir, "sellers.csv"), parse_dates=["signup_date"])
    products_df = pd.read_csv(os.path.join(export_dir, "products.csv"))
    addr_log = pd.read_csv(os.path.join(export_dir, "address_sharing_log.csv"))
    dev_log = pd.read_csv(os.path.join(export_dir, "device_sharing_log.csv"))

    t0 = time.time()

    fake_listing_res = run_fake_listing_audit(listings_df, sellers_df, products_df)
    return_fraud_res = run_return_fraud_audit(returns_df, orders_df, buyers_df, sellers_df)
    transaction_res = run_transaction_fraud_audit(
        orders_df, listings_df, returns_df, buyers_df, sellers_df, products_df, addr_log, dev_log
    )

    elapsed = round(time.time() - t0, 2)
    print(f"\nAll experiments completed in {elapsed}s.")

    combined_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": elapsed,
            "python_version": sys.version,
            "models_audited": ["fake_listing_detector", "return_fraud_detector", "phase3_combined_graph_model"],
        },
        "fake_listing_detector": fake_listing_res,
        "return_fraud_detector": return_fraud_res,
        "phase3_transaction_model": transaction_res,
    }

    out_json = os.path.join(_REPORTS_DIR, "phase3_experiments_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    print(f"Results successfully persisted to: {out_json}")


if __name__ == "__main__":
    main()
