"""
TrustShield AI — Reproducible Multi-Split Model Evaluation Script.

Stage 1 / Phase 2: Independent, leak-free evaluation of all trained model artifacts
across frozen Chronological splits (Train, Validation, and Holdout Test).

Split definitions (Chronological out-of-time cutoffs):
- Train split:       order_date <= 2025-08-31 (Months 1-8, 22,848 orders)
- Validation split:  2025-08-31 < order_date <= 2025-10-31 (Months 9-10, 10,266 orders)
- Test split:        order_date > 2025-10-31 (Months 11-12, 16,886 orders)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Setup paths
_ROOT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
_PROJECT_DIR = os.path.join(_ROOT_DIR, "trustshield_project")
_MODELS_DIR = os.path.join(_ROOT_DIR, "models")
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


def compute_sha256(filepath: str) -> str:
    """Compute sha256 checksum of an artifact file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=cast(Any, 0)))
    rec = float(recall_score(y_true, y_pred, zero_division=cast(Any, 0)))
    f1 = float(f1_score(y_true, y_pred, zero_division=cast(Any, 0)))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(calculate_expected_calibration_error(y_true, y_prob))

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
    }


def main() -> Dict[str, Any]:
    print("=" * 80)
    print("TRUSTSHIELD AI — REPRODUCIBLE MULTI-SPLIT MODEL EVALUATION")
    print("=" * 80)

    # 1. Verify existence of datasets
    export_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else os.environ.get("DATA_DIR", os.path.join(_PROJECT_DIR, "synthetic_data_export"))
    required_csvs = [
        "orders.csv", "listings.csv", "returns.csv",
        "buyers.csv", "sellers.csv", "products.csv",
        "address_sharing_log.csv", "device_sharing_log.csv",
    ]
    for csv_name in required_csvs:
        csv_path = os.path.join(export_dir, csv_name)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing required dataset: {csv_path}")

    print("Loading exported marketplace datasets...")
    orders_df = pd.read_csv(os.path.join(export_dir, "orders.csv"), parse_dates=["order_date"])
    listings_df = pd.read_csv(os.path.join(export_dir, "listings.csv"), parse_dates=["listing_date"])
    returns_df = pd.read_csv(os.path.join(export_dir, "returns.csv"), parse_dates=["return_date"])
    buyers_df = pd.read_csv(os.path.join(export_dir, "buyers.csv"), parse_dates=["signup_date"])
    sellers_df = pd.read_csv(os.path.join(export_dir, "sellers.csv"), parse_dates=["signup_date"])
    products_df = pd.read_csv(os.path.join(export_dir, "products.csv"))
    addr_log = pd.read_csv(os.path.join(export_dir, "address_sharing_log.csv"))
    dev_log = pd.read_csv(os.path.join(export_dir, "device_sharing_log.csv"))

    # 2. Build temporal tabular features
    print("Constructing leak-free tabular features via backward as-of merges...")
    df, tabular_cols = build_features(
        orders_df, listings_df, returns_df, buyers_df, sellers_df, products_df
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    leakage_audit(df, tabular_cols)

    # Split masks
    train_mask = df["order_date"] <= TRAIN_END
    val_mask = (df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)
    test_mask = df["order_date"] > VAL_END

    print(f"Dataset split counts: Train={train_mask.sum():,} | Val={val_mask.sum():,} | Test={test_mask.sum():,}")

    # 3. Construct strictly isolated graph snapshots per split
    print("Constructing temporally isolated relationship graphs...")
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

    # 4. Load production artifacts
    print("Loading serialized production model artifacts...")
    p3_meta = joblib.load(os.path.join(_MODELS_DIR, "feature_meta.joblib"))
    p3_model = joblib.load(os.path.join(_MODELS_DIR, "combined_graph_model.joblib"))
    p3_calibrator = joblib.load(os.path.join(_MODELS_DIR, "calibrator.joblib"))

    p5_model = joblib.load(os.path.join(_MODELS_DIR, "hybrid_model.joblib"))
    p5_meta = joblib.load(os.path.join(_MODELS_DIR, "phase5_feature_meta.joblib"))
    buyer_embs = joblib.load(os.path.join(_MODELS_DIR, "buyer_embeddings.joblib"))
    seller_embs = joblib.load(os.path.join(_MODELS_DIR, "seller_embeddings.joblib"))
    p5_calibrator = joblib.load(os.path.join(_MODELS_DIR, "phase5_calibrator.joblib"))

    p3_cols = p3_meta["all_feature_cols"]
    p5_cols = p5_meta["hybrid_feature_cols"]

    # Attach GNN embeddings for Phase 5
    for i in range(16):
        df[f"gnn_buyer_emb_{i}"] = df["buyer_id"].map(
            lambda b: buyer_embs.get(b, np.zeros(16))[i] if isinstance(buyer_embs.get(b), np.ndarray) else 0.0
        )
        df[f"gnn_seller_emb_{i}"] = df["seller_id"].map(
            lambda s: seller_embs.get(s, np.zeros(16))[i] if isinstance(seller_embs.get(s), np.ndarray) else 0.0
        )

    # 5. Evaluate Phase 3 Combined Graph Model
    results: Dict[str, Any] = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_counts": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "models": {},
    }

    print("\n" + "-" * 60)
    print("1. EVALUATING PHASE 3: COMBINED GRAPH MODEL (TABULAR + NETWORKX)")
    print("-" * 60)
    p3_eval: Dict[str, Any] = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        sub = df.loc[mask]
        X = sub.reindex(columns=p3_cols, fill_value=0.0).fillna(0.0)
        y = sub["is_fraudulent"].astype(int).to_numpy()
        raw_prob = np.asarray(p3_model.predict_proba(X))[:, 1]
        cal_prob = p3_calibrator.predict_proba(raw_prob)
        p3_eval[name] = {
            "raw": evaluate_binary_predictions(y, raw_prob),
            "calibrated": evaluate_binary_predictions(y, cal_prob),
        }
        print(f"  [{name.upper()}] ROC-AUC={p3_eval[name]['calibrated']['roc_auc']:.4f} | "
              f"PR-AUC={p3_eval[name]['calibrated']['pr_auc']:.4f} | "
              f"F1={p3_eval[name]['calibrated']['f1']:.4f} | "
              f"Brier={p3_eval[name]['calibrated']['brier_score']:.4f} | "
              f"ECE={p3_eval[name]['calibrated']['ece']:.4f}")

    # Phase 3 Degraded Check (tabular only, zero-filling graph features)
    # Explains the 0.6029 score reported in docs/ROBUSTNESS_REPORT.md!
    sub_test = df.loc[test_mask].copy()
    for gc in p3_meta["graph_cols"]:
        sub_test[gc] = 0.0
    X_degraded = sub_test.reindex(columns=p3_cols, fill_value=0.0).fillna(0.0)
    y_test = sub_test["is_fraudulent"].astype(int).to_numpy()
    p3_degraded_raw = np.asarray(p3_model.predict_proba(X_degraded))[:, 1]
    p3_degraded_cal = p3_calibrator.predict_proba(p3_degraded_raw)
    p3_eval["test_degraded_zero_graph_features"] = {
        "raw": evaluate_binary_predictions(y_test, p3_degraded_raw),
        "calibrated": evaluate_binary_predictions(y_test, p3_degraded_cal),
    }
    print(f"  [TEST DEGRADED (Graph=0, reproduces 0.6029)] ROC-AUC={p3_eval['test_degraded_zero_graph_features']['calibrated']['roc_auc']:.4f} | "
          f"PR-AUC={p3_eval['test_degraded_zero_graph_features']['calibrated']['pr_auc']:.4f}")

    results["models"]["phase3_combined_graph_model"] = p3_eval

    # 6. Evaluate Phase 5 Hybrid GraphSAGE + XGBoost Model
    print("\n" + "-" * 60)
    print("2. EVALUATING PHASE 5: HYBRID GRAPHSAGE + XGBOOST MODEL")
    print("-" * 60)
    p5_eval: Dict[str, Any] = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        sub = df.loc[mask]
        X = sub.reindex(columns=p5_cols, fill_value=0.0).fillna(0.0)
        y = sub["is_fraudulent"].astype(int).to_numpy()
        raw_prob = np.asarray(p5_model.predict_proba(X))[:, 1]
        cal_prob = p5_calibrator.predict_proba(raw_prob)
        p5_eval[name] = {
            "raw": evaluate_binary_predictions(y, raw_prob),
            "calibrated": evaluate_binary_predictions(y, cal_prob),
        }
        print(f"  [{name.upper()}] ROC-AUC={p5_eval[name]['calibrated']['roc_auc']:.4f} | "
              f"PR-AUC={p5_eval[name]['calibrated']['pr_auc']:.4f} | "
              f"F1={p5_eval[name]['calibrated']['f1']:.4f} | "
              f"Brier={p5_eval[name]['calibrated']['brier_score']:.4f} | "
              f"ECE={p5_eval[name]['calibrated']['ece']:.4f}")

    results["models"]["phase5_hybrid_model"] = p5_eval

    # 7. Evaluate Specialized Detectors
    print("\n" + "-" * 60)
    print("3. EVALUATING SPECIALIZED DETECTORS (LISTING & RETURN)")
    print("-" * 60)
    from phase2_specialized_models import (
        TRAIN_END as P2_TRAIN_END,
        VAL_END as P2_VAL_END,
        build_listing_features,
        build_return_features,
    )

    fake_listing_model = joblib.load(os.path.join(_MODELS_DIR, "fake_listing_model.joblib"))
    listing_meta = p3_meta.get("listing_feature_cols", [
        "price_vs_base_price_ratio", "price_vs_category_median_ratio",
        "seller_age_days_at_listing", "seller_listings_before",
        "multimodal_similarity_score",
    ])
    listings_feat, _ = build_listing_features(listings_df, sellers_df, products_df)
    
    # Pure out-of-time test (> VAL_END, Months 11-12)
    listing_test_pure = listings_feat[listings_feat["listing_date"] > P2_VAL_END]
    X_list_pure = listing_test_pure.reindex(listing_meta, axis=1, fill_value=0.0).fillna(0.0)
    y_list_pure = listing_test_pure["is_fraudulent"].astype(int).to_numpy()
    p_list_pure = np.asarray(fake_listing_model.predict_proba(X_list_pure))[:, 1]
    list_metrics_pure = evaluate_binary_predictions(y_list_pure, p_list_pure)

    # Post-train holdout (> TRAIN_END, Val + Test combined)
    listing_test_post_train = listings_feat[listings_feat["listing_date"] > P2_TRAIN_END]
    X_list_pt = listing_test_post_train.reindex(listing_meta, axis=1, fill_value=0.0).fillna(0.0)
    y_list_pt = listing_test_post_train["is_fraudulent"].astype(int).to_numpy()
    p_list_pt = np.asarray(fake_listing_model.predict_proba(X_list_pt))[:, 1]
    list_metrics_pt = evaluate_binary_predictions(y_list_pt, p_list_pt)

    print(f"  [FAKE LISTING PURE TEST (> VAL_END, N={len(listing_test_pure):,})] ROC-AUC={list_metrics_pure['roc_auc']:.4f} | PR-AUC={list_metrics_pure['pr_auc']:.4f} | F1={list_metrics_pure['f1']:.4f}")
    print(f"  [FAKE LISTING POST-TRAIN (> TRAIN_END, N={len(listing_test_post_train):,})] ROC-AUC={list_metrics_pt['roc_auc']:.4f} | PR-AUC={list_metrics_pt['pr_auc']:.4f} | F1={list_metrics_pt['f1']:.4f}")
    
    results["models"]["fake_listing_detector"] = {
        "test_pure_months_11_12": {**list_metrics_pure, "n_samples": int(len(listing_test_pure))},
        "post_train_holdout": {**list_metrics_pt, "n_samples": int(len(listing_test_post_train))},
        **list_metrics_pt,  # Preserve legacy flat structure for backwards compatibility
    }

    return_fraud_model = joblib.load(os.path.join(_MODELS_DIR, "return_fraud_model.joblib"))
    return_meta = p3_meta.get("return_feature_cols", [
        "days_to_return", "buyer_age_days_at_return", "seller_age_days_at_return", "order_amount",
        "buyer_prior_returns", "buyer_orders_before_return", "buyer_return_rate_before",
        "seller_prior_returns", "seller_orders_before_return", "seller_return_rate_before",
        "reason_changed_mind", "reason_defective", "reason_size_issue", "reason_wrong_item_received",
    ])
    returns_feat, _ = build_return_features(returns_df, orders_df, buyers_df, sellers_df)
    
    # Pure out-of-time test (> VAL_END, Months 11-12, uncensored orders placed <= 2025-12-10)
    returns_test_pure = returns_feat[
        (returns_feat["return_date"] > P2_VAL_END) &
        (returns_feat["order_date"] <= pd.Timestamp("2025-12-31") - pd.Timedelta(days=21))
    ]
    X_ret_pure = returns_test_pure.reindex(return_meta, axis=1, fill_value=0.0).fillna(0.0)
    y_ret_pure = returns_test_pure["is_fraudulent"].astype(int).to_numpy()
    p_ret_pure = np.asarray(return_fraud_model.predict_proba(X_ret_pure))[:, 1]
    ret_metrics_pure = evaluate_binary_predictions(y_ret_pure, p_ret_pure)

    # Post-train holdout (> TRAIN_END, Val + Test combined)
    returns_test_pt = returns_feat[returns_feat["return_date"] > P2_TRAIN_END]
    X_ret_pt = returns_test_pt.reindex(return_meta, axis=1, fill_value=0.0).fillna(0.0)
    y_ret_pt = returns_test_pt["is_fraudulent"].astype(int).to_numpy()
    p_ret_pt = np.asarray(return_fraud_model.predict_proba(X_ret_pt))[:, 1]
    ret_metrics_pt = evaluate_binary_predictions(y_ret_pt, p_ret_pt)

    print(f"  [RETURN FRAUD PURE TEST (> VAL_END, N={len(returns_test_pure):,})] ROC-AUC={ret_metrics_pure['roc_auc']:.4f} | PR-AUC={ret_metrics_pure['pr_auc']:.4f} | F1={ret_metrics_pure['f1']:.4f}")
    print(f"  [RETURN FRAUD POST-TRAIN (> TRAIN_END, N={len(returns_test_pt):,})] ROC-AUC={ret_metrics_pt['roc_auc']:.4f} | PR-AUC={ret_metrics_pt['pr_auc']:.4f} | F1={ret_metrics_pt['f1']:.4f}")
    
    results["models"]["return_fraud_detector"] = {
        "test_pure_months_11_12": {**ret_metrics_pure, "n_samples": int(len(returns_test_pure))},
        "post_train_holdout": {**ret_metrics_pt, "n_samples": int(len(returns_test_pt))},
        **ret_metrics_pt,  # Preserve legacy flat structure for backwards compatibility
    }

    # Save manifest
    report_path = os.path.join(_MODELS_DIR, "reproduced_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nReproducible evaluation manifest saved to: {report_path}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    main()
