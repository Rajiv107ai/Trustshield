"""
TrustShield AI — Phase 2
Two specialized detectors, split out from the Phase 1C combined baseline:
  1. Fake Listing Detector   — listing-level, predicts listings_df.is_fraudulent
  2. Return Fraud Detector   — return-level, predicts returns_df.is_fraudulent

Design reference: design.md — Phase 2 = "baseline listing/return models with
evaluation", specialized versions of the Phase 1C transaction-level baseline.
Each detector uses only features relevant to ITS entity and ITS decision
point in time (a listing is scored when it's created; a return is scored
when it's filed) — not the blended order-level feature set from Phase 1C.

Same rules carried over from baseline_model.py:
- All history features are strictly-before via merge_asof(direction=
  "backward", allow_exact_matches=False).
- No fraud-mechanism columns (price_anomaly, image_mismatch, fraud_ring_id)
  used as features.
- Temporal split (Month 1-8 train / 9-10 val / 11-12 test), by the
  entity's own date (listing_date for the listing model, return_date for
  the return model) — not order_date, since that's not when this detector
  would actually be asked to make a decision.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from utils import evaluate, find_cost_optimal_threshold

# See baseline_model.py's import block for why this is auto-detected rather
# than a hard dependency — same reasoning applies here.
try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

from entity_generator import build_base_entities
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud

TRAIN_END = pd.Timestamp("2025-08-31")
VAL_END = pd.Timestamp("2025-10-31")



def evaluate_at_cost_threshold(y_test, score_test, amount_test, threshold, fp_cost, label):
    pred = (score_test >= threshold).astype(int)
    print(f"\n--- {label}: cost-optimal threshold = {threshold:.2f} (chosen on val, applied to test) ---")
    evaluate(y_test, pred, score_test, f"{label} @ cost-optimal threshold")
    y_arr, amt_arr = y_test.to_numpy(), amount_test.to_numpy()
    fn_mask = (y_arr == 1) & (pred == 0)
    fp_mask = (y_arr == 0) & (pred == 1)
    test_cost = amt_arr[fn_mask].sum() + fp_cost * fp_mask.sum()
    print(f"Realized test-set cost @ this threshold: {test_cost:,.0f}  "
          f"(FN amount lost: {amt_arr[fn_mask].sum():,.0f} + FP count {fp_mask.sum()} x {fp_cost} review cost)")


def train_eval(train, val, test, feature_cols, label_prefix, amount_col=None, fp_cost=None):
    Xtr, ytr = train[feature_cols].fillna(0), train["y"]
    Xval, yval = val[feature_cols].fillna(0), val["y"]
    Xte, yte = test[feature_cols].fillna(0), test["y"]

    scaler = StandardScaler().fit(Xtr)
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    logreg.fit(scaler.transform(Xtr), ytr)
    lr_test_scores = logreg.predict_proba(scaler.transform(Xte))[:, 1]
    evaluate(yte, (lr_test_scores >= 0.5).astype(int), lr_test_scores, f"{label_prefix} — Logistic Regression (test)")

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced_subsample",
                                 random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    rf_val_scores = rf.predict_proba(Xval)[:, 1]
    rf_test_scores = rf.predict_proba(Xte)[:, 1]
    evaluate(yte, (rf_test_scores >= 0.5).astype(int), rf_test_scores, f"{label_prefix} — Random Forest (test)")

    if amount_col is not None and fp_cost is not None:
        best_t, best_cost, default_cost = find_cost_optimal_threshold(
            yval, rf_val_scores, val[amount_col], fp_cost
        )
        print(f"\n[{label_prefix}] Cost sweep on VAL (fp_cost={fp_cost}): "
              f"default-0.5 cost={default_cost:,.0f} -> best cost={best_cost:,.0f} at threshold={best_t:.2f}")
        evaluate_at_cost_threshold(yte, rf_test_scores, test[amount_col], best_t, fp_cost, label_prefix)

    print(f"\n{label_prefix} — Random Forest feature importances:")
    print(pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).round(3))

    if _HAS_XGBOOST:
        scale_pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
        xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                             scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                             random_state=42, n_jobs=-1)
        xgb.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        xgb_val_scores = xgb.predict_proba(Xval)[:, 1]
        xgb_test_scores = xgb.predict_proba(Xte)[:, 1]
        evaluate(yte, (xgb_test_scores >= 0.5).astype(int), xgb_test_scores, f"{label_prefix} — XGBoost (test)")
        print(f"\n{label_prefix} — XGBoost feature importances:")
        print(pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False).round(3))

        if amount_col is not None and fp_cost is not None:
            best_t_xgb, best_cost_xgb, default_cost_xgb = find_cost_optimal_threshold(
                yval, xgb_val_scores, val[amount_col], fp_cost
            )
            print(f"\n[{label_prefix} XGBoost] Cost sweep on VAL (fp_cost={fp_cost}): "
                  f"default-0.5 cost={default_cost_xgb:,.0f} -> best cost={best_cost_xgb:,.0f} at threshold={best_t_xgb:.2f}")
            evaluate_at_cost_threshold(yte, xgb_test_scores, test[amount_col], best_t_xgb, fp_cost, f"{label_prefix} XGBoost")
    else:
        print(f"\n[{label_prefix}] XGBoost not installed — skipped (Random Forest results above stand). "
              f"`pip install xgboost` and re-run to include it automatically.")

    return {"logreg": logreg, "rf": rf, "scaler": scaler}


# ---------------------------------------------------------------------------
# 1. Fake Listing Detector (listing-level)
# ---------------------------------------------------------------------------

def build_listing_features(listings_df, sellers_df, products_df):
    df = listings_df.merge(products_df[["product_id", "base_price"]], on="product_id", how="left")
    df = df.merge(sellers_df[["seller_id", "signup_date"]].rename(columns={"signup_date": "seller_signup_date"}),
                   on="seller_id", how="left")
    df["listing_date"] = pd.to_datetime(df["listing_date"])
    df["seller_signup_date"] = pd.to_datetime(df["seller_signup_date"])

    df["price_vs_base_price_ratio"] = df["price"] / df["base_price"]

    train_mask = df["listing_date"] <= TRAIN_END
    category_median = df.loc[train_mask].groupby("category")["price"].median()
    df["price_vs_category_median_ratio"] = df["price"] / df["category"].map(category_median)

    df["seller_age_days_at_listing"] = (df["listing_date"] - df["seller_signup_date"]).dt.days

    # Seller's cumulative listing count strictly before this listing.
    sorted_listings = df.sort_values("listing_date", kind="mergesort")
    running_count = sorted_listings.groupby("seller_id").cumcount()
    df["seller_listings_before"] = running_count.reindex(df.index)

    feature_cols = [
        "price_vs_base_price_ratio", "price_vs_category_median_ratio",
        "seller_age_days_at_listing", "seller_listings_before",
    ]
    df["y"] = df["is_fraudulent"].astype(int)
    return df, feature_cols


def run_fake_listing_detector(listings_df, sellers_df, products_df):
    print("\n" + "=" * 70)
    print("FAKE LISTING DETECTOR (listing-level)")
    print("=" * 70)

    df, feature_cols = build_listing_features(listings_df, sellers_df, products_df)

    # Leakage checks specific to this entity
    print("\n--- Leakage audit (listing model) ---")
    neg_age = (df["seller_age_days_at_listing"] < 0).sum()
    print(f"  [{'FAIL' if neg_age else 'PASS'}] no negative seller age at listing time: {neg_age} violations")
    banned = {"is_fraudulent", "fraud_type", "price_anomaly", "image_mismatch", "displayed_product_id"}
    print(f"  [{'FAIL' if banned & set(feature_cols) else 'PASS'}] no fraud-mechanism columns used: "
          f"{banned & set(feature_cols) or 'none'}\n")

    train = df[df["listing_date"] <= TRAIN_END]
    val = df[(df["listing_date"] > TRAIN_END) & (df["listing_date"] <= VAL_END)]
    test = df[df["listing_date"] > VAL_END]
    print(f"Split sizes: train={len(train)} ({train['y'].mean():.2%} fake), "
          f"val={len(val)} ({val['y'].mean():.2%} fake), test={len(test)} ({test['y'].mean():.2%} fake)")

    return train_eval(train, val, test, feature_cols, "Fake Listing Detector",
                       amount_col="price", fp_cost=50)


# ---------------------------------------------------------------------------
# 2. Return Fraud Detector (return-level)
# ---------------------------------------------------------------------------

def build_return_features(returns_df, orders_df, buyers_df, sellers_df):
    df = returns_df.merge(
        orders_df[["order_id", "order_date", "amount", "buyer_id"]].rename(columns={"buyer_id": "_ob"}),
        on="order_id", how="left"
    )
    df = df.merge(buyers_df[["buyer_id", "signup_date"]].rename(columns={"signup_date": "buyer_signup_date"}),
                   on="buyer_id", how="left")
    df = df.merge(sellers_df[["seller_id", "signup_date"]].rename(columns={"signup_date": "seller_signup_date"}),
                   on="seller_id", how="left")
    df["return_date"] = pd.to_datetime(df["return_date"])
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["buyer_signup_date"] = pd.to_datetime(df["buyer_signup_date"])
    df["seller_signup_date"] = pd.to_datetime(df["seller_signup_date"])

    df["days_to_return"] = (df["return_date"] - df["order_date"]).dt.days
    df["buyer_age_days_at_return"] = (df["return_date"] - df["buyer_signup_date"]).dt.days
    df["seller_age_days_at_return"] = (df["return_date"] - df["seller_signup_date"]).dt.days
    df["order_amount"] = df["amount"]

    # Buyer's return history strictly before THIS return (by return_date).
    sorted_returns = df.sort_values("return_date", kind="mergesort").copy()
    sorted_returns["running_count"] = sorted_returns.groupby("buyer_id").cumcount()
    df["buyer_prior_returns"] = sorted_returns.set_index(sorted_returns.index)["running_count"].reindex(df.index)

    # Buyer's total orders strictly before this return's date (merge_asof against orders history).
    orders_sorted = orders_df[["buyer_id", "order_date"]].sort_values("order_date", kind="mergesort").copy()
    orders_sorted["running_count"] = orders_sorted.groupby("buyer_id").cumcount() + 1
    left = df[["buyer_id", "return_date"]].sort_values("return_date", kind="mergesort").copy()
    left["_orig_index"] = left.index
    merged = pd.merge_asof(left, orders_sorted, left_on="return_date", right_on="order_date",
                            by="buyer_id", direction="backward", allow_exact_matches=False)
    df["buyer_orders_before_return"] = merged.set_index("_orig_index")["running_count"].reindex(df.index).fillna(0)
    df["buyer_return_rate_before"] = df["buyer_prior_returns"] / df["buyer_orders_before_return"].clip(lower=1)

    # Seller-side history strictly before this return: how return-prone has
    # this seller's buyer base been so far. Previously missing entirely —
    # a seller with an unusually high running return rate is a real signal
    # this detector should have access to, not just buyer-side behavior.
    seller_returns_sorted = df[["seller_id", "return_date"]].sort_values("return_date", kind="mergesort").copy()
    seller_returns_sorted["running_count"] = seller_returns_sorted.groupby("seller_id").cumcount()
    df["seller_prior_returns"] = seller_returns_sorted.set_index(seller_returns_sorted.index)["running_count"].reindex(df.index)

    seller_orders_sorted = orders_df[["seller_id", "order_date"]].sort_values("order_date", kind="mergesort").copy()
    seller_orders_sorted["running_count"] = seller_orders_sorted.groupby("seller_id").cumcount() + 1
    left_s = df[["seller_id", "return_date"]].sort_values("return_date", kind="mergesort").copy()
    left_s["_orig_index"] = left_s.index
    merged_s = pd.merge_asof(left_s, seller_orders_sorted, left_on="return_date", right_on="order_date",
                              by="seller_id", direction="backward", allow_exact_matches=False)
    df["seller_orders_before_return"] = merged_s.set_index("_orig_index")["running_count"].reindex(df.index).fillna(0)
    df["seller_return_rate_before"] = df["seller_prior_returns"] / df["seller_orders_before_return"].clip(lower=1)

    reason_dummies = pd.get_dummies(df["reason"], prefix="reason")
    df = pd.concat([df, reason_dummies], axis=1)

    feature_cols = [
        "days_to_return", "buyer_age_days_at_return", "seller_age_days_at_return", "order_amount",
        "buyer_prior_returns", "buyer_orders_before_return", "buyer_return_rate_before",
        "seller_prior_returns", "seller_orders_before_return", "seller_return_rate_before",
    ] + list(reason_dummies.columns)

    df["y"] = df["is_fraudulent"].astype(int)
    return df, feature_cols


def run_return_fraud_detector(returns_df, orders_df, buyers_df, sellers_df):
    print("\n" + "=" * 70)
    print("RETURN FRAUD DETECTOR (return-level)")
    print("=" * 70)

    df, feature_cols = build_return_features(returns_df, orders_df, buyers_df, sellers_df)

    print("\n--- Leakage audit (return model) ---")
    neg_days = (df["days_to_return"] < 0).sum()
    print(f"  [{'FAIL' if neg_days else 'PASS'}] no return dated before its own order: {neg_days} violations")
    over_count = (df["buyer_prior_returns"] > df["buyer_orders_before_return"]).sum()
    print(f"  [{'FAIL' if over_count else 'PASS'}] buyer_prior_returns never exceeds buyer_orders_before_return: "
          f"{over_count} violations")
    seller_over_count = (df["seller_prior_returns"] > df["seller_orders_before_return"]).sum()
    print(f"  [{'FAIL' if seller_over_count else 'PASS'}] seller_prior_returns never exceeds seller_orders_before_return: "
          f"{seller_over_count} violations")
    banned = {"is_fraudulent", "fraud_type"}
    print(f"  [{'FAIL' if banned & set(feature_cols) else 'PASS'}] no fraud-label columns used\n")

    train = df[df["return_date"] <= TRAIN_END]
    val = df[(df["return_date"] > TRAIN_END) & (df["return_date"] <= VAL_END)]
    test = df[df["return_date"] > VAL_END]
    print(f"Split sizes: train={len(train)} ({train['y'].mean():.2%} fraud), "
          f"val={len(val)} ({val['y'].mean():.2%} fraud), test={len(test)} ({test['y'].mean():.2%} fraud)")

    return train_eval(train, val, test, feature_cols, "Return Fraud Detector",
                       amount_col="order_amount", fp_cost=100)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building full pipeline...")
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"])
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"], catalog["products"],
        base["address_sharing_log"], base["device_sharing_log"],
    )

    run_fake_listing_detector(result["listings"], catalog["sellers"], catalog["products"])
    run_return_fraud_detector(result["returns"], result["orders"], txn["buyers"], catalog["sellers"])
