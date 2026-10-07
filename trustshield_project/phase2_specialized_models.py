"""Specialized fraud detectors: Fake Listing Detector and Return Fraud Detector."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from utils import evaluate, find_cost_optimal_threshold
from entity_generator import build_base_entities
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud
from multimodal_scoring import compute_multimodal_similarity

TRAIN_END = pd.Timestamp("2025-08-31")
VAL_END = pd.Timestamp("2025-10-31")


def _strict_prior_cumcount(df: pd.DataFrame, group_col: str, date_col: str) -> pd.Series:
    """Count of rows per group with a timestamp STRICTLY BEFORE the current row.

    Uses merge_asof with allow_exact_matches=False so that two rows sharing the
    same timestamp are never counted as prior history of each other.
    Same logic as baseline_model._cumulative_count_asof.
    """
    sorted_df = df[[group_col, date_col]].sort_values(by=date_col, kind="mergesort").copy()
    sorted_df["_running"] = sorted_df.groupby(group_col).cumcount() + 1

    left = df[[group_col, date_col]].copy()
    left["_orig_index"] = left.index
    left_sorted = left.sort_values(by=date_col, kind="mergesort")

    merged = pd.merge_asof(
        left_sorted, sorted_df[[group_col, date_col, "_running"]],
        on=date_col, by=group_col,
        direction="backward",
        allow_exact_matches=False,
    )
    return pd.Series(
        merged.set_index("_orig_index")["_running"].reindex(df.index).fillna(0),
    )


def evaluate_at_cost_threshold(y_test, score_test, amount_test, threshold, fp_cost, label):
    """Evaluates test performance at cost-optimal decision threshold."""
    pred = (score_test >= threshold).astype(int)
    evaluate(y_test, pred, score_test, f"{label} @ cost-optimal threshold ({threshold:.2f})")
    y_arr, amt_arr = y_test.to_numpy(), amount_test.to_numpy()
    fn_cost = amt_arr[(y_arr == 1) & (pred == 0)].sum()
    fp_total = fp_cost * ((y_arr == 0) & (pred == 1)).sum()
    print(f"Test realized cost: {fn_cost + fp_total:,.0f} (Missed fraud: {fn_cost:,.0f} + Review friction: {fp_total:,.0f})")


def train_eval(train, val, test, feature_cols, label_prefix, amount_col=None, fp_cost=None):
    """Trains Logistic Regression, Random Forest, and XGBoost models and evaluates cost-benefit curve."""
    Xtr, ytr = train[feature_cols].fillna(0), train["y"]
    Xval, yval = val[feature_cols].fillna(0), val["y"]
    Xte, yte = test[feature_cols].fillna(0), test["y"]

    scaler = StandardScaler().fit(Xtr)
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    logreg.fit(scaler.transform(Xtr), ytr)
    lr_test_scores = np.asarray(logreg.predict_proba(scaler.transform(Xte)))[:, 1]
    evaluate(yte, (lr_test_scores >= 0.5).astype(int), lr_test_scores, f"{label_prefix} — Logistic Regression (test)")

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    rf_val_scores = np.asarray(rf.predict_proba(Xval))[:, 1]
    rf_test_scores = np.asarray(rf.predict_proba(Xte))[:, 1]
    evaluate(yte, (rf_test_scores >= 0.5).astype(int), rf_test_scores, f"{label_prefix} — Random Forest (test)")

    scale_pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                        random_state=42, n_jobs=-1)
    xgb.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    xgb_val_scores = np.asarray(xgb.predict_proba(Xval))[:, 1]
    xgb_test_scores = np.asarray(xgb.predict_proba(Xte))[:, 1]
    evaluate(yte, (xgb_test_scores >= 0.5).astype(int), xgb_test_scores, f"{label_prefix} — XGBoost (test)")

    if amount_col is not None and fp_cost is not None:
        best_t, best_cost, _ = find_cost_optimal_threshold(yval, xgb_val_scores, val[amount_col], fp_cost)
        evaluate_at_cost_threshold(yte, xgb_test_scores, test[amount_col], best_t, fp_cost, f"{label_prefix} XGBoost")

    return {"logreg": logreg, "rf": rf, "xgb": xgb, "scaler": scaler}


def build_listing_features(listings_df, sellers_df, products_df):
    """Builds listing-level fraud features."""
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

    sorted_listings = df.sort_values("listing_date", kind="mergesort")
    df["seller_listings_before"] = _strict_prior_cumcount(sorted_listings, "seller_id", "listing_date").reindex(df.index)
    df["multimodal_similarity_score"] = compute_multimodal_similarity(listings_df, products_df).to_numpy()

    feature_cols = [
        "price_vs_base_price_ratio", "price_vs_category_median_ratio",
        "seller_age_days_at_listing", "seller_listings_before",
        "multimodal_similarity_score",
    ]
    df["y"] = df["is_fraudulent"].astype(int)
    return df, feature_cols


def run_fake_listing_detector(listings_df, sellers_df, products_df):
    """Executes fake listing detection pipeline."""
    df, feature_cols = build_listing_features(listings_df, sellers_df, products_df)
    train = df[df["listing_date"] <= TRAIN_END]
    val = df[(df["listing_date"] > TRAIN_END) & (df["listing_date"] <= VAL_END)]
    test = df[df["listing_date"] > VAL_END]

    print(f"\nFake Listing Detector — Train: {len(train)} ({train['y'].mean():.2%}), Val: {len(val)}, Test: {len(test)}")
    return train_eval(train, val, test, feature_cols, "Fake Listing Detector", amount_col="price", fp_cost=50)


def build_return_features(returns_df, orders_df, buyers_df, sellers_df):
    """Builds return-level fraud features."""
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

    df["buyer_prior_returns"] = _strict_prior_cumcount(df, "buyer_id", "return_date")

    orders_sorted = orders_df[["buyer_id", "order_date"]].sort_values("order_date", kind="mergesort").copy()
    orders_sorted["running_count"] = orders_sorted.groupby("buyer_id").cumcount() + 1
    left = df[["buyer_id", "return_date"]].sort_values("return_date", kind="mergesort").copy()
    left["_orig_index"] = left.index
    merged = pd.merge_asof(left, orders_sorted, left_on="return_date", right_on="order_date",
                            by="buyer_id", direction="backward", allow_exact_matches=False)
    df["buyer_orders_before_return"] = merged.set_index("_orig_index")["running_count"].reindex(df.index).fillna(0)
    df["buyer_return_rate_before"] = df["buyer_prior_returns"] / df["buyer_orders_before_return"].clip(lower=1)

    df["seller_prior_returns"] = _strict_prior_cumcount(df, "seller_id", "return_date")

    seller_orders_sorted = orders_df[["seller_id", "order_date"]].sort_values("order_date", kind="mergesort").copy()
    seller_orders_sorted["running_count"] = seller_orders_sorted.groupby("seller_id").cumcount() + 1
    left_s = df[["seller_id", "return_date"]].sort_values("return_date", kind="mergesort").copy()
    left_s["_orig_index"] = left_s.index
    merged_s = pd.merge_asof(left_s, seller_orders_sorted, left_on="return_date", right_on="order_date",
                              by="seller_id", direction="backward", allow_exact_matches=False)
    df["seller_orders_before_return"] = merged_s.set_index("_orig_index")["running_count"].reindex(df.index).fillna(0)
    df["seller_return_rate_before"] = df["seller_prior_returns"] / df["seller_orders_before_return"].clip(lower=1)

    from order_return_generator import RETURN_REASONS
    reason_cat = pd.Categorical(df["reason"], categories=RETURN_REASONS)
    reason_dummies = pd.get_dummies(reason_cat, prefix="reason", dtype=float)
    df = pd.concat([df, reason_dummies], axis=1)

    feature_cols = [
        "days_to_return", "buyer_age_days_at_return", "seller_age_days_at_return", "order_amount",
        "buyer_prior_returns", "buyer_orders_before_return", "buyer_return_rate_before",
        "seller_prior_returns", "seller_orders_before_return", "seller_return_rate_before",
    ] + list(reason_dummies.columns)

    df["y"] = df["is_fraudulent"].astype(int)
    return df, feature_cols


def run_return_fraud_detector(returns_df, orders_df, buyers_df, sellers_df):
    """Executes return fraud detection pipeline."""
    df, feature_cols = build_return_features(returns_df, orders_df, buyers_df, sellers_df)
    train = df[df["return_date"] <= TRAIN_END]
    val = df[(df["return_date"] > TRAIN_END) & (df["return_date"] <= VAL_END)]
    test = df[df["return_date"] > VAL_END]

    print(f"\nReturn Fraud Detector — Train: {len(train)} ({train['y'].mean():.2%}), Val: {len(val)}, Test: {len(test)}")
    return train_eval(train, val, test, feature_cols, "Return Fraud Detector", amount_col="order_amount", fp_cost=100)


if __name__ == "__main__":
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"])
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"], catalog["products"],
        base["address_sharing_log"], base["device_sharing_log"],
    )

    run_fake_listing_detector(result["listings"], catalog["sellers"], catalog["products"])
    run_return_fraud_detector(result["returns"], result["orders"], txn["buyers"], catalog["sellers"])
