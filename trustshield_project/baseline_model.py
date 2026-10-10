"""Baseline transaction fraud models: Rule-based, Logistic Regression, Random Forest, and XGBoost."""

import numpy as np
import pandas as pd
from typing import cast
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from utils import evaluate
from entity_generator import build_base_entities
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud

TRAIN_END = pd.Timestamp("2025-08-31")
VAL_END = pd.Timestamp("2025-10-31")


def _cumulative_count_asof(orders_df: pd.DataFrame, group_col: str, date_col: str = "order_date") -> pd.Series:
    """Prior row count per group strictly before current row's timestamp.

    Uses merge_asof with allow_exact_matches=False so that two rows with the
    *same* timestamp for the same entity are never counted as prior history of
    each other — preventing temporal leakage on same-day/same-second events.
    """
    # Build a running-count lookup keyed by (group_col, date_col)
    sorted_df = cast(pd.DataFrame, orders_df[[group_col, date_col]]).sort_values(by=date_col, kind="mergesort").copy()
    sorted_df["_running"] = sorted_df.groupby(group_col).cumcount() + 1  # 1-indexed count after current row

    left = cast(pd.DataFrame, orders_df[[group_col, date_col]]).copy()
    left["_orig_index"] = left.index
    left_sorted = left.sort_values(by=date_col, kind="mergesort")

    merged = pd.merge_asof(
        left_sorted, sorted_df[[group_col, date_col, "_running"]],
        on=date_col, by=group_col,
        direction="backward",
        allow_exact_matches=False,  # strictly BEFORE current timestamp
    )
    return pd.Series(
        merged.set_index("_orig_index")["_running"].reindex(orders_df.index).fillna(0),
        name=date_col,
    )


def _asof_cumulative_from_events(orders_df: pd.DataFrame, events_df: pd.DataFrame, group_col: str, 
                                 event_date_col: str, order_date_col: str = "order_date") -> pd.Series:
    """Count of prior events per group strictly before order date."""
    if events_df.empty:
        return pd.Series(0, index=orders_df.index)

    events_sorted = events_df.sort_values(by=event_date_col, kind="mergesort").copy()
    events_sorted["running_count"] = events_sorted.groupby(group_col).cumcount() + 1
    events_sorted = events_sorted.rename(columns={event_date_col: "_event_date"})

    orders_sub = pd.DataFrame(orders_df[[group_col, order_date_col]])
    orders_sorted = orders_sub.sort_values(by=order_date_col, kind="mergesort").copy()
    orders_sorted["_orig_index"] = orders_sorted.index

    merged = pd.merge_asof(
        orders_sorted, events_sorted[[group_col, "_event_date", "running_count"]],
        left_on=order_date_col, right_on="_event_date", by=group_col, direction="backward",
        allow_exact_matches=False,
    )
    return pd.Series(merged.set_index("_orig_index")["running_count"].reindex(orders_df.index).fillna(0))


def _device_shared_buyer_count_asof(
    orders_df: pd.DataFrame,
    device_col: str = "device_id",
    buyer_col: str = "buyer_id",
    date_col: str = "order_date",
) -> pd.Series:
    """Distinct buyers seen on device strictly before the current order's timestamp.

    Eliminates temporal leakage where future orders on the same device would
    statically inflate device sharing degree for earlier transactions.
    Uses merge_asof with allow_exact_matches=False so that only distinct buyers
    observed at timestamps strictly before current order date are counted.
    """
    if orders_df.empty or device_col not in orders_df.columns or buyer_col not in orders_df.columns:
        return pd.Series(0.0, index=orders_df.index)

    orders_sub = orders_df[[device_col, buyer_col, date_col]].copy()
    orders_sub[date_col] = pd.to_datetime(orders_sub[date_col])

    # Earliest order date for each (device_id, buyer_id) pair
    first_use = (
        orders_sub.groupby([device_col, buyer_col], as_index=False)[date_col]
        .min()
        .sort_values(by=date_col, kind="mergesort")
        .reset_index(drop=True)
    )
    # Running count of distinct buyers on device up to that first_use date
    first_use["_distinct_buyers"] = first_use.groupby(device_col).cumcount() + 1

    left = orders_sub[[device_col, date_col]].copy()
    left["_orig_idx"] = left.index
    left_sorted = left.sort_values(by=date_col, kind="mergesort")

    # Match datetime units for pandas merge_asof
    dt_left = left_sorted[date_col].dt.as_unit("ns")
    dt_right = first_use[date_col].dt.as_unit("ns")
    left_sorted["_dt_key"] = dt_left
    first_use["_dt_key"] = dt_right

    merged = pd.merge_asof(
        left_sorted,
        first_use[[device_col, "_dt_key", "_distinct_buyers"]],
        on="_dt_key",
        by=device_col,
        direction="backward",
        allow_exact_matches=False,
    )
    return pd.Series(
        merged.set_index("_orig_idx")["_distinct_buyers"].reindex(orders_df.index).fillna(0.0),
        name="device_shared_buyer_count",
    )


def build_features(orders_df, listings_df, returns_df, buyers_df, sellers_df, products_df):
    """Engineers temporal-safe features for fraud detection."""
    df = orders_df.merge(
        listings_df[["listing_id", "category", "listing_date", "product_id"]],
        on="listing_id", suffixes=("", "_listing")
    )
    df = df.merge(products_df[["product_id", "base_price"]], on="product_id", how="left")
    df = df.merge(sellers_df[["seller_id", "signup_date"]].rename(columns={"signup_date": "seller_signup_date"}),
                  on="seller_id", how="left")
    df = df.merge(buyers_df[["buyer_id", "signup_date"]].rename(columns={"signup_date": "buyer_signup_date"}),
                  on="buyer_id", how="left")

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["seller_signup_date"] = pd.to_datetime(df["seller_signup_date"])
    df["buyer_signup_date"] = pd.to_datetime(df["buyer_signup_date"])

    df["price_vs_base_price_ratio"] = (
        df["amount"] / df["base_price"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    train_mask = df["order_date"] <= TRAIN_END
    category_median = df.loc[train_mask].groupby("category")["amount"].median()
    df["price_vs_category_median_ratio"] = (
        df["amount"] / df["category"].map(category_median).replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["seller_age_days"] = (df["order_date"] - df["seller_signup_date"]).dt.days
    df["buyer_age_days"] = (df["order_date"] - df["buyer_signup_date"]).dt.days
    df["seller_total_listings_before"] = _asof_cumulative_from_events(df, listings_df, "seller_id", "listing_date")

    df["buyer_orders_before"] = _cumulative_count_asof(df, "buyer_id")
    df["buyer_returns_before"] = _asof_cumulative_from_events(df, returns_df, "buyer_id", "return_date")
    df["buyer_return_rate_before"] = df["buyer_returns_before"] / df["buyer_orders_before"].clip(lower=1)

    # Point-in-time device sharing count (strictly prior unique buyers on device)
    df["device_shared_buyer_count"] = _device_shared_buyer_count_asof(df)

    feature_cols = [
        "price_vs_base_price_ratio", "price_vs_category_median_ratio",
        "seller_age_days", "seller_total_listings_before",
        "buyer_age_days", "buyer_orders_before", "buyer_returns_before",
        "buyer_return_rate_before", "device_shared_buyer_count", "amount",
    ]
    return df, feature_cols


def leakage_audit(df, feature_cols):
    """Sanity checks features against target leakage and chronological constraints.

    Banned columns fall into two categories:
    1. Ground-truth labels — columns that ARE the answer (fraud labels, ring IDs).
    2. Model-output columns — columns that are DERIVED FROM a model's prediction
       and must not be fed back as features, which would create a feedback loop
       (the model would be predicting its own past outputs).
    """
    banned = {
        # Ground-truth label columns
        "is_fraudulent", "fraud_type", "price_anomaly", "image_mismatch",
        "fraud_ring_id", "displayed_product_id",
        # Model-output columns (FIX-04: feedback loop guard)
        "trust_score", "risk_score", "overall_fraud_probability",
        "predicted_fraud", "decision", "model_reason_code", "model_probability",
    }
    used_banned = banned.intersection(feature_cols)
    assert not used_banned, f"Leakage: banned target columns in features: {used_banned}"
    violations = (df["buyer_returns_before"] > df["buyer_orders_before"]).sum()
    assert violations == 0, f"Leakage: buyer_returns_before > buyer_orders_before ({violations} violations)"
    neg_age = (df["seller_age_days"] < 0).sum() + (df["buyer_age_days"] < 0).sum()
    assert neg_age == 0, f"Leakage: negative account age ({neg_age} violations)"


def rule_based_predict(df: pd.DataFrame) -> pd.Series:
    """Heuristic fraud baseline."""
    return (
        (df["price_vs_base_price_ratio"] < 0.5) |
        (df["buyer_return_rate_before"] > 0.5) |
        (df["device_shared_buyer_count"] >= 2)
    ).astype(int)


def run_phase_1c():
    """Trains and evaluates baseline transaction fraud models."""
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(
        buyers_df=base["buyers"], sellers_df=catalog["sellers"],
        listings_df=catalog["listings"], device_mapping_df=base["device_mapping"],
    )
    result = inject_all_fraud(
        listings_df=catalog["listings"], orders_df=txn["orders"], returns_df=txn["returns"],
        buyers_df=txn["buyers"], products_df=catalog["products"],
        address_sharing_log=base["address_sharing_log"], device_sharing_log=base["device_sharing_log"],
    )

    df, feature_cols = build_features(
        result["orders"], result["listings"], result["returns"],
        txn["buyers"], catalog["sellers"], catalog["products"]
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["y"] = df["is_fraudulent"].astype(int)
    leakage_audit(df, feature_cols)

    train = df[df["order_date"] <= TRAIN_END]
    val = df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)]
    test = df[df["order_date"] > VAL_END]

    print(f"Split sizes: train={len(train)} ({train['y'].mean():.2%} fraud), "
          f"val={len(val)} ({val['y'].mean():.2%} fraud), "
          f"test={len(test)} ({test['y'].mean():.2%} fraud)")

    # Heuristic baseline
    evaluate(test["y"], rule_based_predict(test), None, "Rule-based baseline (test)")

    # Logistic Regression
    X_train, y_train = train[feature_cols].fillna(0), train["y"]
    X_val, y_val = val[feature_cols].fillna(0), val["y"]
    X_test, y_test = test[feature_cols].fillna(0), test["y"]

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s, X_test_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    logreg.fit(X_train_s, y_train)
    val_scores = np.asarray(logreg.predict_proba(X_val_s))[:, 1]
    test_scores = np.asarray(logreg.predict_proba(X_test_s))[:, 1]
    evaluate(y_val, (val_scores >= 0.5).astype(int), val_scores, "Logistic Regression (val)")
    evaluate(y_test, (test_scores >= 0.5).astype(int), test_scores, "Logistic Regression (test)")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_val_scores = np.asarray(rf.predict_proba(X_val))[:, 1]
    rf_test_scores = np.asarray(rf.predict_proba(X_test))[:, 1]
    evaluate(y_val, (rf_val_scores >= 0.5).astype(int), rf_val_scores, "Random Forest (val)")
    evaluate(y_test, (rf_test_scores >= 0.5).astype(int), rf_test_scores, "Random Forest (test)")

    # XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val_scores = np.asarray(xgb.predict_proba(X_val))[:, 1]
    xgb_test_scores = np.asarray(xgb.predict_proba(X_test))[:, 1]
    evaluate(y_val, (xgb_val_scores >= 0.5).astype(int), xgb_val_scores, "XGBoost (val)")
    evaluate(y_test, (xgb_test_scores >= 0.5).astype(int), xgb_test_scores, "XGBoost (test)")

    return {
        "df": df, "feature_cols": feature_cols,
        "logreg": logreg, "rf": rf, "xgb": xgb, "scaler": scaler,
        "test": test, "test_scores_logreg": test_scores, 
        "test_scores_rf": rf_test_scores, "test_scores_xgb": xgb_test_scores,
    }


if __name__ == "__main__":
    run_phase_1c()
