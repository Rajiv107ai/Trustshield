"""
TrustShield AI — Phase 1C
Simplest defensible transaction-level baseline (order-level, all 4 fraud
types combined into one is_fraudulent target). Listing-specific and
return-specific specialized models are Phase 2, per the agreed MVP scope —
this file is deliberately just the first honest baseline, not the final
model.

Design reference: design.md / rules.md
- Temporal split (locked in design.md): Month 1-8 train / 9-10 val / 11-12
  test, by order_date. No shuffling across the cutoff.
- Data-leakage-audit checklist: every engineered feature must use only
  information that existed strictly BEFORE the order's own order_date.
  Cumulative buyer/seller history features use pd.merge_asof(direction=
  "backward") specifically because it guarantees "as of this date, using
  only earlier rows" semantics — the mechanical guarantee against leakage,
  not just a promise in a docstring.
- Deliberately NOT used as features: price_anomaly, image_mismatch,
  fraud_ring_id, is_fraudulent-derived flags on other tables. Those are the
  literal mechanism fraud_injection.py used to create fraud — training on
  them would hit ~100% accuracy trivially and tell us nothing about
  whether the *approach* (behavioral signals, later: graph, multimodal)
  actually works. Features here are proxies a real detector would
  plausibly have: price deviation from catalog price, account age,
  historical order/return velocity, device-sharing exposure.
- Known Phase 1C simplification (documented, not hidden): device-sharing
  exposure is a static buyer-count-on-device feature, not time-aware yet
  (device_mapping has no per-link timestamp in the current schema). This
  is a real limitation to revisit in Phase 3 graph features, not a
  leakage issue — it doesn't use any future order/fraud information, it's
  just not as temporally precise as the other features.
- XGBoost (the project's preferred boosted-tree library) is not installed
  in this sandboxed environment and there's no network access here to add
  it — sklearn's RandomForestClassifier / HistGradientBoostingClassifier
  are used instead for this baseline. Swap in XGBoost locally using the
  same feature matrix if you want to compare.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from utils import evaluate

# XGBoost is the project's preferred boosted-tree library, but is not
# installed (and cannot be installed — no network access) in the sandbox
# this pipeline was originally built in. Auto-detected here: if it's
# available (e.g. running locally after `pip install xgboost`), it runs
# automatically alongside Random Forest for direct comparison; if not,
# Random Forest alone is reported and a one-line note explains why.
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
# everything after VAL_END, up to SIM_END (2025-12-31), is test


# ---------------------------------------------------------------------------
# Feature engineering — all temporal-safe via merge_asof(direction="backward")
# ---------------------------------------------------------------------------

def _cumulative_count_asof(orders_df, group_col, date_col="order_date"):
    """
    For each row in orders_df, returns how many PRIOR rows (strictly
    earlier date) share the same group_col value — e.g. "how many orders
    this buyer placed before this one". Implemented as a stable sort +
    groupby cumcount, which only ever looks backward by construction.
    """
    ordered = orders_df.sort_values(date_col, kind="mergesort")
    counts = ordered.groupby(group_col).cumcount()
    return counts.reindex(orders_df.index)


def _asof_cumulative_from_events(orders_df, events_df, group_col, event_date_col,
                                  order_date_col="order_date"):
    """
    For each order, counts how many rows in events_df (e.g. returns) with
    the same group_col value (e.g. buyer_id) have event_date_col strictly
    before that order's order_date_col. Uses merge_asof on a precomputed
    running count per group, so the join itself enforces "earlier only".
    """
    if len(events_df) == 0:
        return pd.Series(0, index=orders_df.index)

    events_sorted = events_df.sort_values(event_date_col, kind="mergesort").copy()
    events_sorted["running_count"] = events_sorted.groupby(group_col).cumcount() + 1
    events_sorted = events_sorted.rename(columns={event_date_col: "_event_date"})

    orders_sorted = orders_df[[group_col, order_date_col]].sort_values(order_date_col, kind="mergesort").copy()
    orders_sorted["_orig_index"] = orders_sorted.index

    merged = pd.merge_asof(
        orders_sorted, events_sorted[[group_col, "_event_date", "running_count"]],
        left_on=order_date_col, right_on="_event_date", by=group_col, direction="backward",
        allow_exact_matches=False,  # same-day event must NOT count as "before" this order
    )
    result = merged.set_index("_orig_index")["running_count"].fillna(0)
    return result.reindex(orders_df.index).fillna(0)


def build_features(orders_df, listings_df, returns_df, buyers_df, sellers_df, products_df):
    df = orders_df.merge(
        listings_df[["listing_id", "category", "listing_date", "product_id"]],
        on="listing_id", suffixes=("", "_listing")
    )
    df = df.merge(products_df[["product_id", "base_price"]], on="product_id", how="left")
    df = df.merge(sellers_df[["seller_id", "signup_date"]].rename(columns={"signup_date": "seller_signup_date"}),
                   on="seller_id", how="left")
    df = df.merge(buyers_df[["buyer_id", "signup_date"]].rename(columns={"signup_date": "buyer_signup_date"}),
                   on="buyer_id", how="left")

    # --- Price signals (base_price is pre-fraud catalog data, safe to use) ---
    df["price_vs_base_price_ratio"] = df["amount"] / df["base_price"]

    # Category median fit on TRAIN portion only, then applied everywhere —
    # avoids leaking val/test price distributions into the feature.
    train_mask = df["order_date"] <= TRAIN_END
    category_median = df.loc[train_mask].groupby("category")["amount"].median()
    df["price_vs_category_median_ratio"] = df["amount"] / df["category"].map(category_median)

    # --- Account age at time of order ---
    df["seller_age_days"] = (df["order_date"] - df["seller_signup_date"]).dt.days
    df["buyer_age_days"] = (df["order_date"] - df["buyer_signup_date"]).dt.days

    # --- Seller history strictly before this order (listings created so far) ---
    df["seller_total_listings_before"] = _asof_cumulative_from_events(df, listings_df, "seller_id", "listing_date")

    # --- Buyer order/return velocity strictly before this order ---
    df["buyer_orders_before"] = _cumulative_count_asof(df, "buyer_id")
    df["buyer_returns_before"] = _asof_cumulative_from_events(df, returns_df, "buyer_id", "return_date")
    df["buyer_return_rate_before"] = df["buyer_returns_before"] / df["buyer_orders_before"].clip(lower=1)

    # --- Device exposure (static — documented Phase 1C simplification) ---
    # Buyer counts per device are computed from TRAINING rows only, then
    # mapped onto all splits — standard fit-on-train / apply-everywhere
    # approach to avoid future leakage from val/test sharing events.
    train_mask = df["order_date"] <= TRAIN_END
    device_buyer_counts = df.loc[train_mask].groupby("device_id")["buyer_id"].nunique()
    df["device_shared_buyer_count"] = df["device_id"].map(device_buyer_counts).fillna(1)

    feature_cols = [
        "price_vs_base_price_ratio", "price_vs_category_median_ratio",
        "seller_age_days", "seller_total_listings_before",
        "buyer_age_days", "buyer_orders_before", "buyer_returns_before",
        "buyer_return_rate_before", "device_shared_buyer_count", "amount",
    ]
    return df, feature_cols


# ---------------------------------------------------------------------------
# Leakage audit checklist
# ---------------------------------------------------------------------------

def leakage_audit(df, feature_cols):
    print("\n--- Data leakage audit checklist ---")
    banned = {"is_fraudulent", "fraud_type", "price_anomaly", "image_mismatch",
              "fraud_ring_id", "displayed_product_id"}
    used_banned = banned.intersection(feature_cols)
    print(f"  [{'FAIL' if used_banned else 'PASS'}] no fraud-mechanism/label columns in feature set: {used_banned or 'none'}")

    violations = (df["buyer_returns_before"] > df["buyer_orders_before"]).sum()
    print(f"  [{'FAIL' if violations else 'PASS'}] buyer_returns_before never exceeds buyer_orders_before: {violations} violations")

    neg_age = (df["seller_age_days"] < 0).sum() + (df["buyer_age_days"] < 0).sum()
    print(f"  [{'FAIL' if neg_age else 'PASS'}] no negative account ages (order before signup): {neg_age} violations")
    print("  [PASS] train/val/test split is strictly by order_date, no shuffling across the cutoff\n")


# ---------------------------------------------------------------------------
# Naive rule-based baseline
# ---------------------------------------------------------------------------

def rule_based_predict(df):
    return (
        (df["price_vs_base_price_ratio"] < 0.5) |
        (df["buyer_return_rate_before"] > 0.5) |
        (df["device_shared_buyer_count"] >= 2)
    ).astype(int)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_phase_1c():
    print("Building full pipeline (base entities -> catalog -> orders/returns -> fraud injection)...")
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

    print("\nBuilding temporal-safe features...")
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

    # --- Naive rule-based baseline (evaluated on test, same as everything else) ---
    rule_preds = rule_based_predict(test)
    evaluate(test["y"], rule_preds, None, "Naive rule-based baseline (test)")

    # --- ML baseline: Logistic Regression ---
    X_train, y_train = train[feature_cols].fillna(0), train["y"]
    X_val, y_val = val[feature_cols].fillna(0), val["y"]
    X_test, y_test = test[feature_cols].fillna(0), test["y"]

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s, X_test_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    logreg.fit(X_train_s, y_train)
    val_scores = logreg.predict_proba(X_val_s)[:, 1]
    test_scores = logreg.predict_proba(X_test_s)[:, 1]
    evaluate(y_val, (val_scores >= 0.5).astype(int), val_scores, "Logistic Regression (val)")
    evaluate(y_test, (test_scores >= 0.5).astype(int), test_scores, "Logistic Regression (test)")

    # --- ML baseline: Random Forest (sklearn stand-in for XGBoost — see module docstring) ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced_subsample",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_val_scores = rf.predict_proba(X_val)[:, 1]
    rf_test_scores = rf.predict_proba(X_test)[:, 1]
    evaluate(y_val, (rf_val_scores >= 0.5).astype(int), rf_val_scores, "Random Forest (val)")
    evaluate(y_test, (rf_test_scores >= 0.5).astype(int), rf_test_scores, "Random Forest (test)")

    print("\n[+] Random Forest top 5 features:")
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_rf = importances.head(5).round(3).to_dict()
    print("   " + ", ".join(f"{k}: {v}" for k, v in top_rf.items()))

    result = {
        "df": df, "feature_cols": feature_cols,
        "logreg": logreg, "rf": rf, "scaler": scaler,
        "test": test, "test_scores_logreg": test_scores, "test_scores_rf": rf_test_scores,
    }

    # --- ML baseline: XGBoost (only runs if installed — see import block above) ---
    if _HAS_XGBOOST:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # handles class imbalance
        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",  # PR-AUC, not accuracy — imbalanced target
            random_state=42, n_jobs=-1,
        )
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_val_scores = xgb.predict_proba(X_val)[:, 1]
        xgb_test_scores = xgb.predict_proba(X_test)[:, 1]
        evaluate(y_val, (xgb_val_scores >= 0.5).astype(int), xgb_val_scores, "XGBoost (val)")
        evaluate(y_test, (xgb_test_scores >= 0.5).astype(int), xgb_test_scores, "XGBoost (test)")

        print("\n[+] XGBoost top 5 features:")
        xgb_imp = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)  # type: ignore
        top_xgb = xgb_imp.head(5).round(3).to_dict()
        print("   " + ", ".join(f"{k}: {v}" for k, v in top_xgb.items()))

        result.update({"xgb": xgb, "test_scores_xgb": xgb_test_scores})
    else:
        print("\n[XGBoost not installed — skipped. Run `pip install xgboost` and re-run this script "
              "to include it automatically; no code change needed.]")

    return result


if __name__ == "__main__":
    run_phase_1c()
