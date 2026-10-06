"""Model training and artifact serialization script for API serving."""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..", "trustshield_project")
sys.path.insert(0, PROJECT_DIR)

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from entity_generator import build_base_entities, SIM_START
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud
from phase2_specialized_models import (
    build_listing_features,
    build_return_features,
    TRAIN_END as P2_TRAIN_END,
)
from graph_features import (
    build_relationship_graph,
    compute_relationship_features,
    build_monthly_snapshots,
    attach_snapshot_features,
    add_edge_weight_before,
    detect_fraud_rings,
)
from baseline_model import build_features as build_tabular_features, TRAIN_END, VAL_END, leakage_audit

MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _make_model(n_train_pos: int = 100, n_train_total: int = 1000) -> XGBClassifier:
    neg = n_train_total - n_train_pos
    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / max(n_train_pos, 1),
        eval_metric="aucpr", random_state=42,
        n_jobs=-1, verbosity=0,
    )


def main():
    print("Generating training pipeline...")
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"])
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"],
        catalog["products"], base["address_sharing_log"], base["device_sharing_log"],
    )

    print("Training Fake Listing Detector...")
    listings_feat, listing_cols = build_listing_features(result["listings"], catalog["sellers"], catalog["products"])
    listings_feat["y"] = listings_feat["is_fraudulent"].astype(int)
    listing_train = pd.DataFrame(listings_feat[listings_feat["listing_date"] <= P2_TRAIN_END])
    rf_listing = _make_model(int(np.count_nonzero(listing_train["y"])), len(listing_train))
    rf_listing.fit(pd.DataFrame(listing_train[listing_cols]).fillna(0), listing_train["y"])
    joblib.dump(rf_listing, os.path.join(MODELS_DIR, "fake_listing_model.joblib"))

    print("Training Return Fraud Detector...")
    returns_feat, return_cols = build_return_features(result["returns"], result["orders"], txn["buyers"], catalog["sellers"])
    returns_feat["y"] = returns_feat["is_fraudulent"].astype(int)
    return_train = pd.DataFrame(returns_feat[returns_feat["return_date"] <= P2_TRAIN_END])
    rf_return = _make_model(int(np.count_nonzero(return_train["y"])), len(return_train))
    rf_return.fit(pd.DataFrame(return_train[return_cols]).fillna(0), return_train["y"])
    joblib.dump(rf_return, os.path.join(MODELS_DIR, "return_fraud_model.joblib"))

    print("Training Combined Graph Model...")
    df, tabular_cols = build_tabular_features(
        result["orders"], result["listings"], result["returns"],
        txn["buyers"], catalog["sellers"], catalog["products"]
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["y"] = df["is_fraudulent"].astype(int)
    leakage_audit(df, tabular_cols)

    # TS-AUD-02: Build temporally-isolated relationship graphs per split (prevent future leakage)
    rel_graph_train = build_relationship_graph(
        base["address_sharing_log"], base["device_sharing_log"], cutoff_date=TRAIN_END
    )
    rel_graph_val = build_relationship_graph(
        base["address_sharing_log"], base["device_sharing_log"], cutoff_date=VAL_END
    )
    rel_graph = rel_graph_val

    train_mask = df["order_date"] <= TRAIN_END
    val_test_mask = ~train_mask

    rel_feat_train = compute_relationship_features(rel_graph_train, df.loc[train_mask, "buyer_id"].unique())
    rel_feat_val = compute_relationship_features(rel_graph_val, df.loc[val_test_mask, "buyer_id"].unique())

    df.loc[train_mask, "share_degree"] = 0.0
    df.loc[train_mask, "share_component_size"] = 0.0
    df.loc[val_test_mask, "share_degree"] = 0.0
    df.loc[val_test_mask, "share_component_size"] = 0.0

    tmp_train = df.loc[train_mask, ["buyer_id"]].merge(rel_feat_train, on="buyer_id", how="left")
    df.loc[train_mask, "share_degree"] = tmp_train["share_degree"].fillna(0).values
    df.loc[train_mask, "share_component_size"] = tmp_train["share_component_size"].fillna(0).values

    tmp_val = df.loc[val_test_mask, ["buyer_id"]].merge(rel_feat_val, on="buyer_id", how="left")
    df.loc[val_test_mask, "share_degree"] = tmp_val["share_degree"].fillna(0).values
    df.loc[val_test_mask, "share_component_size"] = tmp_val["share_component_size"].fillna(0).values

    snapshots, months = build_monthly_snapshots(
        result["orders"].assign(order_date=pd.to_datetime(result["orders"]["order_date"])), SIM_START
    )
    df = attach_snapshot_features(df, snapshots, months)
    df = add_edge_weight_before(df)

    graph_cols = [
        "share_degree", "share_component_size", "buyer_seller_degree", "buyer_pagerank",
        "seller_buyer_degree", "seller_pagerank", "seller_buyer_concentration_hhi",
        "buyer_seller_edge_weight_before",
    ]
    all_feature_cols = tabular_cols + graph_cols

    train = pd.DataFrame(df[df["order_date"] <= TRAIN_END])
    val = pd.DataFrame(df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)])
    test = pd.DataFrame(df[df["order_date"] > VAL_END])

    rf_combined = _make_model(int(np.count_nonzero(train["y"])), len(train))
    rf_combined.fit(pd.DataFrame(train[all_feature_cols]).fillna(0), train["y"])
    joblib.dump(rf_combined, os.path.join(MODELS_DIR, "combined_graph_model.joblib"))

    all_scores = np.concatenate([
        np.asarray(rf_combined.predict_proba(pd.DataFrame(split[all_feature_cols]).fillna(0)))[:, 1]
        for split in [train, val, test]
    ])
    df_scored = pd.concat([train, val, test], ignore_index=True).assign(fraud_score=all_scores)

    rings_df = detect_fraud_rings(rel_graph, df_scored, score_col="fraud_score", min_ring_size=2)
    joblib.dump(rings_df, os.path.join(MODELS_DIR, "fraud_rings.joblib"))

    feature_meta = {
        "tabular_cols": tabular_cols,
        "graph_cols": graph_cols,
        "all_feature_cols": all_feature_cols,
        "listing_feature_cols": listing_cols,
        "return_feature_cols": return_cols,
    }
    joblib.dump(feature_meta, os.path.join(MODELS_DIR, "feature_meta.joblib"))
    print("All production model artifacts successfully saved to models/")


if __name__ == "__main__":
    main()
