"""
TrustShield AI — Model Training and Serialization Script

Runs the full data pipeline once, trains all models (Phase 2 + Phase 3),
and saves every artifact the API needs to ../models/ via joblib.

Run once before starting the FastAPI server:
    cd trustshield_full_handoff
    python scripts/train_and_save_models.py

Artifacts saved:
    models/fake_listing_model.joblib   -- Phase 2 Fake Listing RF
    models/return_fraud_model.joblib   -- Phase 2 Return Fraud RF
    models/combined_graph_model.joblib -- Phase 3 tabular+graph RF
    models/fraud_rings.joblib          -- ranked rings DataFrame
    models/feature_meta.joblib         -- dict of feature column lists
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..", "trustshield_project")
sys.path.insert(0, PROJECT_DIR)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# XGBoost auto-detect — uses XGBoost for all models when available
try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
    print("[train_and_save] XGBoost detected — using XGBoost for all models")
except ImportError:
    _HAS_XGBOOST = False
    print("[train_and_save] XGBoost not found — using RandomForest fallback")

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


def _make_model(n_train_pos=None, n_train_total=None):
    """Return XGBoost if available, otherwise RandomForest."""
    if _HAS_XGBOOST:
        neg = (n_train_total or 1000) - (n_train_pos or 100)
        pos = n_train_pos or 100
        return XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg / max(pos, 1),
            eval_metric="aucpr", random_state=42,
            n_jobs=-1, verbosity=0,
        )
    return RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )


def main():
    print("=" * 60)
    print("TrustShield -- Training all models for API serving")
    print("=" * 60)

    print("\n[1/5] Running data pipeline...")
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(
        base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"]
    )
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"],
        catalog["products"], base["address_sharing_log"], base["device_sharing_log"],
    )

    print("\n[2/5] Training Phase 2 -- Fake Listing Detector...")
    listings_feat, listing_cols = build_listing_features(result["listings"], catalog["sellers"], catalog["products"])
    listings_feat = listings_feat.copy()
    listings_feat["y"] = listings_feat["is_fraudulent"].astype(int)
    listing_train = listings_feat[listings_feat["listing_date"] <= P2_TRAIN_END]
    rf_listing = _make_model(int(listing_train["y"].sum()), len(listing_train))
    rf_listing.fit(listing_train[listing_cols].fillna(0), listing_train["y"])
    print(f"  Trained on {len(listing_train)} samples ({listing_train['y'].mean():.2%} fraud rate)")
    joblib.dump(rf_listing, os.path.join(MODELS_DIR, "fake_listing_model.joblib"))
    print("  Saved: models/fake_listing_model.joblib")

    print("\n[3/5] Training Phase 2 -- Return Fraud Detector...")
    returns_feat, return_cols = build_return_features(result["returns"], result["orders"], txn["buyers"], catalog["sellers"])
    returns_feat = returns_feat.copy()
    returns_feat["y"] = returns_feat["is_fraudulent"].astype(int)
    return_train = returns_feat[returns_feat["return_date"] <= P2_TRAIN_END]
    rf_return = _make_model(int(return_train["y"].sum()), len(return_train))
    rf_return.fit(return_train[return_cols].fillna(0), return_train["y"])
    print(f"  Trained on {len(return_train)} samples ({return_train['y'].mean():.2%} fraud rate)")
    joblib.dump(rf_return, os.path.join(MODELS_DIR, "return_fraud_model.joblib"))
    print("  Saved: models/return_fraud_model.joblib")

    print("\n[4/5] Training Phase 3 -- Combined graph model + ring detection...")
    df, tabular_cols = build_tabular_features(
        result["orders"], result["listings"], result["returns"],
        txn["buyers"], catalog["sellers"], catalog["products"]
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["y"] = df["is_fraudulent"].astype(int)
    leakage_audit(df, tabular_cols)

    rel_graph = build_relationship_graph(base["address_sharing_log"], base["device_sharing_log"])
    rel_features = compute_relationship_features(rel_graph, df["buyer_id"].unique())
    df = df.merge(rel_features, on="buyer_id", how="left")
    df[["share_degree", "share_component_size"]] = df[["share_degree", "share_component_size"]].fillna(0)

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

    train = df[df["order_date"] <= TRAIN_END]
    val   = df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)]
    test  = df[df["order_date"] > VAL_END]

    rf_combined = _make_model(int(train["y"].sum()), len(train))
    rf_combined.fit(train[all_feature_cols].fillna(0), train["y"])
    print(f"  Trained on {len(train)} samples ({train['y'].mean():.2%} fraud rate)")
    joblib.dump(rf_combined, os.path.join(MODELS_DIR, "combined_graph_model.joblib"))
    print("  Saved: models/combined_graph_model.joblib")

    all_scores = np.concatenate([
        rf_combined.predict_proba(split[all_feature_cols].fillna(0))[:, 1]
        for split in [train, val, test]
    ])
    df_scored = pd.concat([train, val, test]).copy()
    df_scored["fraud_score"] = all_scores
    assert "fraud_ring_id" not in df_scored.columns

    rings_df = detect_fraud_rings(rel_graph, df_scored, score_col="fraud_score", min_ring_size=2)
    joblib.dump(rings_df, os.path.join(MODELS_DIR, "fraud_rings.joblib"))
    print(f"  Ring detection: {len(rings_df)} rings, {(rings_df['avg_risk_score'] >= 0.3).sum()} high-risk")
    print("  Saved: models/fraud_rings.joblib")

    print("\n[5/5] Saving feature metadata...")
    feature_meta = {
        "tabular_cols": tabular_cols,
        "graph_cols": graph_cols,
        "all_feature_cols": all_feature_cols,
        "listing_feature_cols": listing_cols,
        "return_feature_cols": return_cols,
    }
    joblib.dump(feature_meta, os.path.join(MODELS_DIR, "feature_meta.joblib"))
    print("  Saved: models/feature_meta.joblib")

    print("\nAll models saved. Start the API with:")
    print("    cd backend && uvicorn main:app --reload")


if __name__ == "__main__":
    main()


