"""
TrustShield AI — pytest shared fixtures

Builds the full synthetic pipeline ONCE per test session (entity generation →
catalog → orders/returns → fraud injection → feature engineering) and caches
the result so individual test modules don't each spend ~30s regenerating data.
"""

import pandas as pd
import pytest

from entity_generator import build_base_entities
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud
from baseline_model import build_features, TRAIN_END, VAL_END


@pytest.fixture(scope="session")
def pipeline():
    """Build and return the full pipeline output as a dict."""
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(
        base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"]
    )
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"], catalog["products"],
        base["address_sharing_log"], base["device_sharing_log"],
    )

    df, feature_cols = build_features(
        result["orders"], result["listings"], result["returns"],
        txn["buyers"], catalog["sellers"], catalog["products"]
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["y"] = df["is_fraudulent"].astype(int)

    return {
        "base": base,
        "catalog": catalog,
        "txn": txn,
        "result": result,             # post-fraud-injection tables
        "df": df,                      # feature-engineered order-level df
        "feature_cols": feature_cols,
        "train": df[df["order_date"] <= TRAIN_END],
        "val": df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)],
        "test": df[df["order_date"] > VAL_END],
    }
