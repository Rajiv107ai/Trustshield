import os
import sys

# Ensure current directory is on sys.path for direct module imports
_project_dir = os.path.dirname(os.path.abspath(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import pandas as pd
import pytest

from entity_generator import build_base_entities
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud
from baseline_model import build_features, TRAIN_END, VAL_END


@pytest.fixture(scope="session")
def pipeline():
    """Generates synthetic dataset and feature matrices once per test session."""
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
        "result": result,
        "df": df,
        "feature_cols": feature_cols,
        "train": df[df["order_date"] <= TRAIN_END],
        "val": df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)],
        "test": df[df["order_date"] > VAL_END],
    }
