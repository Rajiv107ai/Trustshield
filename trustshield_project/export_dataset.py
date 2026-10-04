"""Exports generated pipeline tables to disk or database."""

import os
import argparse
from entity_generator import build_base_entities
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_data_export")


def export_dataset(db_url: str | None = None):
    """Executes pipeline generation and exports tabular results."""
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"])
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"], catalog["products"],
        base["address_sharing_log"], base["device_sharing_log"],
    )

    tables = {
        "addresses": base["addresses"],
        "devices": base["devices"],
        "sellers": txn["sellers"],
        "buyers": txn["buyers"],
        "device_mapping": base["device_mapping"],
        "address_sharing_log": base["address_sharing_log"],
        "device_sharing_log": base["device_sharing_log"],
        "products": catalog["products"],
        "listings": result["listings"],
        "orders": result["orders"],
        "returns": result["returns"],
        "fraud_ground_truth": result["fraud_ground_truth"],
    }

    if db_url:
        from sqlalchemy import create_engine
        engine = create_engine(db_url)
        for name, df in tables.items():
            print(f"Exporting {name} ({len(df)} rows) to database...")
            df_export = df.copy()
            df_export.columns = [c.lower() for c in df_export.columns]
            df_export.to_sql(name, engine, if_exists="replace", index=False)
        print("Database export complete.")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for name, df in tables.items():
            path = os.path.join(OUTPUT_DIR, f"{name}.csv")
            df.to_csv(path, index=False)
            print(f"Exported {name}.csv ({len(df):,} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export synthetic dataset")
    parser.add_argument("--db-url", type=str, default=None, help="Database connection string")
    args = parser.parse_args()
    export_dataset(db_url=args.db_url)
