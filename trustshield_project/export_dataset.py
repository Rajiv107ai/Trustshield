"""Exports generated pipeline tables to disk or database."""

import os
import argparse
from entity_generator import build_base_entities
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_data_export")


def export_dataset(output_dir: str | None = None):
    """Executes pipeline generation and exports tabular results as CSV."""
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

    target_dir = output_dir or OUTPUT_DIR
    os.makedirs(target_dir, exist_ok=True)
    for name, df in tables.items():
        path = os.path.join(target_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"Exported {name}.csv ({len(df):,} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export synthetic dataset as CSV files")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save CSV files")
    args = parser.parse_args()
    export_dataset(output_dir=args.output_dir)
