"""
TrustShield AI — Dataset export

Runs the full pipeline once (entity generation -> catalog -> orders/returns
-> fraud injection) and writes every generated table to CSV files under
synthetic_data_export/, so the synthetic dataset actually exists as files on disk —
not just as in-memory objects printed by each phase script.

Run this AFTER confirming the individual phase scripts work, from inside
trustshield_project/:

    python export_dataset.py

Output: a synthetic_data_export/ folder (created next to this script) containing one CSV
per table. fraud_ground_truth.csv is the ONLY file that contains
fraud_ring_id — it's the ground-truth ledger, kept separate by design so
it never accidentally gets used as a feature (see fraud_injection.py's
module docstring for why).
"""

import os
import argparse

from entity_generator import build_base_entities
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_data_export")


def export_dataset(db_url=None):
    print("Building full pipeline...")
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
        "sellers": txn["sellers"],          # final version, includes total_listings & total_orders_received backfills
        "buyers": txn["buyers"],                # final version, includes total_orders/total_returns backfill
        "device_mapping": base["device_mapping"],
        "address_sharing_log": base["address_sharing_log"],
        "device_sharing_log": base["device_sharing_log"],
        "products": catalog["products"],
        "listings": result["listings"],         # final version, includes fraud flags
        "orders": result["orders"],             # final version, includes fraud flags
        "returns": result["returns"],           # final version, includes fraud flags
        "fraud_ground_truth": result["fraud_ground_truth"],  # ONLY place fraud_ring_id lives
    }
    
    if db_url:
        print(f"\nWriting {len(tables)} tables to PostgreSQL database...")
        try:
            from sqlalchemy import create_engine
            engine = create_engine(db_url)
            for name, df in tables.items():
                print(f"  Uploading {name} to Postgres (schema replacement) ...")
                # Work on a copy so the original pipeline DataFrames are not mutated.
                df_export = df.copy()
                df_export.columns = [c.lower() for c in df_export.columns]
                df_export.to_sql(name, engine, if_exists="replace", index=False)
            print(f"\nDone. All tables uploaded to {db_url}.")
        except ImportError:
            print("ERROR: sqlalchemy or psycopg2 not installed. Run: pip install sqlalchemy psycopg2-binary")
        except Exception as e:
            print(f"ERROR exporting to Postgres: {e}")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"\nWriting {len(tables)} tables to {OUTPUT_DIR}/ ...")
        for name, df in tables.items():
            path = os.path.join(OUTPUT_DIR, f"{name}.csv")
            df.to_csv(path, index=False)
            print(f"  {name}.csv  —  {len(df):,} rows, {len(df.columns)} columns")
    
        print("\nDone. These CSVs are a POINT-IN-TIME snapshot of one generator run — "
              "since the pipeline is seeded (RNG_SEED=42 in entity_generator.py), "
              "re-running this script reproduces the exact same files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TrustShield synthetic dataset")
    parser.add_argument("--db-url", type=str, default=None, 
                        help="Optional PostgreSQL connection URI (e.g., postgresql://user:pass@localhost:5432/trustshield)")
    args = parser.parse_args()
    
    export_dataset(db_url=args.db_url)
