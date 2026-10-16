"""TrustShield AI — Deterministic Historical Redis Backfill & Reconciliation Tool.

Phase 4.2B Isolated Implementation:
- Ingests verified historical events from dataset CSVs into Redis sliding-window stores.
- Supports dry-run validation without modifying Redis.
- Performs end-to-end reconciliation checks confirming event counts and entity totals.
- Idempotent: Can be run multiple times safely.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for p in [_ROOT_DIR, os.path.join(_ROOT_DIR, "trustshield_project"), os.path.join(_ROOT_DIR, "backend")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.services.redis_causal_service import RedisCausalService


def main():
    parser = argparse.ArgumentParser(description="Deterministic Redis Causal Backfill")
    parser.add_argument("--orders", default=os.path.join(_ROOT_DIR, "data", "synthetic_v2_1", "orders.csv"))
    parser.add_argument("--returns", default=os.path.join(_ROOT_DIR, "data", "synthetic_v2_1", "returns.csv"))
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    parser.add_argument("--window-days", type=int, default=35, help="Only backfill recent N days")
    parser.add_argument("--in-memory", action="store_true", help="Test backfill in in-memory store")
    args = parser.parse_args()

    print("=" * 80)
    print("TRUSTSHIELD PHASE 4.2B — HISTORICAL REDIS BACKFILL & RECONCILIATION")
    print("=" * 80)
    print(f"Orders Source : {args.orders}")
    print(f"Returns Source: {args.returns}")
    print(f"Dry Run Mode  : {args.dry_run}")
    print(f"In-Memory Mode: {args.in_memory}")

    if not os.path.isfile(args.orders):
        print(f"ERROR: Orders file not found: {args.orders}")
        sys.exit(1)

    print("\n[Step 1] Loading verified orders...")
    df_orders = pd.read_csv(args.orders)
    df_orders["order_date"] = pd.to_datetime(df_orders["order_date"])

    df_returns = None
    if os.path.isfile(args.returns):
        print("[Step 1] Loading returns...")
        df_returns = pd.read_csv(args.returns)
        df_returns["return_date"] = pd.to_datetime(df_returns["return_date"])

    if args.window_days > 0:
        max_date = df_orders["order_date"].max()
        cutoff_date = max_date - pd.Timedelta(days=args.window_days)
        print(f"Applying {args.window_days}-day window: cutoff {cutoff_date} to {max_date}")
        df_orders = df_orders[df_orders["order_date"] >= cutoff_date].copy()
        if df_returns is not None:
            df_returns = df_returns[df_returns["return_date"] >= cutoff_date].copy()

    svc = RedisCausalService(in_memory_mode=args.in_memory)

    print(f"\n[Step 2] Executing backfill ({len(df_orders)} orders, {len(df_returns) if df_returns is not None else 0} returns)...")
    report = svc.backfill_from_dataframe(
        orders_df=df_orders,
        returns_df=df_returns,
        dry_run=args.dry_run,
    )

    print("\n--- BACKFILL RECONCILIATION REPORT ---")
    for k, v in report.items():
        print(f"  {k:25s}: {v}")

    if report["reconciliation_status"] in ("verified", "dry_run_success"):
        print("\nRECONCILIATION RESULT: PASS (All events verified)")
    else:
        print("\nRECONCILIATION RESULT: WARNING / MISMATCH")


if __name__ == "__main__":
    main()
