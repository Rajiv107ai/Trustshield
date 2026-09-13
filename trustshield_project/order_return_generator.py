"""
TrustShield AI — Synthetic Data Generator
Part 3: Order + Return generator

Design reference: design.md
- ~50,000 orders target scale, gradual growth over the 12-month timeline
  (mirrors buyer/seller onboarding growth rather than being uniform).
- Temporal integrity is the whole point of this module: an order can only
  reference a listing that already existed on that date, and only a device
  already linked to that buyer by that date. This is what makes the later
  timestamp-snapshotted graph features (Phase 3) valid instead of leaking
  future information into past predictions.
- NO fraud is injected here. Returns generated here are organic/baseline
  returns only (defective item, changed mind, size issue, wrong item) at a
  realistic base rate. Return-abuse fraud (excessive/coordinated returns)
  is layered on top later by the separate fraud_injection module — this
  keeps "what does a fraud scenario add on top of normal behavior" clean
  and measurable, rather than baking fraud into the base generator.
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from entity_generator import rng, SIM_START, SIM_END, build_base_entities
from product_listing_generator import build_catalog_and_listings

TARGET_TOTAL_ORDERS = 50000
BASE_RETURN_RATE = 0.07          # organic returns only; fraud layer added later
RETURN_WINDOW_DAYS = 21          # max days between order and organic return

RETURN_REASONS = ["defective", "changed_mind", "size_issue", "wrong_item_received"]
RETURN_REASON_WEIGHTS = [0.30, 0.35, 0.20, 0.15]


# ---------------------------------------------------------------------------
# Per-buyer order counts (skewed, like seller listing counts)
# ---------------------------------------------------------------------------

def assign_order_counts(buyers_df, total_orders=TARGET_TOTAL_ORDERS, skew_sigma=0.7):
    """
    Most buyers place a handful of orders; a smaller set of frequent buyers
    place many more — lognormal weighting rather than a flat split, scaled
    to hit total_orders exactly (mirrors assign_listing_counts()).
    """
    n_buyers = len(buyers_df)
    weights = rng.lognormal(mean=0.0, sigma=skew_sigma, size=n_buyers)
    weights = weights / weights.sum()
    counts = np.maximum(0, (weights * total_orders).round().astype(int))

    diff = int(total_orders - counts.sum())
    if diff > 0:
        idx = rng.choice(n_buyers, size=diff, replace=True)
        np.add.at(counts, idx, 1)
    elif diff < 0:
        # Only subtract from buyers that have at least 1 order so
        # np.maximum(0, counts) cannot silently undo the subtraction.
        eligible = np.flatnonzero(counts > 0)
        if len(eligible) > 0:
            n_sub = min(abs(diff), len(eligible))
            idx = rng.choice(eligible, size=n_sub, replace=False)
            np.add.at(counts, idx, -1)
    counts = np.maximum(0, counts)

    return counts


# ---------------------------------------------------------------------------
# Order date candidates (concave growth curve from buyer signup to sim end)
# ---------------------------------------------------------------------------

def generate_order_date_candidates(signup_date, n, sim_end=SIM_END):
    window_days = max(1, (sim_end - signup_date).days)
    u = rng.random(n)
    offsets = (u ** 1.3) * window_days
    return [signup_date + timedelta(days=int(d)) for d in offsets]


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def generate_orders(buyers_df, listings_df, device_mapping_df, total_orders=TARGET_TOTAL_ORDERS):
    """
    Each order: a buyer purchasing one listing on a date on/after both the
    buyer's signup and that listing's listing_date, using a device already
    linked to that buyer (primary or shared) as of that date.
    """
    listings_sorted = listings_df.sort_values("listing_date").reset_index(drop=True)
    listing_dates_ordinal = np.array([d.toordinal() for d in listings_sorted["listing_date"]])
    global_min_listing_date = listings_sorted["listing_date"].iloc[0]

    # Group buyer -> list of device_ids available to them (primary + shared)
    buyer_devices = device_mapping_df.groupby("buyer_id")["device_id"].apply(list)

    counts = assign_order_counts(buyers_df, total_orders)
    rows = []
    order_counter = 0

    for (_, buyer), n_orders in zip(buyers_df.iterrows(), counts):
        if n_orders == 0:
            continue

        effective_signup = max(buyer["signup_date"], global_min_listing_date)
        candidate_dates = generate_order_date_candidates(effective_signup, n_orders)
        devices_for_buyer = buyer_devices.get(buyer["buyer_id"], [None])

        for order_date in candidate_dates:
            # Only listings that existed by order_date are eligible —
            # this is the temporal-integrity guarantee for this module.
            cutoff_idx = np.searchsorted(listing_dates_ordinal, order_date.toordinal(), side="right")
            if cutoff_idx == 0:
                order_date = global_min_listing_date
                cutoff_idx = np.searchsorted(listing_dates_ordinal, order_date.toordinal(), side="right")
                if cutoff_idx == 0:
                    continue

            listing_row = listings_sorted.iloc[rng.integers(0, cutoff_idx)]
            device_id = devices_for_buyer[rng.integers(0, len(devices_for_buyer))]

            rows.append({
                "order_id": f"ORDER_{order_counter:06d}",
                "buyer_id": buyer["buyer_id"],
                "seller_id": listing_row["seller_id"],
                "listing_id": listing_row["listing_id"],
                "product_id": listing_row["product_id"],
                "order_date": order_date,
                "amount": listing_row["price"],
                "device_id": device_id,
                "status": "completed",
            })
            order_counter += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Returns (organic/baseline only — fraud-driven return abuse added later)
# ---------------------------------------------------------------------------

def generate_returns(orders_df, base_return_rate=BASE_RETURN_RATE):
    n_orders = len(orders_df)
    is_returned = rng.random(n_orders) < base_return_rate
    returned_orders = orders_df[is_returned]

    if len(returned_orders) == 0:
        return pd.DataFrame(columns=[
            "return_id", "order_id", "buyer_id", "seller_id",
            "return_date", "reason", "status"
        ])

    delay_days = rng.integers(1, RETURN_WINDOW_DAYS + 1, size=len(returned_orders))
    return_dates = [
        od + timedelta(days=int(d))
        for od, d in zip(returned_orders["order_date"], delay_days)
    ]
    reasons = rng.choice(RETURN_REASONS, size=len(returned_orders), p=RETURN_REASON_WEIGHTS)

    df = pd.DataFrame({
        "return_id": [f"RETURN_{i:06d}" for i in range(len(returned_orders))],
        "order_id": returned_orders["order_id"].values,
        "buyer_id": returned_orders["buyer_id"].values,
        "seller_id": returned_orders["seller_id"].values,
        "return_date": return_dates,
        "reason": reasons,
        "status": "approved",
    })
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_orders_and_returns(buyers_df, sellers_df, listings_df, device_mapping_df):
    print("Generating orders...")
    orders_df = generate_orders(buyers_df, listings_df, device_mapping_df)

    print("Generating returns...")
    returns_df = generate_returns(orders_df)

    # Backfill buyer/seller aggregate counters (kept in sync, not recomputed ad hoc)
    buyers_df = buyers_df.copy()
    sellers_df = sellers_df.copy()

    order_counts_by_buyer = orders_df.groupby("buyer_id").size()
    return_counts_by_buyer = returns_df.groupby("buyer_id").size() if len(returns_df) else pd.Series(dtype=int)
    order_counts_by_seller = orders_df.groupby("seller_id").size()

    buyers_df["total_orders"] = buyers_df["buyer_id"].map(order_counts_by_buyer).fillna(0).astype(int)
    buyers_df["total_returns"] = buyers_df["buyer_id"].map(return_counts_by_buyer).fillna(0).astype(int)
    sellers_df["total_orders_received"] = sellers_df["seller_id"].map(order_counts_by_seller).fillna(0).astype(int)

    return {
        "orders": orders_df,
        "returns": returns_df,
        "buyers": buyers_df,
        "sellers": sellers_df,
    }


if __name__ == "__main__":
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    result = build_orders_and_returns(
        buyers_df=base["buyers"],
        sellers_df=catalog["sellers"],
        listings_df=catalog["listings"],
        device_mapping_df=base["device_mapping"],
    )

    print("\n--- Orders & Returns Summary ---")
    for name in ["orders", "returns"]:
        df = result[name]
        print(f"✅ {name.capitalize()}: {len(df):,} rows")

    print(f"\nOverall return rate: {len(result['returns']) / len(result['orders']):.3%}")

    print("\nTemporal integrity check (order_date >= listing_date for sampled orders):")
    merged_check = result["orders"].merge(
        catalog["listings"][["listing_id", "listing_date"]], on="listing_id"
    )
    violations = (merged_check["order_date"] < merged_check["listing_date"]).sum()
    print(f"  violations found: {violations} (should be 0)")

    print("\nReturn date after order date check:")
    ret_check = result["returns"].merge(
        result["orders"][["order_id", "order_date"]], on="order_id"
    )
    ret_violations = (ret_check["return_date"] <= ret_check["order_date"]).sum()
    print(f"  violations found: {ret_violations} (should be 0)")

    print("\n✅ Orders per buyer — distribution check:")
    desc = result["orders"].groupby("buyer_id").size().describe()
    print(f"   Orders per buyer: mean={desc['mean']:.1f}, max={desc['max']:.0f}")
