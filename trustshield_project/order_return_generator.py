"""Temporal-consistent order and return generator."""

import numpy as np
import pandas as pd
from datetime import timedelta

from entity_generator import rng, SIM_START, SIM_END, build_base_entities
from product_listing_generator import build_catalog_and_listings

TARGET_TOTAL_ORDERS = 50000
BASE_RETURN_RATE = 0.07
RETURN_WINDOW_DAYS = 21

RETURN_REASONS = ["defective", "changed_mind", "size_issue", "wrong_item_received"]
RETURN_REASON_WEIGHTS = [0.30, 0.35, 0.20, 0.15]


def assign_order_counts(buyers_df: pd.DataFrame, total_orders: int = TARGET_TOTAL_ORDERS,
                        skew_sigma: float = 0.7, rng: np.random.Generator | None = None) -> np.ndarray:
    """Allocates order volumes per buyer using a lognormal distribution."""
    gen = rng if rng is not None else globals()["rng"]
    n_buyers = len(buyers_df)
    weights = gen.lognormal(mean=0.0, sigma=skew_sigma, size=n_buyers)
    weights /= weights.sum()
    counts = np.maximum(0, (weights * total_orders).round().astype(int))

    diff = int(total_orders - counts.sum())
    if diff > 0:
        idx = gen.choice(n_buyers, size=diff, replace=True)
        np.add.at(counts, idx, 1)
    elif diff < 0:
        eligible = np.flatnonzero(counts > 0)
        if len(eligible) > 0:
            n_sub = min(abs(diff), len(eligible))
            idx = gen.choice(eligible, size=n_sub, replace=False)
            np.add.at(counts, idx, -1)
    return np.maximum(0, counts)


def generate_order_date_candidates(signup_date, n: int, sim_end=SIM_END,
                                   rng: np.random.Generator | None = None) -> list:
    """Generates order dates respecting buyer registration date."""
    gen = rng if rng is not None else globals()["rng"]
    window_days = max(1, (sim_end - signup_date).days)
    offsets = (gen.random(n) ** 1.3) * window_days
    return [signup_date + timedelta(days=int(d)) for d in offsets]


def generate_orders(buyers_df: pd.DataFrame, listings_df: pd.DataFrame, 
                    device_mapping_df: pd.DataFrame, total_orders: int = TARGET_TOTAL_ORDERS,
                    rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generates transaction orders enforcing strict temporal validity against listings and devices."""
    gen = rng if rng is not None else globals()["rng"]
    listings_sorted = listings_df.sort_values("listing_date").reset_index(drop=True)
    listing_dates_ordinal = np.array([d.toordinal() for d in listings_sorted["listing_date"]])
    global_min_listing_date = listings_sorted["listing_date"].iloc[0]

    buyer_devices = device_mapping_df.groupby("buyer_id")["device_id"].apply(list)
    counts = assign_order_counts(buyers_df, total_orders, rng=gen)
    rows = []
    order_counter = 0

    for (_, buyer), n_orders in zip(buyers_df.iterrows(), counts):
        if n_orders == 0:
            continue

        effective_signup = max(buyer["signup_date"], global_min_listing_date)
        candidate_dates = generate_order_date_candidates(effective_signup, n_orders, rng=gen)
        devices_raw = buyer_devices.get(buyer["buyer_id"])
        devices_for_buyer: list = list(devices_raw) if isinstance(devices_raw, (list, tuple)) else [None]

        for order_date in candidate_dates:
            cutoff_idx = np.searchsorted(listing_dates_ordinal, order_date.toordinal(), side="right")
            if cutoff_idx == 0:
                order_date = global_min_listing_date
                cutoff_idx = np.searchsorted(listing_dates_ordinal, order_date.toordinal(), side="right")
                if cutoff_idx == 0:
                    continue

            listing_row = listings_sorted.iloc[int(gen.integers(0, cutoff_idx))]
            device_id = devices_for_buyer[int(gen.integers(0, len(devices_for_buyer)))]

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


def generate_returns(orders_df: pd.DataFrame, base_return_rate: float = BASE_RETURN_RATE,
                     rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generates baseline organic returns."""
    gen = rng if rng is not None else globals()["rng"]
    returned_orders = pd.DataFrame(orders_df[gen.random(len(orders_df)) < base_return_rate])
    if returned_orders.empty:
        return pd.DataFrame(columns=[
            "return_id", "order_id", "buyer_id", "seller_id",
            "return_date", "reason", "status"
        ])

    delay_days = gen.integers(1, RETURN_WINDOW_DAYS + 1, size=len(returned_orders))
    return_dates = [min(od + timedelta(days=int(d)), SIM_END) for od, d in zip(returned_orders["order_date"], delay_days)]
    reasons = gen.choice(RETURN_REASONS, size=len(returned_orders), p=RETURN_REASON_WEIGHTS)

    return pd.DataFrame({
        "return_id": [f"RETURN_{i:06d}" for i in range(len(returned_orders))],
        "order_id": returned_orders["order_id"].to_numpy(),
        "buyer_id": returned_orders["buyer_id"].to_numpy(),
        "seller_id": returned_orders["seller_id"].to_numpy(),
        "return_date": return_dates,
        "reason": reasons,
        "status": "approved",
    })


def build_orders_and_returns(buyers_df: pd.DataFrame, sellers_df: pd.DataFrame, 
                             listings_df: pd.DataFrame, device_mapping_df: pd.DataFrame,
                             rng: np.random.Generator | None = None) -> dict:
    """Builds order transactions and corresponding returns."""
    gen = rng if rng is not None else globals()["rng"]
    orders_df = generate_orders(buyers_df, listings_df, device_mapping_df, rng=gen)
    returns_df = generate_returns(orders_df, rng=gen)

    buyers_df = buyers_df.copy()
    sellers_df = sellers_df.copy()

    order_counts_by_buyer = orders_df.groupby("buyer_id").size().to_dict()
    return_counts_by_buyer = returns_df.groupby("buyer_id").size().to_dict() if not returns_df.empty else {}
    order_counts_by_seller = orders_df.groupby("seller_id").size().to_dict()

    buyers_df["total_orders"] = buyers_df["buyer_id"].map(lambda x: order_counts_by_buyer.get(x, 0)).astype(int)
    buyers_df["total_returns"] = buyers_df["buyer_id"].map(lambda x: return_counts_by_buyer.get(x, 0)).astype(int)
    sellers_df["total_orders_received"] = sellers_df["seller_id"].map(lambda x: order_counts_by_seller.get(x, 0)).astype(int)

    return {
        "orders": orders_df,
        "returns": returns_df,
        "buyers": buyers_df,
        "sellers": sellers_df,
        "rng": gen,
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
    print(f"Orders: {len(result['orders']):,} | Returns: {len(result['returns']):,}")
