"""Base entity generation: Addresses, Devices, Sellers, and Buyers with device/address sharing."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

SIM_START = datetime(2025, 1, 1)
SIM_END = datetime(2025, 12, 31)
SIM_DAYS = (SIM_END - SIM_START).days

N_SELLERS = 500
N_BUYERS = 5000


def generate_onboarding_dates(n: int, early_share: float = 0.40, early_window_days: int = 60) -> list:
    """Generates staggered signup dates over simulation window."""
    n_early = int(n * early_share)
    early_days = ((rng.random(n_early) ** 1.6) * early_window_days).astype(int)
    rest_days = ((rng.random(n - n_early) ** 0.8) * SIM_DAYS).astype(int)
    all_days = np.concatenate([early_days, rest_days])
    rng.shuffle(all_days)
    return [SIM_START + timedelta(days=int(d)) for d in all_days]


def generate_addresses(n_addresses: int) -> pd.DataFrame:
    """Generates synthetic delivery and registration addresses."""
    regions = [f"REGION_{i:02d}" for i in range(1, 31)]
    return pd.DataFrame({
        "address_id": [f"ADDR_{i:06d}" for i in range(n_addresses)],
        "pincode_region": rng.choice(regions, size=n_addresses),
        "first_seen_date": generate_onboarding_dates(n_addresses),
    })


def generate_devices(n_devices: int) -> pd.DataFrame:
    """Generates synthetic device fingerprints."""
    return pd.DataFrame({
        "device_id": [f"DEV_{i:06d}" for i in range(n_devices)],
        "device_fingerprint": [f"FP_{rng.integers(10**9, 10**10-1)}" for _ in range(n_devices)],
        "first_seen_date": generate_onboarding_dates(n_devices),
    })


def generate_sellers(n_sellers: int, addresses_df: pd.DataFrame) -> pd.DataFrame:
    """Generates seller merchant profiles."""
    categories = [
        "Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports",
        "Toys", "Books", "Grocery", "Automotive", "Furniture"
    ]
    return pd.DataFrame({
        "seller_id": [f"SELLER_{i:05d}" for i in range(n_sellers)],
        "signup_date": generate_onboarding_dates(n_sellers),
        "address_id": rng.choice(addresses_df["address_id"], size=n_sellers, replace=False),
        "category_focus": rng.choice(categories, size=n_sellers),
        "trust_score_current": 70.0,
        "trust_score_history": [[] for _ in range(n_sellers)],
        "total_listings": 0,
        "total_orders_received": 0,
        "account_status": "active",
    })


def generate_buyers(n_buyers: int, addresses_df: pd.DataFrame) -> pd.DataFrame:
    """Generates buyer accounts."""
    return pd.DataFrame({
        "buyer_id": [f"BUYER_{i:05d}" for i in range(n_buyers)],
        "signup_date": generate_onboarding_dates(n_buyers),
        "address_id": rng.choice(addresses_df["address_id"], size=n_buyers, replace=True),
        "trust_score_current": 70.0,
        "total_orders": 0,
        "total_returns": 0,
    })


def assign_primary_devices(buyers_df: pd.DataFrame, devices_df: pd.DataFrame) -> pd.DataFrame:
    """Assigns 1-to-1 primary devices to buyer accounts."""
    return pd.DataFrame({
        "buyer_id": buyers_df["buyer_id"].values,
        "device_id": rng.choice(devices_df["device_id"], size=len(buyers_df), replace=False),
        "is_primary": True,
    })


def assign_shared_addresses(buyers_df: pd.DataFrame, share_rate: float = 0.12, legit_share: float = 0.70):
    """Simulates physical address sharing across buyer accounts.

    The ``first_seen_date`` column records when the sharing relationship was
    first observed.  It is drawn uniformly between the later of the two buyers'
    signup dates and SIM_END, so the relationship is always discoverable *after*
    both accounts exist.  Use this timestamp to filter the sharing graph when
    building features for a specific decision point.
    """
    buyers_df = buyers_df.copy()
    n = len(buyers_df)
    n_sharing = int(n * share_rate)

    sharing_idx = rng.choice(n, size=n_sharing, replace=False)
    partner_idx = rng.choice(n, size=n_sharing, replace=False)
    mask = sharing_idx == partner_idx
    while mask.any():
        partner_idx[mask] = rng.choice(n, size=mask.sum(), replace=False)
        mask = sharing_idx == partner_idx

    is_fraud_link = rng.random(n_sharing) >= legit_share

    # Determine first_seen_date: uniform between max(signup_a, signup_b) and SIM_END
    signup_a = pd.to_datetime(buyers_df.iloc[sharing_idx]["signup_date"].values)
    signup_b = pd.to_datetime(buyers_df.iloc[partner_idx]["signup_date"].values)
    latest_signup = np.maximum(signup_a.asi8, signup_b.asi8)
    sim_end_ns = pd.Timestamp(SIM_END).value
    # Clamp: if latest_signup >= SIM_END, set first_seen = SIM_END
    span_ns = np.maximum(sim_end_ns - latest_signup, 0)
    offsets_ns = (rng.random(n_sharing) * span_ns).astype(np.int64)
    first_seen = pd.to_datetime(latest_signup + offsets_ns)

    sharing_log = pd.DataFrame({
        "buyer_id": buyers_df.iloc[sharing_idx]["buyer_id"].values,
        "shared_with_buyer_id": buyers_df.iloc[partner_idx]["buyer_id"].values,
        "shared_address_id": buyers_df.iloc[partner_idx]["address_id"].values,
        "share_type": np.where(is_fraud_link, "fraud_linked", "legitimate"),
        "first_seen_date": first_seen,
    })
    buyers_df.loc[sharing_idx, "address_id"] = sharing_log["shared_address_id"].values
    return buyers_df, sharing_log


def assign_shared_devices(device_mapping_df: pd.DataFrame, buyers_df: pd.DataFrame,
                          share_rate: float = 0.08, legit_share: float = 0.50):
    """Simulates device reuse across buyer accounts.

    Like ``assign_shared_addresses``, a ``first_seen_date`` is generated for
    each sharing relationship so that graph features can be filtered to only
    relationships known before a given decision timestamp.
    """
    n = len(buyers_df)
    n_sharing = int(n * share_rate)

    sharing_idx = rng.choice(n, size=n_sharing, replace=False)
    partner_idx = rng.choice(n, size=n_sharing, replace=False)
    mask = sharing_idx == partner_idx
    while mask.any():
        partner_idx[mask] = rng.choice(n, size=mask.sum(), replace=False)
        mask = sharing_idx == partner_idx

    is_fraud_link = rng.random(n_sharing) >= legit_share
    sharer_buyer_ids = buyers_df.iloc[sharing_idx]["buyer_id"].values
    partner_buyer_ids = buyers_df.iloc[partner_idx]["buyer_id"].values
    partner_device = device_mapping_df.set_index("buyer_id").loc[partner_buyer_ids, "device_id"].values

    # first_seen_date: uniform between max(signup_a, signup_b) and SIM_END
    signup_a = pd.to_datetime(buyers_df.iloc[sharing_idx]["signup_date"].values)
    signup_b = pd.to_datetime(buyers_df.iloc[partner_idx]["signup_date"].values)
    latest_signup = np.maximum(signup_a.asi8, signup_b.asi8)
    sim_end_ns = pd.Timestamp(SIM_END).value
    span_ns = np.maximum(sim_end_ns - latest_signup, 0)
    offsets_ns = (rng.random(n_sharing) * span_ns).astype(np.int64)
    first_seen = pd.to_datetime(latest_signup + offsets_ns)

    sharing_log = pd.DataFrame({
        "buyer_id": sharer_buyer_ids,
        "shared_with_buyer_id": partner_buyer_ids,
        "shared_device_id": partner_device,
        "share_type": np.where(is_fraud_link, "fraud_linked", "legitimate"),
        "first_seen_date": first_seen,
    })
    extra_rows = pd.DataFrame({
        "buyer_id": sharer_buyer_ids,
        "device_id": partner_device,
        "is_primary": False,
    })
    return pd.concat([device_mapping_df, extra_rows], ignore_index=True), sharing_log


def build_base_entities() -> dict:
    """Builds base entity tables and sharing relationships."""
    addresses_df = generate_addresses(n_addresses=int((N_SELLERS + N_BUYERS) * 0.9))
    devices_df = generate_devices(n_devices=N_BUYERS)
    sellers_df = generate_sellers(N_SELLERS, addresses_df)
    buyers_df = generate_buyers(N_BUYERS, addresses_df)
    device_mapping_df = assign_primary_devices(buyers_df, devices_df)
    buyers_df, address_sharing_log = assign_shared_addresses(buyers_df)
    device_mapping_df, device_sharing_log = assign_shared_devices(device_mapping_df, buyers_df)

    return {
        "addresses": addresses_df,
        "devices": devices_df,
        "sellers": sellers_df,
        "buyers": buyers_df,
        "device_mapping": device_mapping_df,
        "address_sharing_log": address_sharing_log,
        "device_sharing_log": device_sharing_log,
    }


if __name__ == "__main__":
    data = build_base_entities()
    for name, df in data.items():
        print(f"{name}: {len(df):,} rows")
