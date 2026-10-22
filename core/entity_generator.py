"""Base entity generation: Addresses, Devices, Sellers, and Buyers with device/address sharing."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)


def reset_rng(seed: int = RNG_SEED) -> np.random.Generator:
    """Resets the module-level RNG in-place so all importing modules stay in sync."""
    fresh = np.random.default_rng(seed)
    rng.bit_generator.state = fresh.bit_generator.state
    return rng


CURRENCY = "INR"

SIM_START = datetime(2025, 1, 1)
SIM_END = datetime(2025, 12, 31)
SIM_DAYS = (SIM_END - SIM_START).days

N_SELLERS = 500
N_BUYERS = 5000


def generate_onboarding_dates(n: int, early_share: float = 0.40, early_window_days: int = 60,
                             rng: np.random.Generator | None = None) -> list:
    """Generates staggered signup dates over simulation window."""
    gen = rng if rng is not None else globals()["rng"]
    n_early = int(n * early_share)
    early_days = ((gen.random(n_early) ** 1.6) * early_window_days).astype(int)
    rest_days = ((gen.random(n - n_early) ** 0.8) * SIM_DAYS).astype(int)
    all_days = np.concatenate([early_days, rest_days])
    gen.shuffle(all_days)
    return [SIM_START + timedelta(days=int(d)) for d in all_days]


def generate_addresses(n_addresses: int, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generates synthetic delivery and registration addresses."""
    gen = rng if rng is not None else globals()["rng"]
    regions = [f"REGION_{i:02d}" for i in range(1, 31)]
    return pd.DataFrame({
        "address_id": [f"ADDR_{i:06d}" for i in range(n_addresses)],
        "pincode_region": gen.choice(regions, size=n_addresses),
        "first_seen_date": generate_onboarding_dates(n_addresses, rng=gen),
    })


def generate_devices(n_devices: int, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generates synthetic device fingerprints."""
    gen = rng if rng is not None else globals()["rng"]
    return pd.DataFrame({
        "device_id": [f"DEV_{i:06d}" for i in range(n_devices)],
        "device_fingerprint": [f"FP_{gen.integers(10**9, 10**10-1)}" for _ in range(n_devices)],
        "first_seen_date": generate_onboarding_dates(n_devices, rng=gen),
    })


def generate_sellers(n_sellers: int, addresses_df: pd.DataFrame, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generates seller merchant profiles."""
    gen = rng if rng is not None else globals()["rng"]
    categories = [
        "Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports",
        "Toys", "Books", "Grocery", "Automotive", "Furniture"
    ]
    seller_addr_pool = addresses_df["address_id"].iloc[:n_sellers].values if len(addresses_df) >= n_sellers else addresses_df["address_id"].values
    return pd.DataFrame({
        "seller_id": [f"SELLER_{i:05d}" for i in range(n_sellers)],
        "signup_date": generate_onboarding_dates(n_sellers, rng=gen),
        "address_id": gen.choice(np.asarray(seller_addr_pool), size=n_sellers, replace=False),
        "category_focus": gen.choice(categories, size=n_sellers),
        "trust_score_current": 70.0,
        "trust_score_history": [[] for _ in range(n_sellers)],
        "total_listings": 0,
        "total_orders_received": 0,
        "account_status": "active",
    })


def generate_buyers(n_buyers: int, addresses_df: pd.DataFrame, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generates buyer accounts sampling addresses without replacement to eliminate unintended collisions."""
    gen = rng if rng is not None else globals()["rng"]
    # Sample buyer addresses without replacement from dedicated pool so only the logged pairs share
    if len(addresses_df) >= N_SELLERS + n_buyers:
        buyer_addr_pool = addresses_df["address_id"].iloc[N_SELLERS:N_SELLERS + n_buyers].values
    elif len(addresses_df) >= n_buyers:
        buyer_addr_pool = addresses_df["address_id"].iloc[:n_buyers].values
    else:
        buyer_addr_pool = addresses_df["address_id"].values
    return pd.DataFrame({
        "buyer_id": [f"BUYER_{i:05d}" for i in range(n_buyers)],
        "signup_date": generate_onboarding_dates(n_buyers, rng=gen),
        "address_id": gen.choice(np.asarray(buyer_addr_pool), size=n_buyers, replace=False),
        "trust_score_current": 70.0,
        "total_orders": 0,
        "total_returns": 0,
    })


def assign_primary_devices(buyers_df: pd.DataFrame, devices_df: pd.DataFrame,
                           rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Assigns 1-to-1 primary devices to buyer accounts."""
    gen = rng if rng is not None else globals()["rng"]
    return pd.DataFrame({
        "buyer_id": buyers_df["buyer_id"].values,
        "device_id": gen.choice(devices_df["device_id"], size=len(buyers_df), replace=False),
        "is_primary": True,
    })


def assign_shared_addresses(buyers_df: pd.DataFrame, share_rate: float = 0.12, legit_share: float = 0.70,
                            rng: np.random.Generator | None = None):
    """Simulates physical address sharing across buyer accounts.

    The ``first_seen_date`` column records when the sharing relationship was
    first observed.  It is drawn uniformly between the later of the two buyers'
    signup dates and SIM_END, so the relationship is always discoverable *after*
    both accounts exist.  Use this timestamp to filter the sharing graph when
    building features for a specific decision point.
    """
    gen = rng if rng is not None else globals()["rng"]
    buyers_df = buyers_df.copy()
    n = len(buyers_df)
    n_sharing = int(n * share_rate)

    sharing_idx = gen.choice(n, size=n_sharing, replace=False)
    partner_idx = gen.choice(n, size=n_sharing, replace=False)
    mask = sharing_idx == partner_idx
    while mask.any():
        partner_idx[mask] = gen.choice(n, size=mask.sum(), replace=False)
        mask = sharing_idx == partner_idx
    sharing_idx = sharing_idx.tolist()
    partner_idx = partner_idx.tolist()

    is_fraud_link = gen.random(n_sharing) >= legit_share

    # Determine first_seen_date: uniform between max(signup_a, signup_b) and SIM_END
    signup_a = pd.to_datetime(buyers_df.iloc[sharing_idx]["signup_date"].values)
    signup_b = pd.to_datetime(buyers_df.iloc[partner_idx]["signup_date"].values)
    latest_signup = pd.to_datetime(np.maximum(signup_a.values, signup_b.values))
    sim_end_dt = pd.Timestamp(SIM_END)
    # Clamp: if latest_signup >= SIM_END, set first_seen = SIM_END
    span_seconds = np.maximum((sim_end_dt - latest_signup).total_seconds(), 0)
    offsets_seconds = gen.random(n_sharing) * span_seconds
    first_seen = latest_signup + pd.to_timedelta(offsets_seconds, unit="s")

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
                          share_rate: float = 0.08, legit_share: float = 0.50,
                          rng: np.random.Generator | None = None):
    """Simulates device reuse across buyer accounts.

    Like ``assign_shared_addresses``, a ``first_seen_date`` is generated for
    each sharing relationship so that graph features can be filtered to only
    relationships known before a given decision timestamp.
    """
    gen = rng if rng is not None else globals()["rng"]
    n = len(buyers_df)
    n_sharing = int(n * share_rate)

    sharing_idx = gen.choice(n, size=n_sharing, replace=False)
    partner_idx = gen.choice(n, size=n_sharing, replace=False)
    mask = sharing_idx == partner_idx
    while mask.any():
        partner_idx[mask] = gen.choice(n, size=mask.sum(), replace=False)
        mask = sharing_idx == partner_idx
    sharing_idx = sharing_idx.tolist()
    partner_idx = partner_idx.tolist()

    is_fraud_link = gen.random(n_sharing) >= legit_share
    sharer_buyer_ids = buyers_df.iloc[sharing_idx]["buyer_id"].values
    partner_buyer_ids = buyers_df.iloc[partner_idx]["buyer_id"].values
    partner_device = device_mapping_df.set_index("buyer_id").loc[partner_buyer_ids, "device_id"].values

    # Determine first_seen_date: uniform between max(signup_a, signup_b) and SIM_END
    signup_a = pd.to_datetime(buyers_df.iloc[sharing_idx]["signup_date"].values)
    signup_b = pd.to_datetime(buyers_df.iloc[partner_idx]["signup_date"].values)
    latest_signup = pd.to_datetime(np.maximum(signup_a.values, signup_b.values))
    sim_end_dt = pd.Timestamp(SIM_END)
    # Clamp: if latest_signup >= SIM_END, set first_seen = SIM_END
    span_seconds = np.maximum((sim_end_dt - latest_signup).total_seconds(), 0)
    offsets_seconds = gen.random(n_sharing) * span_seconds
    first_seen = latest_signup + pd.to_timedelta(offsets_seconds, unit="s")

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


def build_base_entities(seed: int | None = RNG_SEED, rng: np.random.Generator | None = None) -> dict:
    """Builds base entity tables and sharing relationships deterministically.

    Parameters
    ----------
    seed : int or None, default RNG_SEED (42)
        Controlled seed to reset module RNG in-place when rng is not explicitly provided.
    rng : np.random.Generator or None, default None
        Custom generator. If provided, used directly without mutating global RNG state.
    """
    if rng is None and seed is not None:
        rng = reset_rng(seed)
    elif rng is None:
        rng = globals()["rng"]

    addresses_df = generate_addresses(n_addresses=N_SELLERS + N_BUYERS, rng=rng)
    devices_df = generate_devices(n_devices=N_BUYERS, rng=rng)
    sellers_df = generate_sellers(N_SELLERS, addresses_df, rng=rng)
    buyers_df = generate_buyers(N_BUYERS, addresses_df, rng=rng)
    device_mapping_df = assign_primary_devices(buyers_df, devices_df, rng=rng)
    buyers_df, address_sharing_log = assign_shared_addresses(buyers_df, rng=rng)
    device_mapping_df, device_sharing_log = assign_shared_devices(device_mapping_df, buyers_df, rng=rng)

    return {
        "addresses": addresses_df,
        "devices": devices_df,
        "sellers": sellers_df,
        "buyers": buyers_df,
        "device_mapping": device_mapping_df,
        "address_sharing_log": address_sharing_log,
        "device_sharing_log": device_sharing_log,
        "rng": rng,
    }



def generate_full_pipeline(seed: int = RNG_SEED, rng: np.random.Generator | None = None,
                           abo_metadata_path: str | None = None,
                           ring_coherent: bool = False) -> dict:
    """Executes the full upstream generation pipeline deterministically starting from a controlled seed."""
    base = build_base_entities(seed=seed, rng=rng)
    p_rng = base["rng"]
    from product_listing_generator import build_catalog_and_listings
    from order_return_generator import build_orders_and_returns
    from fraud_injection import inject_all_fraud

    catalog = build_catalog_and_listings(base["sellers"], abo_metadata_path=abo_metadata_path, rng=p_rng)
    txn = build_orders_and_returns(
        base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"], rng=p_rng
    )
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"], catalog["products"],
        base["address_sharing_log"], base["device_sharing_log"], rng=p_rng,
        ring_coherent=ring_coherent
    )
    if ring_coherent and "device_sharing_log" in result:
        base["device_sharing_log"] = result["device_sharing_log"]
    return {
        "base": base,
        "catalog": catalog,
        "txn": txn,
        "result": result,
    }


if __name__ == "__main__":
    data = build_base_entities()
    for name, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"{name}: {len(df):,} rows")
