"""
TrustShield AI — Synthetic Data Generator
Part 1: Base entity generators (Address, Device, Seller, Buyer)

Design reference: design.md
- Scale: ~500 sellers, ~5,000 buyers
- Address sharing: ~12% of buyers share with another buyer (70% legit / 30% fraud-linked)
- Device sharing: ~8% of buyers share with another buyer (50% legit / 50% fraud-linked)
- Timeline: 12-month simulated span, gradual onboarding (~40% in first 2 months, rest phased in)
"""

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

# ---------------------------------------------------------------------------
# Onboarding timeline helper
# ---------------------------------------------------------------------------

def generate_onboarding_dates(n, early_share=0.40, early_window_days=60):
    """
    Generates signup dates over the simulation span with gradual onboarding:
    ~early_share of the population signs up within the first early_window_days
    (itself front-loaded within that window, not uniform), and the rest are
    spread across the *entire* span using a smooth growth curve — so there's
    no hard cliff at the early-window boundary; the early spike blends into
    the ongoing growth curve instead of being a separate, disjoint segment.
    """
    n_early = int(n * early_share)
    n_rest = n - n_early

    # Early adopters: front-loaded within the early window (more signups near
    # the very start, tapering off toward the end of the window) via a
    # concave curve, rather than uniform across the window.
    u_early = rng.random(n_early)
    early_days = (u_early ** 1.6) * early_window_days
    early_days = early_days.astype(int)

    # Remaining signups: drawn across the *full* simulation span with a mild
    # growth skew (more mass later), so density smoothly continues rising
    # from wherever the early cohort leaves off rather than restarting from
    # early_window_days.
    u_rest = rng.random(n_rest)
    rest_days = (u_rest ** 0.8) * SIM_DAYS
    rest_days = rest_days.astype(int)

    all_days = np.concatenate([early_days, rest_days])
    rng.shuffle(all_days)
    return [SIM_START + timedelta(days=int(d)) for d in all_days]


# ---------------------------------------------------------------------------
# Address
# ---------------------------------------------------------------------------

def generate_addresses(n_addresses):
    """
    Synthetic addresses — no real PII, just a region code + a synthetic id.
    We generate more addresses than (sellers+buyers) initially, then assign
    sharing on top (see assign_shared_addresses).
    """
    regions = [f"REGION_{i:02d}" for i in range(1, 31)]  # 30 synthetic regions
    df = pd.DataFrame({
        "address_id": [f"ADDR_{i:06d}" for i in range(n_addresses)],
        "pincode_region": rng.choice(regions, size=n_addresses),
        "first_seen_date": generate_onboarding_dates(n_addresses),
    })
    return df


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def generate_devices(n_devices):
    df = pd.DataFrame({
        "device_id": [f"DEV_{i:06d}" for i in range(n_devices)],
        "device_fingerprint": [f"FP_{rng.integers(10**9, 10**10-1)}" for _ in range(n_devices)],
        "first_seen_date": generate_onboarding_dates(n_devices),
    })
    return df


# ---------------------------------------------------------------------------
# Seller
# ---------------------------------------------------------------------------

def generate_sellers(n_sellers, addresses_df):
    categories = [
        "Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports",
        "Toys", "Books", "Grocery", "Automotive", "Furniture"
    ]
    signup_dates = generate_onboarding_dates(n_sellers)

    df = pd.DataFrame({
        "seller_id": [f"SELLER_{i:05d}" for i in range(n_sellers)],
        "signup_date": signup_dates,
        "address_id": rng.choice(addresses_df["address_id"], size=n_sellers, replace=False),
        "category_focus": rng.choice(categories, size=n_sellers),
        "trust_score_current": 70.0,   # starting trust score, evolves later (Phase 3)
        "trust_score_history": [[] for _ in range(n_sellers)],  # filled in later phases
        "total_listings": 0,           # filled in once listings are generated
        "total_orders_received": 0,    # filled in once orders are generated
        "account_status": "active",
    })
    return df


# ---------------------------------------------------------------------------
# Buyer
# ---------------------------------------------------------------------------

def generate_buyers(n_buyers, addresses_df):
    signup_dates = generate_onboarding_dates(n_buyers)

    df = pd.DataFrame({
        "buyer_id": [f"BUYER_{i:05d}" for i in range(n_buyers)],
        "signup_date": signup_dates,
        "address_id": rng.choice(addresses_df["address_id"], size=n_buyers, replace=True),
        "trust_score_current": 70.0,
        "total_orders": 0,
        "total_returns": 0,
    })
    return df


# ---------------------------------------------------------------------------
# Device assignment (each buyer gets >=1 device; sharing handled separately)
# ---------------------------------------------------------------------------

def assign_primary_devices(buyers_df, devices_df):
    """
    Assigns each buyer exactly one primary device to start.
    Shared devices (buyer-to-buyer overlap) are layered on top in
    assign_shared_devices().
    Returns a buyer_device mapping dataframe (many-to-many capable).
    """
    n = len(buyers_df)
    primary_devices = rng.choice(devices_df["device_id"], size=n, replace=False)
    mapping = pd.DataFrame({
        "buyer_id": buyers_df["buyer_id"].values,
        "device_id": primary_devices,
        "is_primary": True,
    })
    return mapping


# ---------------------------------------------------------------------------
# Shared address / device injection
# ---------------------------------------------------------------------------

def assign_shared_addresses(buyers_df, share_rate=0.12, legit_share=0.70):
    """
    Selects share_rate of buyers to share an address with another buyer.
    legit_share of those pairs are "legitimate" (family/roommate — just a
    label for now, ground truth only), the rest are fraud-ring-linked.

    Returns:
        updated buyers_df (address_id column modified for the sharing buyers)
        sharing_log: dataframe recording each sharing pair + its legit/fraud label
    """
    buyers_df = buyers_df.copy()
    n = len(buyers_df)
    n_sharing = int(n * share_rate)

    sharing_idx = rng.choice(n, size=n_sharing, replace=False)
    partner_idx = rng.choice(n, size=n_sharing, replace=False)
    self_pair_mask = sharing_idx == partner_idx
    while self_pair_mask.any():
        partner_idx[self_pair_mask] = rng.choice(n, size=self_pair_mask.sum(), replace=False)
        self_pair_mask = sharing_idx == partner_idx

    is_fraud_link = rng.random(n_sharing) >= legit_share  # True = fraud-linked

    sharing_log = pd.DataFrame({
        "buyer_id": buyers_df.iloc[sharing_idx]["buyer_id"].values,
        "shared_with_buyer_id": buyers_df.iloc[partner_idx]["buyer_id"].values,
        "shared_address_id": buyers_df.iloc[partner_idx]["address_id"].values,
        "share_type": np.where(is_fraud_link, "fraud_linked", "legitimate"),
    })

    buyers_df.loc[sharing_idx, "address_id"] = sharing_log["shared_address_id"].values

    return buyers_df, sharing_log


def assign_shared_devices(device_mapping_df, buyers_df, share_rate=0.08, legit_share=0.50):
    """
    share_rate of buyers get an additional device link to another buyer's
    primary device (representing shared device usage), split legit/fraud.

    Returns:
        updated device_mapping_df (with extra shared rows appended)
        sharing_log: dataframe recording each sharing pair + its legit/fraud label
    """
    n = len(buyers_df)
    n_sharing = int(n * share_rate)

    sharing_idx = rng.choice(n, size=n_sharing, replace=False)
    partner_idx = rng.choice(n, size=n_sharing, replace=False)
    self_pair_mask = sharing_idx == partner_idx
    while self_pair_mask.any():
        partner_idx[self_pair_mask] = rng.choice(n, size=self_pair_mask.sum(), replace=False)
        self_pair_mask = sharing_idx == partner_idx

    is_fraud_link = rng.random(n_sharing) >= legit_share

    sharer_buyer_ids = buyers_df.iloc[sharing_idx]["buyer_id"].values
    partner_buyer_ids = buyers_df.iloc[partner_idx]["buyer_id"].values

    partner_device = device_mapping_df.set_index("buyer_id").loc[partner_buyer_ids, "device_id"].values

    sharing_log = pd.DataFrame({
        "buyer_id": sharer_buyer_ids,
        "shared_with_buyer_id": partner_buyer_ids,
        "shared_device_id": partner_device,
        "share_type": np.where(is_fraud_link, "fraud_linked", "legitimate"),
    })

    extra_rows = pd.DataFrame({
        "buyer_id": sharer_buyer_ids,
        "device_id": partner_device,
        "is_primary": False,
    })
    updated_mapping = pd.concat([device_mapping_df, extra_rows], ignore_index=True)

    return updated_mapping, sharing_log


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_base_entities():
    print("Generating addresses...")
    addresses_df = generate_addresses(n_addresses=int((N_SELLERS + N_BUYERS) * 0.9))

    print("Generating devices...")
    devices_df = generate_devices(n_devices=N_BUYERS)

    print("Generating sellers...")
    sellers_df = generate_sellers(N_SELLERS, addresses_df)

    print("Generating buyers...")
    buyers_df = generate_buyers(N_BUYERS, addresses_df)

    print("Assigning primary devices...")
    device_mapping_df = assign_primary_devices(buyers_df, devices_df)

    print("Injecting shared addresses...")
    buyers_df, address_sharing_log = assign_shared_addresses(buyers_df)

    print("Injecting shared devices...")
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
    print("\n--- Entity Generation Summary ---")
    for name, df in data.items():
        print(f"✅ {name.replace('_', ' ').capitalize()}: {len(df):,} rows")
