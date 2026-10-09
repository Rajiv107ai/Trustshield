"""
TrustShield — Realistic Synthetic Data Generator v2.1 (Stage 3.1.1 Integrity Fixes).

Remediates all implementation gaps identified in the Stage 3.1 review:
1. Category-Constrained Product & Multimodal Perturbations:
   - All fake listing perturbations are strictly constrained to the SAME category and compatible product types.
   - Eliminates cross-catalog product ID swaps.
2. Generator-Label & Persona Leakage Prevention:
   - Excludes generator-only metadata (buyer_persona, seller_persona, compromise_date) from public tables.
   - Persona metadata is stored strictly in generator_personas.csv and fraud_ground_truth.csv.
   - Public schemas match production schemas exactly.
3. Rigorous Return-Date Semantics & Right-Censoring:
   - Every return occurs strictly after its order (days_to_return >= 1).
   - Zero-day returns are mathematically prevented.
   - Orders too close to SIM_END whose return would fall in 2026 are treated as right-censored observations (not recorded in returns.csv).
   - Sampled delays are never silently shortened.
4. Observable Seller Behaviors:
   - Disposable Sellers: Rapid listing burst within 2-7 days of signup, followed by scam transactions and dormancy.
   - Takeover (ATO) Sellers: Established accounts with steady history in category C1 that experience an anomalous listing surge in high-risk category C2 after an explicit compromise date.
   - Measurable in observable timestamps and category distributions without relying on persona labels.
5. Point-in-Time Prediction Feature Safety:
   - Entity table summary aggregates (total_orders, total_returns, total_listings) are documented as simulation-end summaries and forbidden at prediction time.
   - Models must strictly use point-in-time backward as-of merges.
6. Unified, Non-Label-Revealing Identifiers:
   - All return IDs format as "RETURN_{i:06d}" (zero "RETURN_FRAUD_" strings).
7. Comprehensive Automated Validation Suite:
   - Verifies foreign keys, nulls, unique IDs, chronological invariants, date boundaries, distribution overlaps, and ID leakage.
   - Outputs data/synthetic_v2_1/validation_report.json and dataset_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

# Path resolutions
_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_PROJECT_DIR = os.path.join(_ROOT_DIR, "trustshield_project")
_OUTPUT_DIR = os.path.join(_ROOT_DIR, "data", "synthetic_v2_1")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from product_listing_generator import CATEGORIES, CATEGORY_PRICE_PARAMS, generate_product_catalog


# ---------------------------------------------------------------------------
# Global Constants & Simulation Parameters
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
SIM_START = datetime(2025, 1, 1)
SIM_END = datetime(2025, 12, 31)
SIM_DAYS = (SIM_END - SIM_START).days

N_SELLERS = 500
N_BUYERS = 5000
N_PRODUCTS = 3000
TARGET_LISTINGS = 20000
TARGET_ORDERS = 50000
TARGET_FRAUD_RATE = 0.07

FRAUD_TYPE_SHARE = {
    "fake_listing": 0.40,
    "return_abuse": 0.30,
    "coordinated_fraud": 0.20,
    "seller_buyer_collusion": 0.10,
}

RETURN_REASONS = ["defective", "changed_mind", "size_issue", "wrong_item_received"]
RETURN_REASON_WEIGHTS = [0.30, 0.35, 0.20, 0.15]


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Stage 1: Base Entities (Addresses, Devices, Sellers, Buyers)
# ---------------------------------------------------------------------------

def generate_base_entities_v2_1(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Generates base entities with separate persona tracking to prevent leakage."""
    rng = np.random.default_rng(seed)

    # 1. Addresses
    n_addresses = int((N_SELLERS + N_BUYERS) * 0.9)
    regions = [f"REGION_{i:02d}" for i in range(1, 31)]
    address_offsets = ((rng.random(n_addresses) ** 0.9) * SIM_DAYS).astype(int)
    addresses_df = pd.DataFrame({
        "address_id": [f"ADDR_{i:06d}" for i in range(n_addresses)],
        "pincode_region": rng.choice(regions, size=n_addresses),
        "first_seen_date": [SIM_START + timedelta(days=int(d)) for d in address_offsets],
    })

    # 2. Devices
    n_devices = N_BUYERS
    device_offsets = ((rng.random(n_devices) ** 0.9) * SIM_DAYS).astype(int)
    devices_df = pd.DataFrame({
        "device_id": [f"DEV_{i:06d}" for i in range(n_devices)],
        "device_fingerprint": [f"FP_{rng.integers(10**9, 10**10 - 1)}" for _ in range(n_devices)],
        "first_seen_date": [SIM_START + timedelta(days=int(d)) for d in device_offsets],
    })

    # 3. Sellers
    # Personas for simulation control (segregated from public table):
    # - established (~80%, 400 sellers): onboarded early, stable listing behavior
    # - disposable (~15%, 75 sellers): onboarded throughout year, short active burst
    # - takeover (~5%, 25 sellers): onboarded early, normal initially, compromised later
    n_established = int(N_SELLERS * 0.80)
    n_disposable = int(N_SELLERS * 0.15)
    n_takeover = N_SELLERS - n_established - n_disposable

    seller_personas = (
        ["established"] * n_established +
        ["disposable"] * n_disposable +
        ["takeover"] * n_takeover
    )
    rng.shuffle(seller_personas)

    seller_signup_dates = []
    seller_compromise_dates: List[Optional[datetime]] = []

    for stype in seller_personas:
        if stype in ["established", "takeover"]:
            # Onboarded mostly in the first 4 months
            days = int((rng.random() ** 1.8) * 110)
            signup = SIM_START + timedelta(days=days)
            seller_signup_dates.append(signup)
            if stype == "takeover":
                # Compromise occurs between Month 6 and Month 10
                comp_days = int(rng.uniform(150, 280))
                seller_compromise_dates.append(SIM_START + timedelta(days=comp_days))
            else:
                seller_compromise_dates.append(None)
        else:
            # Disposable sellers onboarded throughout the year
            days = int(rng.uniform(40, SIM_DAYS - 25))
            signup = SIM_START + timedelta(days=days)
            seller_signup_dates.append(signup)
            seller_compromise_dates.append(None)

    # Public Sellers table (Production Schema: NO persona column!)
    sellers_public = pd.DataFrame({
        "seller_id": [f"SELLER_{i:05d}" for i in range(N_SELLERS)],
        "signup_date": seller_signup_dates,
        "address_id": rng.choice(addresses_df["address_id"], size=N_SELLERS, replace=False),
        "category_focus": rng.choice(CATEGORIES, size=N_SELLERS),
        "trust_score_current": 70.0,
        "total_listings": 0,
        "total_orders_received": 0,
        "account_status": "active",
    })

    # Internal Seller Metadata (Saved separately)
    seller_metadata = pd.DataFrame({
        "seller_id": sellers_public["seller_id"],
        "seller_persona": seller_personas,
        "compromise_date": seller_compromise_dates,
    })

    # 4. Buyers
    # Personas for simulation control (segregated from public table):
    # - standard (~75%, 3750): normal return rate (5-8%)
    # - high_return_legit (~15%, 750): fit-check / sizing shoppers (22-38% returns)
    # - abuser (~10%, 500): opportunistic return abusers (25-45% returns)
    n_std_buyers = int(N_BUYERS * 0.75)
    n_high_ret = int(N_BUYERS * 0.15)
    n_abuser = N_BUYERS - n_std_buyers - n_high_ret

    buyer_personas = (
        ["standard"] * n_std_buyers +
        ["high_return_legit"] * n_high_ret +
        ["abuser"] * n_abuser
    )
    rng.shuffle(buyer_personas)

    buyer_offsets = ((rng.random(N_BUYERS) ** 0.9) * (SIM_DAYS - 10)).astype(int)
    buyers_public = pd.DataFrame({
        "buyer_id": [f"BUYER_{i:05d}" for i in range(N_BUYERS)],
        "signup_date": [SIM_START + timedelta(days=int(d)) for d in buyer_offsets],
        "address_id": rng.choice(addresses_df["address_id"], size=N_BUYERS, replace=True),
        "trust_score_current": 70.0,
        "total_orders": 0,
        "total_returns": 0,
    })

    buyer_metadata = pd.DataFrame({
        "buyer_id": buyers_public["buyer_id"],
        "buyer_persona": buyer_personas,
    })

    # 5. Primary Device Mapping (1-to-1)
    device_mapping_df = pd.DataFrame({
        "buyer_id": buyers_public["buyer_id"].values,
        "device_id": rng.choice(devices_df["device_id"], size=N_BUYERS, replace=False),
        "is_primary": True,
    })

    # 6. Shared Addresses (12% sharing rate, 70% legit household sharing, 30% fraud-linked)
    n_addr_sharing = int(N_BUYERS * 0.12)
    s_idx = rng.choice(N_BUYERS, size=n_addr_sharing, replace=False)
    p_idx = rng.choice(N_BUYERS, size=n_addr_sharing, replace=False)
    mask = s_idx == p_idx
    while mask.any():
        p_idx[mask] = rng.choice(N_BUYERS, size=mask.sum(), replace=False)
        mask = s_idx == p_idx

    is_addr_fraud = rng.random(n_addr_sharing) >= 0.70
    signup_a = pd.to_datetime(buyers_public.iloc[s_idx]["signup_date"].values)
    signup_b = pd.to_datetime(buyers_public.iloc[p_idx]["signup_date"].values)
    latest_signup = np.maximum(signup_a.asi8, signup_b.asi8)
    sim_end_ns = pd.Timestamp(SIM_END).as_unit("ns").value
    span_ns = np.maximum(sim_end_ns - latest_signup, 0)
    offsets_ns = (rng.random(n_addr_sharing) * span_ns).astype(np.int64)
    addr_first_seen = pd.to_datetime(latest_signup + offsets_ns)

    address_sharing_log = pd.DataFrame({
        "buyer_id": buyers_public.iloc[s_idx]["buyer_id"].values,
        "shared_with_buyer_id": buyers_public.iloc[p_idx]["buyer_id"].values,
        "shared_address_id": buyers_public.iloc[p_idx]["address_id"].values,
        "share_type": np.where(is_addr_fraud, "fraud_linked", "legitimate"),
        "first_seen_date": addr_first_seen,
    })
    buyers_public.loc[s_idx, "address_id"] = address_sharing_log["shared_address_id"].values

    # 7. Shared Devices (8% sharing rate, 50% legit family/shared, 50% fraud-linked)
    n_dev_sharing = int(N_BUYERS * 0.08)
    ds_idx = rng.choice(N_BUYERS, size=n_dev_sharing, replace=False)
    dp_idx = rng.choice(N_BUYERS, size=n_dev_sharing, replace=False)
    mask = ds_idx == dp_idx
    while mask.any():
        dp_idx[mask] = rng.choice(N_BUYERS, size=mask.sum(), replace=False)
        mask = ds_idx == dp_idx

    is_dev_fraud = rng.random(n_dev_sharing) >= 0.50
    sharer_bids = buyers_public.iloc[ds_idx]["buyer_id"].values
    partner_bids = buyers_public.iloc[dp_idx]["buyer_id"].values
    partner_dev = device_mapping_df.set_index("buyer_id").loc[partner_bids, "device_id"].values

    dsignup_a = pd.to_datetime(buyers_public.iloc[ds_idx]["signup_date"].values)
    dsignup_b = pd.to_datetime(buyers_public.iloc[dp_idx]["signup_date"].values)
    dlatest = np.maximum(dsignup_a.asi8, dsignup_b.asi8)
    dspan = np.maximum(sim_end_ns - dlatest, 0)
    doffsets = (rng.random(n_dev_sharing) * dspan).astype(np.int64)
    dev_first_seen = pd.to_datetime(dlatest + doffsets)

    device_sharing_log = pd.DataFrame({
        "buyer_id": sharer_bids,
        "shared_with_buyer_id": partner_bids,
        "shared_device_id": partner_dev,
        "share_type": np.where(is_dev_fraud, "fraud_linked", "legitimate"),
        "first_seen_date": dev_first_seen,
    })
    extra_dev_rows = pd.DataFrame({
        "buyer_id": sharer_bids,
        "device_id": partner_dev,
        "is_primary": False,
    })
    device_mapping_df = pd.concat([device_mapping_df, extra_dev_rows], ignore_index=True)

    return {
        "addresses": addresses_df,
        "devices": devices_df,
        "sellers": sellers_public,
        "buyers": buyers_public,
        "device_mapping": device_mapping_df,
        "address_sharing_log": address_sharing_log,
        "device_sharing_log": device_sharing_log,
        "internal_seller_metadata": seller_metadata,
        "internal_buyer_metadata": buyer_metadata,
    }


# ---------------------------------------------------------------------------
# Stage 2: Product Catalog & Observable Seller Listings
# ---------------------------------------------------------------------------

def generate_catalog_and_listings_v2_1(
    sellers_df: pd.DataFrame,
    seller_metadata: pd.DataFrame,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Generates catalog and listings modeling measurable seller behaviors:
    - Disposable sellers: 100% of listings created in a short burst (2-7 days from signup).
    - Takeover sellers: Pre-compromise normal listing in focus category C1, followed by
      an anomalous burst in a different high-risk category C2 post-compromise.
    - Established sellers: Normal staggered listing curve across the year.
    """
    rng = np.random.default_rng(seed + 1)
    products_df = generate_product_catalog(rng=rng)

    persona_map = seller_metadata.set_index("seller_id")["seller_persona"].to_dict()
    comp_date_map = seller_metadata.set_index("seller_id")["compromise_date"].to_dict()

    n_sellers = len(sellers_df)
    weights = rng.lognormal(mean=0.0, sigma=0.8, size=n_sellers)
    weights /= weights.sum()
    counts = np.maximum(1, (weights * TARGET_LISTINGS).round().astype(int))
    diff = TARGET_LISTINGS - counts.sum()
    if diff != 0:
        idx = rng.choice(n_sellers, size=abs(diff), replace=True)
        counts[idx] += np.sign(diff)
        counts = np.maximum(1, counts)

    products_by_cat = {
        c: pd.DataFrame(products_df[products_df["category"] == c]).to_dict(orient="records")
        for c in CATEGORIES
    }

    rows = []
    counter = 0

    for (_, seller), n_list in zip(sellers_df.iterrows(), counts):
        s_id = seller["seller_id"]
        signup = seller["signup_date"]
        persona = persona_map.get(s_id, "established")
        comp_date = comp_date_map.get(s_id)
        focus_cat = seller["category_focus"]

        listing_dates = []
        categories = []

        if persona == "disposable":
            # Measurable Behavior 1: Short-Tenure Listing Burst
            # All listings published within 2 to 7 days of account registration!
            burst_days = int(rng.integers(2, 8))
            offsets = rng.uniform(0, burst_days, size=n_list)
            listing_dates = [signup + timedelta(days=float(d)) for d in offsets]
            # Concentrated in high-value categories
            categories = [rng.choice(["Electronics", "Automotive", "Furniture", focus_cat]) for _ in range(n_list)]

        elif persona == "takeover" and comp_date is not None:
            # Measurable Behavior 2: Account Takeover Surge
            # Split listings: 40% pre-compromise in focus_cat, 60% post-compromise in anomalous category
            n_pre = max(1, int(n_list * 0.40))
            n_post = max(1, n_list - n_pre)

            # Pre-compromise: steady pace between signup and compromise date
            pre_span = max(1, (comp_date - signup).days - 5)
            pre_offsets = (rng.random(n_pre) ** 1.1) * pre_span
            listing_dates.extend([signup + timedelta(days=float(d)) for d in pre_offsets])
            categories.extend([focus_cat] * n_pre)

            # Post-compromise: Sudden anomalous burst in high-risk category within 1-5 days
            alt_cats = [c for c in CATEGORIES if c != focus_cat]
            ato_cat = str(rng.choice(["Electronics", "Beauty", "Sports"] if focus_cat not in ["Electronics", "Beauty"] else alt_cats))
            post_offsets = rng.uniform(0, 5, size=n_post)
            listing_dates.extend([comp_date + timedelta(days=float(d)) for d in post_offsets])
            categories.extend([ato_cat] * n_post)

        else:
            # Established sellers: Normal staggered activity over the remaining year
            avail_days = max(1, (SIM_END - signup).days)
            offsets = (rng.random(n_list) ** 1.3) * avail_days
            listing_dates = [signup + timedelta(days=float(d)) for d in offsets]
            match_mask = rng.random(n_list) < 0.85
            categories = [focus_cat if match_mask[i] else str(rng.choice(CATEGORIES)) for i in range(n_list)]

        for i in range(len(listing_dates)):
            l_date = min(listing_dates[i], SIM_END)
            cat = categories[i]
            pool = products_by_cat.get(cat, [])
            prod = pool[int(rng.integers(0, len(pool)))] if pool else products_df.iloc[0].to_dict()

            p_factor = rng.lognormal(mean=0.0, sigma=0.15)
            price = round(float(prod["base_price"]) * p_factor, 2)

            rows.append({
                "listing_id": f"LISTING_{counter:06d}",
                "seller_id": s_id,
                "product_id": prod["product_id"],
                "displayed_product_id": prod["product_id"],
                "category": cat,
                "listing_date": l_date,
                "price": price,
                "status": "active",
            })
            counter += 1

    listings_df = pd.DataFrame(rows)
    sellers_df = sellers_df.copy()
    l_counts = listings_df.groupby("seller_id").size().to_dict()
    sellers_df["total_listings"] = sellers_df["seller_id"].map(lambda s: l_counts.get(s, 0)).astype(int)

    return {
        "products": products_df,
        "listings": listings_df,
        "sellers": sellers_df,
    }


# ---------------------------------------------------------------------------
# Stage 3: Orders & Baseline Organic Returns
# ---------------------------------------------------------------------------

def generate_orders_and_returns_v2_1(
    buyers_df: pd.DataFrame,
    buyer_metadata: pd.DataFrame,
    sellers_df: pd.DataFrame,
    listings_df: pd.DataFrame,
    device_mapping_df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Generates transactions and organic returns with right-censored end-of-year return handling."""
    rng = np.random.default_rng(seed + 2)

    listings_sorted = listings_df.sort_values("listing_date").reset_index(drop=True)
    listing_timestamps = pd.to_datetime(listings_sorted["listing_date"]).values.astype("datetime64[ns]").astype(np.int64)
    min_listing_date = listings_sorted["listing_date"].iloc[0]

    buyer_devices = device_mapping_df.groupby("buyer_id")["device_id"].apply(list).to_dict()

    n_buyers = len(buyers_df)
    weights = rng.lognormal(mean=0.0, sigma=0.7, size=n_buyers)
    weights /= weights.sum()
    order_counts = np.maximum(0, (weights * TARGET_ORDERS).round().astype(int))
    diff = int(TARGET_ORDERS - order_counts.sum())
    if diff > 0:
        idx = rng.choice(n_buyers, size=diff, replace=True)
        np.add.at(order_counts, idx, 1)
    elif diff < 0:
        elig = np.flatnonzero(order_counts > 0)
        idx = rng.choice(elig, size=min(abs(diff), len(elig)), replace=False)
        np.add.at(order_counts, idx, -1)

    order_rows = []
    order_counter = 0

    for (_, buyer), n_ord in zip(buyers_df.iterrows(), order_counts):
        if n_ord == 0:
            continue
        eff_signup = max(buyer["signup_date"], min_listing_date)
        window = max(1, (SIM_END - eff_signup).days)
        cand_offsets = (rng.random(n_ord) ** 1.3) * window
        cand_dates = [eff_signup + timedelta(days=float(d)) for d in cand_offsets]
        dev_list = buyer_devices.get(buyer["buyer_id"], [None])

        for o_date in cand_dates:
            o_date = min(o_date, SIM_END)
            o_ts = pd.Timestamp(o_date).as_unit("ns").value
            c_idx = np.searchsorted(listing_timestamps, o_ts, side="right")
            if c_idx == 0:
                l_row = listings_sorted.iloc[0]
                o_date = l_row["listing_date"] + timedelta(minutes=int(rng.integers(5, 60)))
            else:
                l_row = listings_sorted.iloc[int(rng.integers(0, c_idx))]
                if o_date < l_row["listing_date"]:
                    o_date = l_row["listing_date"] + timedelta(minutes=int(rng.integers(5, 60)))
            # Enforce buyer signup invariant and SIM_END ceiling
            o_date = max(o_date, buyer["signup_date"] + timedelta(minutes=1))
            o_date = min(o_date, SIM_END)
            dev_id = dev_list[int(rng.integers(0, len(dev_list)))]

            order_rows.append({
                "order_id": f"ORDER_{order_counter:06d}",
                "buyer_id": buyer["buyer_id"],
                "seller_id": l_row["seller_id"],
                "listing_id": l_row["listing_id"],
                "product_id": l_row["product_id"],
                "order_date": o_date,
                "amount": l_row["price"],
                "device_id": dev_id,
                "status": "completed",
            })
            order_counter += 1

    orders_df = pd.DataFrame(order_rows)

    # Organic Returns with Strict Right-Censoring Policy
    persona_map = buyer_metadata.set_index("buyer_id")["buyer_persona"].to_dict()
    return_rows = []
    return_counter = 0
    right_censored_count = 0

    for _, order in orders_df.iterrows():
        persona = persona_map.get(order["buyer_id"], "standard")
        # Persona return probabilities:
        # - high_return_legit: 28% probability (fit check, sizing)
        # - standard: 5% probability
        # - abuser (organic baseline): 6% probability
        ret_prob = 0.28 if persona == "high_return_legit" else (0.05 if persona == "standard" else 0.06)

        if rng.random() < ret_prob:
            # Overlapping Return Turnaround Delay:
            # 35% fast returns (1 to 5 days, defective or wrong size)
            # 65% normal returns (4 to 21 days)
            # DELAY IS ALWAYS AT LEAST 1 DAY!
            if rng.random() < 0.35:
                delay = int(rng.integers(1, 6))
            else:
                delay = int(rng.integers(4, 22))

            scheduled_return_date = order["order_date"] + timedelta(days=delay)

            # RIGHT-CENSORING POLICY:
            # If the scheduled return occurs after SIM_END (in 2026),
            # it falls outside the longitudinal observation window!
            # It is NOT recorded in returns.csv. It is NEVER clamped to create a 0-day return.
            if scheduled_return_date <= SIM_END:
                reason = rng.choice(RETURN_REASONS, p=RETURN_REASON_WEIGHTS)
                return_rows.append({
                    "return_id": f"RETURN_{return_counter:06d}",
                    "order_id": order["order_id"],
                    "buyer_id": order["buyer_id"],
                    "seller_id": order["seller_id"],
                    "return_date": scheduled_return_date,
                    "reason": reason,
                    "status": "approved",
                    "is_fraudulent": False,
                    "fraud_type": None,
                })
                return_counter += 1
            else:
                right_censored_count += 1

    returns_df = pd.DataFrame(return_rows)

    buyers_df = buyers_df.copy()
    sellers_df = sellers_df.copy()
    b_ocount = orders_df.groupby("buyer_id").size().to_dict()
    b_rcount = returns_df.groupby("buyer_id").size().to_dict()
    s_ocount = orders_df.groupby("seller_id").size().to_dict()

    buyers_df["total_orders"] = buyers_df["buyer_id"].map(lambda b: b_ocount.get(b, 0)).astype(int)
    buyers_df["total_returns"] = buyers_df["buyer_id"].map(lambda b: b_rcount.get(b, 0)).astype(int)
    sellers_df["total_orders_received"] = sellers_df["seller_id"].map(lambda s: s_ocount.get(s, 0)).astype(int)

    return {
        "orders": orders_df,
        "returns": returns_df,
        "buyers": buyers_df,
        "sellers": sellers_df,
        "right_censored_organic_returns": right_censored_count,
    }


# ---------------------------------------------------------------------------
# Stage 4: Realistic Fraud Injection with Category-Constrained Perturbations
# ---------------------------------------------------------------------------

def inject_realistic_fraud_v2_1(
    catalog: Dict[str, Any],
    txn: Dict[str, Any],
    base: Dict[str, Any],
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Injects realistic fraud patterns with:
    - Category-constrained fake listing perturbations.
    - True observable disposable and takeover seller behavior.
    - Overlapping return abuse with strict right-censoring.
    - Strict date clamping for burst rings.
    """
    rng = np.random.default_rng(seed + 3)

    listings_df = catalog["listings"].copy()
    orders_df = txn["orders"].copy()
    returns_df = txn["returns"].copy()
    buyers_df = txn["buyers"].copy()
    sellers_df = catalog["sellers"].copy()
    products_df = catalog["products"].copy()
    addr_log = base["address_sharing_log"].copy()
    dev_log = base["device_sharing_log"].copy()

    seller_meta = base["internal_seller_metadata"]
    buyer_meta = base["internal_buyer_metadata"]
    seller_persona_map = seller_meta.set_index("seller_id")["seller_persona"].to_dict()
    buyer_persona_map = buyer_meta.set_index("buyer_id")["buyer_persona"].to_dict()

    total_orders = len(orders_df)
    total_fraud_target = int(total_orders * TARGET_FRAUD_RATE)
    target_fake_listings = int(len(listings_df) * 0.025)  # 500 fake listings

    ground_truth_ledgers = []

    # =======================================================================
    # 1. Realistic Fake Listings: Category-Constrained & Disposable Concentration
    # =======================================================================
    # In v2.1:
    # - 65% from disposable sellers (exhibiting short tenure burst)
    # - 20% from takeover sellers (exhibiting anomalous post-compromise category surge)
    # - 15% from established sellers
    # - Perturbations match within the SAME category and compatible product types!
    listings_df["_seller_persona"] = listings_df["seller_id"].map(seller_persona_map)

    disp_lids = listings_df[listings_df["_seller_persona"] == "disposable"]["listing_id"].to_numpy()
    ato_lids = listings_df[listings_df["_seller_persona"] == "takeover"]["listing_id"].to_numpy()
    other_lids = listings_df[listings_df["_seller_persona"] == "established"]["listing_id"].to_numpy()

    n_disp = min(int(target_fake_listings * 0.65), len(disp_lids))
    n_ato = min(int(target_fake_listings * 0.20), len(ato_lids))
    n_other = min(target_fake_listings - n_disp - n_ato, len(other_lids))

    chosen_fake_lids = np.concatenate([
        rng.choice(disp_lids, size=n_disp, replace=False) if n_disp > 0 else [],
        rng.choice(ato_lids, size=n_ato, replace=False) if n_ato > 0 else [],
        rng.choice(other_lids, size=n_other, replace=False) if n_other > 0 else [],
    ])
    chosen_fake_mask = listings_df["listing_id"].isin(chosen_fake_lids)

    # Category-Constrained Product Perturbations with Compatible Subtype Matching
    prods_by_cat_df = {c: products_df[products_df["category"] == c] for c in CATEGORIES}

    updated_prices = {}
    perturbed_evidence_log = []

    subtypes_keywords = [
        "case", "cover", "shoe", "heel", "boot", "cable", "charger", "adapter",
        "speaker", "headphone", "lamp", "light", "pillow", "sheet", "knife",
        "pan", "pot", "shirt", "dress", "watch", "toy", "game", "book", "drawer"
    ]

    for idx in listings_df[chosen_fake_mask].index:
        cat = listings_df.loc[idx, "category"]
        curr_price = listings_df.loc[idx, "price"]
        actual_pid = listings_df.loc[idx, "product_id"]

        cat_pool = prods_by_cat_df.get(cat, products_df[products_df["category"] == cat])
        other_candidates = cat_pool[cat_pool["product_id"] != actual_pid]

        swapped_pid = actual_pid
        pert_type = "same_category_variant_mismatch"

        if len(other_candidates) > 0:
            # Check compatible subtype by inspecting claimed product title
            curr_row = cat_pool[cat_pool["product_id"] == actual_pid]
            curr_title = str(curr_row["title"].iloc[0]).lower() if len(curr_row) > 0 else ""
            matched_st = [st for st in subtypes_keywords if st in curr_title]

            if matched_st:
                st_kw = matched_st[0]
                st_matches = other_candidates[other_candidates["title"].str.lower().str.contains(st_kw, regex=False)]
                if len(st_matches) > 0:
                    swapped_pid = str(rng.choice(st_matches["product_id"].to_numpy()))
                    pert_type = f"same_category_compatible_subtype_{st_kw}"
                else:
                    swapped_pid = str(rng.choice(other_candidates["product_id"].to_numpy()))
                    pert_type = "same_category_variant_mismatch"
            else:
                swapped_pid = str(rng.choice(other_candidates["product_id"].to_numpy()))
                pert_type = "same_category_variant_mismatch"

        # Assign displayed_product_id for observable listing/image evidence
        listings_df.loc[idx, "displayed_product_id"] = swapped_pid

        # Subtle pricing: 75% slight discount (10-20% off), 25% scalper markup (15-30% up)
        if rng.random() < 0.75:
            price_factor = rng.uniform(0.80, 0.92)
        else:
            price_factor = rng.uniform(1.15, 1.30)

        new_price = round(curr_price * price_factor, 2)
        listings_df.loc[idx, "price"] = new_price
        updated_prices[listings_df.loc[idx, "listing_id"]] = new_price

        # Record perturbation evidence in internal audit log
        perturbed_evidence_log.append({
            "listing_id": listings_df.loc[idx, "listing_id"],
            "claimed_product_id": actual_pid,
            "category": cat,
            "perturbed_product_id": swapped_pid,
            "price_factor": round(float(price_factor), 3),
            "perturbation_type": pert_type,
        })

    # Propagate updated prices to orders
    affected_o_mask = orders_df["listing_id"].isin(chosen_fake_lids)
    orders_df.loc[affected_o_mask, "amount"] = orders_df.loc[affected_o_mask, "listing_id"].map(updated_prices)

    listings_df["is_fraudulent"] = False
    listings_df["fraud_type"] = None
    listings_df.loc[chosen_fake_mask, "is_fraudulent"] = True
    listings_df.loc[chosen_fake_mask, "fraud_type"] = "fake_listing"

    fake_order_ids = set(orders_df.loc[affected_o_mask, "order_id"])

    # Ledger
    for s_id, group in listings_df[chosen_fake_mask].groupby("seller_id"):
        ring_id = f"RING_FAKE_{s_id}"
        for lid in group["listing_id"]:
            ground_truth_ledgers.append({
                "fraud_ring_id": ring_id, "fraud_type": "fake_listing",
                "entity_type": "listing", "entity_id": lid
            })

    # =======================================================================
    # 2. Realistic Return Abuse: Overlapping Delay & Strict Right-Censoring
    # =======================================================================
    rem_budget = max(0, total_fraud_target - len(fake_order_ids))
    target_abuse_orders = int(rem_budget * (FRAUD_TYPE_SHARE["return_abuse"] / (1.0 - FRAUD_TYPE_SHARE["fake_listing"])))

    abuser_buyer_pool = buyers_df[buyers_df["buyer_id"].map(buyer_persona_map) == "abuser"]["buyer_id"].to_numpy()
    rng.shuffle(abuser_buyer_pool)

    abuse_order_ids = set()
    new_return_rows = []
    return_counter = len(returns_df)
    right_censored_abuse_count = 0

    eligible_orders = orders_df[~orders_df["order_id"].isin(fake_order_ids)]
    already_ret_ids = set(returns_df["order_id"])

    running_abuse = 0
    for b_id in abuser_buyer_pool:
        if running_abuse >= target_abuse_orders:
            break
        b_orders = eligible_orders[eligible_orders["buyer_id"] == b_id]
        if len(b_orders) == 0:
            continue

        # Moderate return rate: 25% to 45%
        rate = rng.uniform(0.25, 0.45)
        n_take = max(1, int(round(len(b_orders) * rate)))
        taken_orders = b_orders.sample(n=min(n_take, len(b_orders)), random_state=int(rng.integers(0, 2**31)))

        for _, o in taken_orders.iterrows():
            abuse_order_ids.add(o["order_id"])
            running_abuse += 1

            if o["order_id"] in already_ret_ids:
                mask = returns_df["order_id"] == o["order_id"]
                returns_df.loc[mask, "is_fraudulent"] = True
                returns_df.loc[mask, "fraud_type"] = "return_abuse"
            else:
                # Overlapping Return Turnaround Delay (mixture):
                # 40% quick (2 to 6 days)
                # 60% normal (5 to 18 days)
                # NEVER 0 DAYS! DELAY >= 2 DAYS ALWAYS!
                if rng.random() < 0.40:
                    delay = int(rng.integers(2, 7))
                else:
                    delay = int(rng.integers(5, 19))

                scheduled_ret_date = o["order_date"] + timedelta(days=delay)

                # STRICT RIGHT-CENSORING:
                if scheduled_ret_date <= SIM_END:
                    new_return_rows.append({
                        "return_id": f"RETURN_{return_counter:06d}",
                        "order_id": o["order_id"],
                        "buyer_id": o["buyer_id"],
                        "seller_id": o["seller_id"],
                        "return_date": scheduled_ret_date,
                        "reason": rng.choice(RETURN_REASONS, p=RETURN_REASON_WEIGHTS),
                        "status": "approved",
                        "is_fraudulent": True,
                        "fraud_type": "return_abuse",
                    })
                    return_counter += 1
                else:
                    right_censored_abuse_count += 1

        ring_id = f"RING_ABUSE_{b_id}"
        ground_truth_ledgers.append({
            "fraud_ring_id": ring_id, "fraud_type": "return_abuse",
            "entity_type": "buyer", "entity_id": b_id
        })

    if new_return_rows:
        returns_df = pd.concat([returns_df, pd.DataFrame(new_return_rows)], ignore_index=True)

    # =======================================================================
    # 3. Coordinated Fraud with STRICT Date Clamping
    # =======================================================================
    rem_budget2 = max(0, total_fraud_target - len(fake_order_ids) - len(abuse_order_ids))
    target_coord = int(rem_budget2 * (FRAUD_TYPE_SHARE["coordinated_fraud"] / (FRAUD_TYPE_SHARE["coordinated_fraud"] + FRAUD_TYPE_SHARE["seller_buyer_collusion"])))

    f_pairs = dev_log[dev_log["share_type"] == "fraud_linked"]
    adj: Dict[str, set] = {}
    for b1, b2 in zip(f_pairs["buyer_id"], f_pairs["shared_with_buyer_id"]):
        adj.setdefault(b1, set()).add(b2)
        adj.setdefault(b2, set()).add(b1)

    visited = set()
    rings = []
    for b in sorted(adj.keys()):
        if b not in visited:
            comp = []
            queue = [b]
            visited.add(b)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for nxt in adj.get(curr, set()):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            if len(comp) >= 2:
                rings.append(sorted(comp))

    coord_order_ids = set()
    resched_map = {}
    already_used = fake_order_ids | abuse_order_ids | set(returns_df["order_id"])
    elig_coord_orders = orders_df[~orders_df["order_id"].isin(already_used)]

    l_dates = listings_df.set_index("listing_id")["listing_date"].to_dict()
    s_dates = buyers_df.set_index("buyer_id")["signup_date"].to_dict()

    running_coord = 0
    rng.shuffle(rings)

    for ring_members in rings:
        if running_coord >= target_coord:
            break
        r_orders = elig_coord_orders[elig_coord_orders["buyer_id"].isin(ring_members)]
        if len(r_orders) == 0:
            continue

        n_take = min(len(r_orders), max(2, int(rng.integers(2, 6))))
        taken = r_orders.sample(n=n_take, random_state=int(rng.integers(0, 2**31)))

        min_floor = max(
            max(l_dates[lid] for lid in taken["listing_id"]),
            max(s_dates[bid] for bid in taken["buyer_id"])
        )
        max_offset = max(1, (SIM_END - min_floor).days - 10)
        burst_start = min_floor + timedelta(days=int(rng.integers(1, max_offset + 1)))
        burst_span = int(rng.integers(3, 8))

        for oid in taken["order_id"]:
            offset = int(rng.integers(0, burst_span))
            # STRICT DATE CLAMPING TO SIM_END
            new_date = min(burst_start + timedelta(days=offset), SIM_END)
            resched_map[oid] = new_date
            coord_order_ids.add(oid)
            running_coord += 1

        ring_label = f"RING_COORD_{ring_members[0]}"
        for m in ring_members:
            ground_truth_ledgers.append({
                "fraud_ring_id": ring_label, "fraud_type": "coordinated_fraud",
                "entity_type": "buyer", "entity_id": m
            })

    if resched_map:
        mask = orders_df["order_id"].isin(resched_map.keys())
        orders_df.loc[mask, "order_date"] = orders_df.loc[mask, "order_id"].map(resched_map)

    # =======================================================================
    # 4. Seller-Buyer Collusion with STRICT Date Clamping & Right-Censoring
    # =======================================================================
    rem_budget3 = max(0, total_fraud_target - len(fake_order_ids) - len(abuse_order_ids) - len(coord_order_ids))
    target_collusion = rem_budget3

    already_coll = fake_order_ids | abuse_order_ids | coord_order_ids | set(returns_df["order_id"])
    elig_coll = orders_df[~orders_df["order_id"].isin(already_coll)]
    sellers_with_buyers = elig_coll.groupby("seller_id")["buyer_id"].nunique()
    candidate_coll_sellers = sellers_with_buyers[sellers_with_buyers >= 3].index.to_numpy()
    rng.shuffle(candidate_coll_sellers)

    collusion_order_ids = set()
    coll_resched_map = {}
    coll_return_rows = []
    running_coll = 0

    for s_id in candidate_coll_sellers:
        if running_coll >= target_collusion:
            break
        s_orders = elig_coll[elig_coll["seller_id"] == s_id]
        b_pool = s_orders["buyer_id"].unique()
        if len(b_pool) < 3:
            continue

        g_size = min(len(b_pool), int(rng.integers(3, 7)))
        c_buyers = rng.choice(b_pool, size=g_size, replace=False)
        g_orders = s_orders[s_orders["buyer_id"].isin(c_buyers)]

        n_take = min(len(g_orders), max(3, int(rng.integers(3, 8))))
        taken = g_orders.sample(n=n_take, random_state=int(rng.integers(0, 2**31)))

        min_floor = max(
            max(l_dates[lid] for lid in taken["listing_id"]),
            max(s_dates[bid] for bid in taken["buyer_id"])
        )
        max_offset = max(1, (SIM_END - min_floor).days - 12)
        burst_start = min_floor + timedelta(days=int(rng.integers(1, max_offset + 1)))
        burst_span = int(rng.integers(4, 12))

        for oid in taken["order_id"]:
            offset = int(rng.integers(0, burst_span))
            # STRICT DATE CLAMPING TO SIM_END
            new_date = min(burst_start + timedelta(days=offset), SIM_END)
            coll_resched_map[oid] = new_date

        for _, o in taken.iterrows():
            collusion_order_ids.add(o["order_id"])
            running_coll += 1
            o_date = coll_resched_map[o["order_id"]]

            if rng.random() < 0.70:
                ret_delay = int(rng.integers(2, 6))
                scheduled_coll_return = o_date + timedelta(days=ret_delay)
                # STRICT RIGHT CENSORING
                if scheduled_coll_return <= SIM_END:
                    coll_return_rows.append({
                        "return_id": f"RETURN_{return_counter:06d}",
                        "order_id": o["order_id"],
                        "buyer_id": o["buyer_id"],
                        "seller_id": o["seller_id"],
                        "return_date": scheduled_coll_return,
                        "reason": rng.choice(RETURN_REASONS, p=RETURN_REASON_WEIGHTS),
                        "status": "approved",
                        "is_fraudulent": True,
                        "fraud_type": "seller_buyer_collusion",
                    })
                    return_counter += 1

        ring_label = f"RING_COLLUSION_{s_id}"
        ground_truth_ledgers.append({
            "fraud_ring_id": ring_label, "fraud_type": "seller_buyer_collusion",
            "entity_type": "seller", "entity_id": s_id
        })
        for cb in c_buyers:
            ground_truth_ledgers.append({
                "fraud_ring_id": ring_label, "fraud_type": "seller_buyer_collusion",
                "entity_type": "buyer", "entity_id": cb
            })

    if coll_resched_map:
        mask = orders_df["order_id"].isin(coll_resched_map.keys())
        orders_df.loc[mask, "order_date"] = orders_df.loc[mask, "order_id"].map(coll_resched_map)

    if coll_return_rows:
        returns_df = pd.concat([returns_df, pd.DataFrame(coll_return_rows)], ignore_index=True)

    # Consolidate Orders Fraud Flags
    orders_df["is_fraudulent"] = False
    orders_df["fraud_type"] = None

    orders_df.loc[orders_df["order_id"].isin(fake_order_ids), "is_fraudulent"] = True
    orders_df.loc[orders_df["order_id"].isin(fake_order_ids), "fraud_type"] = "fake_listing"

    orders_df.loc[orders_df["order_id"].isin(abuse_order_ids), "is_fraudulent"] = True
    orders_df.loc[orders_df["order_id"].isin(abuse_order_ids), "fraud_type"] = "return_abuse"

    orders_df.loc[orders_df["order_id"].isin(coord_order_ids), "is_fraudulent"] = True
    orders_df.loc[orders_df["order_id"].isin(coord_order_ids), "fraud_type"] = "coordinated_fraud"

    orders_df.loc[orders_df["order_id"].isin(collusion_order_ids), "is_fraudulent"] = True
    orders_df.loc[orders_df["order_id"].isin(collusion_order_ids), "fraud_type"] = "seller_buyer_collusion"

    # Schema Sanitation: Drop internal generator columns from public tables
    public_listings = listings_df.drop(columns=["_seller_persona"], errors="ignore")

    fraud_ground_truth = pd.DataFrame(ground_truth_ledgers)
    perturbation_audit_df = pd.DataFrame(perturbed_evidence_log)

    return {
        "listings": public_listings,
        "orders": orders_df,
        "returns": returns_df,
        "fraud_ground_truth": fraud_ground_truth,
        "perturbation_audit": perturbation_audit_df,
    }


# ---------------------------------------------------------------------------
# Stage 5: Rigorous Automated Validation Suite
# ---------------------------------------------------------------------------

def validate_generated_dataset_v2_1(
    data: Dict[str, pd.DataFrame],
    internal_meta: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Validates data types, foreign keys, chronological ordering, right-censoring,
    duplicate prevention, ID leakage, seller behaviors, and distribution overlap.
    """
    orders = data["orders"].copy()
    listings = data["listings"].copy()
    returns = data["returns"].copy()
    buyers = data["buyers"].copy()
    sellers = data["sellers"].copy()
    products = data["products"].copy()

    orders["order_date"] = pd.to_datetime(orders["order_date"])
    returns["return_date"] = pd.to_datetime(returns["return_date"])
    listings["listing_date"] = pd.to_datetime(listings["listing_date"])
    buyers["signup_date"] = pd.to_datetime(buyers["signup_date"])
    sellers["signup_date"] = pd.to_datetime(sellers["signup_date"])

    val_results: Dict[str, Any] = {"status": "PASS", "checks": {}}

    # 1. Foreign Key Integrity
    fk_errors = {
        "orders_buyer_id_missing": int((~orders["buyer_id"].isin(buyers["buyer_id"])).sum()),
        "orders_seller_id_missing": int((~orders["seller_id"].isin(sellers["seller_id"])).sum()),
        "orders_listing_id_missing": int((~orders["listing_id"].isin(listings["listing_id"])).sum()),
        "orders_product_id_missing": int((~orders["product_id"].isin(products["product_id"])).sum()),
        "returns_order_id_missing": int((~returns["order_id"].isin(orders["order_id"])).sum()),
        "returns_buyer_id_missing": int((~returns["buyer_id"].isin(buyers["buyer_id"])).sum()),
        "returns_seller_id_missing": int((~returns["seller_id"].isin(sellers["seller_id"])).sum()),
        "listings_seller_id_missing": int((~listings["seller_id"].isin(sellers["seller_id"])).sum()),
        "listings_product_id_missing": int((~listings["product_id"].isin(products["product_id"])).sum()),
    }
    val_results["checks"]["foreign_keys"] = {
        "errors": fk_errors,
        "passed": bool(sum(fk_errors.values()) == 0),
    }

    # 2. Null Checks on Required Operational Columns
    null_counts = {
        "orders": int(orders[["order_id", "buyer_id", "seller_id", "listing_id", "product_id", "order_date", "amount"]].isna().sum().sum()),
        "listings": int(listings[["listing_id", "seller_id", "product_id", "category", "listing_date", "price"]].isna().sum().sum()),
        "returns": int(returns[["return_id", "order_id", "buyer_id", "seller_id", "return_date", "reason"]].isna().sum().sum()),
        "buyers": int(buyers[["buyer_id", "signup_date", "address_id"]].isna().sum().sum()),
        "sellers": int(sellers[["seller_id", "signup_date", "address_id", "category_focus"]].isna().sum().sum()),
    }
    val_results["checks"]["null_values"] = {
        "null_counts": null_counts,
        "passed": bool(sum(null_counts.values()) == 0),
    }

    # 3. Uniqueness of Primary Keys
    dup_counts = {
        "orders": int(orders["order_id"].duplicated().sum()),
        "listings": int(listings["listing_id"].duplicated().sum()),
        "returns": int(returns["return_id"].duplicated().sum()),
        "buyers": int(buyers["buyer_id"].duplicated().sum()),
        "sellers": int(sellers["seller_id"].duplicated().sum()),
        "products": int(products["product_id"].duplicated().sum()),
    }
    val_results["checks"]["primary_key_uniqueness"] = {
        "duplicates": dup_counts,
        "passed": bool(sum(dup_counts.values()) == 0),
    }

    # 4. Strict Date Boundaries & Clamping
    sim_start_ts = pd.Timestamp(SIM_START)
    sim_end_ts = pd.Timestamp(SIM_END)

    o_overflow = int((orders["order_date"] > sim_end_ts).sum() + (orders["order_date"] < sim_start_ts).sum())
    r_overflow = int((returns["return_date"] > sim_end_ts).sum() + (returns["return_date"] < sim_start_ts).sum())
    l_overflow = int((listings["listing_date"] > sim_end_ts).sum() + (listings["listing_date"] < sim_start_ts).sum())

    val_results["checks"]["date_boundary_clamping"] = {
        "orders_range": [str(orders["order_date"].min()), str(orders["order_date"].max())],
        "returns_range": [str(returns["return_date"].min()), str(returns["return_date"].max())],
        "listings_range": [str(listings["listing_date"].min()), str(listings["listing_date"].max())],
        "out_of_bounds_count": o_overflow + r_overflow + l_overflow,
        "passed": bool(o_overflow == 0 and r_overflow == 0 and l_overflow == 0),
    }

    # 5. Chronological Ordering & Return Delay Semantics (NO Zero-Day Returns!)
    merged_ol = orders.merge(listings[["listing_id", "listing_date"]], on="listing_id")
    order_before_listing = int((merged_ol["order_date"] < merged_ol["listing_date"]).sum())

    merged_ro = returns.merge(orders[["order_id", "order_date"]], on="order_id")
    delay_days = (merged_ro["return_date"] - merged_ro["order_date"]).dt.days
    return_before_order = int((delay_days < 0).sum())
    zero_day_returns = int((delay_days == 0).sum())

    merged_ob = orders.merge(buyers[["buyer_id", "signup_date"]], on="buyer_id")
    order_before_buyer = int((merged_ob["order_date"] < merged_ob["signup_date"]).sum())

    val_results["checks"]["chronological_invariants"] = {
        "order_before_listing_violations": order_before_listing,
        "return_before_order_violations": return_before_order,
        "zero_day_returns_count": zero_day_returns,
        "min_delay_days": int(delay_days.min()) if len(delay_days) > 0 else None,
        "order_before_buyer_signup_violations": order_before_buyer,
        "passed": bool(order_before_listing == 0 and return_before_order == 0 and zero_day_returns == 0 and order_before_buyer == 0),
    }

    # 6. ID Label Leakage Check
    banned_words = ["FRAUD", "ABUSE", "COLLUSION", "RING"]
    leaking_ids = {}
    for table_name, df, id_col in [
        ("orders", orders, "order_id"),
        ("listings", listings, "listing_id"),
        ("returns", returns, "return_id"),
        ("buyers", buyers, "buyer_id"),
        ("sellers", sellers, "seller_id"),
    ]:
        matches = int(df[id_col].str.contains("|".join(banned_words), case=False).sum())
        leaking_ids[table_name] = matches

    val_results["checks"]["id_leakage"] = {
        "leaking_id_counts": leaking_ids,
        "passed": bool(sum(leaking_ids.values()) == 0),
    }

    # 7. Verification of Observable Seller Behaviors
    s_meta = internal_meta["internal_seller_metadata"]
    disp_sellers = set(s_meta[s_meta["seller_persona"] == "disposable"]["seller_id"])
    ato_sellers = set(s_meta[s_meta["seller_persona"] == "takeover"]["seller_id"])
    est_sellers = set(s_meta[s_meta["seller_persona"] == "established"]["seller_id"])

    # Disposable sellers: Active span (max listing date - signup date) must be short!
    disp_listings = listings[listings["seller_id"].isin(disp_sellers)].merge(
        sellers[["seller_id", "signup_date"]], on="seller_id"
    )
    disp_tenures = (disp_listings["listing_date"] - disp_listings["signup_date"]).dt.days
    disp_median_tenure = float(disp_tenures.median()) if len(disp_tenures) > 0 else 0.0

    est_listings = listings[listings["seller_id"].isin(est_sellers)].merge(
        sellers[["seller_id", "signup_date"]], on="seller_id"
    )
    est_tenures = (est_listings["listing_date"] - est_listings["signup_date"]).dt.days
    est_median_tenure = float(est_tenures.median()) if len(est_tenures) > 0 else 0.0

    val_results["checks"]["observable_seller_behavior"] = {
        "disposable_sellers_count": len(disp_sellers),
        "disposable_median_listing_tenure_days": round(disp_median_tenure, 1),
        "established_median_listing_tenure_days": round(est_median_tenure, 1),
        "takeover_sellers_count": len(ato_sellers),
        "passed": bool(disp_median_tenure <= 7.0 and est_median_tenure >= 50.0),
    }

    # 8. Distribution Overlaps (Ensure non-disjoint realistic distributions)
    legit_delays = delay_days[~merged_ro["is_fraudulent"]].to_numpy()
    fraud_delays = delay_days[merged_ro["is_fraudulent"]].to_numpy()

    ks_stat_delay, _ = stats.ks_2samp(legit_delays, fraud_delays)
    cohen_d_delay = float((np.mean(fraud_delays) - np.mean(legit_delays)) / (np.sqrt((np.var(fraud_delays) + np.var(legit_delays)) / 2) + 1e-9))
    auc_delay = float(roc_auc_score(merged_ro["is_fraudulent"].astype(int), -delay_days))

    val_results["checks"]["return_delay_overlap"] = {
        "legit_mean": round(float(np.mean(legit_delays)), 2),
        "fraud_mean": round(float(np.mean(fraud_delays)), 2),
        "ks_statistic": round(float(ks_stat_delay), 4),
        "cohens_d": round(cohen_d_delay, 4),
        "single_feature_auc": round(auc_delay, 4),
        "passed": bool(auc_delay < 0.70),  # Baseline was 0.8441
    }

    # 9. Category-Constrained Perturbations Verification
    pert_audit = internal_meta.get("perturbation_audit", pd.DataFrame())
    cat_map = products.set_index("product_id")["category"].to_dict()
    if len(pert_audit) > 0:
        claimed_cats = pert_audit["claimed_product_id"].map(cat_map)
        pert_cats = pert_audit["perturbed_product_id"].map(cat_map)
        cross_cat_violations = int((claimed_cats != pert_cats).sum())
        subtype_matches = int(pert_audit["perturbation_type"].str.contains("compatible_subtype").sum())
        variant_matches = len(pert_audit) - subtype_matches
    else:
        cross_cat_violations = 0
        subtype_matches = 0
        variant_matches = 0

    val_results["checks"]["category_constrained_perturbations"] = {
        "total_perturbed_listings": len(pert_audit),
        "cross_category_violations": cross_cat_violations,
        "compatible_subtype_matches": subtype_matches,
        "same_category_variants": variant_matches,
        "passed": bool(cross_cat_violations == 0 and len(pert_audit) > 0),
    }

    # 10. Generator-Label & Metadata Segregation Check
    banned_in_buyers = [c for c in ["buyer_persona", "is_fraudulent", "fraud_type"] if c in buyers.columns]
    banned_in_sellers = [c for c in ["seller_persona", "compromise_date", "is_fraudulent", "fraud_type"] if c in sellers.columns]
    banned_in_listings = [c for c in ["price_anomaly", "image_mismatch", "_seller_persona"] if c in listings.columns]

    val_results["checks"]["generator_metadata_segregation"] = {
        "banned_in_buyers": banned_in_buyers,
        "banned_in_sellers": banned_in_sellers,
        "banned_in_listings": banned_in_listings,
        "passed": bool(len(banned_in_buyers) == 0 and len(banned_in_sellers) == 0 and len(banned_in_listings) == 0),
    }

    # 11. Fraud Prevalence & Subtype Target Verification
    total_orders = len(orders)
    fraud_orders_count = int(orders["is_fraudulent"].sum())
    actual_fraud_rate = float(fraud_orders_count / total_orders)
    subtype_counts = orders[orders["is_fraudulent"]]["fraud_type"].value_counts().to_dict()

    val_results["checks"]["fraud_prevalence_targets"] = {
        "total_orders": total_orders,
        "fraud_orders": fraud_orders_count,
        "actual_fraud_rate": round(actual_fraud_rate, 4),
        "target_fraud_rate": TARGET_FRAUD_RATE,
        "subtype_counts": subtype_counts,
        "passed": bool(abs(actual_fraud_rate - TARGET_FRAUD_RATE) <= 0.015),
    }

    # 12. Return-Date Semantics & End-of-Period Policy (Dec 30 / Dec 31)
    dec30_orders = set(orders[orders["order_date"].dt.date == datetime(2025, 12, 30).date()]["order_id"])
    dec31_orders = set(orders[orders["order_date"].dt.date == datetime(2025, 12, 31).date()]["order_id"])
    dec30_returns = returns[returns["order_id"].isin(dec30_orders)]
    dec31_returns = returns[returns["order_id"].isin(dec31_orders)]

    # Returns for Dec 30 orders must have return_date strictly after order_date (e.g. Dec 31)
    dec30_invalid = int((dec30_returns["return_date"] <= pd.Timestamp(2025, 12, 30)).sum()) if len(dec30_returns) > 0 else 0
    # Returns for Dec 31 orders cannot occur in 2025 (must be 0, all right-censored!)
    dec31_return_count = len(dec31_returns)

    val_results["checks"]["return_date_semantics_december"] = {
        "dec30_order_count": len(dec30_orders),
        "dec30_returns_recorded": len(dec30_returns),
        "dec30_invalid_timing_violations": dec30_invalid,
        "dec31_order_count": len(dec31_orders),
        "dec31_returns_recorded": dec31_return_count,
        "policy": "Right-censoring: returns scheduled after 2025-12-31 are unobserved (excluded) rather than clamped to 0 days.",
        "passed": bool(dec30_invalid == 0 and dec31_return_count == 0),
    }

    # Overall Verdict
    all_passed = all(c.get("passed", False) for c in val_results["checks"].values())
    val_results["status"] = "PASS" if all_passed else "FAIL"

    return val_results


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("TRUSTSHIELD STAGE 3.1.1 — GENERATOR INTEGRITY FIXES & V2.1 EXPORT")
    print("=" * 80)

    t0 = time.time()

    print("\n1. Generating base entities with separate persona tracking...")
    base = generate_base_entities_v2_1(seed=DEFAULT_SEED)

    print("2. Generating catalog and listings with observable disposable & ATO seller behaviors...")
    catalog = generate_catalog_and_listings_v2_1(
        base["sellers"], base["internal_seller_metadata"], seed=DEFAULT_SEED
    )

    print("3. Generating transactions and organic returns with right-censored end-of-period policy...")
    txn = generate_orders_and_returns_v2_1(
        base["buyers"], base["internal_buyer_metadata"],
        catalog["sellers"], catalog["listings"], base["device_mapping"], seed=DEFAULT_SEED
    )
    print(f"  Right-censored organic returns (in 2026): {txn['right_censored_organic_returns']}")

    print("4. Injecting realistic fraud with category-constrained perturbations & clamped bursts...")
    fraud_data = inject_realistic_fraud_v2_1(catalog, txn, base, seed=DEFAULT_SEED)

    # Assemble complete public dataset (100% clean of internal persona/generator tags!)
    full_dataset: Dict[str, pd.DataFrame] = {
        "addresses": base["addresses"],
        "devices": base["devices"],
        "device_mapping": base["device_mapping"],
        "address_sharing_log": base["address_sharing_log"],
        "device_sharing_log": base["device_sharing_log"],
        "buyers": txn["buyers"],
        "sellers": catalog["sellers"],
        "products": catalog["products"],
        "listings": fraud_data["listings"],
        "orders": fraud_data["orders"],
        "returns": fraud_data["returns"],
        "fraud_ground_truth": fraud_data["fraud_ground_truth"],
    }

    internal_meta = {
        "internal_seller_metadata": base["internal_seller_metadata"],
        "internal_buyer_metadata": base["internal_buyer_metadata"],
        "perturbation_audit": fraud_data["perturbation_audit"],
    }

    print("\n5. Running comprehensive Stage 3.1.1 validation suite...")
    val_report = validate_generated_dataset_v2_1(full_dataset, internal_meta)
    print(f"Validation Status: {val_report['status']}")
    for k, v in val_report["checks"].items():
        print(f"  - {k}: {'PASS' if v.get('passed') else 'FAIL'} {v}")

    # Write Public CSVs to isolated directory data/synthetic_v2_1/
    print(f"\n6. Writing output CSVs to isolated directory: {_OUTPUT_DIR}...")
    manifest_entries = {}
    for name, df in full_dataset.items():
        csv_path = os.path.join(_OUTPUT_DIR, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        manifest_entries[f"{name}.csv"] = {
            "row_count": len(df),
            "file_size": os.path.getsize(csv_path),
            "sha256": compute_file_sha256(csv_path),
        }
        print(f"  Exported {name}.csv ({len(df):,} rows, {manifest_entries[f'{name}.csv']['file_size']:,} bytes)")

    # Write separate generator personas metadata
    meta_path = os.path.join(_OUTPUT_DIR, "generator_personas.csv")
    combined_personas = pd.concat([
        base["internal_seller_metadata"].assign(entity_type="seller").rename(columns={"seller_id": "entity_id", "seller_persona": "persona"}),
        base["internal_buyer_metadata"].assign(entity_type="buyer", compromise_date=None).rename(columns={"buyer_id": "entity_id", "buyer_persona": "persona"}),
    ], ignore_index=True)
    combined_personas.to_csv(meta_path, index=False)
    manifest_entries["generator_personas.csv"] = {
        "row_count": len(combined_personas),
        "file_size": os.path.getsize(meta_path),
        "sha256": compute_file_sha256(meta_path),
    }

    # Write perturbation audit log
    pert_path = os.path.join(_OUTPUT_DIR, "perturbation_audit.csv")
    fraud_data["perturbation_audit"].to_csv(pert_path, index=False)
    manifest_entries["perturbation_audit.csv"] = {
        "row_count": len(fraud_data["perturbation_audit"]),
        "file_size": os.path.getsize(pert_path),
        "sha256": compute_file_sha256(pert_path),
    }

    # Write Manifest & Validation Report
    manifest_path = os.path.join(_OUTPUT_DIR, "dataset_manifest.json")
    manifest = {
        "generator": "generate_realistic_synthetic_data_v2_1.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": DEFAULT_SEED,
        "elapsed_seconds": round(time.time() - t0, 2),
        "files": manifest_entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    val_report_path = os.path.join(_OUTPUT_DIR, "validation_report.json")
    with open(val_report_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)

    print(f"\nManifest saved to: {manifest_path}")
    print(f"Validation report saved to: {val_report_path}")
    print(f"\nStage 3.1.1 execution completed successfully in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
