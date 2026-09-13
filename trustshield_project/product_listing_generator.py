"""
TrustShield AI — Synthetic Data Generator
Part 2: Product catalog + Listing generator

Design reference: design.md
- ~20,000 listings across ~500 sellers (target from design.md scale)
- Real product catalog intended: Amazon Berkeley Objects (ABO) images + text
  for realism. This module exposes a clean ABO-loading interface
  (load_abo_catalog) with a synthetic fallback so the pipeline runs
  end-to-end even before the real ABO metadata is downloaded/mounted.
- NO fraud is injected here. This is clean base data only — fake listings,
  price anomalies, image-text mismatches, copied images etc. are injected
  by a separate fraud_injection module downstream (per rules.md: keep
  fraud logic isolated from base generation, and keep fraud_ring_id /
  is_fraudulent out of anything that could leak into model features here).
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import timedelta

from entity_generator import rng, SIM_START, SIM_END, SIM_DAYS, build_base_entities


# ---------------------------------------------------------------------------
# ABO dataset auto-detection
# ---------------------------------------------------------------------------

def _resolve_abo_path():
    """
    Resolves the ABO dataset root directory in priority order:
      1. ABO_DATA_DIR environment variable (absolute or relative to cwd)
      2. data/external/abo  relative to THIS file's directory
      3. data/external/abo  one level UP from this file's directory
      4. ../../trustshield_project/data/external/abo  — handles the actual
         layout in this repo where code lives in:
           trustshield_full_handoff/trustshield_full_handoff/trustshield_project/
         and data lives in:
           trustshield_full_handoff/trustshield_project/data/external/abo/
    Returns the first path that exists and contains listings/metadata/, or
    None so callers fall through to the synthetic fallback gracefully.
    """
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("ABO_DATA_DIR", ""),
        os.path.join(_this_dir, "data", "external", "abo"),
        os.path.join(_this_dir, "..", "data", "external", "abo"),
        os.path.join(_this_dir, "..", "..", "trustshield_project", "data", "external", "abo"),
    ]
    for p in candidates:
        p = p.strip()
        if not p:
            continue
        p = os.path.normpath(p)
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "listings", "metadata")):
            return p
    return None


_DEFAULT_ABO_PATH = _resolve_abo_path()

N_PRODUCTS = 3000
TARGET_TOTAL_LISTINGS = 20000

CATEGORIES = [
    "Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports",
    "Toys", "Books", "Grocery", "Automotive", "Furniture"
]

# Rough per-category base price ranges (median, spread) — synthetic but
# ordered sensibly (Electronics/Furniture pricier than Books/Grocery).
CATEGORY_PRICE_PARAMS = {
    "Electronics":    (2500, 0.9),
    "Fashion":        (800,  0.7),
    "Home & Kitchen": (1200, 0.6),
    "Beauty":         (500,  0.5),
    "Sports":         (1000, 0.6),
    "Toys":           (600,  0.5),
    "Books":          (350,  0.4),
    "Grocery":        (250,  0.3),
    "Automotive":     (1800, 0.7),
    "Furniture":      (4500, 0.8),
}


# ---------------------------------------------------------------------------
# Product catalog
# ---------------------------------------------------------------------------

import gzip
import csv

# Maps ABO's raw product_type strings (ABO's own taxonomy) to TrustShield's
# 10 categories. Confirmed against real downloaded data. Anything unmatched
# falls back to a random TrustShield category (tracked, not silently
# dropped) rather than inventing an "Other" category the rest of the
# pipeline doesn't know about.
_ABO_PRODUCT_TYPE_MAP = {
    "CELLULAR_PHONE_CASE": "Electronics", "HEADPHONES": "Electronics",
    "SPEAKER": "Electronics", "TABLET_COMPUTER": "Electronics",
    "LAPTOP_COMPUTER": "Electronics", "CAMERA": "Electronics",
    "TELEVISION": "Electronics", "COMPUTER_COMPONENT": "Electronics",
    "BATTERY_CHARGER": "Electronics", "CABLE": "Electronics",
    "SHOES": "Fashion", "DRESS": "Fashion", "SHIRT": "Fashion",
    "PANTS": "Fashion", "JEWELRY": "Fashion", "HANDBAG": "Fashion",
    "OUTERWEAR_COAT": "Fashion", "UNDERGARMENT": "Fashion",
    "SUNGLASSES": "Fashion", "WATCH": "Fashion",
    "SOFA": "Home & Kitchen", "KITCHEN": "Home & Kitchen",
    "COOKWARE": "Home & Kitchen", "HOME_BED_AND_BATH": "Home & Kitchen",
    "CURTAIN": "Home & Kitchen", "LAMP": "Home & Kitchen", "RUG": "Home & Kitchen",
    "SKIN_CARE_PRODUCT": "Beauty", "HAIR_CARE_PRODUCT": "Beauty",
    "PERFUME": "Beauty", "MAKEUP": "Beauty",
    "SPORTING_GOODS": "Sports", "EXERCISE_EQUIPMENT": "Sports", "BICYCLE": "Sports",
    "TOY_FIGURE": "Toys", "TOYS_AND_GAMES": "Toys", "GAME": "Toys",
    "ABIS_BOOK": "Books", "PUBLICATION_BINDING": "Books",
    "GROCERY": "Grocery", "FOOD_BEVERAGE": "Grocery",
    "AUTO_ACCESSORY": "Automotive", "AUTO_PART": "Automotive",
    "CHAIR": "Furniture", "TABLE": "Furniture", "FURNITURE": "Furniture", "BED": "Furniture",
}
_PREFERRED_LANGUAGES = ["en_US", "en_GB", "en_CA", "en_AU"]


def _abo_first_localized_value(field):
    """ABO text fields are lists of {language_tag, value}; many listings
    are locale-only (e.g. nl_NL/de_DE with no English entry at all) — fall
    back to whatever language is present rather than dropping the product."""
    if not field:
        return None
    by_lang = {e.get("language_tag"): e.get("value") for e in field if isinstance(e, dict)}
    for lang in _PREFERRED_LANGUAGES:
        if lang in by_lang and by_lang[lang]:
            return by_lang[lang]
    first = field[0]
    return first.get("value") if isinstance(first, dict) else None


def _abo_extract_product_type(raw):
    """Real ABO data stores this as [{"value": <str>}], not a plain <str>
    as the README documents — handle both."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list) and raw:
        first = raw[0]
        return first.get("value") if isinstance(first, dict) else first
    return None


def _abo_load_image_id_to_path(abo_metadata_path):
    """images/metadata/images.csv.gz maps image_id -> path (e.g.
    "14/14fe8812.jpg", relative to images/small/) — main_image_id does
    NOT match the filename directly, this lookup is required."""
    csv_path = os.path.join(abo_metadata_path, "images", "metadata", "images.csv.gz")
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found — real images will not be linked.")
        return {}
    mapping = {}
    with gzip.open(csv_path, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["image_id"]] = row["path"]
    return mapping


def load_abo_catalog(abo_metadata_path):
    """
    Real-data path. abo_metadata_path should point at the ABO root
    directory containing:
        listings/metadata/listings_0.json.gz ... listings_f.json.gz
        images/metadata/images.csv.gz
        images/small/<xx>/<hash>.jpg

    Returns a DataFrame in the same schema as
    generate_synthetic_product_catalog() (product_id, category, title,
    description, base_price, image_ref, source), or None if the path
    doesn't exist / has no usable records, so callers fall back to the
    synthetic catalog without crashing the pipeline.

    Note: ABO has no price field at all — base_price is generated
    synthetically per mapped category (same distributions as the fully-
    synthetic fallback) even when title/description/image are real.
    """
    if not abo_metadata_path or not os.path.isdir(abo_metadata_path):
        return None

    listings_dir = os.path.join(abo_metadata_path, "listings", "metadata")
    if not os.path.isdir(listings_dir):
        return None

    image_id_to_path = _abo_load_image_id_to_path(abo_metadata_path)

    rows = []
    n_missing_image = 0
    n_unmapped_type = 0
    n_no_english_text = 0

    shard_files = sorted(
        f for f in os.listdir(listings_dir)
        if f.startswith("listings_") and f.endswith(".json.gz")
    )
    for shard_file in shard_files:
        with gzip.open(os.path.join(listings_dir, shard_file), "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                item_id = record.get("item_id")
                if not item_id:
                    continue

                title = _abo_first_localized_value(record.get("item_name"))
                if title is None:
                    continue  # no usable name in any language — skip, don't fabricate

                name_langs = {e.get("language_tag") for e in (record.get("item_name") or []) if isinstance(e, dict)}
                if not (name_langs & set(_PREFERRED_LANGUAGES)):
                    n_no_english_text += 1

                description = _abo_first_localized_value(record.get("product_description")) or ""

                raw_type = _abo_extract_product_type(record.get("product_type"))
                category = _ABO_PRODUCT_TYPE_MAP.get(raw_type) if raw_type else None
                if category is None:
                    category = rng.choice(CATEGORIES)
                    n_unmapped_type += 1

                main_image_id = record.get("main_image_id")
                image_rel_path = image_id_to_path.get(main_image_id) if main_image_id else None
                if image_rel_path is None:
                    n_missing_image += 1
                    image_ref = f"PLACEHOLDER_IMG_{item_id}.jpg"
                else:
                    image_ref = os.path.join("images", "small", image_rel_path)

                rows.append({
                    "product_id": item_id,
                    "category": category,
                    "title": title,
                    "description": description,
                    "image_ref": image_ref,
                    "source": "abo_real",
                })

                if len(rows) >= N_PRODUCTS:
                    break
        if len(rows) >= N_PRODUCTS:
            break

    if not rows:
        print("  WARNING: ABO path existed but no usable listing records were parsed.")
        return None

    df = pd.DataFrame(rows)
    df["base_price"] = [
        round(max(50, rng.lognormal(
            mean=np.log(CATEGORY_PRICE_PARAMS[c][0]), sigma=CATEGORY_PRICE_PARAMS[c][1]
        )), 2)
        for c in df["category"]
    ]

    print(f"  Loaded {len(df)} real ABO products "
          f"({n_unmapped_type} unmapped product_type -> random category, "
          f"{n_missing_image} missing image -> placeholder, "
          f"{n_no_english_text} no English title -> used first available language).")

    return df[["product_id", "category", "title", "description", "base_price", "image_ref", "source"]]


def generate_synthetic_product_catalog(n_products):
    """
    Fallback catalog: same schema real ABO-backed products would have
    (product_id, category, title, description, base_price, image_ref),
    so downstream code (listings, fraud injection, multimodal features)
    doesn't need to change when real ABO data is swapped in later.
    """
    categories = rng.choice(CATEGORIES, size=n_products)
    base_prices = np.array([
        max(50, rng.lognormal(
            mean=np.log(CATEGORY_PRICE_PARAMS[c][0]),
            sigma=CATEGORY_PRICE_PARAMS[c][1]
        ))
        for c in categories
    ]).round(2)

    df = pd.DataFrame({
        "product_id": [f"PROD_{i:06d}" for i in range(n_products)],
        "category": categories,
        "title": [f"{c} Item #{i:05d}" for i, c in enumerate(categories)],
        "description": [f"Synthetic placeholder description for a {c} product." for c in categories],
        "base_price": base_prices,
        "image_ref": [f"PLACEHOLDER_IMG_{i:06d}.jpg" for i in range(n_products)],
        "source": "synthetic",
    })
    return df


def generate_product_catalog(n_products=N_PRODUCTS, abo_metadata_path=None):
    # If caller did not supply an explicit path, fall back to auto-detected
    # _DEFAULT_ABO_PATH (set at import time via _resolve_abo_path()).
    resolved_path = abo_metadata_path if abo_metadata_path is not None else _DEFAULT_ABO_PATH
    real_catalog = load_abo_catalog(resolved_path)
    if real_catalog is not None:
        return real_catalog
    if resolved_path:
        print(f"  WARNING: ABO path '{resolved_path}' found but produced no usable records "
              "— using synthetic placeholder catalog.")
    else:
        print("  (no ABO metadata found — using synthetic placeholder catalog)")
    return generate_synthetic_product_catalog(n_products)


# ---------------------------------------------------------------------------
# Listing counts per seller
# ---------------------------------------------------------------------------

def assign_listing_counts(sellers_df, total_listings=TARGET_TOTAL_LISTINGS, skew_sigma=0.8):
    """
    Most sellers list a modest number of items; a smaller set of "power
    sellers" list a lot more — modeled with a lognormal weight distribution
    rather than a uniform split, then scaled to hit total_listings exactly.
    """
    n_sellers = len(sellers_df)
    weights = rng.lognormal(mean=0.0, sigma=skew_sigma, size=n_sellers)
    weights = weights / weights.sum()
    counts = np.maximum(1, (weights * total_listings).round().astype(int))

    # Reconcile rounding drift so the total matches exactly.
    diff = total_listings - counts.sum()
    if diff != 0:
        idx = rng.choice(n_sellers, size=abs(diff), replace=True)
        counts[idx] += np.sign(diff)
        counts = np.maximum(1, counts)

    return counts


# ---------------------------------------------------------------------------
# Listing dates (per seller, starting after their signup)
# ---------------------------------------------------------------------------

def generate_listing_dates_for_seller(signup_date, n, sim_end=SIM_END):
    """
    A seller typically lists an initial batch soon after joining, then adds
    more sporadically over time — a concave curve from signup_date to
    sim_end, mirroring the onboarding shape used for signups.
    """
    window_days = max(1, (sim_end - signup_date).days)
    u = rng.random(n)
    offsets = (u ** 1.4) * window_days
    return [signup_date + timedelta(days=int(d)) for d in offsets]


# ---------------------------------------------------------------------------
# Listings
# ---------------------------------------------------------------------------

def generate_listings(sellers_df, products_df, total_listings=TARGET_TOTAL_LISTINGS,
                       category_match_rate=0.80, price_variance_sigma=0.15):
    """
    Each listing belongs to one seller and is based_on one catalog product.
    category_match_rate of listings are pulled from the seller's own
    category_focus (realistic specialization); the rest are cross-category
    (marketplace noise — sellers occasionally list outside their niche).

    Listing price = product base_price * lognormal noise, so identical
    products get realistic price variation across sellers (this is also
    the natural hook point for price-anomaly fraud injection later).
    """
    counts = assign_listing_counts(sellers_df, total_listings)
    products_by_category = {c: products_df[products_df["category"] == c] for c in CATEGORIES}

    rows = []
    listing_counter = 0
    for (_, seller), n_listings in zip(sellers_df.iterrows(), counts):
        dates = generate_listing_dates_for_seller(seller["signup_date"], n_listings)
        match_mask = rng.random(n_listings) < category_match_rate

        for i in range(n_listings):
            category = seller["category_focus"] if match_mask[i] else rng.choice(CATEGORIES)
            pool = products_by_category[category]
            product = pool.iloc[rng.integers(0, len(pool))]

            price = round(product["base_price"] * rng.lognormal(mean=0.0, sigma=price_variance_sigma), 2)

            rows.append({
                "listing_id": f"LISTING_{listing_counter:06d}",
                "seller_id": seller["seller_id"],
                "product_id": product["product_id"],
                "category": category,
                "listing_date": dates[i],
                "price": price,
                "status": "active",
            })
            listing_counter += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_catalog_and_listings(sellers_df, abo_metadata_path=None):
    # abo_metadata_path=None → generate_product_catalog will use _DEFAULT_ABO_PATH
    resolved_path = abo_metadata_path if abo_metadata_path is not None else _DEFAULT_ABO_PATH
    if resolved_path:
        print(f"Generating product catalog (ABO data: {resolved_path})...")
    else:
        print("Generating product catalog (synthetic fallback — ABO data not found)...")
    products_df = generate_product_catalog(abo_metadata_path=resolved_path)

    print("Generating listings...")
    listings_df = generate_listings(sellers_df, products_df)

    # Backfill seller.total_listings (kept in sync, not recomputed ad hoc downstream)
    counts = listings_df.groupby("seller_id").size()
    sellers_df = sellers_df.copy()
    sellers_df["total_listings"] = sellers_df["seller_id"].map(counts).fillna(0).astype(int)

    return {
        "products": products_df,
        "listings": listings_df,
        "sellers": sellers_df,  # updated total_listings
    }


if __name__ == "__main__":
    if _DEFAULT_ABO_PATH:
        print(f"[ABO auto-detected] Using real ABO dataset at: {_DEFAULT_ABO_PATH}")
    else:
        print("[ABO not found] Set ABO_DATA_DIR env var or place data at "
              "data/external/abo/ — falling back to synthetic catalog.")

    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])

    print("\n--- Generation Summary ---")
    for name in ["products", "listings"]:
        df = catalog[name]
        print(f"✅ {name.capitalize()} generated: {len(df):,} rows")
        if "source" in df.columns:
            counts = df['source'].value_counts().to_dict()
            counts_str = ", ".join(f"{k}: {v:,}" for k, v in counts.items())
            print(f"   Source mix: {counts_str}")

    print("\n✅ Listings distribution:")
    top_cats = catalog["listings"]["category"].value_counts().head(5).to_dict()
    top_cats_str = ", ".join(f"{k}: {v:,}" for k, v in top_cats.items())
    print(f"   Top 5 categories: {top_cats_str}")

    desc = catalog["listings"].groupby("seller_id").size().describe()
    print(f"   Listings per seller: mean={desc['mean']:.1f}, min={desc['min']:.0f}, max={desc['max']:.0f}")
