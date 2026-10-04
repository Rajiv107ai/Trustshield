"""Product catalog and seller listing generator with Amazon Berkeley Objects (ABO) support."""

import os
import json
import gzip
import csv
import numpy as np
import pandas as pd
from datetime import timedelta

from entity_generator import rng, SIM_START, SIM_END, SIM_DAYS, build_base_entities


def _resolve_abo_path():
    """Locates ABO dataset directory from env or local project hierarchy."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("ABO_DATA_DIR", ""),
        os.path.join(this_dir, "data", "external", "abo"),
        os.path.join(this_dir, "..", "data", "external", "abo"),
        os.path.join(this_dir, "..", "..", "trustshield_project", "data", "external", "abo"),
    ]
    for p in candidates:
        p = p.strip()
        if p and os.path.isdir(p) and os.path.isdir(os.path.join(p, "listings", "metadata")):
            return os.path.normpath(p)
    return None


_DEFAULT_ABO_PATH = _resolve_abo_path()
N_PRODUCTS = 3000
TARGET_TOTAL_LISTINGS = 20000

CATEGORIES = [
    "Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports",
    "Toys", "Books", "Grocery", "Automotive", "Furniture"
]

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
    if not field:
        return None
    by_lang = {e.get("language_tag"): e.get("value") for e in field if isinstance(e, dict)}
    for lang in _PREFERRED_LANGUAGES:
        if lang in by_lang and by_lang[lang]:
            return by_lang[lang]
    first = field[0]
    return first.get("value") if isinstance(first, dict) else None


def _abo_extract_product_type(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list) and raw:
        first = raw[0]
        return first.get("value") if isinstance(first, dict) else first
    return None


def _abo_load_image_id_to_path(abo_metadata_path):
    csv_path = os.path.join(abo_metadata_path, "images", "metadata", "images.csv.gz")
    if not os.path.exists(csv_path):
        return {}
    mapping = {}
    with gzip.open(csv_path, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["image_id"]] = row["path"]
    return mapping


def load_abo_catalog(abo_metadata_path, rng: np.random.Generator | None = None):
    """Loads product records from local Amazon Berkeley Objects (ABO) files."""
    if not abo_metadata_path or not os.path.isdir(abo_metadata_path):
        return None

    gen = rng if rng is not None else globals()["rng"]
    listings_dir = os.path.join(abo_metadata_path, "listings", "metadata")
    if not os.path.isdir(listings_dir):
        return None

    image_id_to_path = _abo_load_image_id_to_path(abo_metadata_path)
    rows = []
    shard_files = sorted(f for f in os.listdir(listings_dir) if f.startswith("listings_") and f.endswith(".json.gz"))

    for shard_file in shard_files:
        with gzip.open(os.path.join(listings_dir, shard_file), "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                item_id = record.get("item_id")
                if not item_id:
                    continue

                title = _abo_first_localized_value(record.get("item_name"))
                if title is None:
                    continue

                description = _abo_first_localized_value(record.get("product_description")) or ""
                raw_type = _abo_extract_product_type(record.get("product_type"))
                category = (_ABO_PRODUCT_TYPE_MAP.get(raw_type) if raw_type is not None else None) or gen.choice(CATEGORIES)

                main_image_id = record.get("main_image_id")
                image_rel_path = image_id_to_path.get(main_image_id) if main_image_id else None
                image_ref = os.path.join("images", "small", image_rel_path) if image_rel_path else f"PLACEHOLDER_IMG_{item_id}.jpg"

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
        return None

    df = pd.DataFrame(rows)
    df["base_price"] = [
        round(max(50, gen.lognormal(mean=np.log(CATEGORY_PRICE_PARAMS[c][0]), sigma=CATEGORY_PRICE_PARAMS[c][1])), 2)
        for c in df["category"]
    ]
    return pd.DataFrame(df[["product_id", "category", "title", "description", "base_price", "image_ref", "source"]])


def generate_synthetic_product_catalog(n_products: int, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generates synthetic product catalog when real ABO data is absent."""
    gen = rng if rng is not None else globals()["rng"]
    categories = gen.choice(CATEGORIES, size=n_products)
    base_prices = np.array([
        max(50, gen.lognormal(mean=np.log(CATEGORY_PRICE_PARAMS[c][0]), sigma=CATEGORY_PRICE_PARAMS[c][1]))
        for c in categories
    ]).round(2)

    return pd.DataFrame({
        "product_id": [f"PROD_{i:06d}" for i in range(n_products)],
        "category": categories,
        "title": [f"{c} Item #{i:05d}" for i, c in enumerate(categories)],
        "description": [f"Synthetic description for a {c} product." for c in categories],
        "base_price": base_prices,
        "image_ref": [f"PLACEHOLDER_IMG_{i:06d}.jpg" for i in range(n_products)],
        "source": "synthetic",
    })


def generate_product_catalog(n_products: int = N_PRODUCTS, abo_metadata_path: str | None = None,
                             rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Resolves catalog source (real ABO or synthetic fallback)."""
    resolved_path = abo_metadata_path if abo_metadata_path is not None else _DEFAULT_ABO_PATH
    real_catalog = load_abo_catalog(resolved_path, rng=rng)
    if real_catalog is not None:
        return real_catalog
    return generate_synthetic_product_catalog(n_products, rng=rng)


def assign_listing_counts(sellers_df: pd.DataFrame, total_listings: int = TARGET_TOTAL_LISTINGS,
                          skew_sigma: float = 0.8, rng: np.random.Generator | None = None) -> np.ndarray:
    """Allocates listings across sellers using a lognormal activity curve."""
    gen = rng if rng is not None else globals()["rng"]
    n_sellers = len(sellers_df)
    weights = gen.lognormal(mean=0.0, sigma=skew_sigma, size=n_sellers)
    weights /= weights.sum()
    counts = np.maximum(1, (weights * total_listings).round().astype(int))

    diff = total_listings - counts.sum()
    if diff != 0:
        idx = gen.choice(n_sellers, size=abs(diff), replace=True)
        counts[idx] += np.sign(diff)
        counts = np.maximum(1, counts)
    return counts


def generate_listing_dates_for_seller(signup_date, n: int, sim_end=SIM_END,
                                      rng: np.random.Generator | None = None) -> list:
    """Staggers listing creation dates following seller signup."""
    gen = rng if rng is not None else globals()["rng"]
    window_days = max(1, (sim_end - signup_date).days)
    offsets = (gen.random(n) ** 1.4) * window_days
    return [signup_date + timedelta(days=int(d)) for d in offsets]


def generate_listings(sellers_df: pd.DataFrame, products_df: pd.DataFrame, 
                      total_listings: int = TARGET_TOTAL_LISTINGS,
                      category_match_rate: float = 0.80, 
                      price_variance_sigma: float = 0.15,
                      rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generates seller listings with realistic price variance against base catalog price."""
    gen = rng if rng is not None else globals()["rng"]
    counts = assign_listing_counts(sellers_df, total_listings, rng=gen)
    products_by_category = {
        c: pd.DataFrame(products_df[products_df["category"] == c]).to_dict(orient="records")
        for c in CATEGORIES
    }

    rows = []
    listing_counter = 0
    for (_, seller), n_listings in zip(sellers_df.iterrows(), counts):
        dates = generate_listing_dates_for_seller(seller["signup_date"], n_listings, rng=gen)
        match_mask = gen.random(n_listings) < category_match_rate

        for i in range(n_listings):
            category = str(seller["category_focus"]) if match_mask[i] else str(gen.choice(CATEGORIES))
            pool = products_by_category[category]
            product = pool[int(gen.integers(0, len(pool)))]
            price = round(float(product["base_price"]) * gen.lognormal(mean=0.0, sigma=price_variance_sigma), 2)

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


def build_catalog_and_listings(sellers_df: pd.DataFrame, abo_metadata_path: str | None = None,
                               rng: np.random.Generator | None = None) -> dict:
    """Builds product catalog and populates listings."""
    gen = rng if rng is not None else globals()["rng"]
    products_df = generate_product_catalog(abo_metadata_path=abo_metadata_path, rng=gen)
    listings_df = generate_listings(sellers_df, products_df, rng=gen)

    counts = listings_df.groupby("seller_id").size().to_dict()
    sellers_updated = sellers_df.copy()
    sellers_updated["total_listings"] = sellers_updated["seller_id"].map(lambda x: counts.get(x, 0)).astype(int)

    return {
        "products": products_df,
        "listings": listings_df,
        "sellers": sellers_updated,
        "rng": gen,
    }


if __name__ == "__main__":
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    print(f"Products: {len(catalog['products']):,} | Listings: {len(catalog['listings']):,}")
