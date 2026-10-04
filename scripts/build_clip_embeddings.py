"""Offline CLIP embedding pre-computation script -- Phase 4.

Run this script once after installing the project to pre-compute and cache
OpenAI CLIP embeddings for the full ABO product catalog (~3 000 items).

Usage
-----
    python scripts/build_clip_embeddings.py
    python scripts/build_clip_embeddings.py --n-products 50   # quick smoke test
    python scripts/build_clip_embeddings.py --force           # force re-encode

The script:
  1. Resolves the ABO dataset from disk (env var ABO_DATA_DIR or auto-detected).
  2. Loads the full product catalog via product_listing_generator.
  3. Encodes all product texts and real ABO images with CLIP
     (openai/clip-vit-base-patch32).
  4. Writes the compressed embedding cache to --cache-dir (default:
     models/clip_cache/).

On CPU this takes ~3-8 minutes for 3 000 products. Subsequent calls to
compute_multimodal_similarity(..., cache_dir=...) or
get_clip_scorer(products_df, cache_dir=...) load the cache in seconds.

Embedding cache layout
----------------------
    models/clip_cache/
        clip_meta.npz      -- {fingerprint, version, product_id keys/values}
        clip_text_emb.npy  -- float32 (N, 512) CLIP text embeddings
        clip_image_emb.npy -- float32 (N, 512) CLIP image embeddings

The fingerprint is a 16-char SHA-256 prefix of the sorted product ID list.
If the catalog changes, delete the cache directory and re-run this script.
"""

import argparse
import logging
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..", "trustshield_project")
sys.path.insert(0, PROJECT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_clip_embeddings")


def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-compute CLIP embeddings for the ABO product catalog."
    )
    p.add_argument(
        "--cache-dir",
        default=os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models", "clip_cache")),
        help="Directory to write embedding cache. (default: models/clip_cache)",
    )
    p.add_argument("--n-products", type=int, default=None,
                   help="Limit to first N products -- useful for quick smoke tests.")
    p.add_argument("--batch-size", type=int, default=32,
                   help="CLIP encoding batch size per GPU/CPU pass. (default: 32)")
    p.add_argument("--device", default=None,
                   help="Torch device override, e.g. 'cpu' or 'cuda:0'. Auto-detected if omitted.")
    p.add_argument("--force", action="store_true",
                   help="Force re-computation even if a valid cache already exists.")
    return p.parse_args()


def _run_sanity_check(products_df, id_to_idx, text_emb, image_emb):
    """Verify that same-product cosine > cross-product cosine (signal sanity)."""
    import numpy as np
    import pandas as pd
    from multimodal_scoring import CLIPMultimodalScorer

    n_check = min(20, len(products_df))
    sample = products_df.head(n_check).copy()

    sc = CLIPMultimodalScorer.__new__(CLIPMultimodalScorer)
    sc.id_to_idx = id_to_idx
    sc.text_embeddings = text_emb
    sc.image_embeddings = image_emb
    sc._fitted = True

    genuine = sample[["product_id"]].copy()
    genuine["displayed_product_id"] = genuine["product_id"]

    fake = sample[["product_id"]].copy()
    fake["displayed_product_id"] = (
        sample["product_id"].iloc[1:].tolist() + [sample["product_id"].iloc[0]]
    )

    sims_genuine = sc.score_listings(genuine)
    sims_fake = sc.score_listings(fake)
    genuine_mean = float(sims_genuine.mean())
    fake_mean = float(sims_fake.mean())
    ok = genuine_mean > fake_mean

    logger.info(
        "Sanity check (%d products) -- genuine mean: %.3f | fake mean: %.3f | separation: %s",
        n_check, genuine_mean, fake_mean, "PASS" if ok else "FAIL -- check ABO image paths",
    )
    if not ok:
        logger.warning(
            "Sanity check FAILED. Most products may have fallen back to text-perturbed "
            "image embeddings (ABO images not found). "
            "Check ABO_DATA_DIR or trustshield_project/data/external/abo/."
        )


def main():
    args = parse_args()
    cache_dir = os.path.normpath(args.cache_dir)

    try:
        import transformers
        from PIL import Image
        import torch
        logger.info("Dependencies OK -- torch %s, transformers %s",
                    torch.__version__, transformers.__version__)
    except ImportError as exc:
        logger.error("Missing dependency: %s\nInstall: pip install transformers>=4.37 Pillow>=10.0 torch>=2.0", exc)
        sys.exit(1)

    from product_listing_generator import generate_product_catalog
    from multimodal_scoring import CLIPMultimodalScorer, CLIPEmbeddingCache, _product_fingerprint

    logger.info("Loading ABO product catalog...")
    products_df = generate_product_catalog()
    if args.n_products is not None:
        products_df = products_df.head(args.n_products)
        logger.info("Limited to first %d products (smoke-test mode).", args.n_products)

    n_products = len(products_df)
    real_imgs = (~products_df["image_ref"].str.startswith("PLACEHOLDER")).sum()
    logger.info("Catalog: %d products (%d with real ABO images, %d placeholders).",
                n_products, real_imgs, n_products - real_imgs)

    product_ids = list(products_df["product_id"].astype(str))
    fingerprint = _product_fingerprint(product_ids)
    logger.info("Catalog fingerprint: %s", fingerprint)

    if not args.force:
        cache = CLIPEmbeddingCache(cache_dir)
        if cache.exists():
            cached = cache.load(fingerprint)
            if cached is not None:
                _id_to_idx, text_emb, image_emb = cached
                logger.info("Valid cache already exists at '%s' (%d products). Pass --force to recompute.",
                            cache_dir, len(_id_to_idx))
                _run_sanity_check(products_df, _id_to_idx, text_emb, image_emb)
                return

    logger.info("Starting CLIP encoding (%d products) -> cache_dir='%s'...", n_products, cache_dir)
    t0 = time.perf_counter()

    scorer = CLIPMultimodalScorer(batch_size=args.batch_size, device=args.device, cache_dir=cache_dir)
    scorer.fit(products_df)

    elapsed = time.perf_counter() - t0
    logger.info("Encoding complete in %.1f s | text_emb: %s | image_emb: %s",
                elapsed, scorer.text_embeddings.shape, scorer.image_embeddings.shape)
    logger.info("Cache written to '%s'.", cache_dir)

    _run_sanity_check(products_df, scorer.id_to_idx, scorer.text_embeddings, scorer.image_embeddings)


if __name__ == "__main__":
    main()
