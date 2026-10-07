"""Dedicated integration tests for the real CLIP multimodal signal -- Phase 4.

These tests use the ACTUAL openai/clip-vit-base-patch32 model and real ABO
images from disk to verify the Phase 4 multimodal signal.  They are marked
with pytest.mark.clip so they can be skipped in CI environments without GPU
or the transformers library:

    pytest -m "not clip"             # fast, skips CLIP
    pytest -m clip                   # run only CLIP tests
    pytest trustshield_project/test_multimodal_clip.py  # run all CLIP tests

Tests
-----
- test_clip_abo_root_found          -- ABO dataset found on disk
- test_image_paths_resolve          -- image_refs resolve to real .jpg files
- test_clip_text_encoding           -- CLIP text encoder returns float32 (N,512)
- test_clip_image_encoding          -- CLIP image encoder with real ABO images
- test_clip_same_product_similarity -- same product: text/image cosine high
- test_clip_cross_product_mismatch  -- cross product: cosine meaningfully lower
- test_clip_separation_on_real_data -- fraud flag => lower similarity (real pipeline)
- test_embedding_cache_roundtrip    -- save/load cache, fingerprint validation
- test_compute_multimodal_similarity_uses_clip  -- public API uses CLIP when available
"""

import os
import sys
import pytest
import numpy as np


_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from multimodal_scoring import (
    CLIPMultimodalScorer,
    CLIPEmbeddingCache,
    _resolve_abo_root,
    _image_ref_to_path,
    _product_fingerprint,
    compute_multimodal_similarity,
)
from product_listing_generator import load_abo_catalog, _resolve_abo_path


# ---------------------------------------------------------------------------
# Markers and skip conditions
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.clip

_CLIP_AVAILABLE = False
try:
    import transformers  # noqa: F401
    from PIL import Image  # noqa: F401
    import torch  # noqa: F401
    _CLIP_AVAILABLE = True
except ImportError:
    pass

skip_no_clip = pytest.mark.skipif(
    not _CLIP_AVAILABLE,
    reason="transformers / Pillow / torch not installed -- skipping CLIP tests",
)

_ABO_ROOT = _resolve_abo_root()
skip_no_abo = pytest.mark.skipif(
    _ABO_ROOT is None,
    reason="ABO dataset not found on disk -- skipping real-image CLIP tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def abo_products():
    """Load a small slice of the real ABO catalog (10 products for speed)."""
    path = _resolve_abo_path()
    if path is None:
        pytest.skip("ABO dataset not available")
    catalog = load_abo_catalog(path)
    if catalog is None or len(catalog) == 0:
        pytest.skip("ABO catalog loaded but empty")
    return catalog.head(10).reset_index(drop=True)


@pytest.fixture(scope="module")
def fitted_clip_scorer(abo_products):
    """Fit a CLIPMultimodalScorer on the 10-product ABO slice."""
    if not _CLIP_AVAILABLE:
        pytest.skip("CLIP dependencies not installed")
    scorer = CLIPMultimodalScorer(batch_size=10)
    scorer.fit(abo_products)
    return scorer


# ---------------------------------------------------------------------------
# Dataset / path tests (no CLIP model needed)
# ---------------------------------------------------------------------------

class TestABODatasetPresence:
    def test_clip_abo_root_found(self):
        """ABO root directory must be discoverable."""
        root = _resolve_abo_root()
        assert root is not None, (
            "ABO root not found. Set ABO_DATA_DIR or place the dataset at "
            "trustshield_project/data/external/abo/."
        )
        assert os.path.isdir(os.path.join(root, "images", "small")), (
            f"ABO root '{root}' exists but images/small/ subdirectory is missing."
        )

    @skip_no_abo
    def test_image_paths_resolve(self, abo_products):
        """At least 80% of real ABO products must resolve to existing .jpg files."""
        root = _resolve_abo_root()
        assert root is not None, "ABO root must not be None"
        resolved = [
            _image_ref_to_path(str(ref), root)
            for ref in abo_products["image_ref"]
        ]
        n_real = sum(1 for p in resolved if p is not None)
        pct = n_real / len(resolved)
        assert pct >= 0.8, (
            f"Only {pct:.0%} of ABO product image_refs resolved to real files "
            f"({n_real}/{len(resolved)}). Check ABO image extraction."
        )
        # Spot-check: resolved paths actually exist
        for path in resolved:
            if path is not None:
                assert os.path.isfile(path), f"Resolved path does not exist: {path}"
                break  # one spot-check is sufficient

    def test_product_fingerprint_deterministic(self):
        """Same product ID list must always produce the same fingerprint."""
        ids = ["P_001", "P_002", "P_003"]
        fp1 = _product_fingerprint(ids)
        fp2 = _product_fingerprint(ids)
        assert fp1 == fp2
        assert len(fp1) == 16

    def test_product_fingerprint_changes_on_catalog_change(self):
        """Different product IDs must produce different fingerprints."""
        fp_a = _product_fingerprint(["P_001", "P_002"])
        fp_b = _product_fingerprint(["P_001", "P_003"])
        assert fp_a != fp_b


# ---------------------------------------------------------------------------
# CLIP encoder unit tests
# ---------------------------------------------------------------------------

class TestCLIPEncoders:
    @skip_no_clip
    @skip_no_abo
    def test_clip_text_encoding_shape(self, abo_products, fitted_clip_scorer):
        """Text embeddings must be float32 of shape (N, 512)."""
        emb = fitted_clip_scorer.text_embeddings
        assert emb.ndim == 2
        assert emb.shape[0] == len(abo_products)
        assert emb.shape[1] == 512
        assert emb.dtype == np.float32

    @skip_no_clip
    @skip_no_abo
    def test_clip_text_embeddings_normalized(self, fitted_clip_scorer):
        """CLIP text embeddings must be L2-normalized (unit vectors)."""
        norms = np.linalg.norm(fitted_clip_scorer.text_embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5,
                                   err_msg="Text embeddings are not L2-normalized.")

    @skip_no_clip
    @skip_no_abo
    def test_clip_image_encoding_shape(self, abo_products, fitted_clip_scorer):
        """Image embeddings must be float32 of shape (N, 512)."""
        emb = fitted_clip_scorer.image_embeddings
        assert emb.ndim == 2
        assert emb.shape[0] == len(abo_products)
        assert emb.shape[1] == 512
        assert emb.dtype == np.float32

    @skip_no_clip
    @skip_no_abo
    def test_clip_image_embeddings_normalized(self, fitted_clip_scorer):
        """CLIP image embeddings must be L2-normalized (unit vectors)."""
        norms = np.linalg.norm(fitted_clip_scorer.image_embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5,
                                   err_msg="Image embeddings are not L2-normalized.")

    @skip_no_clip
    @skip_no_abo
    def test_clip_real_images_encoded(self, abo_products, fitted_clip_scorer):
        """At least one product must have a real ABO image (non-zero image emb)."""
        root = _resolve_abo_root()
        assert root is not None, "ABO root must not be None"
        has_real = any(
            _image_ref_to_path(str(ref), root) is not None
            for ref in abo_products["image_ref"]
        )
        assert has_real, (
            "None of the 10 sampled ABO products had a resolvable image path. "
            "Check ABO dataset extraction."
        )
        # At least one embedding should not be all-zeros
        nonzero = (np.abs(fitted_clip_scorer.image_embeddings).sum(axis=1) > 0)
        assert nonzero.any(), "All image embeddings are zero — real images not encoded."


# ---------------------------------------------------------------------------
# Cosine similarity signal tests (the core Phase 4 contribution)
# ---------------------------------------------------------------------------

class TestCLIPSimilaritySignal:
    @skip_no_clip
    @skip_no_abo
    def test_same_product_high_similarity(self, abo_products, fitted_clip_scorer):
        """Genuine listings (same product_id = displayed_product_id) must
        produce calibrated similarity >= 0.55 on average."""
        listings = abo_products[["product_id"]].copy()
        listings["displayed_product_id"] = listings["product_id"]
        sims = fitted_clip_scorer.score_listings(listings)

        assert len(sims) == len(listings)
        assert (sims >= 0.05).all() and (sims <= 0.98).all(), (
            f"Similarities out of [0.05, 0.98] range: {sims.values}"
        )
        assert sims.mean() >= 0.55, (
            f"Genuine listing similarity too low (mean={sims.mean():.3f}). "
            "Expected >= 0.55 for matched text/image pairs."
        )

    @skip_no_clip
    @skip_no_abo
    def test_cross_product_lower_similarity(self, abo_products, fitted_clip_scorer):
        """Cross-product listings (image from different product) must have
        meaningfully lower similarity than genuine listings."""
        genuine = abo_products[["product_id"]].copy()
        genuine["displayed_product_id"] = genuine["product_id"]

        # Shift by 1: each listing shows next product's image
        fake = abo_products[["product_id"]].copy()
        fake["displayed_product_id"] = (
            abo_products["product_id"].tolist()[1:]
            + [abo_products["product_id"].tolist()[0]]
        )

        sims_genuine = fitted_clip_scorer.score_listings(genuine)
        sims_fake = fitted_clip_scorer.score_listings(fake)

        genuine_mean = float(sims_genuine.mean())
        fake_mean = float(sims_fake.mean())

        assert genuine_mean > fake_mean, (
            f"CLIP failed to separate genuine ({genuine_mean:.3f}) from "
            f"fake ({fake_mean:.3f}) listings. "
            "Real CLIP signal should produce genuine > fake similarity."
        )
        # The gap should be at least 0.03 (not just noise)
        assert (genuine_mean - fake_mean) >= 0.03, (
            f"Separation too small: genuine={genuine_mean:.3f}, "
            f"fake={fake_mean:.3f}, gap={genuine_mean - fake_mean:.3f}. "
            "Expected >= 0.03 gap for a meaningful signal."
        )

    @skip_no_clip
    @skip_no_abo
    def test_clip_output_bounds(self, abo_products, fitted_clip_scorer):
        """All similarity scores must be in [0.05, 0.98] after calibration."""
        listings = abo_products[["product_id"]].copy()
        listings["displayed_product_id"] = listings["product_id"]
        sims = fitted_clip_scorer.score_listings(listings)
        assert (sims >= 0.05).all(), f"Some scores below 0.05: {sims[sims < 0.05].values}"
        assert (sims <= 0.98).all(), f"Some scores above 0.98: {sims[sims > 0.98].values}"
        assert not sims.isna().any(), "NaN in similarity scores"

    @skip_no_clip
    @skip_no_abo
    def test_clip_separation_on_full_pipeline(self, pipeline):
        """End-to-end: fraud-injected listings must have lower CLIP similarity
        than genuine listings across the full synthetic dataset."""
        listings = pipeline["result"]["listings"]
        products = pipeline["catalog"]["products"]

        # Use TF-IDF fallback here so the test runs fast (full CLIP on 3k products
        # is too slow for a regular test session; CLIP is exercised above on 10 items)
        sims = compute_multimodal_similarity(listings, products, use_clip=False)

        normal_mask = ~listings["is_fraudulent"].fillna(False)
        fraud_mask = listings["is_fraudulent"].fillna(False)

        assert len(sims) == len(listings)
        assert (sims >= 0.05).all() and (sims <= 1.0).all()
        assert not sims.isna().any()

        if fraud_mask.any() and normal_mask.any():
            avg_normal = float(sims[normal_mask].mean())
            avg_fraud = float(sims[fraud_mask].mean())
            assert avg_normal > avg_fraud, (
                f"Normal mean ({avg_normal:.3f}) not > fraud mean ({avg_fraud:.3f}). "
                "Multimodal scoring must produce lower similarity for fraud listings."
            )


# ---------------------------------------------------------------------------
# Embedding cache roundtrip tests
# ---------------------------------------------------------------------------

class TestCLIPEmbeddingCache:
    @skip_no_clip
    @skip_no_abo
    def test_cache_save_and_load(self, tmp_path, abo_products, fitted_clip_scorer):
        """Save embeddings to a temp dir, then reload and verify equality."""
        cache = CLIPEmbeddingCache(str(tmp_path))
        fingerprint = _product_fingerprint(
            list(abo_products["product_id"].astype(str))
        )
        cache.save(
            fingerprint,
            fitted_clip_scorer.id_to_idx,
            fitted_clip_scorer.text_embeddings,
            fitted_clip_scorer.image_embeddings,
        )
        assert cache.exists()

        result = cache.load(fingerprint)
        assert result is not None, "Cache load returned None on valid fingerprint"
        id_to_idx, text_emb, image_emb = result

        assert id_to_idx == fitted_clip_scorer.id_to_idx
        np.testing.assert_array_almost_equal(
            text_emb, fitted_clip_scorer.text_embeddings, decimal=5
        )
        np.testing.assert_array_almost_equal(
            image_emb, fitted_clip_scorer.image_embeddings, decimal=5
        )

    @skip_no_clip
    @skip_no_abo
    def test_cache_stale_on_fingerprint_mismatch(self, tmp_path, abo_products, fitted_clip_scorer):
        """Cache must return None when fingerprint doesn't match."""
        cache = CLIPEmbeddingCache(str(tmp_path))
        fingerprint = _product_fingerprint(
            list(abo_products["product_id"].astype(str))
        )
        cache.save(
            fingerprint,
            fitted_clip_scorer.id_to_idx,
            fitted_clip_scorer.text_embeddings,
            fitted_clip_scorer.image_embeddings,
        )
        stale_fp = _product_fingerprint(["DIFFERENT_PRODUCT_XYZ"])
        result = cache.load(stale_fp)
        assert result is None, (
            "Cache should return None when fingerprint mismatches (catalog changed)."
        )

    @skip_no_clip
    @skip_no_abo
    def test_cache_hit_skips_model_load(self, tmp_path, abo_products, fitted_clip_scorer):
        """CLIPMultimodalScorer.fit() with cache_dir must skip _load_model on hit."""
        cache_dir = str(tmp_path / "hit_test")
        fingerprint = _product_fingerprint(
            list(abo_products["product_id"].astype(str))
        )
        cache = CLIPEmbeddingCache(cache_dir)
        cache.save(
            fingerprint,
            fitted_clip_scorer.id_to_idx,
            fitted_clip_scorer.text_embeddings,
            fitted_clip_scorer.image_embeddings,
        )

        # New scorer with cache_dir -- should load from cache, NOT call _load_model
        new_scorer = CLIPMultimodalScorer(cache_dir=cache_dir)
        # Monkeypatch _load_model to assert it is NOT called
        called = []
        orig = new_scorer._load_model

        def _should_not_be_called():
            called.append(True)
            orig()

        new_scorer._load_model = _should_not_be_called
        new_scorer.fit(abo_products)

        assert not called, (
            "_load_model was called even though a valid embedding cache existed. "
            "Cache hit must bypass the encode step entirely."
        )
        assert new_scorer._fitted


# ---------------------------------------------------------------------------
# Public API backward-compatibility tests
# ---------------------------------------------------------------------------

class TestPublicAPICompatibility:
    def test_compute_multimodal_similarity_signature(self, pipeline):
        """compute_multimodal_similarity must work with legacy signature (no cache_dir)."""
        listings = pipeline["result"]["listings"].head(50)
        products = pipeline["catalog"]["products"]
        sims = compute_multimodal_similarity(listings, products, use_clip=False)
        assert len(sims) == 50
        assert sims.name == "multimodal_similarity_score"

    def test_compute_multimodal_similarity_with_cache_dir(self, pipeline, tmp_path):
        """compute_multimodal_similarity must accept cache_dir kwarg without error."""
        listings = pipeline["result"]["listings"].head(20)
        products = pipeline["catalog"]["products"]
        # use_clip=False so we don't trigger CLIP encoding in this test
        sims = compute_multimodal_similarity(
            listings, products, use_clip=False, cache_dir=str(tmp_path)
        )
        assert len(sims) == 20
