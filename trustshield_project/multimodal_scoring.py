"""Multimodal image-text semantic alignment scoring and investigator narrative — Phase 4.

Architecture
------------
Two scorer classes are provided, used in order of preference:

1. ``CLIPMultimodalScorer`` — real CLIP (openai/clip-vit-base-patch32) embeddings.
   Requires: transformers, Pillow, torch.
   Uses the actual ABO product images from disk + ABO listing text to compute
   cosine similarity in CLIP's joint vision-language embedding space.  This is
   the real multimodal signal described in Phase 4.

   Embedding Cache
   ~~~~~~~~~~~~~~
   Computing CLIP embeddings for 3 000 products (text + image) takes ~3-5 min on
   CPU.  ``CLIPMultimodalScorer`` supports optional on-disk caching via
   ``CLIPEmbeddingCache``::

       scorer = CLIPMultimodalScorer(cache_dir="models/clip_cache")
       scorer.fit(products_df)   # loads cache on first call; skips re-encode

   The cache stores ``id_to_idx``, ``text_embeddings``, and ``image_embeddings``
   as compressed NumPy archives alongside a SHA-256 fingerprint of the product
   IDs list.  If the fingerprint mismatches (e.g. catalog changes), the cache
   is invalidated and embeddings are re-computed.

2. ``MultimodalScorer`` — TF-IDF surrogate (CPU-only, no extra deps).
   Used as a fallback when CLIP is unavailable or images are missing.
   Behaviour is identical to the pre-Phase-4 implementation so all existing
   tests continue to pass.

The public ``compute_multimodal_similarity`` function and ``explain_listing_risk``
function remain unchanged in signature and are backward-compatible.

Public helpers
--------------
- ``compute_multimodal_similarity(listings_df, products_df, use_clip=True)``
- ``build_and_cache_clip_scorer(products_df, cache_dir)``  — pre-compute & persist
- ``get_clip_scorer(products_df, cache_dir)``              — load-or-fit singleton
- ``explain_listing_risk(features, risk_score, threshold)``
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
_ABO_ROOT: Optional[str] = None


def _resolve_abo_root() -> Optional[str]:
    """Locates the ABO dataset root (containing images/small/…)."""
    global _ABO_ROOT
    if _ABO_ROOT is not None:
        return _ABO_ROOT
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("ABO_DATA_DIR", ""),
        os.path.join(this_dir, "data", "external", "abo"),
        os.path.join(this_dir, "..", "data", "external", "abo"),
    ]
    for p in candidates:
        p = p.strip()
        if p and os.path.isdir(p) and os.path.isdir(os.path.join(p, "images", "small")):
            _ABO_ROOT = os.path.normpath(p)
            return _ABO_ROOT
    return None


def _image_ref_to_path(image_ref: str, abo_root: str) -> Optional[str]:
    """Resolves a relative image_ref to an absolute path if the file exists.

    Handles both forward-slash and backslash separators that can appear in
    ``image_ref`` values produced by ``product_listing_generator`` on Windows
    (``os.path.join`` emits backslashes there, but the ABO CSV paths use
    forward slashes for the 2-char shard prefix).
    """
    if not image_ref or image_ref.startswith("PLACEHOLDER"):
        return None
    # Normalise mixed separators so os.path.join works on every OS.
    image_ref_norm = image_ref.replace("\\", "/")
    full = os.path.normpath(os.path.join(abo_root, image_ref_norm))
    return full if os.path.isfile(full) else None


# ---------------------------------------------------------------------------
# Embedding cache (Phase 4 — avoid re-encoding on every startup)
# ---------------------------------------------------------------------------

_CACHE_VERSION = "v1"  # bump when embedding format changes


def _product_fingerprint(product_ids: list[str]) -> str:
    """SHA-256 hex digest of the sorted product ID list for cache invalidation."""
    h = hashlib.sha256("|".join(sorted(product_ids)).encode()).hexdigest()
    return h[:16]  # 16 hex chars (64 bits) is more than sufficient


class CLIPEmbeddingCache:
    """Persists CLIP embeddings and id_to_idx to a directory as .npz files.

    Layout inside *cache_dir*::

        clip_meta.npz       — {fingerprint, version, id_to_idx keys+values}
        clip_text_emb.npy   — float32 array (N, 512)
        clip_image_emb.npy  — float32 array (N, 512)
    """

    META_FILE = "clip_meta.npz"
    TEXT_FILE = "clip_text_emb.npy"
    IMAGE_FILE = "clip_image_emb.npy"

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def _paths(self):
        return (
            os.path.join(self.cache_dir, self.META_FILE),
            os.path.join(self.cache_dir, self.TEXT_FILE),
            os.path.join(self.cache_dir, self.IMAGE_FILE),
        )

    def exists(self) -> bool:
        return all(os.path.isfile(p) for p in self._paths())

    def load(self, fingerprint: str):
        """Returns (id_to_idx, text_emb, image_emb) or None if cache is stale."""
        meta_path, text_path, image_path = self._paths()
        meta = np.load(meta_path, allow_pickle=False)
        raw_fp = meta["fingerprint"].item() if hasattr(meta["fingerprint"], "item") else meta["fingerprint"]
        cached_fp = raw_fp.decode("utf-8") if isinstance(raw_fp, bytes) else str(raw_fp)
        raw_ver = meta["version"].item() if hasattr(meta["version"], "item") else meta["version"]
        cached_ver = raw_ver.decode("utf-8") if isinstance(raw_ver, bytes) else str(raw_ver)

        if cached_fp != fingerprint or cached_ver != _CACHE_VERSION:
            logger.info(
                "CLIP cache fingerprint/version mismatch — will re-encode. "
                "(cached=%s/%s, current=%s/%s)",
                cached_fp, cached_ver, fingerprint, _CACHE_VERSION,
            )
            return None
        keys = [str(k) for k in meta["keys"]]
        values = [int(v) for v in meta["values"]]
        id_to_idx: Dict[str, int] = dict(zip(keys, values))
        text_emb = np.load(text_path)
        image_emb = np.load(image_path)
        logger.info("CLIP embeddings loaded from cache (%s products).", len(id_to_idx))
        return id_to_idx, text_emb, image_emb

    def save(
        self,
        fingerprint: str,
        id_to_idx: Dict[str, int],
        text_emb: np.ndarray,
        image_emb: np.ndarray,
    ) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        meta_path, text_path, image_path = self._paths()
        keys_arr = np.array(list(id_to_idx.keys()))
        vals_arr = np.array(list(id_to_idx.values()), dtype=np.int32)
        np.savez(
            meta_path,
            fingerprint=fingerprint,
            version=_CACHE_VERSION,
            keys=keys_arr,
            values=vals_arr,
        )
        np.save(text_path, text_emb.astype(np.float32))
        np.save(image_path, image_emb.astype(np.float32))
        logger.info("CLIP embeddings saved to cache at '%s'.", self.cache_dir)


# ---------------------------------------------------------------------------
# Real CLIP scorer (Phase 4)
# ---------------------------------------------------------------------------

class CLIPMultimodalScorer:
    """Computes real image-text cosine similarity using OpenAI CLIP.

    On ``fit``: encodes all product texts into CLIP text embeddings and, where
    real ABO images exist on disk, encodes the product images into CLIP vision
    embeddings.  Products with missing images fall back to a noisy text clone
    (identical behaviour to the TF-IDF surrogate).

    On ``score_listings``: computes cosine similarity between the *real* catalog
    item text embedding and the *displayed* product image embedding.  For fake
    listings ``displayed_product_id`` ≠ ``product_id``, so the text/image come
    from different products, driving similarity down — exactly the signal we want.
    """

    def __init__(
        self,
        model_id: str = _CLIP_MODEL_ID,
        batch_size: int = 32,
        device: Optional[str] = None,
        random_state: int = 42,
        cache_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.random_state = random_state
        self.cache_dir = cache_dir
        self._fitted = False
        self.id_to_idx: Dict[str, int] = {}
        self.text_embeddings: np.ndarray = np.empty((0, 512))
        self.image_embeddings: np.ndarray = np.empty((0, 512))

        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Internal batch-encoding helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy-load CLIP model + processor (deferred to first fit call)."""
        from transformers import CLIPModel, CLIPProcessor
        logger.info("Loading CLIP model %s on %s…", self.model_id, self.device)
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(
            self.model_id, use_safetensors=False
        ).to(self.device)
        self._model.eval()
        logger.info("CLIP model loaded.")

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        all_embeds = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self._processor.tokenizer(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)
            with torch.no_grad():
                out = self._model.get_text_features(**inputs)
                # transformers ≥5.x may return a dataclass; extract the tensor.
                feats: torch.Tensor = out if isinstance(out, torch.Tensor) else out.pooler_output
                feats = F.normalize(feats.float(), dim=-1)
            all_embeds.append(feats.cpu().numpy())
        return np.vstack(all_embeds)

    def _encode_images(
        self, image_paths: list[Optional[str]], text_fallbacks: np.ndarray
    ) -> np.ndarray:
        """Encode images; fall back to a noise-perturbed text embedding where image is missing."""
        import torch
        import torch.nn.functional as F
        from PIL import Image

        rng = np.random.default_rng(self.random_state)
        result = np.zeros_like(text_fallbacks)

        # Group indices by whether we have a real image
        real_indices = [i for i, p in enumerate(image_paths) if p is not None]
        fake_indices = [i for i, p in enumerate(image_paths) if p is None]

        # Encode real images in batches
        for batch_start in range(0, len(real_indices), self.batch_size):
            batch_idxs = real_indices[batch_start : batch_start + self.batch_size]
            pil_imgs = []
            valid_idxs = []
            for idx in batch_idxs:
                try:
                    img = Image.open(image_paths[idx]).convert("RGB")  # type: ignore[arg-type]
                    pil_imgs.append(img)
                    valid_idxs.append(idx)
                except Exception:
                    fake_indices.append(idx)
            if pil_imgs:
                inputs = self._processor.image_processor(images=pil_imgs, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self._model.get_image_features(**inputs)
                    # transformers ≥5.x may return a dataclass; extract the tensor.
                    feats: torch.Tensor = out if isinstance(out, torch.Tensor) else out.pooler_output
                    feats = F.normalize(feats.float(), dim=-1)
                for local_i, global_i in enumerate(valid_idxs):
                    result[global_i] = feats[local_i].cpu().numpy()

        # Fall back for missing images: text + noise (same as surrogate)
        for idx in fake_indices:
            noise = rng.normal(0, 0.45, size=text_fallbacks[idx].shape)
            noisy = text_fallbacks[idx] + noise
            norm = np.linalg.norm(noisy)
            result[idx] = noisy / max(norm, 1e-8)

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, products_df: pd.DataFrame) -> "CLIPMultimodalScorer":
        """Encode all products' texts and images into CLIP embedding space.

        If ``cache_dir`` was provided at construction time, the method first
        checks for a valid on-disk cache before running the expensive
        encode step.  A cache miss triggers encoding and then writes the
        results back to disk for future calls.
        """
        product_ids = list(products_df["product_id"].astype(str))
        fingerprint = _product_fingerprint(product_ids)

        # ── Try cache hit ────────────────────────────────────────────────
        if self.cache_dir is not None:
            cache = CLIPEmbeddingCache(self.cache_dir)
            if cache.exists():
                cached = cache.load(fingerprint)
                if cached is not None:
                    self.id_to_idx, self.text_embeddings, self.image_embeddings = cached
                    self._fitted = True
                    return self

        # ── Cache miss: encode from scratch ──────────────────────────────
        self._load_model()

        abo_root = _resolve_abo_root()
        texts = (
            products_df["category"].fillna("").astype(str) + " "
            + products_df["title"].fillna("").astype(str) + " "
            + products_df["description"].fillna("").astype(str)
        ).tolist()

        logger.info("Encoding %d product texts with CLIP…", len(texts))
        self.text_embeddings = self._encode_texts(texts)

        image_paths: list[Optional[str]] = []
        if abo_root and "image_ref" in products_df.columns:
            image_paths = [
                _image_ref_to_path(str(ref), abo_root)
                for ref in products_df["image_ref"]
            ]
            n_real = sum(1 for p in image_paths if p is not None)
            logger.info(
                "ABO images: %d real on-disk, %d fallback (placeholder/missing)",
                n_real, len(image_paths) - n_real,
            )
        else:
            image_paths = [None] * len(texts)
            logger.warning(
                "ABO image directory not found. All products will use text-fallback image embeddings."
            )

        self.image_embeddings = self._encode_images(image_paths, self.text_embeddings)
        self.id_to_idx = {pid: i for i, pid in enumerate(product_ids)}
        self._fitted = True

        # ── Persist to cache ─────────────────────────────────────────────
        if self.cache_dir is not None:
            try:
                CLIPEmbeddingCache(self.cache_dir).save(
                    fingerprint, self.id_to_idx, self.text_embeddings, self.image_embeddings
                )
            except Exception as exc:
                logger.warning("Could not write CLIP cache: %s", exc)

        return self

    def score_listings(self, listings_df: pd.DataFrame) -> pd.Series:
        """Compute [0, 1] cosine similarity between listing text and displayed image."""
        if not self._fitted:
            raise ValueError("CLIPMultimodalScorer must be fitted before scoring.")

        fallback_idx = 0
        prod_indices = [
            self.id_to_idx.get(str(pid), fallback_idx) if pd.notna(pid) else fallback_idx
            for pid in listings_df["product_id"]
        ]

        displayed_col = (
            listings_df["displayed_product_id"]
            if "displayed_product_id" in listings_df.columns
            else listings_df["product_id"]
        )
        disp_indices = [
            self.id_to_idx.get(str(dpid), pidx) if pd.notna(dpid) else pidx
            for dpid, pidx in zip(displayed_col, prod_indices)
        ]

        # text from the *real* product_id, image from the *displayed* product
        u = self.text_embeddings[prod_indices]
        v = self.image_embeddings[disp_indices]

        raw_sim = np.sum(u * v, axis=1)
        # CLIP raw cosine is roughly in [−1, 1]; calibrate to [0.05, 0.98]
        calibrated = np.clip(0.50 + 0.50 * raw_sim, 0.05, 0.98).round(4)
        return pd.Series(calibrated, index=listings_df.index, name="multimodal_similarity_score")


# ---------------------------------------------------------------------------
# TF-IDF surrogate scorer (fallback / legacy)
# ---------------------------------------------------------------------------

class MultimodalScorer:
    """TF-IDF surrogate for multimodal similarity — CPU-only, no extra deps.

    Behaviour is identical to the pre-Phase-4 implementation.  All existing
    tests rely on this class and continue to pass unchanged.
    """

    def __init__(self, n_components: int = 64, random_state: int = 42):
        self.vectorizer = TfidfVectorizer(max_features=n_components, stop_words="english")
        self.random_state = random_state
        self._fitted = False
        self.id_to_idx: Dict[str, int] = {}
        self.text_embeddings: np.ndarray = np.empty((0, n_components))
        self.image_embeddings: np.ndarray = np.empty((0, n_components))

    def fit(self, products_df: pd.DataFrame) -> "MultimodalScorer":
        texts = (
            products_df["category"].fillna("").astype(str) + " "
            + products_df["title"].fillna("").astype(str) + " "
            + products_df["description"].fillna("").astype(str)
        ).tolist()

        X_text = self.vectorizer.fit_transform(texts).toarray()  # type: ignore[union-attr]
        norms = np.linalg.norm(X_text, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_text = X_text / norms

        rng = np.random.default_rng(self.random_state)
        visual_noise = rng.normal(0, 0.45, size=X_text.shape)
        X_img = X_text + visual_noise
        img_norms = np.linalg.norm(X_img, axis=1, keepdims=True)
        img_norms[img_norms == 0] = 1.0
        X_img = X_img / img_norms

        self.text_embeddings = X_text
        self.image_embeddings = X_img
        self.id_to_idx = {str(pid): i for i, pid in enumerate(products_df["product_id"])}
        self._fitted = True
        return self

    def score_listings(self, listings_df: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise ValueError("MultimodalScorer must be fitted before scoring listings.")

        fallback_idx = 0
        prod_indices = [
            self.id_to_idx.get(str(pid), fallback_idx) if pd.notna(pid) else fallback_idx
            for pid in listings_df["product_id"]
        ]

        displayed_col = (
            listings_df["displayed_product_id"]
            if "displayed_product_id" in listings_df.columns
            else listings_df["product_id"]
        )
        disp_indices = [
            self.id_to_idx.get(str(dpid), pidx) if pd.notna(dpid) else pidx
            for dpid, pidx in zip(displayed_col, prod_indices)
        ]

        u = self.text_embeddings[prod_indices]
        v = self.image_embeddings[disp_indices]

        raw_sim = np.sum(u * v, axis=1)
        calibrated = np.clip(0.55 + 0.90 * raw_sim, 0.05, 0.98).round(4)
        return pd.Series(calibrated, index=listings_df.index, name="multimodal_similarity_score")


# ---------------------------------------------------------------------------
# Public convenience functions
# ---------------------------------------------------------------------------

def compute_multimodal_similarity(
    listings_df: pd.DataFrame,
    products_df: pd.DataFrame,
    use_clip: bool = True,
    cache_dir: Optional[str] = None,
) -> pd.Series:
    """Compute multimodal alignment scores for listings.

    Attempts to use real CLIP embeddings (``CLIPMultimodalScorer``).  Falls
    back transparently to the TF-IDF surrogate when CLIP is unavailable (no
    transformers, no Pillow) or when ABO images are not on disk.

    Parameters
    ----------
    listings_df:
        Must contain columns ``product_id`` and optionally ``displayed_product_id``.
    products_df:
        Must contain columns ``product_id``, ``category``, ``title``, ``description``
        and optionally ``image_ref``.
    use_clip:
        Set to ``False`` to skip CLIP and always use the TF-IDF surrogate.
    cache_dir:
        Optional directory for caching CLIP embeddings to disk.  Pass a path
        such as ``"models/clip_cache"`` to avoid re-encoding on every call.
    """
    if use_clip:
        # Check default cache directory if not explicitly provided
        effective_cache_dir = cache_dir
        if effective_cache_dir is None:
            default_dir = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "clip_cache")
            )
            if os.path.isdir(default_dir):
                effective_cache_dir = default_dir

        # If catalog is large and we don't have a valid cache hit, avoid long CPU freezes
        if len(products_df) > 100:
            has_cache = False
            if effective_cache_dir:
                try:
                    c = CLIPEmbeddingCache(effective_cache_dir)
                    if c.exists():
                        fp = _product_fingerprint(list(products_df["product_id"].astype(str)))
                        if c.load(fp) is not None:
                            has_cache = True
                except Exception:
                    pass
            if not has_cache:
                logger.info(
                    "Large catalog (%d products) without precomputed CLIP cache. "
                    "Using TF-IDF surrogate for fast scoring. "
                    "Run scripts/build_clip_embeddings.py to enable real CLIP.",
                    len(products_df),
                )
                scorer = MultimodalScorer()
                return scorer.fit(products_df).score_listings(listings_df)

        try:
            import transformers  # noqa: F401
            from PIL import Image  # noqa: F401
            scorer: CLIPMultimodalScorer | MultimodalScorer = CLIPMultimodalScorer(
                cache_dir=effective_cache_dir
            )
            result = scorer.fit(products_df).score_listings(listings_df)
            logger.info(
                "CLIP multimodal similarity computed (mean=%.3f, std=%.3f)",
                float(result.mean()),
                float(result.std()),
            )
            return result
        except Exception as exc:
            logger.warning(
                "CLIP scorer unavailable (%s). Falling back to TF-IDF surrogate.", exc
            )

    scorer = MultimodalScorer()
    return scorer.fit(products_df).score_listings(listings_df)


def build_and_cache_clip_scorer(
    products_df: pd.DataFrame,
    cache_dir: str,
) -> CLIPMultimodalScorer:
    """Pre-compute CLIP embeddings for *products_df* and persist to *cache_dir*.

    This is the offline training-time function.  Run it once after updating
    the product catalog; subsequent calls to ``compute_multimodal_similarity``
    or ``get_clip_scorer`` will load from cache instead of re-encoding.

    Parameters
    ----------
    products_df:
        Full product catalog DataFrame with ``product_id``, ``category``,
        ``title``, ``description``, and ``image_ref`` columns.
    cache_dir:
        Directory where the embedding cache is written.
        Recommended: ``"models/clip_cache"``.

    Returns
    -------
    Fitted ``CLIPMultimodalScorer`` instance.
    """
    scorer = CLIPMultimodalScorer(cache_dir=cache_dir)
    scorer.fit(products_df)
    logger.info(
        "CLIP scorer built and cached at '%s' (%d products).",
        cache_dir, len(products_df),
    )
    return scorer


# Module-level singleton (populated by get_clip_scorer for API serving)
_CLIP_SCORER_SINGLETON: Optional[CLIPMultimodalScorer] = None


def get_clip_scorer(
    products_df: pd.DataFrame,
    cache_dir: Optional[str] = None,
) -> CLIPMultimodalScorer:
    """Return a fitted ``CLIPMultimodalScorer``, reusing a module-level singleton.

    Intended for the FastAPI prediction endpoint so that the model is loaded
    only once per worker process.  The first call fits the scorer (or loads
    from *cache_dir*); subsequent calls return the cached instance directly.

    Parameters
    ----------
    products_df:
        Full product catalog DataFrame.  Only used on the first call.
    cache_dir:
        Optional cache directory forwarded to ``CLIPMultimodalScorer``.
    """
    global _CLIP_SCORER_SINGLETON
    if _CLIP_SCORER_SINGLETON is not None and _CLIP_SCORER_SINGLETON._fitted:
        return _CLIP_SCORER_SINGLETON
    _CLIP_SCORER_SINGLETON = CLIPMultimodalScorer(cache_dir=cache_dir)
    _CLIP_SCORER_SINGLETON.fit(products_df)
    return _CLIP_SCORER_SINGLETON


# ---------------------------------------------------------------------------
# Investigator narrative (unchanged)
# ---------------------------------------------------------------------------

def explain_listing_risk(
    features: Dict[str, Any],
    risk_score: float,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Generates structured natural-language explanation of risk signals for fraud investigators."""
    is_high_risk = risk_score >= threshold
    risk_label = "HIGH" if risk_score >= 0.5 else ("MEDIUM" if risk_score >= 0.3 else "LOW")

    evidence = []

    # Multimodal alignment analysis
    sim_score = float(features.get("multimodal_similarity_score", 0.8))
    if sim_score < 0.45:
        evidence.append(
            f"Severe visual-textual mismatch (similarity: {sim_score:.2f}). "
            "Displayed product image does not match catalog item title/description."
        )
    elif sim_score < 0.65:
        evidence.append(f"Moderate visual discrepancy detected (similarity: {sim_score:.2f}).")
    else:
        evidence.append(
            f"Product image is visually consistent with item description (similarity: {sim_score:.2f})."
        )

    # Price anomaly analysis
    price_ratio = float(features.get("price_vs_base_price_ratio", 1.0))
    if price_ratio < 0.60:
        evidence.append(
            f"Abnormal discount detected: listing is priced at {price_ratio:.1%} of catalog base price."
        )
    elif price_ratio > 1.60:
        evidence.append(f"Significant price markup detected ({price_ratio:.1%} of base price).")

    # Seller history analysis
    listings_before = int(features.get("seller_listings_before", 0))
    seller_age = float(features.get("seller_age_days_at_listing", 0))
    if listings_before < 2 and seller_age < 14:
        evidence.append(
            f"New seller account ({seller_age:.0f} days old) with limited track record "
            f"({listings_before} prior listings)."
        )

    # Recommended action
    if is_high_risk:
        action = "Suspend listing immediately and request merchant proof of inventory & supplier invoices."
    elif risk_label == "MEDIUM":
        action = "Queue for manual visual review by Trust & Safety before order disbursement."
    else:
        action = "Standard approval; no anomalous visual or pricing signals detected."

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_label,
        "summary": f"{risk_label} RISK (score {risk_score:.2f}) — " + (
            "Potential fake listing flagged by multimodal discrepancy and pricing anomalies."
            if is_high_risk else "Listing within normal operating parameters."
        ),
        "key_evidence": evidence,
        "recommended_action": action,
    }
