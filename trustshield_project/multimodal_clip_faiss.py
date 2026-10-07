"""Multimodal Fake Listing Detection with CLIP and FAISS for TrustShield.

Combines vision-language representations and vector similarity search:
- CLIP Text & Image embedding extraction / cosine similarity
- FAISS IndexFlatIP (Cosine Similarity) for fast nearest neighbor search
- Detection of:
    - Image-Text semantic mismatch
    - Visual image reuse across unlinked sellers (scam/counterfeit indicator)
    - Near-duplicate product image clustering
- Ablation benchmarking across Tabular vs Vision vs Text vs Combined representations.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss: Any = None
    FAISS_AVAILABLE = False


class MultimodalFAISSIndex:
    """FAISS-powered vector similarity search for product listing visual embeddings."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.listing_ids: List[str] = []
        self.seller_ids: List[str] = []
        if FAISS_AVAILABLE:
            # Inner product on L2-normalized vectors is exact cosine similarity
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None
            self.vectors: List[np.ndarray] = []

    def add_listings(
        self,
        listing_ids: List[str],
        seller_ids: List[str],
        embeddings: np.ndarray,
    ) -> None:
        """Add normalized image embeddings to the index with metadata."""
        embs = np.asarray(embeddings, dtype=np.float32)
        # L2-normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_embs = embs / norms

        self.listing_ids.extend(listing_ids)
        self.seller_ids.extend(seller_ids)

        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(norm_embs)
        else:
            if not hasattr(self, "vectors") or self.vectors is None:
                self.vectors = []
            self.vectors.append(norm_embs)

    def query_similarity_features(
        self,
        query_embs: np.ndarray,
        query_seller_ids: List[str],
        query_listing_ids: Optional[List[str]] = None,
        k: int = 5,
    ) -> List[Dict[str, float]]:
        """Query top-k nearest neighbor listings and extract visual reuse fraud features.
        
        Temporal Policy:
            For strict historical consistency, indexed embeddings must have been observed
            prior to the query's decision timestamp. Self-matches are strictly excluded
            using explicit listing identity mapping.
        """
        q = np.asarray(query_embs, dtype=np.float32)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q_norm = q / norms

        results = []
        n_queries = len(q)
        search_k = min(k + 1, max(len(self.seller_ids), 1))

        if FAISS_AVAILABLE and self.index is not None:
            D, I = self.index.search(q_norm, search_k)
        else:
            # Fallback exact dot product
            all_vecs = np.vstack(self.vectors) if self.vectors else np.empty((0, self.dim))
            if len(all_vecs) == 0:
                return [{"faiss_max_image_similarity": 0.0, "faiss_near_duplicate_count": 0.0, "faiss_cross_seller_image_reuse": 0.0}] * n_queries
            sims = np.dot(q_norm, all_vecs.T)
            I = np.argsort(-sims, axis=1)[:, :search_k]
            D = np.take_along_axis(sims, I, axis=1)

        for i in range(n_queries):
            cur_seller = query_seller_ids[i] if i < len(query_seller_ids) else None
            cur_listing = query_listing_ids[i] if (query_listing_ids and i < len(query_listing_ids)) else None
            neighbor_sims = D[i]
            neighbor_indices = I[i]

            max_sim = 0.0
            near_dups = 0
            cross_seller_reuse = 0.0
            valid_neighbors = 0

            for sim, idx in zip(neighbor_sims, neighbor_indices):
                if idx < 0 or idx >= len(self.seller_ids):
                    continue
                # Exclude self-match using real listing ID identity
                if cur_listing is not None and idx < len(self.listing_ids) and self.listing_ids[idx] == cur_listing:
                    continue
                if cur_listing is None and idx == i and len(query_embs) == len(self.listing_ids):
                    continue

                valid_neighbors += 1
                s = float(sim)
                if s > max_sim:
                    max_sim = s
                if s >= 0.95:
                    near_dups += 1
                    neighbor_seller = self.seller_ids[idx]
                    if cur_seller and neighbor_seller != cur_seller:
                        cross_seller_reuse = 1.0

                if valid_neighbors >= k:
                    break

            results.append({
                "faiss_max_image_similarity": round(max_sim, 4),
                "faiss_near_duplicate_count": float(near_dups),
                "faiss_cross_seller_image_reuse": float(cross_seller_reuse),
            })

        return results


def extract_multimodal_listing_features(
    listings_df: pd.DataFrame,
    image_embs: np.ndarray,
    text_embs: np.ndarray,
    faiss_index: Optional[MultimodalFAISSIndex] = None,
) -> pd.DataFrame:
    """Compute rich vision-language features for listing fraud detection."""
    df = listings_df.copy()

    # 1. Cosine similarity between image and text
    img_norm = image_embs / np.maximum(np.linalg.norm(image_embs, axis=1, keepdims=True), 1e-8)
    txt_norm = text_embs / np.maximum(np.linalg.norm(text_embs, axis=1, keepdims=True), 1e-8)
    img_txt_sim = np.sum(img_norm * txt_norm, axis=1)
    sim = np.clip(img_txt_sim, -1.0, 1.0).astype(np.float32)

    df = df.assign(
        clip_image_text_similarity=sim.tolist(),
        clip_semantic_mismatch=(sim < 0.50).astype(np.float32).tolist(),
        clip_extreme_mismatch=(sim < 0.25).astype(np.float32).tolist(),
    )

    # 2. FAISS visual reuse features if index provided
    if faiss_index is not None:
        listing_ids = list(df["listing_id"].fillna("")) if "listing_id" in df.columns else None
        sim_feats = faiss_index.query_similarity_features(
            image_embs, list(df["seller_id"].fillna("")), query_listing_ids=listing_ids
        )
        for k in sim_feats[0]:
            df[k] = [r[k] for r in sim_feats]
    else:
        df["faiss_max_image_similarity"] = 0.0
        df["faiss_near_duplicate_count"] = 0.0
        df["faiss_cross_seller_image_reuse"] = 0.0

    return df


def run_multimodal_ablation_study(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_sets: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    """Execute ablation experiments comparing Tabular, Vision, Text, and Combined models."""
    results = {}
    for name, cols in feature_sets.items():
        avail_cols = [c for c in cols if c in X_train.columns]
        if not avail_cols:
            continue

        clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
        clf.fit(X_train[avail_cols].fillna(0.0), y_train)

        probs = np.asarray(clf.predict_proba(X_val[avail_cols].fillna(0.0)))[:, 1]
        preds = (probs >= 0.5).astype(int)

        roc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.5
        pr = average_precision_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.0
        f1 = f1_score(y_val, preds, zero_division=cast(Any, 0))

        results[name] = {
            "roc_auc": round(float(roc), 4),
            "pr_auc": round(float(pr), 4),
            "f1": round(float(f1), 4),
            "n_features": len(avail_cols),
        }

    return results
