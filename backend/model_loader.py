"""
TrustShield AI — Model loader for the FastAPI serving layer.

Loads pre-trained model artifacts (joblib files) at application startup so
requests don't trigger a multi-minute pipeline re-run.

Phase 3 artifacts (required):
    combined_graph_model.joblib   — tabular + NetworkX topology RF
    fake_listing_model.joblib     — Phase 2 listing-level RF
    return_fraud_model.joblib     — Phase 2 return-level RF
    fraud_rings.joblib            — pre-ranked fraud ring DataFrame
    feature_meta.joblib           — Phase 3 feature column lists

Phase 4 artifacts (optional — CLIP multimodal scorer):
    clip_cache/                   — pre-computed embedding cache directory
                                    (built by scripts/build_clip_embeddings.py)
    The scorer loads from cache in seconds. If absent, the /listing/analyze
    endpoint uses the caller-supplied multimodal_similarity_score field instead.

Phase 5 artifacts (optional — auto-detected):
    hybrid_model.joblib           — tabular + graph + GNN embedding XGBoost/RF
    buyer_embeddings.joblib       — dict {buyer_id  → np.array(16,)}
    seller_embeddings.joblib      — dict {seller_id → np.array(16,)}
    phase5_feature_meta.joblib    — Phase 5 feature column lists + metadata

If Phase 3 artifacts are missing, raises FileNotFoundError with instructions.
If Phase 4/5 artifacts are absent the server continues serving Phase 3
predictions without error — no configuration needed.
"""

import os
import sys
import joblib
import pandas as pd
from typing import Optional, Any

_PROJECT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trustshield_project")
)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

_REQUIRED = [
    "combined_graph_model.joblib",
    "fake_listing_model.joblib",
    "return_fraud_model.joblib",
    "fraud_rings.joblib",
    "feature_meta.joblib",
]

_PHASE5_OPTIONAL = [
    "hybrid_model.joblib",
    "buyer_embeddings.joblib",
    "seller_embeddings.joblib",
    "phase5_feature_meta.joblib",
]


class ModelStore:
    """Singleton holding all loaded model artifacts."""

    def __init__(self):
        # --- Phase 3 (required) ---
        self.combined_graph_model: Any = None
        self.fake_listing_model: Any   = None
        self.return_fraud_model: Any   = None
        self.rings_df: Optional[pd.DataFrame] = None
        self.feature_meta: Optional[dict] = None
        self._loaded: bool = False

        # --- Phase 4 CLIP scorer (optional) ---
        self.clip_scorer: Any = None          # CLIPMultimodalScorer instance, or None
        self.clip_products_df: Optional[pd.DataFrame] = None
        self._clip_loaded: bool = False
        self._tfidf_scorer: Any = None        # Cached MultimodalScorer surrogate

        # --- Calibration (Phase 6) ---
        self.calibrator: Any = None
        self.phase5_calibrator: Any = None

        # --- Graph Snapshot (API Parity) ---
        self.graph_snapshot: Optional[dict] = None

        # --- Phase 5 (optional) ---
        self.hybrid_model: Any         = None
        self.buyer_embeddings:  dict = {}
        self.seller_embeddings: dict = {}
        self.phase5_meta: Optional[dict] = None
        self._phase5_loaded: bool = False

        # --- Explainability Layer (TreeSHAP) ---
        self.shap_explainer: Any = None
        self.phase5_shap_explainer: Any = None

    def load(self):
        # ---- Phase 3 ----
        missing = [
            f for f in _REQUIRED
            if not os.path.exists(os.path.join(MODELS_DIR, f))
        ]
        if missing:
            raise FileNotFoundError(
                f"Required model artifacts not found: {missing}\n"
                f"Run:  python scripts/train_and_save_models.py"
            )

        self.combined_graph_model = joblib.load(
            os.path.join(MODELS_DIR, "combined_graph_model.joblib")
        )
        self.fake_listing_model   = joblib.load(
            os.path.join(MODELS_DIR, "fake_listing_model.joblib")
        )
        self.return_fraud_model   = joblib.load(
            os.path.join(MODELS_DIR, "return_fraud_model.joblib")
        )
        self.rings_df             = joblib.load(
            os.path.join(MODELS_DIR, "fraud_rings.joblib")
        )
        self.feature_meta         = joblib.load(
            os.path.join(MODELS_DIR, "feature_meta.joblib")
        )
        calibrator_path = os.path.join(MODELS_DIR, "calibrator.joblib")
        if os.path.isfile(calibrator_path):
            self.calibrator = joblib.load(calibrator_path)
            print("[model_loader] Loaded ProbabilityCalibrator from models/calibrator.joblib")

        snap_path = os.path.join(MODELS_DIR, "graph_snapshot.joblib")
        if os.path.isfile(snap_path):
            self.graph_snapshot = joblib.load(snap_path)
            print(f"[model_loader] Loaded graph snapshot from {snap_path}")

        self._loaded = True
        print(f"[model_loader] Loaded Phase 3 models from {MODELS_DIR}")
        if self.rings_df is not None:
            print(
                f"[model_loader] {len(self.rings_df)} fraud rings pre-computed "
                f"({(self.rings_df['avg_risk_score'] >= 0.3).sum()} high-risk)"
            )

        # ---- Explainability: Initialize TreeSHAP Explainer for Phase 3 ----
        try:
            from shap_explainer import TrustShieldSHAPExplainer
            p3_cols = self.feature_meta.get("all_feature_cols", []) if self.feature_meta is not None else []
            self.shap_explainer = TrustShieldSHAPExplainer(
                model=self.combined_graph_model,
                feature_names=p3_cols,
                model_name="Combined Graph Detector (Phase 3)",
            )
            print(
                f"[model_loader] Phase 3 TreeSHAP explainer initialized "
                f"({len(p3_cols)} features)"
            )
        except Exception as exc:
            print(f"[model_loader] TreeSHAP Phase 3 explainer init failed: {exc}")
            self.shap_explainer = None

        # ---- Phase 4: CLIP embedding cache (optional) ----
        clip_cache_dir = os.path.join(MODELS_DIR, "clip_cache")
        clip_meta_path = os.path.join(clip_cache_dir, "clip_meta.npz")
        if os.path.isfile(clip_meta_path):
            try:
                import sys
                _project_dir = os.path.normpath(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trustshield_project")
                )
                if _project_dir not in sys.path:
                    sys.path.insert(0, _project_dir)

                from product_listing_generator import generate_product_catalog
                from multimodal_scoring import (
                    CLIPMultimodalScorer,
                    CLIPEmbeddingCache,
                    _product_fingerprint,
                )

                self.clip_products_df = generate_product_catalog()
                cache = CLIPEmbeddingCache(clip_cache_dir)
                fingerprint = _product_fingerprint(list(self.clip_products_df["product_id"].astype(str)))
                cached = cache.load(fingerprint)
                if cached is None and cache.exists():
                    # Fallback: load available cached product embeddings
                    cached = cache.load(None)
                if cached is not None:
                    self.clip_scorer = CLIPMultimodalScorer(cache_dir=clip_cache_dir)
                    self.clip_scorer.id_to_idx, self.clip_scorer.text_embeddings, self.clip_scorer.image_embeddings = cached
                    self.clip_scorer._fitted = True
                    self._clip_loaded = True
                    print(
                        f"[model_loader] Phase 4 CLIP scorer loaded from cache "
                        f"({len(self.clip_scorer.id_to_idx):,} products, "
                        f"text_emb={self.clip_scorer.text_embeddings.shape}, "
                        f"image_emb={self.clip_scorer.image_embeddings.shape})"
                    )
                else:
                    print(
                        "[model_loader] Phase 4 CLIP cache fingerprint mismatch or stale. "
                        "Run scripts/build_clip_embeddings.py to refresh cache. "
                        "The /listing/analyze endpoint will use caller-supplied "
                        "multimodal_similarity_score instead."
                    )
            except Exception as exc:
                print(
                    f"[model_loader] Phase 4 CLIP scorer unavailable ({exc}). "
                    "The /listing/analyze endpoint will use caller-supplied "
                    "multimodal_similarity_score instead."
                )
        else:
            print(
                "[model_loader] Phase 4 CLIP cache not found — "
                "run scripts/build_clip_embeddings.py to enable real CLIP scoring."
            )

        # ---- Phase 5 (optional — graceful skip if absent) ----
        phase5_present = all(
            os.path.exists(os.path.join(MODELS_DIR, f))
            for f in _PHASE5_OPTIONAL
        )
        if phase5_present:
            self.hybrid_model       = joblib.load(
                os.path.join(MODELS_DIR, "hybrid_model.joblib")
            )
            self.buyer_embeddings   = joblib.load(
                os.path.join(MODELS_DIR, "buyer_embeddings.joblib")
            )
            self.seller_embeddings  = joblib.load(
                os.path.join(MODELS_DIR, "seller_embeddings.joblib")
            )
            self.phase5_meta        = joblib.load(
                os.path.join(MODELS_DIR, "phase5_feature_meta.joblib")
            )
            p5_cal_path = os.path.join(MODELS_DIR, "phase5_calibrator.joblib")
            if os.path.isfile(p5_cal_path):
                self.phase5_calibrator = joblib.load(p5_cal_path)
                print("[model_loader] Loaded Phase 5 ProbabilityCalibrator from models/phase5_calibrator.joblib")
            self._phase5_loaded = True
            meta = self.phase5_meta
            if meta is not None:
                print(
                    f"[model_loader] Phase 5 hybrid model loaded  "
                    f"({len(self.buyer_embeddings):,} buyer embeddings, "
                    f"{len(self.seller_embeddings):,} seller embeddings, "
                    f"{len(meta['hybrid_feature_cols'])} features, "
                    f"classifier={meta.get('classifier', 'XGBoost')})"
                )
                try:
                    from shap_explainer import TrustShieldSHAPExplainer
                    self.phase5_shap_explainer = TrustShieldSHAPExplainer(
                        model=self.hybrid_model,
                        feature_names=meta.get("hybrid_feature_cols", []),
                        model_name="Hybrid GNN Detector (Phase 5)",
                    )
                    print(
                        f"[model_loader] Phase 5 TreeSHAP explainer initialized "
                        f"({len(meta.get('hybrid_feature_cols', []))} features)"
                    )
                except Exception as exc:
                    print(f"[model_loader] TreeSHAP Phase 5 explainer init failed: {exc}")
                    self.phase5_shap_explainer = None
        else:
            print(
                "[model_loader] Phase 5 artifacts not found — "
                "serving Phase 3 predictions.  "
                "Run scripts/train_phase5.py to enable hybrid scoring."
            )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def phase5_loaded(self) -> bool:
        return self._phase5_loaded

    @property
    def clip_loaded(self) -> bool:
        return self._clip_loaded

    @property
    def shap_loaded(self) -> bool:
        return (self.shap_explainer is not None and self.shap_explainer.is_available) or (
            self.phase5_shap_explainer is not None and self.phase5_shap_explainer.is_available
        )


# Global singleton — imported by main.py
store = ModelStore()
