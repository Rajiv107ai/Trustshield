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

If Phase 3 artifacts are missing, raises FileNotFoundError with instructions.
If Phase 4 CLIP cache is absent, the server continues serving Phase 3
predictions without error.
"""

import os
import logging
import sys
from datetime import datetime, timedelta
import joblib
import pandas as pd
from typing import Optional, Any

_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
MODELS_DIR = os.path.join(_ROOT, "models")

logger = logging.getLogger(__name__)

try:
    import trustshield_project.calibration as _cal
    sys.modules.setdefault("calibration", _cal)
except ImportError:
    pass

_REQUIRED = [
    "combined_graph_model.joblib",
    "fake_listing_model.joblib",
    "return_fraud_model.joblib",
    "fraud_rings.joblib",
    "feature_meta.joblib",
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

        # --- Calibration & Conformal Uncertainty ---
        self.calibrator: Any = None
        self.conformal_predictor: Any = None

        # --- Point-in-Time Transaction History & Governed Feature Service ---
        self.transaction_history: Any = None
        self.feature_service: Any = None

        # --- Graph Snapshot (API Parity) ---
        self.graph_snapshot: Optional[dict] = None

        # --- Serving Flags ---
        self._phase5_loaded: bool = False
        self.graph_snapshot_stale: bool = False
        self._graph_snapshot_age_days: Optional[float] = None

        # --- Explainability Layer (TreeSHAP) ---
        self.shap_explainer: Any = None

    def load(self, force: bool = False):
        if self._loaded and not force:
            return
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

        import warnings
        try:
            from sklearn.exceptions import InconsistentVersionWarning
        except ImportError:
            InconsistentVersionWarning = Warning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
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
                logger.info("Loaded ProbabilityCalibrator from models/calibrator.joblib")

        snap_path = os.path.join(MODELS_DIR, "graph_snapshot.joblib")
        if os.path.isfile(snap_path):
            self.graph_snapshot = joblib.load(snap_path)
            logger.info("Loaded graph snapshot from %s", snap_path)
            snap_as_of = None
            if isinstance(self.graph_snapshot, dict):
                snap_as_of = self.graph_snapshot.get("effective_as_of") or self.graph_snapshot.get("snapshot_cutoff")
            if snap_as_of:
                logger.info("Graph snapshot effective_as_of: %s", snap_as_of)
                # --- Graph Snapshot Freshness Gate ---
                max_stale_days = int(os.getenv("TRUSTSHIELD_GRAPH_MAX_STALE_DAYS", "400"))
                try:
                    snap_date = pd.Timestamp(str(snap_as_of)).to_pydatetime()
                    if snap_date.tzinfo is not None:
                        snap_date = snap_date.replace(tzinfo=None)
                    age_days = (datetime.now() - snap_date).days
                    self._graph_snapshot_age_days = float(age_days)
                    if age_days > max_stale_days:
                        self.graph_snapshot_stale = True
                        logger.error(
                            "CRITICAL: Graph snapshot is %d days old (threshold: %d days). "
                            "Graph topology features are stale. Schedule a snapshot rebuild or "
                            "set TRUSTSHIELD_GRAPH_MAX_STALE_DAYS to adjust the threshold.",
                            age_days, max_stale_days,
                        )
                    else:
                        logger.info("Graph snapshot age: %d days (within %d-day threshold)", age_days, max_stale_days)
                except Exception as _age_exc:
                    logger.warning("Could not parse graph snapshot date for freshness check: %s", _age_exc)
            logger.warning(
                "Graph topology is a static snapshot. "
                "Serving requests after the snapshot date receive stale topology features."
            )
        else:
            self.graph_snapshot = None
            logger.warning(
                "graph_snapshot.joblib not found. "
                "All graph features will be accepted from callers without server-side validation."
            )

        # Point-in-time transaction history store (for causal edge weight calculation)
        try:
            from trustshield_project.point_in_time_history import PointInTimeHistoryStore
            data_dir = os.path.join(_ROOT, "data", "synthetic_v2_1")
            orders_path = os.path.join(data_dir, "orders.csv")
            self.transaction_history = PointInTimeHistoryStore(orders_csv_path=orders_path if os.path.isfile(orders_path) else None)
        except Exception as _pit_exc:
            logger.warning("PointInTimeHistoryStore initialization failed: %s", _pit_exc)
            self.transaction_history = None

        # Governed Real-Time Point-in-Time Feature Service (sliding windows & catalog)
        try:
            from backend.services.feature_service import PointInTimeFeatureService
            data_dir = os.path.join(_ROOT, "data", "synthetic_v2_1")
            catalog_path = os.path.join(MODELS_DIR, "product_catalog.parquet")
            self.feature_service = PointInTimeFeatureService(
                data_dir=data_dir if os.path.isdir(data_dir) else None,
                catalog_path=catalog_path if os.path.isfile(catalog_path) else None,
            )
            logger.info("PointInTimeFeatureService initialized (status=%s)", self.feature_service.get_status())
        except Exception as _fs_exc:
            logger.warning("PointInTimeFeatureService initialization failed: %s", _fs_exc)
            self.feature_service = None

        # Conformal Predictor: Load held-out validation calibration data if available
        # Prioritize Phase 3 calibration arrays; fallback to Stage 3.4 validation arrays if missing
        val_pred_path = os.path.join(MODELS_DIR, "phase3_val_predictions.npy")
        val_label_path = os.path.join(MODELS_DIR, "phase3_val_labels.npy")

        if os.path.isfile(val_pred_path) and os.path.isfile(val_label_path):
            try:
                import numpy as _np
                from trustshield_project.advanced_trust_engine import ConformalPredictor
                _p_val = _np.load(val_pred_path)
                _y_val = _np.load(val_label_path)
                cal_src = os.path.basename(val_pred_path)
                self.conformal_predictor = ConformalPredictor(alpha=0.05).calibrate(_p_val, _y_val, source=cal_src)
                logger.info(
                    "ConformalPredictor calibrated on validation data from %s "
                    "(%d samples, q_hat_0=%.4f, q_hat_1=%.4f)",
                    cal_src, len(_y_val),
                    self.conformal_predictor.q_hat_0,
                    self.conformal_predictor.q_hat_1,
                )
            except Exception as _c_exc:
                logger.warning("ConformalPredictor validation init failed: %s", _c_exc)
                self.conformal_predictor = None
        else:
            self.conformal_predictor = None

        self._loaded = True
        logger.info("Loaded Phase 3 models from %s", MODELS_DIR)
        if self.rings_df is not None:
            logger.info(
                "%d fraud rings pre-computed (%d high-risk)",
                len(self.rings_df),
                int((self.rings_df["avg_risk_score"] >= 0.3).sum()),
            )

        # ---- Explainability: Initialize TreeSHAP Explainer for Phase 3 ----
        try:
            from trustshield_project.shap_explainer import TrustShieldSHAPExplainer
            p3_cols = self.feature_meta.get("all_feature_cols", []) if self.feature_meta is not None else []
            self.shap_explainer = TrustShieldSHAPExplainer(
                model=self.combined_graph_model,
                feature_names=p3_cols,
                model_name="Combined Graph Detector (Phase 3)",
            )
            logger.info("Phase 3 TreeSHAP explainer initialized (%d features)", len(p3_cols))
        except Exception as exc:
            logger.warning("TreeSHAP Phase 3 explainer init failed: %s", exc)
            self.shap_explainer = None

        # ---- Phase 4: CLIP embedding cache (optional — feature flag controlled) ----
        clip_enabled = os.getenv("CLIP_ENABLED", "true").lower() in ("true", "1", "yes")
        clip_cache_dir = os.path.join(MODELS_DIR, "clip_cache")
        clip_meta_path = os.path.join(clip_cache_dir, "clip_meta.npz")
        if not clip_enabled:
            logger.info("Phase 4 CLIP scorer disabled via CLIP_ENABLED=false.")
            self._clip_loaded = False
        elif os.path.isfile(clip_meta_path):
            try:
                from trustshield_project.multimodal_scoring import (
                    CLIPMultimodalScorer,
                    CLIPEmbeddingCache,
                    _product_fingerprint,
                )

                catalog_parquet_path = os.path.join(MODELS_DIR, "product_catalog.parquet")
                if os.path.isfile(catalog_parquet_path):
                    self.clip_products_df = pd.read_parquet(catalog_parquet_path)
                else:
                    from trustshield_project.product_listing_generator import generate_product_catalog
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
                    logger.info(
                        "Phase 4 CLIP scorer loaded from cache (%d products, text_emb=%s, image_emb=%s)",
                        len(self.clip_scorer.id_to_idx),
                        self.clip_scorer.text_embeddings.shape,
                        self.clip_scorer.image_embeddings.shape,
                    )
                else:
                    logger.warning(
                        "Phase 4 CLIP cache fingerprint mismatch or stale. "
                        "Run scripts/build_clip_embeddings.py to refresh cache. "
                        "The /listing/analyze endpoint will use caller-supplied "
                        "multimodal_similarity_score instead."
                    )
            except Exception as exc:
                logger.warning(
                    "Phase 4 CLIP scorer unavailable (%s). "
                    "The /listing/analyze endpoint will use caller-supplied "
                    "multimodal_similarity_score instead.",
                    exc,
                )
        else:
            logger.info(
                "Phase 4 CLIP cache not found — "
                "run scripts/build_clip_embeddings.py to enable real CLIP scoring."
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
        return self.shap_explainer is not None and self.shap_explainer.is_available


# Global singleton — imported by main.py
store = ModelStore()
