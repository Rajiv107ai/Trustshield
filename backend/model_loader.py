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

Phase 5 artifacts (optional — auto-detected):
    hybrid_model.joblib           — tabular + graph + GNN embedding XGBoost/RF
    buyer_embeddings.joblib       — dict {buyer_id  → np.array(16,)}
    seller_embeddings.joblib      — dict {seller_id → np.array(16,)}
    phase5_feature_meta.joblib    — Phase 5 feature column lists + metadata

If Phase 3 artifacts are missing, raises FileNotFoundError with instructions.
If Phase 5 artifacts are absent (e.g. torch not yet installed), the server
continues serving Phase 3 predictions without error — no configuration needed.
"""

import os
import joblib
import pandas as pd
from typing import Optional

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
        self.combined_graph_model = None
        self.fake_listing_model   = None
        self.return_fraud_model   = None
        self.rings_df: Optional[pd.DataFrame] = None
        self.feature_meta: Optional[dict] = None
        self._loaded = False

        # --- Phase 5 (optional) ---
        self.hybrid_model         = None
        self.buyer_embeddings:  dict = {}
        self.seller_embeddings: dict = {}
        self.phase5_meta: Optional[dict] = None
        self._phase5_loaded = False

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
        self._loaded = True
        print(f"[model_loader] Loaded Phase 3 models from {MODELS_DIR}")
        print(
            f"[model_loader] {len(self.rings_df)} fraud rings pre-computed "
            f"({(self.rings_df['avg_risk_score'] >= 0.3).sum()} high-risk)"
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
            self._phase5_loaded = True
            meta = self.phase5_meta
            print(
                f"[model_loader] Phase 5 hybrid model loaded  "
                f"({len(self.buyer_embeddings):,} buyer embeddings, "
                f"{len(self.seller_embeddings):,} seller embeddings, "
                f"{len(meta['hybrid_feature_cols'])} features, "
                f"classifier={meta['classifier']})"
            )
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


# Global singleton — imported by main.py
store = ModelStore()
