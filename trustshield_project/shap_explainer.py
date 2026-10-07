"""
TrustShield AI — SHAP (SHapley Additive exPlanations) Explainability Engine.

Provides mathematically grounded, local and global feature attributions for
tree-based fraud models (Phase 3 Combined Graph Model, Phase 5 Hybrid XGBoost,
Phase 2 Fake Listing & Return Fraud models).

Architecture:
1. Fast TreeExplainer computing exact local Shapley values in sub-10ms.
2. Identifies top positive risk contributors (amplifying fraud likelihood)
   and top negative risk dampeners (protective trust factors).
3. Synthesizes evidence-grounded natural language narratives for fraud operations.
4. Integrates with Canonical Trust Engine and Forensic Dossier API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    shap = None
    _SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


# Feature human-readable display names and units
FEATURE_FRIENDLY_NAMES: Dict[str, str] = {
    "share_degree": "Hardware / Address Sharing Degree",
    "share_component_size": "Fraud Ring Component Size",
    "buyer_return_rate_before": "Historical Buyer Return Rate",
    "buyer_returns_before": "Prior Return Count",
    "buyer_orders_before": "Prior Completed Orders",
    "price_vs_base_price_ratio": "Price vs Catalog Base Price Ratio",
    "price_vs_category_median_ratio": "Price vs Category Median Ratio",
    "device_shared_buyer_count": "Device Collision Count (Shared Buyers)",
    "seller_total_listings_before": "Seller Listing Catalog Breadth",
    "seller_age_days": "Seller Account Age (Days)",
    "buyer_age_days": "Buyer Account Age (Days)",
    "amount": "Order Dollar Amount",
    "order_amount": "Order Dollar Amount",
    "buyer_pagerank": "Buyer Graph Centrality (PageRank)",
    "seller_pagerank": "Seller Graph Centrality (PageRank)",
    "buyer_seller_degree": "Buyer-Seller Graph Connectivity",
    "seller_buyer_degree": "Seller-Buyer Graph Connectivity",
    "seller_buyer_concentration_hhi": "Merchant-Buyer Concentration Index (HHI)",
    "buyer_seller_edge_weight_before": "Historical Repeat Transaction Count",
    "multimodal_similarity_score": "Visual-Semantic Listing Consistency (CLIP)",
}


class TrustShieldSHAPExplainer:
    """Production TreeSHAP explainer with feature alignment and narrative synthesis."""

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str = "Combined Graph Detector",
    ):
        self.model = model
        self.feature_names = list(feature_names)
        self.model_name = model_name
        self._explainer: Optional[Any] = None

    @property
    def is_available(self) -> bool:
        return _SHAP_AVAILABLE and self.model is not None

    def _get_explainer(self) -> Any:
        if self._explainer is None:
            if not _SHAP_AVAILABLE or shap is None:
                raise RuntimeError("SHAP library is not installed. Install with `pip install shap`.")
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def explain_instance(
        self,
        features: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Compute exact Shapley values and plain-English narrative for a single transaction.

        Parameters
        ----------
        features : dict or pd.Series
            Transaction features. Missing features are automatically zero-filled.
        top_k : int
            Number of top positive and negative drivers to extract.

        Returns
        -------
        dict containing:
            - base_value: expected model margin / logit
            - attributions: Dict[feature_name, shap_value]
            - top_positive_drivers: List[dict] (features increasing fraud risk)
            - top_negative_dampeners: List[dict] (features reducing fraud risk)
            - narrative: str (plain-English summary for investigators)
        """
        if not self.is_available:
            return {
                "available": False,
                "base_value": 0.0,
                "attributions": {},
                "top_positive_drivers": [],
                "top_negative_dampeners": [],
                "narrative": "SHAP explainability unavailable (model or shap dependency missing).",
            }

        # Align features into single row DataFrame
        if isinstance(features, dict):
            row_data = {col: float(features.get(col, 0.0) or 0.0) for col in self.feature_names}
            X = pd.DataFrame([row_data])[self.feature_names]
        elif isinstance(features, pd.Series):
            row_data = {col: float(features.get(col, 0.0) or 0.0) for col in self.feature_names}
            X = pd.DataFrame([row_data])[self.feature_names]
        elif isinstance(features, pd.DataFrame):
            X = features.reindex(columns=self.feature_names, fill_value=0.0).iloc[[0]]
        else:
            raise ValueError(f"Unsupported features type: {type(features)}")

        explainer = self._get_explainer()
        raw_shap = explainer.shap_values(X)

        # Handle binary classification shapes across XGBoost and scikit-learn
        if isinstance(raw_shap, list):
            # RandomForestClassifier returns [class_0, class_1]
            phi = np.asarray(raw_shap[1])[0]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            phi = raw_shap[0, :, 1]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
            phi = raw_shap[0]
        else:
            phi = np.asarray(raw_shap).flatten()

        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base_val = float(ev[1] if len(ev) > 1 else ev[0])
        else:
            base_val = float(ev)

        # Build feature attribution dictionary
        attributions: Dict[str, float] = {}
        for col, val in zip(self.feature_names, phi):
            attributions[col] = round(float(val), 4)

        # Classify into risk drivers (+) and trust dampeners (-)
        items = []
        for col in self.feature_names:
            feat_val = float(X.iloc[0][col])
            shap_val = attributions[col]
            friendly = FEATURE_FRIENDLY_NAMES.get(col)
            if not friendly:
                if col.startswith("gnn_buyer_emb_"):
                    friendly = f"Buyer GNN Latent Topology (Dim {col.replace('gnn_buyer_emb_', '')})"
                elif col.startswith("gnn_seller_emb_"):
                    friendly = f"Seller GNN Latent Topology (Dim {col.replace('gnn_seller_emb_', '')})"
                else:
                    friendly = col.replace("_", " ").title()
            items.append({
                "feature_name": col,
                "friendly_name": friendly,
                "feature_value": round(feat_val, 4),
                "shap_value": shap_val,
                "abs_impact": abs(shap_val),
            })

        pos_drivers = sorted([it for it in items if it["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:top_k]
        neg_dampeners = sorted([it for it in items if it["shap_value"] < 0], key=lambda x: x["shap_value"])[:top_k]

        # Generate Investigator Narrative
        narrative = self._synthesize_narrative(pos_drivers, neg_dampeners, base_val)

        return {
            "available": True,
            "base_value": round(base_val, 4),
            "attributions": attributions,
            "top_positive_drivers": pos_drivers,
            "top_negative_dampeners": neg_dampeners,
            "narrative": narrative,
        }

    def _synthesize_narrative(
        self,
        pos_drivers: List[Dict[str, Any]],
        neg_dampeners: List[Dict[str, Any]],
        base_val: float,
    ) -> str:
        """Translate Shapley values into a concise, professional investigator summary."""
        parts = []

        if pos_drivers:
            top_reasons = []
            for d in pos_drivers[:3]:
                top_reasons.append(f"{d['friendly_name']} ({d['feature_value']}, SHAP: +{d['shap_value']:.3f})")
            parts.append("Elevated risk is primarily propelled by: " + "; ".join(top_reasons) + ".")
        else:
            parts.append("No strong positive fraud risk factors were observed.")

        if neg_dampeners:
            top_mitigations = []
            for d in neg_dampeners[:2]:
                top_mitigations.append(f"{d['friendly_name']} ({d['feature_value']}, SHAP: {d['shap_value']:.3f})")
            parts.append("Risk is mitigated by legitimate signals: " + "; ".join(top_mitigations) + ".")

        return " ".join(parts)
