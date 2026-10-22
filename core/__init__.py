"""
TrustShield Core Intelligence Engine.

Provides:
- CanonicalTrustEngine & ConformalPredictor (advanced_trust_engine)
- Explainability attributions (shap_explainer)
- Probability calibration (calibration)
- Point-in-time causal history & graph features (point_in_time_history, graph_features)
- Multimodal CLIP scoring (multimodal_scoring)
- Specialized fraud detectors (specialized_fraud_detectors)
- Synthetic graph simulation (entity_generator, fraud_injection, order_return_generator, product_listing_generator)
"""

from .advanced_trust_engine import CanonicalTrustEngine, ConformalPredictor
from .calibration import ProbabilityCalibrator
from .policy_config import RISK_POLICY, RiskPolicy

__all__ = [
    "CanonicalTrustEngine",
    "ConformalPredictor",
    "ProbabilityCalibrator",
    "RISK_POLICY",
    "RiskPolicy",
]
