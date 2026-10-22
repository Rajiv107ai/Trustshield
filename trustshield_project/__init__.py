"""Backward compatibility shim package forwarding to core."""

import core
from core import *  # noqa: F401, F403
from core.advanced_trust_engine import CanonicalTrustEngine, ConformalPredictor
from core.policy_config import RISK_POLICY, RiskPolicy

__all__ = core.__all__
