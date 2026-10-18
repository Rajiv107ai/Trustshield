"""TrustShield AI — Class-Conditional (Mondrian) Conformal Prediction Module.

Provides distribution-free finite-sample error rate guarantees (1 - alpha)
and non-empty prediction sets ({0}, {1}, or {0, 1}) for transaction risk evaluation.
"""

from __future__ import annotations

import math
from typing import Set
import numpy as np

# Canonical implementation lives in advanced_trust_engine; re-export for clean modularity
from trustshield_project.advanced_trust_engine import ConformalPredictor

__all__ = ["ConformalPredictor"]
