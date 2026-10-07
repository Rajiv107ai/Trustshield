"""Unified Trust Engine for TrustShield.

Combines component risk scores (tabular, graph, GNN, multimodal, heuristic)
into a unified calibrated risk assessment with:
- Decision routing: ALLOW, REVIEW, HOLD, BLOCK
- Information-theoretic confidence scoring (via binary Shannon entropy)
- Model disagreement metrics (divergence between component detectors)
- Deterministic explainable reason codes
- Anti-feedback-loop protection guard
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, List, Optional



class Decision(str, Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    HOLD = "HOLD"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class TrustResult:
    risk_score: float                     # [0.0, 1.0] calibrated fraud probability
    trust_score: float                    # [0.0, 100.0] where 100 is maximal trust
    confidence: float                     # [0.0, 1.0] where 1.0 is certainty
    decision: Decision                    # ALLOW, REVIEW, HOLD, BLOCK
    model_disagreement: float             # Spread between highest & lowest component risk
    reason_codes: List[str]               # Structured evidence codes
    component_risks: Dict[str, float]     # Normalized input component scores


class TrustEngine:
    """Unified Risk & Trust Scoring Engine.

    Aggregates multi-entity and multi-detector signals:
    - Tabular behavioral risk (buyer return velocity, price anomalies)
    - Graph topology risk (shared devices, ring membership, PageRank)
    - GNN node embedding risk
    - Multimodal listing anomaly risk

    Thresholds and component weights are validated on holdout data.
    """

    DEFAULT_THRESHOLDS = (0.25, 0.60, 0.85)  # (allow_cutoff, review_cutoff, hold_cutoff)

    DEFAULT_WEIGHTS = {
        "tabular_risk": 0.35,
        "graph_risk": 0.25,
        "gnn_risk": 0.20,
        "multimodal_risk": 0.10,
        "velocity_risk": 0.10,
    }

    FORBIDDEN_FEEDBACK_FEATURES = {
        "trust_score",
        "risk_score",
        "overall_fraud_probability",
        "predicted_fraud",
        "decision",
        "model_reason_code",
        "model_probability",
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: tuple[float, float, float] = DEFAULT_THRESHOLDS,
    ):
        raw_weights = weights if weights is not None else self.DEFAULT_WEIGHTS
        if not raw_weights or sum(raw_weights.values()) <= 0 or any(v < 0 for v in raw_weights.values()):
            raise ValueError(f"Invalid component weights: {raw_weights}")

        total = sum(raw_weights.values())
        self.weights = {k: float(v) / total for k, v in raw_weights.items()}
        self.allow_cutoff, self.review_cutoff, self.hold_cutoff = thresholds

        if not (0.0 <= self.allow_cutoff < self.review_cutoff < self.hold_cutoff <= 1.0):
            raise ValueError(
                f"Thresholds must be strictly increasing in [0, 1]: {thresholds}"
            )

    @staticmethod
    def calculate_confidence(risk_score: float) -> float:
        """Calculate confidence based on binary Shannon entropy.

        Confidence is 1.0 at p=0 (certain legitimate) and p=1 (certain fraud),
        and drops to 0.0 at p=0.5 (maximum uncertainty).
        """
        p = min(max(float(risk_score), 1e-12), 1.0 - 1e-12)
        entropy = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
        return float(max(0.0, min(1.0, 1.0 - entropy)))

    @staticmethod
    def validate_no_feedback_loop(feature_cols: list[str]) -> None:
        """Ensure model-generated output features are not recycled into training sets."""
        overlap = sorted(set(feature_cols) & TrustEngine.FORBIDDEN_FEEDBACK_FEATURES)
        if overlap:
            raise ValueError(
                f"Feedback loop violation: model outputs {overlap} cannot be used as features."
            )

    def score(
        self,
        component_risks: Dict[str, float],
        is_cold_start: bool = False,
        extra_reasons: Optional[List[str]] = None,
    ) -> TrustResult:
        """Combine component risks into a final TrustResult."""
        if not component_risks:
            raise ValueError("Component risks dictionary cannot be empty.")

        # Match components against defined weights (or normalize present subset)
        active_weights = {}
        for k in component_risks:
            active_weights[k] = self.weights.get(k, 1.0 / len(component_risks))

        norm_total = sum(active_weights.values())
        norm_weights = {k: v / norm_total for k, v in active_weights.items()}

        composite_risk = sum(
            norm_weights[k] * float(min(max(component_risks[k], 0.0), 1.0))
            for k in component_risks
        )
        composite_risk = float(min(max(composite_risk, 0.0), 1.0))

        # Trust score on 0-100 scale
        trust_score = round(100.0 * (1.0 - composite_risk), 2)

        # Confidence with cold-start penalty
        base_confidence = self.calculate_confidence(composite_risk)
        if is_cold_start:
            # Reduce confidence for entities with < 3 prior interactions
            confidence = round(base_confidence * 0.70, 4)
        else:
            confidence = round(base_confidence, 4)

        # Model disagreement
        vals = [float(v) for v in component_risks.values()]
        model_disagreement = round(float(max(vals) - min(vals)), 4) if len(vals) > 1 else 0.0

        # Decision routing
        if composite_risk >= self.hold_cutoff:
            decision = Decision.BLOCK
        elif composite_risk >= self.review_cutoff:
            decision = Decision.HOLD
        elif composite_risk >= self.allow_cutoff:
            decision = Decision.REVIEW
        else:
            decision = Decision.ALLOW

        # Reason code generation
        reasons: List[str] = list(extra_reasons or [])
        if is_cold_start:
            reasons.append("COLD_START_INSUFFICIENT_HISTORY")

        if model_disagreement > 0.40:
            reasons.append("HIGH_DETECTOR_DISAGREEMENT")

        for k, v in sorted(component_risks.items(), key=lambda x: x[1], reverse=True):
            if v >= 0.75:
                reasons.append(f"CRITICAL_{k.upper()}_ELEVATED ({v:.2f})")
            elif v >= 0.50:
                reasons.append(f"WARN_{k.upper()}_MODERATE ({v:.2f})")

        return TrustResult(
            risk_score=round(composite_risk, 4),
            trust_score=trust_score,
            confidence=confidence,
            decision=decision,
            model_disagreement=model_disagreement,
            reason_codes=reasons,
            component_risks={k: round(float(v), 4) for k, v in component_risks.items()},
        )
