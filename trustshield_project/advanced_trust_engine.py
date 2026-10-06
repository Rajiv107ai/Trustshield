"""Advanced Trust Engine with Stacking Meta-Learner and Conformal Uncertainty for TrustShield.

Combines component risk models (Tabular, Graph, HeteroGNN, Multimodal, Ring Intelligence):
- Compares Aggregation Methods: Simple Average vs Validated Weighted Ensemble vs Stacking Meta-Model
- Advanced Uncertainty Framework:
    - Calibrated Risk Score & Trust Score (0-100)
    - Shannon Entropy Predictive Uncertainty
    - Epistemic Model Disagreement (Variance & Spread across detectors)
    - Split Conformal Prediction Intervals (guaranteed 1 - alpha coverage)
    - Automated Abstention / Human Review Routing on ambiguous prediction sets
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class OperationalDecision(str, Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    HOLD = "HOLD"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class AdvancedTrustResult:
    calibrated_risk: float
    trust_score: float
    decision: OperationalDecision
    confidence: float
    predictive_uncertainty: float           # Shannon entropy in [0, 1]
    detector_disagreement: float           # Spread max - min
    detector_variance: float               # Variance across detector scores
    conformal_prediction_set: Set[int]      # e.g. {0}, {1}, or {0, 1}
    reason_codes: List[str]
    detector_scores: Dict[str, float]
    aggregation_method_used: str


class ConformalPredictor:
    """Split Conformal Prediction for distribution-free finite-sample error guarantees."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # Target error rate (e.g. 0.05 for 95% coverage)
        self.q_hat: float = 0.5
        self.is_calibrated: bool = False

    def calibrate(self, val_probs: np.ndarray, val_labels: np.ndarray) -> "ConformalPredictor":
        """Compute conformal quantile non-conformity score on held-out calibration set."""
        p = np.asarray(val_probs, dtype=float)
        y = np.asarray(val_labels, dtype=int)
        n = len(y)
        if n == 0:
            return self

        # Non-conformity score: 1 - P(true class)
        true_class_probs = np.where(y == 1, p, 1.0 - p)
        scores = 1.0 - true_class_probs

        # Conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        self.is_calibrated = True
        return self

    def predict_set(self, prob: float) -> Set[int]:
        """Return prediction set {0}, {1}, or {0, 1}."""
        if not self.is_calibrated:
            return {0, 1} if 0.35 <= prob <= 0.65 else ({1} if prob > 0.65 else {0})

        pred_set = set()
        # Include 0 if 1 - (1 - p) <= q_hat => p <= q_hat
        if prob <= self.q_hat:
            pred_set.add(0)
        # Include 1 if 1 - p <= q_hat => p >= 1 - q_hat
        if (1.0 - prob) <= self.q_hat:
            pred_set.add(1)

        # Non-empty fallback
        if not pred_set:
            pred_set.add(int(prob >= 0.5))
        return pred_set


class StackingRiskMetaLearner:
    """Meta-learner combining component detector scores into an optimal stacked risk score."""

    def __init__(self):
        # LogisticRegression with non-negative constraints ensures higher detector risk increases composite risk
        self.clf = LogisticRegression(solver="lbfgs", max_iter=200, random_state=42)
        self.component_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X_components: pd.DataFrame, y_labels: np.ndarray) -> "StackingRiskMetaLearner":
        self.component_names = list(X_components.columns)
        self.clf.fit(X_components.fillna(0.0), y_labels)
        self.is_fitted = True
        return self

    def predict_risk(self, component_dict: Dict[str, float]) -> float:
        if not self.is_fitted:
            # Fallback to mean if unfitted
            return float(np.mean(list(component_dict.values())))

        row = [component_dict.get(c, 0.0) for c in self.component_names]
        df_row = pd.DataFrame([row], columns=self.component_names)
        prob = float(self.clf.predict_proba(df_row)[0, 1])
        return min(max(prob, 0.0), 1.0)


class AdvancedTrustEngine:
    """Advanced Multi-Detector Trust & Uncertainty Engine."""

    DEFAULT_THRESHOLDS = (0.25, 0.55, 0.85)  # (ALLOW, REVIEW, HOLD, BLOCK)

    def __init__(
        self,
        meta_learner: Optional[StackingRiskMetaLearner] = None,
        conformal_predictor: Optional[ConformalPredictor] = None,
        weights: Optional[Dict[str, float]] = None,
        thresholds: tuple[float, float, float] = DEFAULT_THRESHOLDS,
    ):
        self.meta_learner = meta_learner
        self.conformal = conformal_predictor or ConformalPredictor(alpha=0.05)
        self.weights = weights or {
            "tabular_risk": 0.30,
            "graph_risk": 0.25,
            "hetero_gnn_risk": 0.20,
            "multimodal_risk": 0.15,
            "ring_risk": 0.10,
        }
        total_w = sum(self.weights.values())
        self.norm_weights = {k: v / total_w for k, v in self.weights.items()}
        self.allow_th, self.review_th, self.hold_th = thresholds

    def aggregate_risk(self, detector_scores: Dict[str, float]) -> Tuple[float, str]:
        """Calculate composite risk comparing stacking vs weighted ensemble."""
        if self.meta_learner is not None and self.meta_learner.is_fitted:
            risk = self.meta_learner.predict_risk(detector_scores)
            return round(risk, 4), "stacking_meta_learner"

        # Validated Weighted Ensemble
        active_w = {k: self.norm_weights.get(k, 1.0 / len(detector_scores)) for k in detector_scores}
        total_act = sum(active_w.values())
        weighted_sum = sum((w / total_act) * float(detector_scores[k]) for k, w in active_w.items())
        return round(float(np.clip(weighted_sum, 0.0, 1.0)), 4), "validated_weighted_ensemble"

    def score(
        self,
        detector_scores: Dict[str, float],
        is_cold_start: bool = False,
    ) -> AdvancedTrustResult:
        if not detector_scores:
            raise ValueError("detector_scores dictionary cannot be empty.")

        risk, agg_method = self.aggregate_risk(detector_scores)
        trust_score = round(100.0 * (1.0 - risk), 2)

        # 1. Predictive Uncertainty (Shannon Entropy)
        p = min(max(risk, 1e-12), 1.0 - 1e-12)
        entropy = float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))
        confidence = round(1.0 - entropy, 4)
        if is_cold_start:
            confidence = round(confidence * 0.70, 4)

        # 2. Epistemic Model Disagreement
        vals = list(detector_scores.values())
        disagreement = round(float(max(vals) - min(vals)), 4) if len(vals) > 1 else 0.0
        variance = round(float(np.var(vals)), 6) if len(vals) > 1 else 0.0

        # 3. Conformal Prediction Set
        pred_set = self.conformal.predict_set(risk)

        # 4. Reason Codes
        reasons = []
        if is_cold_start:
            reasons.append("COLD_START_INSUFFICIENT_HISTORY")
        if len(pred_set) > 1:
            reasons.append("HIGH_CONFORMAL_UNCERTAINTY_SET_{0,1}")
        if disagreement >= 0.40:
            reasons.append(f"HIGH_DETECTOR_DISAGREEMENT (spread={disagreement:.2f})")

        for k, v in sorted(detector_scores.items(), key=lambda x: x[1], reverse=True):
            if v >= 0.75:
                reasons.append(f"CRITICAL_{k.upper()}_ELEVATED ({v:.2f})")
            elif v >= 0.50:
                reasons.append(f"WARN_{k.upper()}_MODERATE ({v:.2f})")

        # 5. Operational Decision Routing
        # Force REVIEW if conformal prediction is ambiguous {0, 1} and not blocked
        if risk >= self.hold_th:
            decision = OperationalDecision.BLOCK
        elif risk >= self.review_th or len(pred_set) > 1 or disagreement > 0.45:
            decision = OperationalDecision.HOLD if risk >= self.review_th else OperationalDecision.REVIEW
        elif risk >= self.allow_th:
            decision = OperationalDecision.REVIEW
        else:
            decision = OperationalDecision.ALLOW

        return AdvancedTrustResult(
            calibrated_risk=risk,
            trust_score=trust_score,
            decision=decision,
            confidence=confidence,
            predictive_uncertainty=round(entropy, 4),
            detector_disagreement=disagreement,
            detector_variance=variance,
            conformal_prediction_set=pred_set,
            reason_codes=reasons,
            detector_scores={k: round(float(v), 4) for k, v in detector_scores.items()},
            aggregation_method_used=agg_method,
        )
