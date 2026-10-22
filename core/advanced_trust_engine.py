"""Advanced Trust Engine with Conformal Uncertainty for TrustShield.

Routes operational decisions directly via calibrated model probabilities and cost thresholds:
- Eliminates hand-tuned linear blending heuristics and uncalibrated stacking meta-learners
- Direct calibrated probability pass-through preserves empirical calibration guarantees
- Mondrian class-conditional conformal prediction intervals (guaranteed 1 - alpha coverage)
- Graph and ring intelligence signals surfaced strictly as explanatory reason codes
- Automated abstention and human review routing on ambiguous prediction sets
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass
import numpy as np

try:
    from trustshield_project.policy_config import RISK_POLICY
except ImportError:  # pragma: no cover - direct legacy module execution
    from policy_config import RISK_POLICY  # type: ignore


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

    @property
    def risk_score(self) -> float:
        return self.calibrated_risk

    @property
    def model_disagreement(self) -> float:
        return self.detector_disagreement

    @property
    def component_risks(self) -> Dict[str, float]:
        return self.detector_scores


class ConformalPredictor:
    """Class-conditional (Mondrian) Conformal Prediction for distribution-free finite-sample error guarantees."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # Target error rate per class (0.05 for 95% nominal coverage)
        self.q_hat_0: float = 0.5
        self.q_hat_1: float = 0.5
        self.is_calibrated: bool = False
        self.n_cal_0: int = 0
        self.n_cal_1: int = 0
        self.calibration_source: str = "uncalibrated_heuristic"

    @property
    def q_hat(self) -> float:
        """Backward-compatible single scalar quantile (class 1 fraud quantile)."""
        return self.q_hat_1

    @q_hat.setter
    def q_hat(self, val: float):
        self.q_hat_1 = float(val)

    @staticmethod
    def compute_finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
        """Computes the exact finite-sample conformal quantile:
        k = ceil((n + 1) * (1 - alpha)).
        The critical threshold is the k-th smallest value in the sorted non-conformity scores.
        """
        n = len(scores)
        if n == 0:
            return 1.0
        k = math.ceil((n + 1) * (1.0 - alpha))
        if k > n:
            return 1.0
        sorted_scores = np.sort(scores)
        return float(sorted_scores[k - 1])

    def calibrate(self, val_probs: np.ndarray, val_labels: np.ndarray, source: str = "validation_cohort") -> "ConformalPredictor":
        """Compute class-conditional (Mondrian) conformal quantiles on held-out calibration set."""
        p = np.asarray(val_probs, dtype=float)
        y = np.asarray(val_labels, dtype=int)
        mask_0 = (y == 0)
        mask_1 = (y == 1)
        self.n_cal_0 = int(mask_0.sum())
        self.n_cal_1 = int(mask_1.sum())
        if self.n_cal_0 == 0 or self.n_cal_1 == 0:
            return self

        # Non-conformity scores:
        # Class 0: non-conformity is predicted fraud probability p (higher p is worse for non-fraud)
        scores_0 = p[mask_0]
        self.q_hat_0 = self.compute_finite_sample_quantile(scores_0, self.alpha)

        # Class 1: non-conformity is predicted non-fraud probability 1 - p (lower p is worse for fraud)
        scores_1 = 1.0 - p[mask_1]
        self.q_hat_1 = self.compute_finite_sample_quantile(scores_1, self.alpha)

        self.is_calibrated = True
        self.calibration_source = source
        return self

    def predict_set(self, prob: float) -> Set[int]:
        """Return prediction set {0}, {1}, or {0, 1} under Mondrian error guarantees."""
        if not self.is_calibrated:
            return {0, 1} if 0.35 <= prob <= 0.65 else ({1} if prob > 0.65 else {0})

        pred_set = set()
        # Include 0 if non-conformity score p <= q_hat_0
        if prob <= self.q_hat_0:
            pred_set.add(0)
        # Include 1 if non-conformity score 1 - p <= q_hat_1 <=> p >= 1 - q_hat_1
        if (1.0 - prob) <= self.q_hat_1:
            pred_set.add(1)

        # Non-empty fallback
        if not pred_set:
            pred_set.add(int(prob >= 0.5))
        return pred_set


class AdvancedTrustEngine:
    """Authoritative Trust & Decision Engine for TrustShield.
    
    Evaluates calibrated probabilities directly against cost thresholds,
    bounds epistemic risk with Mondrian conformal prediction sets,
    and surfaces graph/device sharing signals purely as explanatory reason codes.
    """

    DEFAULT_THRESHOLDS = (RISK_POLICY.review, RISK_POLICY.hold, RISK_POLICY.block)

    def __init__(
        self,
        meta_learner: Optional[Any] = None,
        conformal_predictor: Optional[ConformalPredictor] = None,
        weights: Optional[Dict[str, float]] = None,
        thresholds: tuple[float, float, float] = DEFAULT_THRESHOLDS,
        **kwargs,
    ):
        if weights is not None:
            if not weights or sum(weights.values()) <= 0 or any(v < 0 for v in weights.values()):
                raise ValueError(f"Invalid component weights: {weights}")
            total_w = sum(weights.values())
            self.weights = dict(weights)
            self.norm_weights = {k: v / total_w for k, v in weights.items()}
        else:
            self.weights = {
                "tabular_risk": 0.40,
                "graph_risk_heuristic": 0.25,
                "ring_risk_heuristic": 0.15,
                "multimodal_risk": 0.20,
            }
            total_w = sum(self.weights.values())
            self.norm_weights = {k: v / total_w for k, v in self.weights.items()}

        if not (0.0 <= thresholds[0] < thresholds[1] < thresholds[2] <= 1.0):
            raise ValueError(f"Thresholds must be strictly increasing in [0, 1]: {thresholds}")

        self.conformal = conformal_predictor or ConformalPredictor(alpha=0.05)
        self.allow_th, self.review_th, self.hold_th = thresholds

    @property
    def allow_cutoff(self) -> float:
        return self.allow_th

    @property
    def review_cutoff(self) -> float:
        return self.review_th

    @property
    def hold_cutoff(self) -> float:
        return self.hold_th

    @staticmethod
    def calculate_confidence(risk_score: float) -> float:
        p = min(max(float(risk_score), 1e-12), 1.0 - 1e-12)
        entropy = float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))
        return round(1.0 - entropy, 4)

    FORBIDDEN_FEEDBACK_FEATURES = {
        "trust_score",
        "risk_score",
        "overall_fraud_probability",
        "predicted_fraud",
        "decision",
        "model_reason_code",
        "model_probability",
    }

    @classmethod
    def validate_no_feedback_loop(cls, feature_names: List[str]) -> None:
        violations = cls.FORBIDDEN_FEEDBACK_FEATURES.intersection(feature_names)
        if violations:
            raise ValueError(f"Feedback loop violation: features contain model outputs: {violations}")

    def aggregate_risk(self, detector_scores: Dict[str, float]) -> Tuple[float, str]:
        """Direct calibrated probability pass-through without heuristic corruption.
        
        Preserves calibrated model probabilities when tabular_risk is present;
        falls back to normalized component weights only when primary model is absent.
        """
        if "tabular_risk" in detector_scores:
            risk = float(detector_scores["tabular_risk"])
            method = "calibrated_probability_pass_through"
        elif detector_scores and hasattr(self, "norm_weights"):
            active_weights = {k: self.norm_weights.get(k, 0.0) for k in detector_scores}
            w_sum = sum(active_weights.values())
            risk = sum((w / w_sum) * float(detector_scores[k]) for k, w in active_weights.items()) if w_sum > 0 else float(next(iter(detector_scores.values())))
            method = "weighted_detector_fallback"
        elif detector_scores:
            risk = float(next(iter(detector_scores.values())))
            method = "single_detector_fallback"
        else:
            risk = 0.0
            method = "zero_risk_default"
        return round(float(np.clip(risk, 0.0, 1.0)), 4), method

    def score(
        self,
        detector_scores: Dict[str, float],
        is_cold_start: bool = False,
        model_probability: Optional[float] = None,
        order_amount: Optional[float] = None,
    ) -> AdvancedTrustResult:
        if not detector_scores and model_probability is None:
            raise ValueError("detector_scores dictionary cannot be empty when model_probability is not provided.")

        # Authoritative calibrated probability pass-through (never diluted by heuristic blending)
        if model_probability is not None:
            risk = float(model_probability)
            agg_method = "calibrated_probability_pass_through"
        elif "tabular_risk" in detector_scores:
            risk = float(detector_scores["tabular_risk"])
            agg_method = "calibrated_probability_pass_through"
        elif detector_scores and hasattr(self, "norm_weights"):
            active_weights = {k: self.norm_weights.get(k, 0.0) for k in detector_scores}
            w_sum = sum(active_weights.values())
            risk = sum((w / w_sum) * float(detector_scores[k]) for k, w in active_weights.items()) if w_sum > 0 else float(next(iter(detector_scores.values())))
            agg_method = "weighted_detector_fallback"
        elif detector_scores:
            risk = float(next(iter(detector_scores.values())))
            agg_method = "calibrated_probability_pass_through"
        else:
            risk = 0.0
            agg_method = "calibrated_probability_pass_through"

        risk = round(float(np.clip(risk, 0.0, 1.0)), 4)
        trust_score = round(100.0 * (1.0 - risk), 2)

        # 1. Predictive Uncertainty (Shannon Entropy)
        p = min(max(risk, 1e-12), 1.0 - 1e-12)
        entropy = float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))
        confidence = round(1.0 - entropy, 4)
        if is_cold_start:
            confidence = round(confidence * 0.70, 4)

        # 2. Epistemic Model Disagreement (for explainability & inspection)
        vals = list(detector_scores.values()) if detector_scores else [risk]
        disagreement = round(float(max(vals) - min(vals)), 4) if len(vals) > 1 else 0.0
        variance = round(float(np.var(vals)), 6) if len(vals) > 1 else 0.0

        # 3. Conformal Prediction Set (Distribution-Free Finite-Sample Guarantee)
        pred_set = self.conformal.predict_set(risk)

        # 4. Reason Codes (Graph & Ring Signals strictly as explanatory evidence)
        reasons: List[str] = []
        if is_cold_start:
            reasons.append("COLD_START_INSUFFICIENT_HISTORY")
        if len(pred_set) > 1 and risk >= 0.15:
            reasons.append("HIGH_CONFORMAL_UNCERTAINTY_SET_{0,1}")
        if disagreement >= 0.40:
            reasons.append(f"HIGH_DETECTOR_DISAGREEMENT (spread={disagreement:.2f})")

        for k, v in sorted(detector_scores.items(), key=lambda x: x[1], reverse=True):
            if v >= 0.75:
                reasons.append(f"CRITICAL_{k.upper()}_ELEVATED ({v:.2f})")
            elif v >= 0.50:
                reasons.append(f"WARN_{k.upper()}_MODERATE ({v:.2f})")

        # 5. Operational Decision Routing via Cost-Optimal Thresholds & Conformal Set
        # Cost-loss: flag for review if p >= C_review / (C_review + order_amount)
        cost_review_flag = False
        if order_amount is not None and order_amount > 0:
            c_review = 20.0  # Estimated manual human review cost ($20)
            cost_threshold = c_review / (c_review + order_amount)
            if risk >= cost_threshold and risk >= 0.05:
                cost_review_flag = True

        if risk >= self.hold_th:
            decision = OperationalDecision.BLOCK
        elif risk >= self.review_th:
            decision = OperationalDecision.HOLD
        elif (len(pred_set) > 1 and risk >= 0.15) or disagreement > 0.45 or cost_review_flag:
            decision = OperationalDecision.REVIEW
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


# Canonical alias for the unified serving Trust Engine
CanonicalTrustEngine = AdvancedTrustEngine

# Backward-compatible aliases for legacy imports
TrustEngine = CanonicalTrustEngine
TrustResult = AdvancedTrustResult
Decision = OperationalDecision
