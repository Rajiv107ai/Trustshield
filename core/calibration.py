"""Probability Calibration and Cost-Optimal Threshold Optimization for TrustShield.

Raw classification scores from decision trees, gradient boosting, and neural networks
frequently suffer from distortion (e.g. over-confidence or squashed sigmoid probabilities).
This module provides:
- ProbabilityCalibrator using isotonic regression or Platt sigmoid scaling (fit strictly on validation data)
- Expected Calibration Error (ECE) and Brier Score metrics
- Cost-sensitive threshold optimizer balancing False Negatives vs False Positives
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
from sklearn.isotonic import IsotonicRegression



def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared difference between predicted probabilities and actual binary outcomes.

    Lower is better: 0.0 indicates perfect calibration.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def calculate_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Calculate Expected Calibration Error (ECE) across uniform confidence bins."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (p >= bin_lower) & (p < bin_upper) if i < n_bins - 1 else (p >= bin_lower) & (p <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(y[in_bin])
            bin_conf = np.mean(p[in_bin])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


class ProbabilityCalibrator:
    """Calibrator for post-processing model risk scores on held-out validation data."""

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"Unknown calibration method: {method}. Choose 'isotonic' or 'sigmoid'.")
        self.method = method
        self.iso_reg: Optional[IsotonicRegression] = None
        self.platt_a: float = 1.0
        self.platt_b: float = 0.0
        self.is_fitted: bool = False

    def fit(self, scores: np.ndarray, y_val: np.ndarray) -> "ProbabilityCalibrator":
        """Fit calibration mapping strictly on validation split."""
        s = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(y_val, dtype=float).ravel()

        if self.method == "isotonic":
            self.iso_reg = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.iso_reg.fit(s, y)
        else:
            # Platt Sigmoid: logistic regression on scores
            # Simple analytical gradient descent for Platt parameters
            lr = 0.05
            epochs = 1000
            a, b = 1.0, 0.0
            for _ in range(epochs):
                z = np.clip(a * s + b, -40.0, 40.0)
                p = 1.0 / (1.0 + np.exp(-z))
                err = p - y
                a -= lr * np.mean(err * s)
                b -= lr * np.mean(err)
            self.platt_a = a
            self.platt_b = b

        self.is_fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Transform raw model scores into calibrated probabilities in [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before predict_proba.")

        s = np.asarray(scores, dtype=float)
        if self.method == "isotonic" and self.iso_reg is not None:
            calibrated = self.iso_reg.predict(s.ravel())
            return np.clip(calibrated.reshape(s.shape), 0.0, 1.0)
        else:
            z = np.clip(self.platt_a * s + self.platt_b, -40.0, 40.0)
            return np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0)


def optimize_cost_sensitive_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    n_candidates: int = 100,
) -> Dict[str, Any]:
    """Find the threshold t that minimizes total operational loss:

    Loss(t) = C_fn * FN(t) + C_fp * FP(t)

    In fraud detection, False Negatives (missed fraud) are typically 3x to 10x
    more expensive than False Positives (manual reviewer cost).
    """
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(scores, dtype=float).ravel()

    thresholds = np.linspace(0.01, 0.99, n_candidates)
    best_cost = float("inf")
    best_metrics = {}

    for t in thresholds:
        pred = (s >= t).astype(int)
        fn = int(np.sum((y == 1) & (pred == 0)))
        fp = int(np.sum((y == 0) & (pred == 1)))
        tp = int(np.sum((y == 1) & (pred == 1)))
        tn = int(np.sum((y == 0) & (pred == 0)))

        cost = fn_cost * fn + fp_cost * fp
        if cost < best_cost:
            best_cost = float(cost)
            best_metrics = {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "cost": float(cost),
                "threshold": float(t),
            }

    return best_metrics


def optimize_dollar_weighted_cost_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    fp_cost: float = 20.0,
    n_candidates: int = 200,
) -> Dict[str, Any]:
    """Finds the decision threshold t that minimizes total net business loss:

    Net Loss(t) = sum_{missed fraud} amount_i + sum_{false alarms} fp_cost

    Parameters
    ----------
    y_true:
        Binary ground-truth labels (1 = fraud, 0 = legit).
    scores:
        Predicted fraud probabilities or continuous risk scores.
    amounts:
        Order values (e.g. INR / USD) per transaction.
    fp_cost:
        Estimated human review friction cost per false positive (default: 20.0).
    n_candidates:
        Number of candidate thresholds evaluated.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    amt = np.asarray(amounts, dtype=float).ravel()

    total_fraud_value = float(np.sum(amt[y == 1])) if np.any(y == 1) else 1e-6
    thresholds = np.linspace(0.01, 0.99, n_candidates)
    best_cost = float("inf")
    best_metrics: Dict[str, Any] = {}

    for t in thresholds:
        pred = (s >= t).astype(int)
        fn_mask = (y == 1) & (pred == 0)
        fp_mask = (y == 0) & (pred == 1)
        tp_mask = (y == 1) & (pred == 1)

        fn_loss = float(np.sum(amt[fn_mask]))
        fp_friction = float(fp_cost * np.sum(fp_mask))
        total_loss = fn_loss + fp_friction

        if total_loss < best_cost:
            best_cost = total_loss
            fraud_value_saved = float(np.sum(amt[tp_mask]))
            best_metrics = {
                "threshold": round(float(t), 4),
                "total_loss": round(total_loss, 2),
                "fraud_value_lost": round(fn_loss, 2),
                "review_friction_cost": round(fp_friction, 2),
                "fraud_value_saved": round(fraud_value_saved, 2),
                "value_recall_pct": round(100.0 * (fraud_value_saved / total_fraud_value), 2),
                "transaction_recall_pct": round(100.0 * (np.sum(tp_mask) / max(np.sum(y == 1), 1)), 2),
                "review_volume_pct": round(100.0 * (np.sum(pred == 1) / len(y)), 2),
            }

    return best_metrics


def compute_cost_weighted_sample_weights(
    y_train: np.ndarray,
    amounts: np.ndarray,
    base_fp_cost: float = 20.0,
    max_weight_multiplier: float = 10.0,
) -> np.ndarray:
    """Computes cost-asymmetric sample weights for gradient boosted trees or random forests.

    Penalizes high-dollar fraud proportionally to transaction amount so that trees
    prioritize splitting on high-value fraud instances.
    """
    y = np.asarray(y_train, dtype=int).ravel()
    amt = np.asarray(amounts, dtype=float).ravel()
    weights = np.ones(len(y), dtype=np.float32)

    fraud_mask = (y == 1)
    if np.any(fraud_mask):
        scaled = np.clip(amt[fraud_mask] / max(base_fp_cost, 1.0), 1.0, max_weight_multiplier)
        weights[fraud_mask] = scaled.astype(np.float32)

    return weights


class CapacityBudgetCalibrator:
    """Calibrates operational decision thresholds against fixed reviewer capacity budgets.

    In real-world Trust & Safety teams, investigator bandwidth is bounded (e.g. 1%, 2%, 5%, 10%
    of daily transaction volume). This calibrator finds the exact empirical thresholds
    on held-out validation cohorts and reports expected recovery metrics.
    """

    def __init__(self, target_budgets: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)):
        self.target_budgets = target_budgets
        self.calibrated_thresholds: Dict[float, float] = {}
        self.budget_metrics: Dict[float, Dict[str, Any]] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        val_scores: np.ndarray,
        val_labels: np.ndarray,
        val_amounts: Optional[np.ndarray] = None,
    ) -> "CapacityBudgetCalibrator":
        """Fits budget thresholds strictly on validation data."""
        s = np.asarray(val_scores, dtype=float).ravel()
        y = np.asarray(val_labels, dtype=int).ravel()
        amt = np.asarray(val_amounts, dtype=float).ravel() if val_amounts is not None else np.ones(len(y))

        total_fraud_count = int(np.sum(y == 1))
        total_fraud_amt = float(np.sum(amt[y == 1])) if total_fraud_count > 0 else 1.0

        for budget in self.target_budgets:
            # Capacity constraint: review at most budget * n_samples transactions
            # Critical threshold is the (1 - budget) quantile of validation scores
            quantile_val = 1.0 - budget
            threshold = float(np.quantile(s, np.clip(quantile_val, 0.0, 1.0)))

            pred = (s >= threshold).astype(int)
            tp_mask = (y == 1) & (pred == 1)
            fp_mask = (y == 0) & (pred == 1)

            tp = int(np.sum(tp_mask))
            fp = int(np.sum(fp_mask))
            caught_amt = float(np.sum(amt[tp_mask]))

            recall = tp / max(total_fraud_count, 1)
            precision = tp / max(tp + fp, 1)
            value_recall = caught_amt / max(total_fraud_amt, 1e-6)

            self.calibrated_thresholds[budget] = round(threshold, 4)
            self.budget_metrics[budget] = {
                "budget_pct": round(budget * 100.0, 1),
                "threshold": round(threshold, 4),
                "realized_volume_pct": round(float(np.mean(pred)) * 100.0, 2),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "fraud_value_recall": round(value_recall, 4),
                "fraud_value_saved": round(caught_amt, 2),
            }

        self.is_fitted = True
        return self

    def get_threshold(self, budget: float) -> float:
        """Returns the calibrated threshold for a specific operational budget."""
        if not self.is_fitted:
            raise RuntimeError("CapacityBudgetCalibrator must be fitted before querying thresholds.")
        if budget in self.calibrated_thresholds:
            return self.calibrated_thresholds[budget]
        closest_b = min(self.calibrated_thresholds.keys(), key=lambda b: abs(b - budget))
        return self.calibrated_thresholds[closest_b]

