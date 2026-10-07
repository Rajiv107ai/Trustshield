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
from typing import Dict, Optional, Any
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
