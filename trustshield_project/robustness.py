"""Robustness, Multi-seed Evaluation, and Confidence Estimation for TrustShield."""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Callable, Any, Tuple


def multiseed_summary(scores: List[float]) -> Dict[str, float]:
    """Compute summary statistics (mean, std, min, max) across multiple random seed runs."""
    arr = np.asarray(scores, dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_seeds": 0}

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_seeds": int(len(arr)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Calculate non-parametric bootstrap confidence interval for any evaluation metric."""
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    n = len(y)
    rng = np.random.default_rng(seed)

    boot_scores = []
    point_estimate = float(metric_fn(y, p))

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        # Skip resamples that lack positive or negative samples
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            boot_scores.append(metric_fn(y[idx], p[idx]))
        except Exception:
            continue

    if not boot_scores:
        return {"estimate": point_estimate, "ci_lower": point_estimate, "ci_upper": point_estimate}

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(boot_scores, 100 * alpha))
    upper = float(np.percentile(boot_scores, 100 * (1.0 - alpha)))

    return {
        "estimate": point_estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "std_err": float(np.std(boot_scores)),
    }


def simulate_prevalence_shift(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_prevalence: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample positive and negative instances to simulate fraud prevalence shifts (e.g. 1%, 3%, 15%)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return y, p

    current_prev = n_pos / (n_pos + n_neg)
    target_p = min(max(float(target_prevalence), 0.001), 0.999)

    if target_p <= current_prev:
        # Downsample positives
        needed_pos = max(1, round(n_neg * target_p / (1.0 - target_p)))
        selected_pos = rng.choice(pos_idx, size=min(needed_pos, n_pos), replace=False)
        selected_neg = neg_idx
    else:
        # Downsample negatives
        needed_neg = max(1, round(n_pos * (1.0 - target_p) / target_p))
        selected_neg = rng.choice(neg_idx, size=min(needed_neg, n_neg), replace=False)
        selected_pos = pos_idx

    combined_idx = np.concatenate([selected_pos, selected_neg])
    rng.shuffle(combined_idx)

    return y[combined_idx], p[combined_idx]
