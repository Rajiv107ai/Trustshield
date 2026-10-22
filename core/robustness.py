"""Robustness, Multi-seed Evaluation, and Confidence Estimation for TrustShield."""

from __future__ import annotations
import numpy as np
from scipy import stats
from xgboost import XGBClassifier
from typing import Dict, List, Callable, Tuple, Any, Sequence

SEEDS_20: List[int] = [
    42, 101, 202, 303, 404, 505, 606, 707, 808, 909,
    1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809, 1910
]



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


def paired_stats_calc(baseline_vals: Sequence[float], new_vals: Sequence[float]) -> Dict[str, Any]:
    """Calculate paired differences, t-statistic, two-tailed p-value, Cohen's d, and 95% CI."""
    diffs = np.array(new_vals) - np.array(baseline_vals)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_d = std_d / np.sqrt(n) if n > 1 else 0.0
    t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 1.96
    ci_lower = mean_d - t_crit * se_d
    ci_upper = mean_d + t_crit * se_d
    cohens_d = (mean_d / std_d) if std_d > 1e-9 else 0.0
    t_res = stats.ttest_rel(new_vals, baseline_vals) if (std_d > 1e-9 and n > 1) else None
    p_val = float(t_res.pvalue) if t_res is not None and not np.isnan(t_res.pvalue) else 1.0
    t_stat = float(t_res.statistic) if t_res is not None and not np.isnan(t_res.statistic) else 0.0
    return {
        "mean_diff": mean_d,
        "std_diff": std_d,
        "ci_95": [float(ci_lower), float(ci_upper)],
        "cohens_d": float(cohens_d),
        "t_stat": t_stat,
        "p_value": p_val,
        "paired_diffs": [float(d) for d in diffs],
    }


def holm_bonferroni_correction(tests_dict: Dict[str, float]) -> List[Dict[str, Any]]:
    """Applies step-down Holm-Bonferroni correction with ascending sort and running max."""
    items = list(tests_dict.items())
    items.sort(key=lambda x: x[1])
    m = len(items)
    adjusted = []
    running_max = 0.0
    for i, (name, raw_p) in enumerate(items):
        rank = i + 1
        adj_p = min(1.0, (m - rank + 1) * raw_p)
        running_max = max(running_max, adj_p)
        adjusted.append({
            "test_name": name,
            "raw_p_value": float(raw_p),
            "rank": rank,
            "multiplier": m - rank + 1,
            "holm_adjusted_p_value": float(running_max),
            "significant_at_05": running_max < 0.05,
        })
    return adjusted


def train_xgb(X_tr: Any, y_tr: Any, random_state: int = 42) -> XGBClassifier:
    """Standardized XGBoost classifier for multi-seed benchmarks."""
    pos = int(y_tr.sum())
    neg = len(y_tr) - pos
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X_tr.fillna(0.0), y_tr)
    return clf
