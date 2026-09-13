"""
TrustShield AI — Shared utilities

Common evaluation and cost-threshold helpers used across Phase 1C,
Phase 2, and Phase 3 — extracted here to eliminate the copy-paste
that previously lived in baseline_model.py, phase2_specialized_models.py,
and graph_features.py.
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, y_score, label):
    """
    Prints precision / recall / F1 / ROC-AUC / PR-AUC / confusion matrix
    for a binary classifier.

    Args:
        y_true  : ground-truth binary labels (array-like)
        y_pred  : predicted binary labels (array-like)
        y_score : predicted probabilities for the positive class, or None
                  (ROC-AUC and PR-AUC are skipped when None)
        label   : section header string printed above the metrics
    """
    print(f"\n=== {label} ===")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.3f}")
    if y_score is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_score):.3f}")
        print(f"PR-AUC:    {average_precision_score(y_true, y_score):.3f}"
              f"  (more informative than ROC-AUC at ~7% base rate)")
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{cm}")


# ---------------------------------------------------------------------------
# Cost-optimal threshold sweep
# ---------------------------------------------------------------------------

def find_cost_optimal_threshold(y_val, score_val, amount_val, fp_cost=50.0):
    """
    Sweeps thresholds in [0.01, 0.99] on the VALIDATION set only (never
    test) and returns the threshold that minimises total business cost:

        cost(t) = sum(amount of missed fraud | FN at t)
                  + fp_cost * count(false positives at t)

    Cost assumptions are explicit, documented estimates for demonstrating
    the methodology — not claimed real business figures:
      - FN cost = the order / listing amount itself (fraud that goes through
        costs roughly what it's worth).
      - FP cost = a flat per-false-alarm review / friction cost (passed in
        by the caller — much smaller than a typical transaction amount).

    Args:
        y_val      : ground-truth labels for the validation set (array-like)
        score_val  : predicted probabilities for the positive class
                     (array-like, same length as y_val)
        amount_val : per-row monetary amounts used for FN cost
                     (array-like or pd.Series, same length as y_val)
        fp_cost    : flat cost per false positive (default $50)

    Returns:
        (best_threshold, best_cost, default_cost_at_0_5)
          best_threshold    : float in [0.01, 0.99]
          best_cost         : total cost at best_threshold
          default_cost_at_0_5 : total cost at the naive 0.5 cutoff
                               (comparison baseline)
    """
    y_arr = np.asarray(y_val)
    amt_arr = np.asarray(amount_val)
    thresholds = np.arange(0.01, 1.00, 0.01)
    costs = []
    for t in thresholds:
        pred = (score_val >= t).astype(int)
        fn_mask = (y_arr == 1) & (pred == 0)
        fp_mask = (y_arr == 0) & (pred == 1)
        costs.append(amt_arr[fn_mask].sum() + fp_cost * fp_mask.sum())
    costs = np.array(costs)
    best_idx = costs.argmin()

    default_pred = (score_val >= 0.5).astype(int)
    default_fn = (y_arr == 1) & (default_pred == 0)
    default_fp = (y_arr == 0) & (default_pred == 1)
    default_cost = amt_arr[default_fn].sum() + fp_cost * default_fp.sum()

    return thresholds[best_idx], costs[best_idx], default_cost
