"""Model evaluation metrics and cost-optimal threshold search."""

from typing import Any, cast
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix


def evaluate(y_true, y_pred, y_score, label: str):
    """Prints binary classification evaluation metrics."""
    print(f"\n=== {label} ===")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=cast(Any, 0)):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=cast(Any, 0)):.3f}")
    print(f"F1:        {f1_score(y_true, y_pred, zero_division=cast(Any, 0)):.3f}")
    if y_score is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_score):.3f}")
        print(f"PR-AUC:    {average_precision_score(y_true, y_score):.3f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")


def find_cost_optimal_threshold(y_val, score_val, amount_val, fp_cost: float = 50.0):
    """Grid searches threshold in [0.01, 0.99] minimizing total cost (FN dollar losses + FP review cost)."""
    y_arr = np.asarray(y_val)
    amt_arr = np.asarray(amount_val)
    thresholds = np.arange(0.01, 1.00, 0.01)

    costs = [
        amt_arr[(y_arr == 1) & (score_val < t)].sum() + fp_cost * ((y_arr == 0) & (score_val >= t)).sum()
        for t in thresholds
    ]
    costs = np.array(costs)
    best_idx = costs.argmin()

    default_cost = amt_arr[(y_arr == 1) & (score_val < 0.5)].sum() + fp_cost * ((y_arr == 0) & (score_val >= 0.5)).sum()
    return thresholds[best_idx], costs[best_idx], default_cost
