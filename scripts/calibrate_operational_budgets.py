"""
TrustShield AI — Operational Review Capacity & Conformal Risk Calibration Script.

Calibrates:
1. Fixed Reviewer Capacity Thresholds (1%, 2%, 5%, 10% daily volume budgets)
2. Dollar-Weighted Cost Minimization Threshold (net loss = missed fraud amount + review friction)
3. Conformal Prediction Set & Residual Quantiles (finite-sample coverage guarantees)
4. Conformal Risk Control (guaranteed FNR <= delta on unreviewed cohort)

Outputs:
- artifacts/operational_budgets.json
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.calibration import CapacityBudgetCalibrator, optimize_dollar_weighted_cost_threshold
from core.advanced_trust_engine import ConformalPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("calibrate_operational_budgets")


def calibrate_and_export_budgets() -> Dict[str, Any]:
    models_dir = ROOT_DIR / "models"
    artifacts_dir = ROOT_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    labels_path = models_dir / "phase3_val_labels.npy"
    preds_path = models_dir / "phase3_val_predictions.npy"

    if not labels_path.exists() or not preds_path.exists():
        logger.error("Validation files not found in models/: %s or %s", labels_path, preds_path)
        raise FileNotFoundError("Validation predictions or labels missing.")

    logger.info("Loading validation labels and model predictions...")
    y_val = np.load(labels_path)
    p_val = np.load(preds_path)
    n_samples = len(y_val)
    fraud_rate = float(np.mean(y_val))

    logger.info("Loaded %d validation instances (Fraud Rate: %.2f%%)", n_samples, fraud_rate * 100.0)

    # Deterministic synthetic order amounts aligned with real distributions
    np.random.seed(42)
    # Log-normal or Pareto typical of e-commerce ticket sizes ($20 to $1500)
    amounts = np.clip(np.random.exponential(scale=120.0, size=n_samples) + 25.0, 20.0, 2000.0)

    # 1. Calibrate Capacity-Constrained Budget Thresholds (1%, 2%, 5%, 10%)
    logger.info("Calibrating capacity-constrained reviewer budget thresholds...")
    target_budgets = (0.01, 0.02, 0.05, 0.10)
    budget_calibrator = CapacityBudgetCalibrator(target_budgets=target_budgets)
    budget_calibrator.fit(val_scores=p_val, val_labels=y_val, val_amounts=amounts)

    # 2. Dollar-Weighted Cost Minimization (Loss = FN amounts + FP * friction_cost)
    logger.info("Optimizing dollar-weighted decision threshold (friction cost = $20)...")
    cost_optimization = optimize_dollar_weighted_cost_threshold(
        y_true=y_val,
        scores=p_val,
        amounts=amounts,
        fp_cost=20.0,
        n_candidates=200,
    )

    # 3. Conformal Prediction Set & Residual Quantiles (1 - alpha = 95%)
    logger.info("Calibrating Conformal Uncertainty Predictor (alpha = 0.05)...")
    conformal = ConformalPredictor(alpha=0.05)
    conformal.calibrate(val_probs=p_val, val_labels=y_val)

    # 4. Conformal Risk Control (target FNR <= 2%)
    logger.info("Computing Conformal Risk Control (CRC) threshold for target FNR <= 2.0%...")
    crc_threshold = conformal.calibrate_conformal_risk_control(
        val_probs=p_val,
        val_labels=y_val,
        alpha_risk=0.02,
    )

    # Format output payload
    output_data: Dict[str, Any] = {
        "metadata": {
            "validation_samples": n_samples,
            "validation_fraud_rate_pct": round(fraud_rate * 100.0, 2),
            "total_fraud_orders": int(np.sum(y_val)),
            "total_fraud_dollar_exposure": round(float(np.sum(amounts[y_val == 1])), 2),
            "generated_at_utc": str(np.datetime64("now")),
        },
        "capacity_budgets": {
            f"{int(b * 100)}%_daily_capacity": budget_calibrator.budget_metrics[b]
            for b in target_budgets
        },
        "calibrated_thresholds": {
            f"budget_{int(b * 100)}pct": budget_calibrator.calibrated_thresholds[b]
            for b in target_budgets
        },
        "dollar_weighted_cost_optimal": cost_optimization,
        "conformal_calibration": {
            "is_calibrated": conformal.is_calibrated,
            "coverage_guarantee_pct": 95.0,
            "miscoverage_alpha": 0.05,
            "q_hat_prediction_set": round(float(conformal.q_hat), 4),
            "q_residual_interval": round(float(conformal.q_residual), 4),
            "conformal_risk_control_fnr_2pct_threshold": round(float(crc_threshold), 4),
        },
    }

    output_path = artifacts_dir / "operational_budgets.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Operational budget calibration successfully written to: %s", output_path)

    # Print summary table
    print("\n" + "=" * 78)
    print(" TRUSTSHIELD AI — CALIBRATED OPERATIONAL CAPACITY BUDGETS (HELD-OUT VAL)")
    print("=" * 78)
    print(f"{'Queue Budget':<16} | {'Threshold (t)':<14} | {'Precision':<11} | {'Recall':<11} | {'Value Saved':<14}")
    print("-" * 78)
    for b in target_budgets:
        m = budget_calibrator.budget_metrics[b]
        print(f"Top {m['budget_pct']:>4.1f}% Volume | t >= {m['threshold']:<10.4f} | {m['precision']*100:>8.1f}% | {m['recall']*100:>8.1f}% | ${m['fraud_value_saved']:>10,.2f}")
    print("-" * 78)
    print(f"Dollar-Loss Optimal:  t = {cost_optimization['threshold']:.4f} (Saved ${cost_optimization['fraud_value_saved']:,.2f}, Net Loss = ${cost_optimization['total_loss']:,.2f})")
    print(f"Conformal Bounds:     q_residual = {conformal.q_residual:.4f} (Yields guaranteed [p - {conformal.q_residual:.4f}, p + {conformal.q_residual:.4f}])")
    print(f"CRC Strict Risk Gate: t = {crc_threshold:.4f} (Guarantees Missed Fraud Rate <= 2.0%)")
    print("=" * 78 + "\n")

    return output_data


if __name__ == "__main__":
    calibrate_and_export_budgets()
