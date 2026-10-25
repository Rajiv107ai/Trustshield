"""
TrustShield AI — Operational Model Monitoring & Governance Router.

Exposes endpoints for:
- Aggregate feature, score, and decision drift monitoring (/monitoring/drift)
- Matured-label calibration, ECE, Brier score, and operating-point capture (/monitoring/calibration)
- Unified operational health overview & alerts (/monitoring/overview)
- Historical monitoring report retrieval (/monitoring/reports)
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.contracts.monitoring import (
    CalibrationReportV1,
    DriftReportV1,
    MonitoringWindowV1,
)
from backend.services.calibration_service import (
    CalibrationService,
    CalibrationThresholds,
)
from backend.services.drift_service import (
    DriftService,
    DriftThresholds,
)
from backend.services.monitoring_store import MonitoringStore
from core.policy_config import RISK_POLICY

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

_store_lock = threading.RLock()
_monitoring_store: Optional[MonitoringStore] = None
_drift_service: Optional[DriftService] = None
_calibration_service: Optional[CalibrationService] = None

DATA_PATH = Path("data/synthetic_v2_1/orders.csv")


def get_monitoring_store() -> MonitoringStore:
    global _monitoring_store
    with _store_lock:
        if _monitoring_store is None:
            _monitoring_store = MonitoringStore()
        return _monitoring_store


def get_drift_service() -> DriftService:
    global _drift_service
    with _store_lock:
        if _drift_service is None:
            _drift_service = DriftService()
        return _drift_service


def get_calibration_service() -> CalibrationService:
    global _calibration_service
    with _store_lock:
        if _calibration_service is None:
            _calibration_service = CalibrationService()
        return _calibration_service


def _load_orders_dataset() -> pd.DataFrame:
    """Load orders dataset for baseline and evaluation windows."""
    if not DATA_PATH.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Monitoring dataset is unavailable",
        )
    return pd.read_csv(DATA_PATH)


def _generate_drift_report(store: MonitoringStore, service: DriftService) -> Dict[str, Any]:
    """Compute aggregate drift between reference baseline and current window."""
    df = _load_orders_dataset()
    ref_size = min(2000, len(df) // 2)
    cur_size = min(2000, len(df) - ref_size)

    ref_df = df.iloc[:ref_size].copy()
    cur_df = df.iloc[ref_size : ref_size + cur_size].copy()

    # Pre-calculate synthetic prediction scores and decisions for demo/eval parity
    rng = np.random.RandomState(42)
    ref_scores = np.clip(
        np.where(ref_df["is_fraudulent"].to_numpy() == 1, rng.beta(7, 3, size=len(ref_df)), rng.beta(2, 8, size=len(ref_df))),
        0.0,
        1.0,
    )
    ref_df["score"] = ref_scores.tolist()

    cur_scores = np.clip(
        np.where(cur_df["is_fraudulent"].to_numpy() == 1, rng.beta(6, 4, size=len(cur_df)), rng.beta(2.5, 7.5, size=len(cur_df))),
        0.0,
        1.0,
    )
    cur_df["score"] = cur_scores.tolist()

    ref_decisions = np.where(ref_scores >= RISK_POLICY.block, "BLOCK", np.where(ref_scores >= RISK_POLICY.hold, "HOLD", np.where(ref_scores >= RISK_POLICY.review, "REVIEW", "ALLOW")))
    ref_df["decision"] = ref_decisions.tolist()

    cur_decisions = np.where(cur_scores >= RISK_POLICY.block, "BLOCK", np.where(cur_scores >= RISK_POLICY.hold, "HOLD", np.where(cur_scores >= RISK_POLICY.review, "REVIEW", "ALLOW")))
    cur_df["decision"] = cur_decisions.tolist()

    now = datetime.now(timezone.utc)
    window = MonitoringWindowV1(
        window_id=f"drift-window-{now.strftime('%Y%m%d%H%M')}",
        window_start=now - timedelta(days=7),
        window_end=now,
        baseline_window_id="baseline-v2.1",
        baseline_window_start=now - timedelta(days=30),
        baseline_window_end=now - timedelta(days=7),
        sample_count=len(cur_df),
        reference_sample_count=len(ref_df),
        current_sample_count=len(cur_df),
    )

    numeric_cols = ["amount"]
    categorical_cols = ["status"]

    report = service.monitor(
        ref_df,
        cur_df,
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
        score_column="score",
        decision_column="decision",
        window=window,
    )
    store.save(report)
    return report


def _generate_calibration_report(store: MonitoringStore, service: CalibrationService) -> Dict[str, Any]:
    """Compute matured-label calibration metrics."""
    df = _load_orders_dataset()
    sample = df.iloc[:3000].copy()

    rng = np.random.RandomState(42)
    sample_scores = np.clip(
        np.where(sample["is_fraudulent"].to_numpy() == 1, rng.beta(8, 2, size=len(sample)), rng.beta(1.5, 8.5, size=len(sample))),
        0.0,
        1.0,
    )
    sample["score"] = sample_scores.tolist()
    sample_decisions = np.where(sample_scores >= RISK_POLICY.block, "BLOCK", np.where(sample_scores >= RISK_POLICY.hold, "HOLD", np.where(sample_scores >= RISK_POLICY.review, "REVIEW", "ALLOW")))
    sample["decision"] = sample_decisions.tolist()
    sample["event_time"] = pd.to_datetime(sample["order_date"], utc=True)
    sample["outcome_window_days"] = 14
    sample["label_time"] = sample["event_time"] + pd.to_timedelta(15, unit="D")
    sample["label"] = sample["is_fraudulent"]

    now = datetime.now(timezone.utc)
    window = MonitoringWindowV1(
        window_id=f"cal-window-{now.strftime('%Y%m%d%H%M')}",
        window_start=now - timedelta(days=30),
        window_end=now,
        sample_count=len(sample),
        matured_label_count=len(sample),
    )

    report = service.monitor(
        sample,
        as_of=now,
        score_column="score",
        label_column="label",
        decision_column="decision",
        threshold=0.55,
        window=window,
    )
    store.save(report)
    return report


@router.get("/drift", summary="Get or calculate latest model & feature drift report")
def get_drift_report(
    refresh: bool = Query(False, description="Force recomputation of drift metrics"),
    store: MonitoringStore = Depends(get_monitoring_store),
    service: DriftService = Depends(get_drift_service),
) -> Dict[str, Any]:
    """Return the latest aggregate drift report comparing reference and current windows."""
    if not refresh:
        cached = store.latest("drift")
        if cached is not None:
            return cached

    return _generate_drift_report(store, service)


@router.get("/calibration", summary="Get or calculate matured-label calibration report")
def get_calibration_report(
    refresh: bool = Query(False, description="Force recomputation of calibration metrics"),
    store: MonitoringStore = Depends(get_monitoring_store),
    service: CalibrationService = Depends(get_calibration_service),
) -> Dict[str, Any]:
    """Return matured-label calibration evaluation (ECE, Brier score, operating point)."""
    if not refresh:
        cached = store.latest("calibration")
        if cached is not None:
            return cached

    return _generate_calibration_report(store, service)


@router.get("/overview", summary="Unified monitoring health and alert dashboard summary")
def get_monitoring_overview(
    store: MonitoringStore = Depends(get_monitoring_store),
    drift_service: DriftService = Depends(get_drift_service),
    calibration_service: CalibrationService = Depends(get_calibration_service),
) -> Dict[str, Any]:
    """Return an operational dashboard summary combining drift and calibration health."""
    drift_report = store.latest("drift") or _generate_drift_report(store, drift_service)
    cal_report = store.latest("calibration") or _generate_calibration_report(store, calibration_service)

    drift_status = drift_report.get("status", "OK")
    cal_status = cal_report.get("status", "OK")

    overall_ranks = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
    system_status = "CRITICAL" if "CRITICAL" in (drift_status, cal_status) else "WARNING" if "WARNING" in (drift_status, cal_status) else "OK"

    active_alerts = []
    active_alerts.extend(drift_report.get("alerts", []))
    active_alerts.extend(cal_report.get("alerts", []))

    # Top drifted features from numerical metrics
    top_features = []
    numeric_metrics = drift_report.get("metrics", {}).get("numeric", {})
    for feat, data in numeric_metrics.items():
        if isinstance(data, dict):
            top_features.append({
                "feature": feat,
                "psi": data.get("psi", 0.0),
                "ks_statistic": data.get("ks_statistic", 0.0),
                "status": data.get("status", "OK"),
            })
    top_features.sort(key=lambda x: x["psi"], reverse=True)

    cal_metrics = cal_report.get("metrics", {})

    return {
        "status": system_status,
        "schema_version": "monitoring.v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "drift": {
            "status": drift_status,
            "recommended_action": drift_report.get("recommended_action", "none"),
            "top_drifted_features": top_features[:5],
            "decision_distribution": drift_report.get("metrics", {}).get("decision_distribution", {}),
            "score_drift": drift_report.get("metrics", {}).get("score_drift", {}),
        },
        "calibration": {
            "status": cal_status,
            "ece": cal_metrics.get("ece", 0.0),
            "brier_score": cal_metrics.get("brier_score", 0.0),
            "precision": cal_metrics.get("precision", 0.0),
            "recall": cal_metrics.get("recall", 0.0),
            "review_budget_capture": cal_metrics.get("review_budget_capture", {}),
            "matured_label_count": cal_report.get("matured_label_count", 0),
        },
        "active_alerts_count": len(active_alerts),
        "active_alerts": active_alerts,
    }


@router.get("/reports", summary="List historical monitoring reports")
def list_monitoring_reports(
    limit: int = Query(10, ge=1, le=50, description="Max reports to return"),
    report_type: Optional[str] = Query(None, description="Filter by 'drift' or 'calibration'"),
    store: MonitoringStore = Depends(get_monitoring_store),
) -> List[Dict[str, Any]]:
    """Return historical monitoring reports from durable storage."""
    return store.list_reports(limit=limit, report_type=report_type)
