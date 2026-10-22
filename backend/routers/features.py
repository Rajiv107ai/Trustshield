"""TrustShield AI — Feature Engineering & Diagnostics API Router.

Exposes real-time inspection endpoints for point-in-time feature resolution,
sliding-window velocity profiles, and Redis feature store operational status.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from backend.auth import authenticate_request
from backend.model_loader import store
from backend.schemas import TransactionScoreRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/features", tags=["features"])


def _get_feature_service() -> Any:
    fs = getattr(store, "feature_service", None)
    if fs is None or not getattr(fs, "is_ready", False):
        # Attempt lazy initialization if store is loaded
        if not store.is_loaded:
            try:
                store.load()
                fs = getattr(store, "feature_service", None)
            except Exception as exc:
                logger.warning("Failed lazy loading store in /features: %s", exc)
    if fs is None or not getattr(fs, "is_ready", False):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "FEATURE_SERVICE_NOT_INITIALIZED",
                "message": "Governed point-in-time feature service is not initialized or still loading.",
            },
        )
    return fs


@router.get("/status")
def get_feature_service_status():
    """Returns operational status, storage backend, and dataset indices for the feature service."""
    fs = getattr(store, "feature_service", None)
    if fs is None:
        return {"status": "uninitialized", "ready": False}
    return {
        "status": "operational" if fs.is_ready else "loading",
        **fs.get_status(),
    }


@router.post("/preview", dependencies=[Depends(authenticate_request)])
def preview_features(req: TransactionScoreRequest):
    """Calculates authoritative point-in-time features and velocity profiles for a request."""
    fs = _get_feature_service()
    snap = getattr(store, "graph_snapshot", None)
    all_cols = getattr(store, "feature_meta", {}).get("all_feature_cols") if store.feature_meta else None

    feature_row, metadata = fs.compute_features(req, graph_snapshot=snap, all_cols=all_cols)
    return {
        "order_id": req.order_id,
        "feature_row": feature_row,
        "metadata": metadata,
    }


@router.get("/buyer/{buyer_id}", dependencies=[Depends(authenticate_request)])
def get_buyer_feature_profile(buyer_id: str, as_of: Optional[str] = None):
    """Retrieves real-time sliding window velocity profile and historical return metrics for a buyer."""
    fs = _get_feature_service()
    from backend.services.feature_service import normalize_timestamp
    target_dt = normalize_timestamp(as_of)

    clean_bid = str(buyer_id).strip()
    orders_before = fs.store.count_buyer_orders_before(clean_bid, target_dt)
    returns_before = fs.store.count_buyer_returns_before(clean_bid, target_dt)
    return_rate = min(1.0, float(returns_before) / max(orders_before, 1))

    v_1h_count, v_1h_amt = fs.store.buyer_velocity(clean_bid, target_dt, window_seconds=3600)
    v_24h_count, v_24h_amt = fs.store.buyer_velocity(clean_bid, target_dt, window_seconds=86400)
    v_7d_count, v_7d_amt = fs.store.buyer_velocity(clean_bid, target_dt, window_seconds=7 * 86400)

    signup = fs.buyer_signups.get(clean_bid)
    age_days = max(0.0, (target_dt - signup).total_seconds() / 86400.0) if signup else 0.0

    return {
        "buyer_id": clean_bid,
        "as_of": target_dt.isoformat(),
        "total_historical_orders": orders_before,
        "total_historical_returns": returns_before,
        "historical_return_rate": return_rate,
        "buyer_age_days": round(age_days, 1),
        "velocities": {
            "window_1h_orders": v_1h_count,
            "window_1h_spend": v_1h_amt,
            "window_24h_orders": v_24h_count,
            "window_24h_spend": v_24h_amt,
            "window_7d_orders": v_7d_count,
            "window_7d_spend": v_7d_amt,
        },
        "engine": "redis" if fs.redis.is_connected else "in_memory",
    }


@router.get("/device/{device_id}", dependencies=[Depends(authenticate_request)])
def get_device_collision_profile(device_id: str, as_of: Optional[str] = None):
    """Retrieves device collision profile and account sharing metrics in the sliding 7-day window."""
    fs = _get_feature_service()
    from backend.services.feature_service import normalize_timestamp
    target_dt = normalize_timestamp(as_of)

    clean_did = str(device_id).strip()
    shared_buyers = fs.store.count_device_shared_buyers(clean_did, target_dt, window_seconds=7 * 86400)

    return {
        "device_id": clean_did,
        "as_of": target_dt.isoformat(),
        "device_shared_buyer_count_7d": shared_buyers,
        "is_shared_device": shared_buyers > 1,
        "engine": "redis" if fs.redis.is_connected else "in_memory",
    }
