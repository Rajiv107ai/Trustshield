"""
TrustShield AI — System, Observability & Benchmark Router.

Endpoints:
- GET /health            -- Liveness probe
- GET /ready             -- Single-container readiness probe
- GET /api/results       -- Canonical audited benchmark metrics
- GET /metrics           -- Structured metrics exposition
- GET /system/benchmark  -- Latency and throughput benchmarks
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response

from backend.auth import Role, require_role
from backend.model_loader import store
from backend.routers.scoring import score_transaction
from backend.schemas import HealthResponse, ReadyResponse, ServiceState, TransactionScoreRequest
from backend.services.investigation_service import investigation_service
from backend.services.metrics_service import metrics_service

logger = logging.getLogger(__name__)

router = APIRouter()

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """
    Liveness check. Returns whether the model artifacts are loaded and how
    many fraud rings are pre-computed and ready to serve.
    """
    return HealthResponse(
        status="ok",
        models_loaded=store.is_loaded,
        phase5_loaded=store.phase5_loaded,
        clip_loaded=store.clip_loaded,
        shap_loaded=store.shap_loaded,
        rings_loaded=store.rings_df is not None,
        n_rings=len(store.rings_df) if store.rings_df is not None else 0,
    )


@router.get("/ready", response_model=ReadyResponse, tags=["system"])
def ready(response: Response):
    """
    Readiness probe for orchestrators. Returns service state:
    - 'ready' when Phase 3 models and precomputed rings are fully loaded
    - 'not_ready' when core models are unavailable
    """
    if not store.is_loaded:
        try:
            store.load()
        except Exception as exc:
            logger.warning("Failed to lazy-load models in /ready: %s", exc)

    models_ready = store.is_loaded and (store.combined_graph_model is not None)
    phase3_ready = store.combined_graph_model is not None
    phase5_ready = False
    clip_ready = getattr(store, "clip_loaded", False)
    shap_ready = getattr(store, "shap_loaded", False)
    rings_ready = store.rings_df is not None
    pit_history_ready = getattr(store, "transaction_history", None) is not None
    feature_service_ready = os.getenv("TRUSTSHIELD_FEATURE_SERVICE_READY", "").strip().lower() in {"1", "true", "yes", "on"}
    pit_required_raw = os.getenv("TRUSTSHIELD_REQUIRE_PIT_HISTORY", "")
    pit_required = (
        pit_required_raw.strip().lower() in {"1", "true", "yes", "on"}
        if pit_required_raw.strip()
        else os.getenv("TRUSTSHIELD_ENV", os.getenv("ENV", "development")).strip().lower() == "production"
    )

    fs_instance = getattr(store, "feature_service", None)
    redis_connected = bool(
        fs_instance is not None
        and getattr(fs_instance, "redis", None) is not None
        and fs_instance.redis.is_connected
    )

    details: Dict[str, Any] = {
        "n_rings": len(store.rings_df) if store.rings_df is not None else 0,
        "phase3_loaded": phase3_ready,
        "clip_loaded": clip_ready,
        "shap_loaded": shap_ready,
        "redis_status": "connected" if redis_connected else "disabled",
        "neo4j_status": "disabled",
        "investigation_agent_ready": investigation_service.agent is not None,
        "pit_history_required": pit_required,
        "trusted_feature_service_ready": feature_service_ready,
    }
    if fs_instance is not None and getattr(fs_instance, "is_ready", False):
        details["feature_service"] = fs_instance.get_status()

    if not pit_history_ready:
        details["pit_history"] = "unavailable — edge weights default to 0.0"

    graph_snap = getattr(store, "graph_snapshot", None)
    if isinstance(graph_snap, dict):
        as_of = graph_snap.get("effective_as_of") or graph_snap.get("snapshot_cutoff")
        if as_of:
            details["graph_snapshot_as_of"] = str(as_of)

    feature_service_required = os.getenv("TRUSTSHIELD_ENV", os.getenv("ENV", "development")).strip().lower() == "production"
    if models_ready and rings_ready and pit_history_ready and (feature_service_ready or not feature_service_required):
        status = ServiceState.READY
    elif models_ready and not pit_required and not feature_service_required:
        status = ServiceState.DEGRADED
    else:
        status = ServiceState.NOT_READY
        response.status_code = 503

    return ReadyResponse(
        status=status,
        models_ready=models_ready,
        phase3_ready=phase3_ready,
        phase5_ready=phase5_ready,
        clip_ready=clip_ready,
        shap_ready=shap_ready,
        rings_ready=rings_ready,
        pit_history_ready=pit_history_ready,
        redis_ready=redis_connected,
        neo4j_ready=False,
        components={
            "models": "ready" if models_ready else "unavailable",
            "explainability": "available" if shap_ready else "unavailable",
            "graph_topology": "ready" if rings_ready else "unavailable",
            "investigation": "available" if investigation_service.agent is not None else "unavailable",
        },
        details=details,
    )


@router.get("/api/results", tags=["system"])
@router.get("/results", tags=["system"])
def get_canonical_results():
    """Returns canonical audited benchmark metrics payload directly from results/results.json."""
    results_path = os.path.join(_ROOT_DIR, "results", "results.json")
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="results.json not found on disk")
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/metrics", tags=["monitoring"])
def prometheus_metrics():
    """
    Lightweight metrics exposition endpoint.
    """
    return Response(content=metrics_service.export_metrics(), media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get("/system/benchmark", tags=["monitoring"], dependencies=[Depends(require_role(Role.ADMIN))])
def system_benchmark(iterations: int = Query(default=25, ge=5, le=50, description="Repetition count (bounded <=50 to prevent denial of service)")):
    """
    Measure empirical p50, p95, and p99 latencies for the scoring pipeline.
    """
    test_req = TransactionScoreRequest(
        order_id="BENCHMARK_PROBE",
        buyer_id="BUYER_000001",
        seller_id="SELLER_000001",
        amount=150.0,
        base_price=150.0,
        category_median_price=150.0,
    )
    score_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = score_transaction(test_req)
        score_latencies.append((time.perf_counter() - t0) * 1000.0)

    score_arr = np.array(score_latencies)
    scoring_bench = {
        "count": float(len(score_arr)),
        "p50_ms": round(float(np.percentile(score_arr, 50)), 3),
        "p95_ms": round(float(np.percentile(score_arr, 95)), 3),
        "p99_ms": round(float(np.percentile(score_arr, 99)), 3),
        "mean_ms": round(float(np.mean(score_arr)), 3),
    }

    return {
        "iterations": iterations,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scoring_pipeline": scoring_bench,
        "redis_feature_store": {"status": "disabled", "p50_ms": 0.0},
        "neo4j_graph_engine": {"status": "disabled", "p50_ms": 0.0},
    }
