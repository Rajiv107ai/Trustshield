"""
TrustShield AI — System, Observability & Benchmark Router.

Endpoints:
- GET /health            -- Liveness probe
- GET /ready             -- Multi-component readiness probe
- GET /api/results       -- Canonical audited benchmark metrics
- GET /metrics           -- Prometheus metrics exposition
- GET /system/benchmark  -- Latency and throughput benchmarks
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST

from backend.auth import Role, require_role
from backend.model_loader import store
from backend.routers.scoring import score_transaction
from backend.schemas import HealthResponse, ReadyResponse, ServiceState, TransactionScoreRequest
from backend.services.investigation_service import investigation_service
from backend.services.metrics_service import metrics_service
from backend.services.neo4j_service import neo4j_service
from backend.services.redis_service import redis_service

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
    - 'ready' when models and graph data are fully loaded
    - 'degraded' when Phase 3 baseline runs without Phase 5/CLIP or with offline infrastructure
    - 'not_ready' when core models are unavailable
    """
    if not store.is_loaded:
        try:
            store.load()
        except Exception as exc:
            logger.warning("Failed to lazy-load models in /ready: %s", exc)

    models_ready = store.is_loaded and (store.combined_graph_model is not None)
    phase3_ready = store.combined_graph_model is not None
    phase5_ready = store.phase5_loaded
    clip_ready = store.clip_loaded
    shap_ready = store.shap_loaded
    rings_ready = store.rings_df is not None
    redis_ready = redis_service.is_available
    neo4j_ready = neo4j_service.is_available

    if models_ready and phase5_ready and rings_ready:
        status = ServiceState.READY
    elif models_ready:
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
        redis_ready=redis_ready,
        neo4j_ready=neo4j_ready,
        components={
            "models": "ready" if models_ready else "unavailable",
            "explainability": "available" if shap_ready else "unavailable",
            "redis": "available" if redis_ready else "unavailable",
            "neo4j": "available" if neo4j_ready else "unavailable",
            "prometheus": "available",
            "investigation": "available" if investigation_service.agent is not None else "unavailable",
        },
        details={
            "n_rings": len(store.rings_df) if store.rings_df is not None else 0,
            "phase5_loaded": phase5_ready,
            "clip_loaded": clip_ready,
            "shap_loaded": shap_ready,
            "redis_status": "connected" if redis_ready else ("disabled" if not redis_service.enabled else "unavailable"),
            "neo4j_status": "connected" if neo4j_ready else ("disabled" if not neo4j_service.enabled else "unavailable"),
            "prometheus_instrumented": True,
            "investigation_agent_ready": investigation_service.agent is not None,
        },
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
    Prometheus metrics exposition endpoint for scraping by Prometheus and Grafana.
    """
    return Response(content=metrics_service.export_metrics(), media_type=CONTENT_TYPE_LATEST)


@router.get("/system/benchmark", tags=["monitoring"], dependencies=[Depends(require_role(Role.ADMIN))])
def system_benchmark(iterations: int = Query(default=25, ge=5, le=50, description="Repetition count (bounded <=50 to prevent denial of service)")):
    """
    Measure empirical p50, p95, and p99 latencies across system components:
    - Model inference
    - Redis embedding cache
    - Neo4j Cypher graph engine
    - Full transaction scoring pipeline
    """
    redis_bench = redis_service.benchmark_latency(iterations=iterations)
    neo4j_bench = neo4j_service.benchmark_latency(iterations=iterations)

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
        "redis_feature_store": redis_bench,
        "neo4j_graph_engine": neo4j_bench,
    }
