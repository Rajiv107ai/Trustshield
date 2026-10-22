"""
TrustShield AI — FastAPI Application Entry Point & Serving Layer.

Composes domain routers, application lifespan, Prometheus metrics middleware,
and security policies:
- Scoring Router:     /transaction/score, /transaction/explain
- Operations Router:  /fraud-rings, /listing/analyze, /return/analyze, /investigation/generate-dossier
- System Router:      /health, /ready, /results, /metrics, /system/benchmark, /version

Start:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.auth import validate_security_configuration
from backend.features import _build_feature_row, _risk_label
from backend.model_loader import store
from backend.routers.actions import router as actions_router
from backend.routers.operations import router as operations_router
from backend.routers.contracts import router as contracts_router
from backend.routers.scoring import router as scoring_router, score_transaction
from backend.routers.system import router as system_router
from backend.routers.version import router as version_router
from backend.routers.monitoring import router as monitoring_router
from backend.routers.features import router as features_router
from backend.routers.champion_challenger import router as champion_challenger_router
from backend.release_manifest import validate_release_manifest_configuration
from backend.services.metrics_service import metrics_service
from trustshield_project.advanced_trust_engine import CanonicalTrustEngine

logger = logging.getLogger(__name__)

# CanonicalTrustEngine is the sole authoritative production engine.
_canonical_trust_engine = CanonicalTrustEngine() if CanonicalTrustEngine is not None else None

# Enforce security configuration immediately at startup (fails fast before app init)
validate_security_configuration()
validate_release_manifest_configuration()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup."""
    validate_security_configuration()
    validate_release_manifest_configuration()
    if _canonical_trust_engine is None:
        logger.error("FATAL: CanonicalTrustEngine failed to load. Server cannot safely score transactions.")
    try:
        store.load()
        logger.info(
            "TrustShield models loaded successfully (Phase 3: %s, CLIP: %s, Rings: %s)",
            store.is_loaded, store.clip_loaded,
            len(store.rings_df) if store.rings_df is not None else 0,
        )
    except Exception as exc:
        logger.warning("Models could not be loaded at startup: %s. Will retry on first request.", exc)

    yield


app = FastAPI(
    title="TrustShield AI — Fraud & Trust Scoring Engine",
    description=(
        "Production AI risk evaluation pipeline for e-commerce transactions, "
        "synthesizing tabular features, graph topology, multimodal CLIP embeddings, "
        "and calibrated trust decisions."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# Prometheus HTTP metrics tracking middleware
@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - t0
    endpoint = request.url.path
    metrics_service.requests_total.labels(
        endpoint=endpoint,
        method=request.method,
        status_code=str(response.status_code),
    ).inc()
    metrics_service.request_duration.labels(endpoint=endpoint).observe(duration)
    return response


# Explicit validated CORS configuration (strict origin whitelist)
_raw_origins = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000",
)
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
_allow_creds = "*" not in _allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allow_creds,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount domain routers
app.include_router(scoring_router)
app.include_router(operations_router)
app.include_router(contracts_router)
app.include_router(actions_router)
app.include_router(system_router)
app.include_router(version_router)
app.include_router(monitoring_router)
app.include_router(features_router)
app.include_router(champion_challenger_router)

# Public symbols preserved for backward compatibility with tests & scripts
__all__ = [
    "app",
    "store",
    "score_transaction",
    "_build_feature_row",
    "_risk_label",
    "_canonical_trust_engine",
]
