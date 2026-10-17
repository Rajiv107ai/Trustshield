"""
TrustShield AI — FastAPI Application Entry Point & Serving Layer.

Composes domain routers, application lifespan, Prometheus metrics middleware,
and security policies:
- Scoring Router:     /transaction/score, /transaction/explain
- Operations Router:  /fraud-rings, /listing/analyze, /return/analyze, /investigation/generate-dossier
- System Router:      /health, /ready, /results, /metrics, /system/benchmark

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

# Ensure repository root is on sys.path
_ROOT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from backend.auth import validate_security_configuration
from backend.features import _build_feature_row, _risk_label
from backend.model_loader import store
from backend.routers.operations import router as operations_router
from backend.routers.scoring import router as scoring_router, score_transaction
from backend.routers.system import router as system_router
from backend.services.metrics_service import metrics_service

try:
    from trustshield_project.advanced_trust_engine import CanonicalTrustEngine
except ImportError:
    try:
        from advanced_trust_engine import CanonicalTrustEngine
    except ImportError:
        CanonicalTrustEngine = None

logger = logging.getLogger(__name__)

# CanonicalTrustEngine is the sole authoritative production engine.
_canonical_trust_engine = CanonicalTrustEngine() if CanonicalTrustEngine is not None else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup."""
    validate_security_configuration()
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
app.include_router(system_router)

# Public symbols preserved for backward compatibility with tests & scripts
__all__ = [
    "app",
    "store",
    "score_transaction",
    "_build_feature_row",
    "_risk_label",
    "_canonical_trust_engine",
]
