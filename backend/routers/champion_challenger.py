"""TrustShield AI — Champion-Challenger Shadow Scoring & Conformal A/B API Router.

Endpoints:
- GET  /models/champion-challenger/status          -- Get live concordance metrics, deployment mode, and uncertainty comparison
- POST /models/champion-challenger/configure       -- Update deployment mode (SHADOW/AB_SPLIT/CANARY) and traffic split
- POST /models/champion-challenger/score-dual      -- Evaluate transaction on both models with side-by-side conformal sets
- GET  /models/champion-challenger/disagreements   -- Retrieve recent edge-case disagreements
- POST /models/champion-challenger/reset-metrics   -- Clear comparison history (operator tool)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from backend.auth import Role, require_role, authenticate_request
from backend.features import _build_feature_row, _resolve_graph_features
from backend.model_loader import store
from backend.schemas import (
    ChampionChallengerComparisonResponse,
    ChampionChallengerConfigureRequest,
    ChampionChallengerScoreDualRequest,
    ChampionChallengerStatusResponse,
    TransactionScoreRequest,
)
from backend.services.champion_challenger_service import champion_challenger_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models/champion-challenger", tags=["champion-challenger"])


@router.get("/status", response_model=ChampionChallengerStatusResponse)
def get_champion_challenger_status():
    """Retrieve live concordance rate, conformal uncertainty comparison, and deployment policy."""
    if not store.is_loaded:
        try:
            store.load()
        except Exception as exc:
            logger.debug("Lazy load in champion-challenger status: %s", exc)

    return champion_challenger_service.get_status()


@router.post(
    "/configure",
    response_model=ChampionChallengerStatusResponse,
    dependencies=[Depends(require_role(Role.OPERATOR))],
)
def configure_deployment_mode(payload: ChampionChallengerConfigureRequest):
    """Update active deployment policy (SHADOW, AB_SPLIT, CANARY) and traffic split percentage."""
    try:
        return champion_challenger_service.configure(
            deployment_mode=payload.deployment_mode,
            traffic_split_pct=payload.traffic_split_pct,
            challenger_model_id=payload.challenger_model_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post(
    "/score-dual",
    response_model=ChampionChallengerComparisonResponse,
    dependencies=[Depends(authenticate_request)],
)
def score_dual_transaction(req: ChampionChallengerScoreDualRequest):
    """Run transaction through both Champion and Challenger models simultaneously.

    Returns side-by-side calibrated probabilities, operational decisions,
    conformal prediction sets, latency diffs, and deterministic routing.
    """
    if not store.is_loaded:
        try:
            store.load()
        except Exception as exc:
            logger.debug("Lazy load in score-dual: %s", exc)

    # Convert request to feature dictionary
    tx_dict: Dict[str, Any] = {}
    if req.transaction_data:
        tx_dict = dict(req.transaction_data)
    else:
        tx_dict = req.model_dump(exclude={"order_id", "transaction_data"})

    order_id = req.order_id or tx_dict.get("order_id", "PROBE_TX")
    tx_dict["order_id"] = order_id

    # Create dummy TransactionScoreRequest to resolve graph features if needed
    try:
        score_req = TransactionScoreRequest(**{
            k: v for k, v in tx_dict.items()
            if k in TransactionScoreRequest.model_fields
        })
        graph_resolved, _, _, _ = _resolve_graph_features(score_req)
        all_cols = store.feature_meta.get("all_feature_cols", []) if store.feature_meta else list(tx_dict.keys())
        feature_row = _build_feature_row(score_req, all_cols, resolved_graph_features=graph_resolved)
    except Exception:
        # Fallback to direct dictionary
        feature_row = tx_dict

    comparison = champion_challenger_service.score_dual(feature_row, order_id=order_id)
    return comparison


@router.get(
    "/disagreements",
    response_model=List[ChampionChallengerComparisonResponse],
    dependencies=[Depends(authenticate_request)],
)
def get_recent_disagreements(limit: int = 20):
    """Retrieve recent transactions where Champion and Challenger produced divergent decisions or conformal sets."""
    return champion_challenger_service.get_disagreements(limit=limit)


@router.post(
    "/reset-metrics",
    response_model=ChampionChallengerStatusResponse,
    dependencies=[Depends(require_role(Role.OPERATOR))],
)
def reset_comparison_metrics():
    """Clear in-memory comparison buffer and reset circuit breakers."""
    champion_challenger_service.reset_metrics()
    return champion_challenger_service.get_status()
