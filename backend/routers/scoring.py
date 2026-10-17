"""
TrustShield AI — Transaction Scoring & TreeSHAP Explainability Router.

Endpoints:
- POST /transaction/score    -- High-throughput, low-latency transaction fraud scoring
- POST /transaction/explain  -- Local TreeSHAP attribution and investigator narrative synthesis
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from backend.auth import authenticate_request
from backend.features import (
    _build_feature_row,
    _resolve_graph_features,
    _risk_label,
)
from backend.model_loader import store
from backend.schemas import (
    FeatureShapDriver,
    TransactionExplainRequest,
    TransactionExplainResponse,
    TransactionScoreRequest,
    TransactionScoreResponse,
)
from backend.services.metrics_service import metrics_service

try:
    from trustshield_project.advanced_trust_engine import CanonicalTrustEngine
except ImportError:
    try:
        from advanced_trust_engine import CanonicalTrustEngine
    except ImportError:
        CanonicalTrustEngine = None

logger = logging.getLogger(__name__)

router = APIRouter()

VALID_SERVING_MODELS = {"phase3"}
_default_canonical_trust_engine = CanonicalTrustEngine() if CanonicalTrustEngine is not None else None


def get_trust_engine() -> CanonicalTrustEngine:
    """Dependency provider for CanonicalTrustEngine."""
    if _default_canonical_trust_engine is not None:
        return _default_canonical_trust_engine
    if CanonicalTrustEngine is not None:
        return CanonicalTrustEngine()
    raise HTTPException(
        status_code=503,
        detail="Canonical Trust Engine is unavailable. Legacy fallback is disabled."
    )


def _get_serving_model_version(req: TransactionScoreRequest, request: Optional[Request] = None) -> str:
    """Determines model version: environment default or authenticated header/body override."""
    env_default = os.getenv("TRUSTSHIELD_SERVING_MODEL", "phase3").strip().lower()
    if env_default not in VALID_SERVING_MODELS:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid server configuration for TRUSTSHIELD_SERVING_MODEL: '{env_default}'. Valid options: {sorted(list(VALID_SERVING_MODELS))}"
        )

    chosen = req.model_version
    if not chosen and request is not None:
        chosen = request.headers.get("x-trustshield-model-version")

    if not chosen:
        chosen = env_default
    else:
        chosen = chosen.strip().lower()

    if chosen not in VALID_SERVING_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model version requested: '{chosen}'. Valid model options: {sorted(list(VALID_SERVING_MODELS))}"
        )
    return chosen


def _evaluate_and_build_response(
    req: TransactionScoreRequest,
    feature_row: Dict[str, Any],
    prob_raw: float,
    prob_cal: float,
    prob_tabular: float,
    model_version: str,
    is_cold_start: bool,
    graph_source: str,
    graph_notes: str,
    client_mismatch: bool,
    engine: Any,
    explainer: Any,
    calibrator_active: bool,
    gnn_embeddings_active: bool,
    infra_sources: Dict[str, str],
    scoring_mode: str,
    model_used_label: str,
    note_prefix: str,
) -> TransactionScoreResponse:
    graph_sig_heuristic = float(min(feature_row.get("share_degree", 0.0) * 0.15 + feature_row.get("buyer_pagerank", 0.0) * 0.35, 1.0))
    ring_sig_heuristic = float(min(max(feature_row.get("share_component_size", 1.0) - 1, 0) * 0.2, 1.0))
    component_risks = {
        "tabular_risk": prob_tabular,
        "graph_risk_heuristic": graph_sig_heuristic,
        "ring_risk_heuristic": ring_sig_heuristic,
    }
    if req.multimodal_similarity_score is not None:
        component_risks["multimodal_risk"] = float(
            max(0.0, min(1.0, 1.0 - req.multimodal_similarity_score))
        )

    t_te0 = time.perf_counter()
    if engine is not None:
        t_res = engine.score(
            component_risks,
            is_cold_start=is_cold_start,
            model_probability=prob_cal,
            order_amount=float(feature_row.get("amount", 0.0) or feature_row.get("order_amount", 0.0)),
        )
        final_risk = getattr(t_res, "calibrated_risk", getattr(t_res, "risk_score", 0.0))
        decision = t_res.decision.value
        trust_score = t_res.trust_score
        confidence = t_res.confidence
        uncertainty = getattr(t_res, "predictive_uncertainty", None)
        model_disagreement = getattr(t_res, "detector_disagreement", getattr(t_res, "model_disagreement", 0.0))
        reason_codes = t_res.reason_codes
        conformal_set = sorted(list(t_res.conformal_prediction_set)) if hasattr(t_res, "conformal_prediction_set") else None
    else:
        final_risk = prob_cal
        decision = "REVIEW" if final_risk >= 0.25 else "ALLOW"
        trust_score = round(100.0 * (1.0 - final_risk), 2)
        confidence = 0.8
        uncertainty = None
        model_disagreement = 0.0
        reason_codes = []
        conformal_set = [0, 1] if 0.35 <= prob_cal <= 0.65 else ([1] if prob_cal > 0.65 else [0])

    metrics_service.trust_engine_seconds.observe(time.perf_counter() - t_te0)
    metrics_service.record_decision(decision, _risk_label(final_risk), model_version)

    shap_attrs = None
    top_drivers = None
    if explainer is not None:
        try:
            shap_exp = explainer.explain_instance(feature_row, top_k=5)
            if shap_exp.get("available"):
                shap_attrs = shap_exp.get("attributions")
                top_drivers = shap_exp.get("top_positive_drivers")
        except Exception as exc:
            logger.warning("TreeSHAP %s explanation failed: %s", model_version, exc)

    conformal_calibrated = bool(engine.conformal.is_calibrated) if engine and hasattr(engine, "conformal") else False
    conformal_q = round(float(engine.conformal.q_hat), 4) if engine and hasattr(engine, "conformal") and engine.conformal.is_calibrated else None

    return TransactionScoreResponse(
        order_id=req.order_id,
        overall_fraud_probability=round(final_risk, 4),
        raw_fraud_probability=round(prob_raw, 4),
        calibrated_fraud_probability=round(prob_cal, 4),
        policy_risk_score=round(final_risk, 4),
        canonical_trust_engine_risk=round(final_risk, 4),
        conformal_prediction_set=conformal_set,
        conformal_is_calibrated=conformal_calibrated,
        conformal_coverage_guarantee=0.95,
        conformal_q_hat=conformal_q,
        risk_label=_risk_label(final_risk),
        decision=decision,
        trust_score=trust_score,
        confidence=confidence,
        predictive_uncertainty=uncertainty,
        model_used=model_used_label,
        model_version=model_version,
        graph_features_source=graph_source,
        client_mismatch=client_mismatch,
        cold_start=is_cold_start,
        model_disagreement=model_disagreement,
        reason_codes=reason_codes,
        evidence_availability={
            "tabular": True,
            "graph": bool(feature_row.get("share_degree", 0) > 0 or feature_row.get("buyer_pagerank", 0) > 0),
            "gnn_embeddings": gnn_embeddings_active,
            "multimodal": req.multimodal_similarity_score is not None,
            "calibrator_active": calibrator_active,
            "shap_active": shap_attrs is not None,
        },
        infrastructure_sources=infra_sources,
        shap_attributions=shap_attrs,
        top_risk_drivers=top_drivers,
        scoring_mode=scoring_mode,
        is_simulation=False,
        note=note_prefix + (f" [{graph_notes}]" if graph_notes else ""),
    )


@router.post("/transaction/score", response_model=TransactionScoreResponse, tags=["scoring"], dependencies=[Depends(authenticate_request)])
def score_transaction(
    req: TransactionScoreRequest,
    request: Request = None,  # type: ignore[assignment]
    engine: CanonicalTrustEngine = Depends(get_trust_engine),
):
    """
    Score an incoming order for fraud using the Phase 3 Combined model
    (tabular + NetworkX graph features).

    Evaluated via Canonical Trust Engine for calibrated risk, trust score (0-100),
    entropy confidence, model disagreement, and decision routing (ALLOW/REVIEW/HOLD/BLOCK).
    """
    if not store.is_loaded:
        try:
            store.load()
        except Exception as exc:
            logger.warning("Failed to lazy-load models in /transaction/score: %s", exc)
    if not store.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet — try again in a moment.")

    target_model = _get_serving_model_version(req, request)

    is_cold_start = bool(
        (req.buyer_orders_before < 3)
        or (req.seller_total_listings_before < 3)
        or (req.buyer_age_days < 7.0)
        or (req.seller_age_days < 7.0)
    )

    if not isinstance(engine, CanonicalTrustEngine):
        engine = get_trust_engine()
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Canonical Trust Engine is unavailable. Legacy fallback is disabled."
        )

    if getattr(store, "conformal_predictor", None) is not None:
        if not getattr(engine.conformal, "is_calibrated", False):
            engine.conformal = store.conformal_predictor

    # Phase 3 model (authoritative production model)
    if store.feature_meta is None or store.combined_graph_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet — try again in a moment.")

    meta = store.feature_meta
    all_cols = meta["all_feature_cols"]

    graph_resolved, graph_source, graph_notes, client_mismatch = _resolve_graph_features(req)
    feature_row = _build_feature_row(req, all_cols, resolved_graph_features=graph_resolved)
    t_infer0 = time.perf_counter()
    X = pd.DataFrame([feature_row])[all_cols].fillna(0.0)

    prob_raw = float(np.asarray(store.combined_graph_model.predict_proba(X))[0, 1])
    metrics_service.model_inference_seconds.labels(model_version="phase3").observe(
        time.perf_counter() - t_infer0
    )
    prob_cal = float(store.calibrator.predict_proba(np.array([prob_raw]))[0]) if store.calibrator is not None else prob_raw
    clf_name = meta.get("classifier", type(store.combined_graph_model).__name__)

    if getattr(store, "transaction_history", None) is not None and req.buyer_id and req.seller_id and getattr(req, "order_date", None) is None:
        store.transaction_history.record_live_transaction(
            req.buyer_id, req.seller_id, order_id=req.order_id
        )

    infra_sources = {
        "graph_topology": "disk_artifact",
        "embeddings": "not_used_in_phase3",
    }

    return _evaluate_and_build_response(
        req=req,
        feature_row=feature_row,
        prob_raw=prob_raw,
        prob_cal=prob_cal,
        prob_tabular=prob_cal,
        model_version="phase3",
        is_cold_start=is_cold_start,
        graph_source=graph_source,
        graph_notes=graph_notes,
        client_mismatch=client_mismatch,
        engine=engine,
        explainer=store.shap_explainer,
        calibrator_active=store.calibrator is not None,
        gnn_embeddings_active=False,
        infra_sources=infra_sources,
        scoring_mode="production_model_phase3_combined",
        model_used_label=f"tabular+graph {clf_name} (Phase 3)",
        note_prefix=f"Scored with Phase 3 tabular+graph model. Graph features source: {graph_source}.",
    )


@router.post("/transaction/explain", response_model=TransactionExplainResponse, tags=["explainability"], dependencies=[Depends(authenticate_request)])
def explain_transaction(req: TransactionExplainRequest):
    """
    Generate local Shapley additive explanations (TreeSHAP) for a given transaction.
    Computes exact marginal contributions, separates risk-amplifying drivers from
    protective trust dampeners, and synthesizes an investigator narrative.
    """
    if not store.is_loaded:
        try:
            store.load()
        except Exception as exc:
            logger.warning("Failed to lazy-load models in /transaction/explain: %s", exc)
    if not store.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    tx_dict: Dict[str, Any] = {}
    if req.transaction_data:
        tx_dict = dict(req.transaction_data)
    else:
        tx_dict = req.model_dump(exclude={"order_id", "transaction_data"})

    order_id = req.order_id or tx_dict.get("order_id", "ORDER_EXPLAIN_PROBE")
    tx_dict["order_id"] = order_id

    score_req = TransactionScoreRequest(**{
        k: v for k, v in tx_dict.items()
        if k in TransactionScoreRequest.model_fields
    })
    score_res = score_transaction(score_req)

    explanation: Dict[str, Any] = {}
    if store.shap_explainer is not None and store.feature_meta is not None:
        all_cols = store.feature_meta.get("all_feature_cols", [])
        feature_row = _build_feature_row(score_req, all_cols)
        explanation = store.shap_explainer.explain_instance(feature_row, top_k=5)
    else:
        explanation = {
            "available": False,
            "base_value": 0.0,
            "attributions": {},
            "top_positive_drivers": [],
            "top_negative_dampeners": [],
            "narrative": "TreeSHAP explainer is not loaded or configured for this environment.",
        }

    pos_drivers = [
        FeatureShapDriver(
            feature_name=d["feature_name"],
            friendly_name=d["friendly_name"],
            feature_value=float(d["feature_value"]),
            shap_value=float(d["shap_value"]),
            abs_impact=float(d["abs_impact"]),
        )
        for d in explanation.get("top_positive_drivers", [])
    ]
    neg_dampeners = [
        FeatureShapDriver(
            feature_name=d["feature_name"],
            friendly_name=d["friendly_name"],
            feature_value=float(d["feature_value"]),
            shap_value=float(d["shap_value"]),
            abs_impact=float(d["abs_impact"]),
        )
        for d in explanation.get("top_negative_dampeners", [])
    ]

    return TransactionExplainResponse(
        order_id=order_id,
        base_value=float(explanation.get("base_value", 0.0)),
        overall_fraud_probability=score_res.overall_fraud_probability,
        decision=score_res.decision,
        model_used=score_res.model_used,
        top_positive_drivers=pos_drivers,
        top_negative_dampeners=neg_dampeners,
        all_attributions=explanation.get("attributions", {}),
        investigator_narrative=explanation.get("narrative", "No narrative generated."),
    )
