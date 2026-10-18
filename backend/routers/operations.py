"""
TrustShield AI — Specialized Models & Operational Workflows Router.

Endpoints:
- GET  /fraud-rings                    -- Pre-computed high-risk graph clusters and rings
- POST /listing/analyze                -- Multimodal fake listing detection (CLIP + XGBoost)
- POST /return/analyze & /return/score -- Return abuse & velocity scoring
- POST /investigation/generate-dossier -- Evidence-grounded forensic dossier generation
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from backend.auth import Role, authenticate_request, require_role
from backend.features import _risk_label
from backend.model_loader import store
from backend.routers.scoring import score_transaction
from backend.schemas import (
    DossierRequest,
    DossierResponse,
    FraudRingItem,
    FraudRingsResponse,
    ListingScoreRequest,
    ListingScoreResponse,
    ReturnScoreRequest,
    ReturnScoreResponse,
    StreamTransactionEvent,
    TransactionScoreRequest,
)
from backend.services.investigation_service import investigation_service
from backend.services.metrics_service import metrics_service

try:
    from trustshield_project.multimodal_scoring import MultimodalScorer as _MultimodalScorer
    from trustshield_project.multimodal_scoring import explain_listing_risk
except ImportError:
    try:
        from multimodal_scoring import MultimodalScorer as _MultimodalScorer
        from multimodal_scoring import explain_listing_risk
    except ImportError:
        explain_listing_risk = None
        _MultimodalScorer = None

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/fraud-rings", response_model=FraudRingsResponse, tags=["rings"], dependencies=[Depends(authenticate_request)])
def get_fraud_rings(
    min_risk_score: float = Query(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Filter: only return rings with avg_risk_score >= this value. "
                    "Use 0.3 for medium+ risk, 0.5 for high risk only.",
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of rings to return (sorted by avg_risk_score desc).",
    ),
):
    """
    Return pre-computed suspected fraud rings, ranked by average predicted
    fraud probability across member orders.
    """
    if not store.is_loaded or store.rings_df is None:
        raise HTTPException(status_code=503, detail="Ring data not loaded yet.")

    df = store.rings_df
    if min_risk_score > 0:
        df = df[df["avg_risk_score"] >= min_risk_score]

    df = df.head(limit)

    ring_items: List[FraudRingItem] = []
    for _, row in df.iterrows():
        r = row.to_dict()
        raw_members = r.get("members")
        members_list = list(raw_members) if isinstance(raw_members, (list, tuple)) else []
        avg_score = float(r.get("avg_risk_score", 0.0))
        ring_items.append(FraudRingItem(
            ring_id=str(r.get("ring_id", "")),
            size=int(r.get("size", 0)),
            n_orders=int(r.get("n_orders", 0)),
            avg_risk_score=avg_score,
            max_risk_score=float(r.get("max_risk_score", 0.0)),
            n_high_risk_orders=int(r.get("n_high_risk_orders", 0)),
            risk_label=_risk_label(avg_score),
            members=members_list,
        ))

    high_risk = int((store.rings_df["avg_risk_score"] >= 0.5).sum())

    return FraudRingsResponse(
        total_rings=len(store.rings_df),
        high_risk_rings=high_risk,
        rings=ring_items,
    )


@router.post("/listing/analyze", response_model=ListingScoreResponse, tags=["scoring"], dependencies=[Depends(authenticate_request)])
def analyze_listing(req: ListingScoreRequest):
    """
    Score a product listing for fraud / fake-listing risk using the Phase 4
    multimodal model (trained on price anomalies, seller profile, and
    image-text alignment via real CLIP embeddings).
    """
    if not store.is_loaded or store.fake_listing_model is None or store.feature_meta is None:
        raise HTTPException(status_code=503, detail="Listing model not loaded yet.")

    model_fn = getattr(store.fake_listing_model, "feature_names_in_", None)
    if model_fn is None and hasattr(store.fake_listing_model, "get_booster"):
        try:
            model_fn = store.fake_listing_model.get_booster().feature_names
        except Exception:
            model_fn = None
    cols = list(model_fn) if model_fn is not None else store.feature_meta.get("listing_feature_cols", [
        "price_vs_base_price_ratio", "price_vs_category_median_ratio",
        "seller_age_days_at_listing", "seller_listings_before",
        "synthetic_mismatch_score",
    ])

    price_base_ratio = req.price / max(req.base_price, 0.01)
    price_cat_ratio = req.price / max(req.category_median_price, 0.01)

    clip_scored = False
    tfidf_scored = False
    sim_score = req.synthetic_mismatch_score if req.synthetic_mismatch_score is not None else req.multimodal_similarity_score

    if req.product_id is not None and store.clip_loaded and store.clip_scorer is not None:
        try:
            displayed_pid = req.displayed_product_id or req.product_id
            listing_row = pd.DataFrame([
                {
                    "product_id": req.product_id,
                    "displayed_product_id": displayed_pid,
                }
            ])
            clip_sim = store.clip_scorer.score_listings(listing_row)
            sim_score = float(clip_sim.iloc[0])
            clip_scored = True
        except Exception as exc:
            logger.warning(
                "CLIP scoring failed for listing %s (%s). Falling back to TF-IDF surrogate.",
                req.listing_id, exc,
            )

    if not clip_scored and req.product_id is not None and store.clip_products_df is not None:
        try:
            if _MultimodalScorer is not None:
                if getattr(store, "_tfidf_scorer", None) is None:
                    _tfidf = _MultimodalScorer()
                    _tfidf.fit(store.clip_products_df)
                    store._tfidf_scorer = _tfidf
                else:
                    _tfidf = store._tfidf_scorer
                displayed_pid = req.displayed_product_id or req.product_id
                listing_row = pd.DataFrame([
                    {
                        "product_id": req.product_id,
                        "displayed_product_id": displayed_pid,
                    }
                ])
                sim_score = float(_tfidf.score_listings(listing_row).iloc[0])
                tfidf_scored = True
        except Exception as exc2:
            logger.warning(
                "TF-IDF fallback also failed for listing %s (%s). Using caller-supplied multimodal_similarity_score=%.3f.",
                req.listing_id, exc2, req.multimodal_similarity_score,
            )

    features = {
        "price_vs_base_price_ratio": price_base_ratio,
        "price_vs_category_median_ratio": price_cat_ratio,
        "seller_age_days_at_listing": req.seller_age_days_at_listing,
        "seller_listings_before": req.seller_listings_before,
        "multimodal_similarity_score": sim_score,
        "synthetic_mismatch_score": sim_score,
    }

    X = pd.DataFrame([{c: features.get(c, 0.0) for c in cols}])[cols].fillna(0.0)
    prob = float(np.asarray(store.fake_listing_model.predict_proba(X))[0, 1])

    narrative = explain_listing_risk(features, prob) if explain_listing_risk is not None else {
        "risk_score": round(prob, 4),
        "risk_level": _risk_label(prob).upper(),
        "summary": f"Listing risk score: {prob:.2f}",
        "key_evidence": [],
        "recommended_action": "Review listing.",
    }

    return ListingScoreResponse(
        listing_id=req.listing_id,
        fake_listing_probability=round(prob, 4),
        risk_label=_risk_label(prob),
        model_used=(
            "Fake Listing Detector (Phase 4 CLIP + XGBoost)"
            if clip_scored
            else (
                "Fake Listing Detector (Phase 4 TF-IDF Fallback + XGBoost)"
                if tfidf_scored
                else "Fake Listing Detector (Phase 4 Multimodal XGBoost, caller-supplied similarity)"
            )
        ),
        clip_scored=clip_scored,
        multimodal_similarity_score=round(sim_score, 4),
        investigator_narrative=narrative,
    )


@router.post("/return/analyze", response_model=ReturnScoreResponse, tags=["scoring"], dependencies=[Depends(authenticate_request)])
@router.post("/return/score", response_model=ReturnScoreResponse, tags=["scoring"], include_in_schema=False, dependencies=[Depends(authenticate_request)])
def analyze_return(req: ReturnScoreRequest):
    """
    Score a return request for fraud / abuse using the Phase 2 specialized Return Fraud Detector
    trained on return velocity, account tenure, order amount, and return reasons.
    """
    if not store.is_loaded or store.return_fraud_model is None or store.feature_meta is None:
        raise HTTPException(status_code=503, detail="Return model not loaded yet.")

    cols = store.feature_meta.get("return_feature_cols", [
        "days_to_return", "buyer_age_days_at_return", "seller_age_days_at_return", "order_amount",
        "buyer_prior_returns", "buyer_orders_before_return", "buyer_return_rate_before",
        "seller_prior_returns", "seller_orders_before_return", "seller_return_rate_before",
        "reason_changed_mind", "reason_defective", "reason_size_issue", "reason_wrong_item_received",
    ])

    clean_reason = req.reason.strip().lower() if isinstance(req.reason, str) else ""
    ret_rate = req.buyer_return_rate_before
    if req.buyer_orders_before_return > 0 and req.buyer_prior_returns > 0 and ret_rate == 0.0:
        ret_rate = float(min(1.0, req.buyer_prior_returns / req.buyer_orders_before_return))

    features = {
        "days_to_return": req.days_to_return,
        "buyer_age_days_at_return": req.buyer_age_days_at_return,
        "seller_age_days_at_return": req.seller_age_days_at_return,
        "order_amount": req.order_amount,
        "buyer_prior_returns": float(req.buyer_prior_returns),
        "buyer_orders_before_return": float(req.buyer_orders_before_return),
        "buyer_return_rate_before": ret_rate,
        "seller_prior_returns": float(req.seller_prior_returns),
        "seller_orders_before_return": float(req.seller_orders_before_return),
        "seller_return_rate_before": req.seller_return_rate_before,
        "reason_changed_mind": 1.0 if clean_reason == "changed_mind" else 0.0,
        "reason_defective": 1.0 if clean_reason == "defective" else 0.0,
        "reason_size_issue": 1.0 if clean_reason == "size_issue" else 0.0,
        "reason_wrong_item_received": 1.0 if clean_reason == "wrong_item_received" else 0.0,
    }

    X = pd.DataFrame([{c: features.get(c, 0.0) for c in cols}])[cols].fillna(0.0)
    prob = float(np.asarray(store.return_fraud_model.predict_proba(X))[0, 1])

    decision = "BLOCK" if prob >= 0.85 else ("HOLD" if prob >= 0.55 else ("REVIEW" if prob >= 0.25 else "ALLOW"))

    reasons = []
    if req.buyer_return_rate_before >= 0.50:
        reasons.append("HIGH_BUYER_RETURN_RATE_BEFORE")
    if req.buyer_prior_returns >= 3:
        reasons.append("REPEAT_RETURN_ABUSER")
    if req.days_to_return <= 1.0:
        reasons.append("RAPID_RETURN_VELOCITY")
    if prob >= 0.50:
        reasons.append("RETURN_FRAUD_MODEL_SCORE_ELEVATED")

    return ReturnScoreResponse(
        return_id=req.return_id,
        return_fraud_probability=round(prob, 4),
        risk_label=_risk_label(prob),
        decision=decision,
        model_used="Return Fraud Detector (Phase 2 Specialized Model)",
        reason_codes=reasons,
    )


@router.post("/investigation/generate-dossier", response_model=DossierResponse, tags=["investigation"], dependencies=[Depends(require_role(Role.OPERATOR))])
def generate_forensic_dossier(req: DossierRequest):
    """
    Generate an evidence-grounded forensic dossier for human fraud operations.
    """
    if not store.is_loaded:
        try:
            store.load()
        except Exception:
            raise HTTPException(status_code=503, detail="Models not loaded yet.")

    t0 = time.perf_counter()
    try:
        dossier = investigation_service.generate_dossier(
            req=req,
            scoring_fn=score_transaction,
            store_instance=store,
        )
        duration = time.perf_counter() - t0
        metrics_service.dossier_duration.observe(duration)
        metrics_service.dossier_total.labels(entity_type=req.entity_type, status="success").inc()
        return dossier
    except HTTPException:
        metrics_service.dossier_total.labels(entity_type=req.entity_type, status="error").inc()
        raise
    except Exception as exc:
        metrics_service.dossier_total.labels(entity_type=req.entity_type, status="error").inc()
        logger.error("Dossier generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal dossier generation failed.")


STREAM_SEEDS = [
    {
        "buyer_id": "BUYER_STREAM_01",
        "seller_id": "SELLER_STREAM_01",
        "amount": 45.0,
        "base_price": 50.0,
        "category_median_price": 48.0,
        "buyer_orders_before": 12,
        "buyer_returns_before": 0,
        "buyer_age_days": 150.0,
        "seller_age_days": 320.0,
        "seller_total_listings_before": 80,
    },
    {
        "buyer_id": "BUYER_STREAM_02",
        "seller_id": "SELLER_STREAM_02",
        "amount": 890.0,
        "base_price": 420.0,
        "category_median_price": 400.0,
        "buyer_orders_before": 1,
        "buyer_returns_before": 0,
        "buyer_age_days": 2.0,
        "seller_age_days": 15.0,
        "seller_total_listings_before": 6,
        "device_shared_buyer_count": 5.0,
        "share_degree": 4.0,
        "share_component_size": 6.0,
    },
    {
        "buyer_id": "BUYER_STREAM_03",
        "seller_id": "SELLER_STREAM_03",
        "amount": 320.0,
        "base_price": 320.0,
        "category_median_price": 280.0,
        "buyer_orders_before": 6,
        "buyer_returns_before": 5,
        "buyer_age_days": 90.0,
        "seller_age_days": 200.0,
        "seller_total_listings_before": 45,
    },
    {
        "buyer_id": "BUYER_STREAM_04",
        "seller_id": "SELLER_STREAM_04",
        "amount": 125.0,
        "base_price": 130.0,
        "category_median_price": 128.0,
        "buyer_orders_before": 5,
        "buyer_returns_before": 0,
        "buyer_age_days": 60.0,
        "seller_age_days": 180.0,
        "seller_total_listings_before": 25,
    },
]


@router.get("/stream/transactions", tags=["streaming"], dependencies=[Depends(authenticate_request)])
@router.get("/transactions/stream", tags=["streaming"], include_in_schema=False, dependencies=[Depends(authenticate_request)])
async def stream_transactions(
    request: Request,
    interval: float = Query(default=3.0, ge=0.5, le=10.0, description="Stream event interval in seconds"),
    limit: Optional[int] = Query(default=None, ge=1, le=1000, description="Optional maximum event count"),
):
    """
    Real-time Server-Sent Events (SSE) live transaction feed.
    Streams continuously evaluated incoming marketplace transactions
    scored on the fly by the Canonical Trust Engine.
    """
    if not store.is_loaded:
        try:
            store.load()
        except Exception:
            raise HTTPException(status_code=503, detail="Models not loaded yet.")

    async def event_generator():
        seq = 1000
        count = 0
        while not await request.is_disconnected():
            if limit is not None and count >= limit:
                break
            tmpl = dict(STREAM_SEEDS[seq % len(STREAM_SEEDS)])
            seq += 1
            count += 1
            ord_id = f"ORD_LIVE_{79000 + seq}"
            tmpl["order_id"] = ord_id

            score_req = TransactionScoreRequest(**tmpl)
            score_res = score_transaction(score_req)
            now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")

            event = StreamTransactionEvent(
                event_id=f"EVT_{ord_id}",
                timestamp=now_str,
                order_id=ord_id,
                buyer_id=score_req.buyer_id or "",
                seller_id=score_req.seller_id or "",
                amount=score_req.amount or 0.0,
                risk_score=score_res.overall_fraud_probability,
                calibrated_risk=float(score_res.calibrated_fraud_probability if score_res.calibrated_fraud_probability is not None else score_res.overall_fraud_probability),
                policy_risk_score=score_res.policy_risk_score,
                risk_level=score_res.risk_label,
                decision=score_res.decision,
                trust_score=score_res.trust_score,
                reason_codes=score_res.reason_codes,
                source="live_stream",
                is_simulation=False,
            )
            data_str = event.model_dump_json()
            yield f"data: {data_str}\n\n"
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
