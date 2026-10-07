"""
TrustShield AI — FastAPI serving layer (MVP)

Three endpoints (as required by phases.md MVP scope):

    GET  /health               -- liveness + model-loaded status
    POST /transaction/score    -- score an incoming order for fraud
    GET  /fraud-rings          -- return pre-computed ranked fraud ring list

Design constraints (per project rules):
- All responses return REAL model output — no hardcoded demo data.
- Models are loaded once at startup from joblib artifacts.
- Run scripts/train_and_save_models.py before starting this server.

Feature mapping:
- The request schema (schemas.py) uses the EXACT feature column names the
  trained model expects.  When a caller supplies raw inputs (amount,
  base_price, category_median_price) instead of pre-computed ratios, this
  module derives the ratio features before scoring.

Start:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Docs auto-generated at: http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

try:
    from backend.model_loader import store
    from backend.schemas import (
        TransactionScoreRequest, TransactionScoreResponse,
        FraudRingsResponse, FraudRingItem,
        HealthResponse, ReadyResponse, ServiceState,
        ListingScoreRequest, ListingScoreResponse,
        ReturnScoreRequest, ReturnScoreResponse,
    )
except ImportError:
    from model_loader import store  # type: ignore[import-not-found]
    from schemas import (  # type: ignore[import-not-found]
        TransactionScoreRequest, TransactionScoreResponse,
        FraudRingsResponse, FraudRingItem,
        HealthResponse, ReadyResponse, ServiceState,
        ListingScoreRequest, ListingScoreResponse,
        ReturnScoreRequest, ReturnScoreResponse,
    )

try:
    from trustshield_project.advanced_trust_engine import CanonicalTrustEngine, OperationalDecision
except ImportError:
    try:
        from advanced_trust_engine import CanonicalTrustEngine, OperationalDecision
    except ImportError:
        CanonicalTrustEngine = None
        OperationalDecision = None

try:
    from trustshield_project.trust_engine import TrustEngine
except ImportError:
    try:
        from trust_engine import TrustEngine
    except ImportError:
        TrustEngine = None

try:
    from trustshield_project.multimodal_scoring import explain_listing_risk, MultimodalScorer as _MultimodalScorer
except ImportError:
    try:
        from multimodal_scoring import explain_listing_risk, MultimodalScorer as _MultimodalScorer
    except ImportError:
        explain_listing_risk = None
        _MultimodalScorer = None

_canonical_trust_engine = CanonicalTrustEngine() if CanonicalTrustEngine is not None else None
_trust_engine = TrustEngine() if TrustEngine is not None else None


def _risk_label(score: float) -> str:
    """Canonical risk label mapping aligned with Trust Engine thresholds (0.25, 0.55)."""
    if score >= 0.55:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def _build_feature_row(req: TransactionScoreRequest, all_cols: list) -> dict:
    """
    Convert a TransactionScoreRequest into a feature dict keyed by the
    exact model column names.

    Priority:
      1. Directly supplied feature fields (e.g. price_vs_base_price_ratio != 0)
         are used as-is.
      2. If a ratio field is 0 (default) AND the raw inputs needed to compute
         it are present, the ratio is derived server-side.
      3. Everything else defaults to 0.

    Amount harmonization:
      - ``amount`` (Optional[float], default None) and ``order_amount`` (float,
        default 0.0) are aliases for the same concept.
      - Whichever field the caller explicitly provides wins.
      - If both are provided and disagree by more than 1e-6, a 400 is raised.
      - ``model_fields_set`` (Pydantic v2) is the authoritative way to detect
        explicit supply; we never rely on sentinel-value comparisons.
    """
    row = req.model_dump()

    # --- Determine which amount alias(es) the caller explicitly provided ---
    # model_fields_set is always available in Pydantic v2; fall back for safety.
    fields_set: set = getattr(req, "model_fields_set", set())
    has_amount       = "amount"       in fields_set
    has_order_amount = "order_amount" in fields_set

    amount_val       = row.get("amount")        # Optional[float], None when not set
    order_amount_val = row.get("order_amount")  # float, 0.0 when not set

    if has_amount and has_order_amount:
        # Both explicitly provided — they must agree.
        amt     = float(amount_val)       if amount_val       is not None else 0.0
        ord_amt = float(order_amount_val) if order_amount_val is not None else 0.0
        if abs(amt - ord_amt) > 1e-6:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Conflicting 'amount' ({amt}) and 'order_amount' ({ord_amt}) "
                    "provided. Supply only one, or ensure they are equal."
                ),
            )
        resolved_amount = amt
    elif has_amount:
        # Only `amount` was explicitly supplied (order_amount at default 0.0).
        resolved_amount = float(amount_val) if amount_val is not None else 0.0
    elif has_order_amount:
        # Only `order_amount` was explicitly supplied.
        resolved_amount = float(order_amount_val) if order_amount_val is not None else 0.0
    else:
        # Neither explicitly supplied — use whatever non-None value we have.
        resolved_amount = float(amount_val) if amount_val is not None else float(order_amount_val or 0.0)

    # Ensure both aliases carry the resolved value so downstream ratio derivation
    # can reference either key without distinction.
    row["amount"]       = resolved_amount
    row["order_amount"] = resolved_amount

    # --- Derive price_vs_base_price_ratio from raw inputs if not given ---
    if row.get("price_vs_base_price_ratio", 0.0) == 0.0:
        base_price = row.get("base_price")
        if resolved_amount and base_price and base_price > 0:
            row["price_vs_base_price_ratio"] = resolved_amount / base_price

    # --- Derive price_vs_category_median_ratio from raw inputs if not given ---
    if row.get("price_vs_category_median_ratio", 0.0) == 0.0:
        cat_median = row.get("category_median_price")
        if resolved_amount and cat_median and cat_median > 0:
            row["price_vs_category_median_ratio"] = resolved_amount / cat_median

    # --- Derive buyer_return_rate_before from counts if not given ---
    if row.get("buyer_return_rate_before", 0.0) == 0.0:
        orders_before = row.get("buyer_orders_before", 0) or 0
        returns_before = row.get("buyer_returns_before", 0) or 0
        row["buyer_return_rate_before"] = min(1.0, float(returns_before) / max(orders_before, 1))

    # Clamp device_shared_buyer_count to at least 1.0 (self) if missing or <= 0
    dev_count = row.get("device_shared_buyer_count")
    if dev_count is None or dev_count <= 0:
        row["device_shared_buyer_count"] = 1.0

    # Build the final feature vector using only the columns the model expects
    return {col: (row.get(col) if row.get(col) is not None else 0.0) for col in all_cols}


def _build_hybrid_feature_row(
    req: TransactionScoreRequest,
    hybrid_cols: list,
    p3_cols: list,
    buyer_embs: dict,
    seller_embs: dict,
    emb_dim: int = 16,
) -> dict:
    """
    Build a feature dict for Phase 5 hybrid scoring by combining tabular+graph
    features with pre-computed GNN buyer/seller embeddings. Cold-start nodes
    default to zero embeddings.
    """
    row = _build_feature_row(req, p3_cols)
    zero = [0.0] * emb_dim
    b_vec = buyer_embs.get(req.buyer_id, zero) if req.buyer_id else zero
    s_vec = seller_embs.get(req.seller_id, zero) if req.seller_id else zero

    for i in range(emb_dim):
        row[f"gnn_buyer_emb_{i}"] = float(b_vec[i])
        row[f"gnn_seller_emb_{i}"] = float(s_vec[i])

    return {col: row.get(col, 0.0) for col in hybrid_cols}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup; nothing to clean up on shutdown."""
    store.load()
    yield


app = FastAPI(
    title="TrustShield AI",
    description=(
        "E-commerce fraud intelligence API. "
        "Returns real model output from Phase 2 (specialized detectors), "
        "Phase 3 (graph-augmented combined model + fraud ring detection), "
        "and Phase 5 (hybrid GNN + XGBoost). "
        "All predictions are from trained XGBoost models — "
        "no hardcoded or mocked responses."
    ),
    version="0.1.0-mvp",
    lifespan=lifespan,
)

# Production security & dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
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
        rings_loaded=store.rings_df is not None,
        n_rings=len(store.rings_df) if store.rings_df is not None else 0,
    )


# ---------------------------------------------------------------------------
# GET /ready - Readiness probe
# ---------------------------------------------------------------------------

@app.get("/ready", response_model=ReadyResponse, tags=["system"])
def ready(response: Response):
    """
    Readiness probe for orchestrators. Returns service state:
    - 'ready' when models and graph data are fully loaded
    - 'degraded' when Phase 3 baseline runs without Phase 5/CLIP
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
    rings_ready = store.rings_df is not None

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
        rings_ready=rings_ready,
        details={
            "n_rings": len(store.rings_df) if store.rings_df is not None else 0,
            "phase5_loaded": phase5_ready,
            "clip_loaded": clip_ready,
        },
    )


# ---------------------------------------------------------------------------
# POST /transaction/score
# ---------------------------------------------------------------------------

@app.post("/transaction/score", response_model=TransactionScoreResponse, tags=["scoring"])
def score_transaction(req: TransactionScoreRequest):
    """
    Score an incoming order for fraud using either the Phase 5 Hybrid model
    (tabular + NetworkX graph features + 32-dim GNN embeddings) when loaded,
    or falling back to the Phase 3 combined tabular+graph Random Forest model.

    Evaluated via Unified Trust Engine for calibrated risk, trust score (0-100),
    entropy confidence, model disagreement, and decision routing (ALLOW/REVIEW/HOLD/BLOCK).
    """
    if not store.is_loaded:
        try:
            store.load()
        except Exception as exc:
            logger.warning("Failed to lazy-load models in /transaction/score: %s", exc)
    if not store.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet — try again in a moment.")

    # Phase 10: Multi-signal cold-start detection (insufficient buyer/seller history)
    is_cold_start = bool(
        (req.buyer_orders_before < 3)
        or (req.seller_total_listings_before < 3)
        or (req.buyer_age_days < 7.0)
        or (req.seller_age_days < 7.0)
    )

    if store.phase5_loaded and store.phase5_meta is not None and store.hybrid_model is not None:
        p5_meta = store.phase5_meta
        hybrid_cols = p5_meta["hybrid_feature_cols"]
        p3_cols = p5_meta["phase3_feature_cols"]
        emb_dim = p5_meta.get("emb_dim", 16)
        row = _build_hybrid_feature_row(
            req, hybrid_cols, p3_cols,
            store.buyer_embeddings, store.seller_embeddings, emb_dim
        )
        X = pd.DataFrame([row])[hybrid_cols].fillna(0.0)
        prob_raw = float(np.asarray(store.hybrid_model.predict_proba(X))[0, 1])
        clf_name = p5_meta.get("classifier", "XGBoost")

        # Phase 6: Calibrated probability
        if store.phase5_calibrator is not None:
            prob_cal = float(store.phase5_calibrator.predict_proba(np.array([prob_raw]))[0])
        else:
            prob_cal = prob_raw

        # TS-AUD-03: Score Phase 3 tabular+graph model distinctly
        if store.combined_graph_model is not None and store.feature_meta is not None:
            p3_all_cols = store.feature_meta.get("all_feature_cols", p3_cols)
            p3_row = _build_feature_row(req, p3_all_cols)
            X_p3 = pd.DataFrame([p3_row])[p3_all_cols].fillna(0.0)
            raw_tab = float(np.asarray(store.combined_graph_model.predict_proba(X_p3))[0, 1])
            prob_tabular = float(store.calibrator.predict_proba(np.array([raw_tab]))[0]) if store.calibrator is not None else raw_tab
        else:
            prob_tabular = float(prob_cal)

        # Canonical Trust Engine scoring
        graph_sig = float(min(req.share_degree * 0.15 + req.buyer_pagerank * 0.35, 1.0))
        ring_sig = float(min(max(req.share_component_size - 1, 0) * 0.2, 1.0))
        component_risks = {
            "tabular_risk": prob_tabular,
            "graph_risk": graph_sig,
            "hetero_gnn_risk": prob_cal,
            "ring_risk": ring_sig,
        }
        if req.multimodal_similarity_score is not None:
            component_risks["multimodal_risk"] = float(
                max(0.0, min(1.0, 1.0 - req.multimodal_similarity_score))
            )

        engine = _canonical_trust_engine or _trust_engine
        if engine is not None:
            t_res = engine.score(component_risks, is_cold_start=is_cold_start)
            final_risk = getattr(t_res, "calibrated_risk", getattr(t_res, "risk_score", 0.0))
            decision = t_res.decision.value
            trust_score = t_res.trust_score
            confidence = t_res.confidence
            uncertainty = getattr(t_res, "predictive_uncertainty", None)
            model_disagreement = getattr(t_res, "detector_disagreement", getattr(t_res, "model_disagreement", 0.0))
            reason_codes = t_res.reason_codes
        else:
            final_risk = prob_cal
            decision = "REVIEW" if final_risk >= 0.25 else "ALLOW"
            trust_score = round(100.0 * (1.0 - final_risk), 2)
            confidence = 0.8
            uncertainty = None
            model_disagreement = 0.0
            reason_codes = []

        return TransactionScoreResponse(
            order_id=req.order_id,
            overall_fraud_probability=round(final_risk, 4),
            raw_fraud_probability=round(prob_raw, 4),
            calibrated_fraud_probability=round(prob_cal, 4),
            canonical_trust_engine_risk=round(final_risk, 4),
            risk_label=_risk_label(final_risk),
            decision=decision,
            trust_score=trust_score,
            confidence=confidence,
            predictive_uncertainty=uncertainty,
            model_used=f"Hybrid GNN + {clf_name} (Phase 5, tabular+graph+GNN embeddings)",
            model_version="phase5-hybrid",
            cold_start=is_cold_start,
            model_disagreement=model_disagreement,
            reason_codes=reason_codes,
            evidence_availability={
                "tabular": True,
                "graph": bool(req.share_degree > 0 or req.buyer_pagerank > 0),
                "gnn_embeddings": bool(req.buyer_id in store.buyer_embeddings or req.seller_id in store.seller_embeddings),
                "multimodal": req.multimodal_similarity_score is not None,
                "calibrator_active": store.phase5_calibrator is not None,
            },
            note=(
                "Scored with Phase 5 hybrid model routed through Canonical Trust Engine "
                "with probability calibration and conformal uncertainty."
            ),
        )

    if store.feature_meta is None or store.combined_graph_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet — try again in a moment.")

    meta = store.feature_meta
    all_cols = meta["all_feature_cols"]

    feature_row = _build_feature_row(req, all_cols)
    X = pd.DataFrame([feature_row])[all_cols].fillna(0.0)

    prob_raw = float(np.asarray(store.combined_graph_model.predict_proba(X))[0, 1])
    if store.calibrator is not None:
        prob_cal = float(store.calibrator.predict_proba(np.array([prob_raw]))[0])
    else:
        prob_cal = prob_raw

    clf_name = meta.get("classifier", type(store.combined_graph_model).__name__)

    graph_sig = float(min(req.share_degree * 0.15 + req.buyer_pagerank * 0.35, 1.0))
    ring_sig = float(min(max(req.share_component_size - 1, 0) * 0.2, 1.0))
    component_risks = {
        "tabular_risk": prob_cal,
        "graph_risk": graph_sig,
        "ring_risk": ring_sig,
    }
    if req.multimodal_similarity_score is not None:
        component_risks["multimodal_risk"] = float(
            max(0.0, min(1.0, 1.0 - req.multimodal_similarity_score))
        )

    engine = _canonical_trust_engine or _trust_engine
    if engine is not None:
        t_res = engine.score(component_risks, is_cold_start=is_cold_start)
        final_risk = getattr(t_res, "calibrated_risk", getattr(t_res, "risk_score", 0.0))
        decision = t_res.decision.value
        trust_score = t_res.trust_score
        confidence = t_res.confidence
        uncertainty = getattr(t_res, "predictive_uncertainty", None)
        model_disagreement = getattr(t_res, "detector_disagreement", getattr(t_res, "model_disagreement", 0.0))
        reason_codes = t_res.reason_codes
    else:
        final_risk = prob_cal
        decision = "REVIEW" if final_risk >= 0.25 else "ALLOW"
        trust_score = round(100.0 * (1.0 - final_risk), 2)
        confidence = 0.8
        uncertainty = None
        model_disagreement = 0.0
        reason_codes = []

    return TransactionScoreResponse(
        order_id=req.order_id,
        overall_fraud_probability=round(final_risk, 4),
        raw_fraud_probability=round(prob_raw, 4),
        calibrated_fraud_probability=round(prob_cal, 4),
        canonical_trust_engine_risk=round(final_risk, 4),
        risk_label=_risk_label(final_risk),
        decision=decision,
        trust_score=trust_score,
        confidence=confidence,
        predictive_uncertainty=uncertainty,
        model_used=f"tabular+graph {clf_name} (Phase 3)",
        model_version="phase3",
        cold_start=is_cold_start,
        model_disagreement=model_disagreement,
        reason_codes=reason_codes,
        evidence_availability={
            "tabular": True,
            "graph": bool(req.share_degree > 0 or req.buyer_pagerank > 0),
            "gnn_embeddings": False,
            "multimodal": req.multimodal_similarity_score is not None,
            "calibrator_active": store.calibrator is not None,
        },
        note=(
            "Graph features default to 0 if not provided. Supply them from a live graph lookup for best accuracy. "
            "Evaluated with Canonical Trust Engine."
        ),
    )


# ---------------------------------------------------------------------------
# GET /fraud-rings
# ---------------------------------------------------------------------------

@app.get("/fraud-rings", response_model=FraudRingsResponse, tags=["rings"])
def get_fraud_rings(
    min_risk_score: float = Query(
        default=0.0,
        ge=0.0, le=1.0,
        description="Filter: only return rings with avg_risk_score >= this value. "
                    "Use 0.3 for medium+ risk, 0.5 for high risk only.",
    ),
    limit: int = Query(
        default=50,
        ge=1, le=500,
        description="Maximum number of rings to return (sorted by avg_risk_score desc).",
    ),
):
    """
    Return pre-computed suspected fraud rings, ranked by average predicted
    fraud probability across member orders.

    Rings are connected components in the buyer sharing graph (device + address
    sharing), scored using the Phase 3 combined model. This is the
    detect_fraud_rings() output from graph_features.py.

    Use min_risk_score=0.3 for a practical high-signal filter.
    """
    if not store.is_loaded or store.rings_df is None:
        raise HTTPException(status_code=503, detail="Ring data not loaded yet.")

    df = store.rings_df
    if min_risk_score > 0:
        df = df[df["avg_risk_score"] >= min_risk_score]

    df = df.head(limit)

    ring_items: List[FraudRingItem] = []
    for _, row in df.iterrows():
        raw_members = row.get("members")
        members_list = list(raw_members) if isinstance(raw_members, (list, tuple)) else []
        ring_items.append(FraudRingItem(
            ring_id=row["ring_id"],
            size=int(row["size"]),
            n_orders=int(row["n_orders"]),
            avg_risk_score=float(row["avg_risk_score"]),
            max_risk_score=float(row["max_risk_score"]),
            n_high_risk_orders=int(row["n_high_risk_orders"]),
            risk_label=_risk_label(row["avg_risk_score"]),
            members=members_list,
        ))

    high_risk = int((store.rings_df["avg_risk_score"] >= 0.5).sum())

    return FraudRingsResponse(
        total_rings=len(store.rings_df),
        high_risk_rings=high_risk,
        rings=ring_items,
    )


# ---------------------------------------------------------------------------
# POST /listing/analyze  (Phase 4 Multimodal Fake Listing Scoring)
# ---------------------------------------------------------------------------

@app.post("/listing/analyze", response_model=ListingScoreResponse, tags=["scoring"])
def analyze_listing(req: ListingScoreRequest):
    """
    Score a product listing for fraud / fake-listing risk using the Phase 4
    multimodal model (trained on price anomalies, seller profile, and
    image-text alignment via real CLIP embeddings).

    **CLIP auto-scoring** (preferred): supply ``product_id`` and
    ``displayed_product_id`` in the request body.  When the CLIP scorer is
    loaded (run ``scripts/build_clip_embeddings.py`` first), the server will
    compute ``multimodal_similarity_score`` from real CLIP image-text cosine
    similarity and return ``clip_scored=true`` in the response.

    **Legacy mode**: omit ``product_id`` and supply ``multimodal_similarity_score``
    directly (backward-compatible with prior API clients).

    Also generates structured natural-language investigator rationale.
    """
    if not store.is_loaded or store.fake_listing_model is None or store.feature_meta is None:
        raise HTTPException(status_code=503, detail="Listing model not loaded yet.")

    cols = store.feature_meta.get("listing_feature_cols", [
        "price_vs_base_price_ratio", "price_vs_category_median_ratio",
        "seller_age_days_at_listing", "seller_listings_before",
        "multimodal_similarity_score",
    ])

    price_base_ratio = req.price / max(req.base_price, 0.01)
    price_cat_ratio = req.price / max(req.category_median_price, 0.01)

    # ------------------------------------------------------------------ #
    # Phase 4: derive multimodal_similarity_score via real CLIP embeddings
    # ------------------------------------------------------------------ #
    clip_scored = False
    tfidf_scored = False
    sim_score = req.multimodal_similarity_score  # default: caller-supplied

    if req.product_id is not None and store.clip_loaded and store.clip_scorer is not None:
        # ── Preferred path: real CLIP scorer ────────────────────────────
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
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "CLIP scoring failed for listing %s (%s). "
                "Falling back to TF-IDF surrogate.",
                req.listing_id, exc,
            )

    if not clip_scored and req.product_id is not None and store.clip_products_df is not None:
        # ── Secondary path: deterministic TF-IDF surrogate ──────────────
        # Used when CLIP is unavailable/stale but product_id was supplied.
        # This is reproducible and avoids the silent 0.85-default trap.
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
                "TF-IDF fallback also failed for listing %s (%s). "
                "Using caller-supplied multimodal_similarity_score=%.3f.",
                req.listing_id, exc2, req.multimodal_similarity_score,
            )

    features = {
        "price_vs_base_price_ratio": price_base_ratio,
        "price_vs_category_median_ratio": price_cat_ratio,
        "seller_age_days_at_listing": req.seller_age_days_at_listing,
        "seller_listings_before": req.seller_listings_before,
        "multimodal_similarity_score": sim_score,
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


# ---------------------------------------------------------------------------
# POST /return/analyze  (Phase 2 Return Fraud Detector)
# ---------------------------------------------------------------------------

@app.post("/return/analyze", response_model=ReturnScoreResponse, tags=["scoring"])
@app.post("/return/score", response_model=ReturnScoreResponse, tags=["scoring"], include_in_schema=False)
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
    features = {
        "days_to_return": req.days_to_return,
        "buyer_age_days_at_return": req.buyer_age_days_at_return,
        "seller_age_days_at_return": req.seller_age_days_at_return,
        "order_amount": req.order_amount,
        "buyer_prior_returns": float(req.buyer_prior_returns),
        "buyer_orders_before_return": float(req.buyer_orders_before_return),
        "buyer_return_rate_before": req.buyer_return_rate_before,
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

