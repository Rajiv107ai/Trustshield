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

from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from model_loader import store
from schemas import (
    TransactionScoreRequest, TransactionScoreResponse,
    FraudRingsResponse, FraudRingItem,
    HealthResponse,
)


def _risk_label(score: float) -> str:
    if score >= 0.5:
        return "high"
    if score >= 0.3:
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
    """
    row = req.model_dump()

    # --- Derive price_vs_base_price_ratio from raw inputs if not given ---
    if row.get("order_amount", 0.0) == 0.0 and row.get("amount") is not None:
        row["order_amount"] = row["amount"]

    if row.get("price_vs_base_price_ratio", 0.0) == 0.0:
        amount = row.get("amount") or row.get("order_amount")
        base_price = row.get("base_price")
        if amount and base_price and base_price > 0:
            row["price_vs_base_price_ratio"] = amount / base_price

    # --- Derive price_vs_category_median_ratio from raw inputs if not given ---
    if row.get("price_vs_category_median_ratio", 0.0) == 0.0:
        amount = row.get("amount") or row.get("order_amount")
        cat_median = row.get("category_median_price")
        if amount and cat_median and cat_median > 0:
            row["price_vs_category_median_ratio"] = amount / cat_median

    # --- Derive buyer_return_rate_before from counts if not given ---
    if row.get("buyer_return_rate_before", 0.0) == 0.0:
        orders_before = row.get("buyer_orders_before", 0) or 0
        returns_before = row.get("buyer_returns_before", 0) or 0
        row["buyer_return_rate_before"] = returns_before / max(orders_before, 1)

    # Build the final feature vector using only the columns the model expects
    return {col: row.get(col, 0.0) for col in all_cols}


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
        "Returns real model output from Phase 2 (specialized detectors) "
        "and Phase 3 (graph-augmented combined model + fraud ring detection). "
        "All predictions are from trained scikit-learn RandomForest models — "
        "no hardcoded or mocked responses."
    ),
    version="0.1.0-mvp",
    lifespan=lifespan,
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
        rings_loaded=store.rings_df is not None,
        n_rings=len(store.rings_df) if store.rings_df is not None else 0,
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

    Supply as many feature fields as you have available — missing ones
    default to 0. The richer the feature vector, the more accurate the score.

    Raw inputs (amount, base_price, category_median_price) are accepted as
    a convenience: the server derives price ratio features from them
    automatically when the direct ratio fields are not supplied.

    Returns a fraud probability in [0, 1] and a risk label (low/medium/high).
    """
    if not store.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet — try again in a moment.")

    if store.phase5_loaded:
        p5_meta = store.phase5_meta
        hybrid_cols = p5_meta["hybrid_feature_cols"]
        p3_cols = p5_meta["phase3_feature_cols"]
        emb_dim = p5_meta.get("emb_dim", 16)
        row = _build_hybrid_feature_row(
            req, hybrid_cols, p3_cols,
            store.buyer_embeddings, store.seller_embeddings, emb_dim
        )
        X = pd.DataFrame([row])[hybrid_cols].fillna(0.0)
        prob = float(store.hybrid_model.predict_proba(X)[0, 1])
        clf_name = p5_meta.get("classifier", "XGBoost")
        return TransactionScoreResponse(
            order_id=req.order_id,
            overall_fraud_probability=round(prob, 4),
            risk_label=_risk_label(prob),
            model_used=f"Hybrid GNN + {clf_name} (Phase 5, tabular+graph+GNN embeddings)",
            model_version="phase5-hybrid",
            note=(
                "Scored with Phase 5 hybrid model (tabular + graph topology + 32-dim GNN embeddings). "
                "Cold-start entities receive zero GNN embeddings automatically."
            ),
        )

    meta = store.feature_meta
    all_cols = meta["all_feature_cols"]

    feature_row = _build_feature_row(req, all_cols)
    X = pd.DataFrame([feature_row])[all_cols].fillna(0.0)

    prob = float(store.combined_graph_model.predict_proba(X)[0, 1])

    clf_name = meta.get("classifier", type(store.combined_graph_model).__name__)

    return TransactionScoreResponse(
        order_id=req.order_id,
        overall_fraud_probability=round(prob, 4),
        risk_label=_risk_label(prob),
        model_used=f"tabular+graph RF (Phase 3, {clf_name})",
        model_version="phase3",
        note=(
            "Graph features (share_degree, share_component_size, etc.) default to 0 "
            "if not provided. Supply them from a live graph lookup for best accuracy."
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
        ring_items.append(FraudRingItem(
            ring_id=row["ring_id"],
            size=int(row["size"]),
            n_orders=int(row["n_orders"]),
            avg_risk_score=float(row["avg_risk_score"]),
            max_risk_score=float(row["max_risk_score"]),
            n_high_risk_orders=int(row["n_high_risk_orders"]),
            risk_label=_risk_label(row["avg_risk_score"]),
            members=list(row.get("members", [])),
        ))

    high_risk = int((store.rings_df["avg_risk_score"] >= 0.5).sum())

    return FraudRingsResponse(
        total_rings=len(store.rings_df),
        high_risk_rings=high_risk,
        rings=ring_items,
    )
