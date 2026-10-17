"""
TrustShield AI — Feature Engineering & Matrix Construction.

Transforms incoming API score requests into sanitized feature dictionaries
matching the exact column schemas of trained Phase 3 and Phase 5 models.
Provides:
- Amount alias resolution and conflict validation
- Real-time server-side ratio derivation
- Authoritative graph feature resolution from server snapshot & PIT history
- Hybrid GNN embedding retrieval via RedisService with disk fallback
"""

import logging
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException

from backend.model_loader import store
from backend.schemas import TransactionScoreRequest
from backend.services.metrics_service import metrics_service

logger = logging.getLogger(__name__)

GRAPH_FEATURE_KEYS = [
    "share_degree",
    "share_component_size",
    "buyer_seller_degree",
    "buyer_pagerank",
    "seller_buyer_degree",
    "seller_pagerank",
    "seller_buyer_concentration_hhi",
    "buyer_seller_edge_weight_before",
]


def _risk_label(score: float) -> str:
    """Canonical risk label mapping aligned with Trust Engine thresholds (0.25, 0.55)."""
    if score >= 0.55:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def _build_feature_row(
    req: TransactionScoreRequest,
    all_cols: list,
    resolved_graph_features: Optional[dict] = None,
) -> dict:
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
    fields_set: set = getattr(req, "model_fields_set", set())
    has_amount = "amount" in fields_set
    has_order_amount = "order_amount" in fields_set

    amount_val = row.get("amount")
    order_amount_val = row.get("order_amount")

    if has_amount and has_order_amount:
        amt = float(amount_val) if amount_val is not None else 0.0
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
        resolved_amount = float(amount_val) if amount_val is not None else 0.0
    elif has_order_amount:
        resolved_amount = float(order_amount_val) if order_amount_val is not None else 0.0
    else:
        resolved_amount = float(amount_val) if amount_val is not None else float(order_amount_val or 0.0)

    row["amount"] = resolved_amount
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

    # Incorporate resolved graph features (server / client / default)
    if resolved_graph_features is not None:
        graph_resolved = resolved_graph_features
    else:
        graph_resolved, _, _, _ = _resolve_graph_features(req)
    row.update(graph_resolved)

    # Build the final feature vector using only the columns the model expects
    return {col: (row.get(col) if row.get(col) is not None else 0.0) for col in all_cols}


def _resolve_graph_features(
    req: TransactionScoreRequest,
) -> Tuple[dict, str, str, bool]:
    """
    Computes share_degree, share_component_size, pagerank and edge-weight features
    server-side from stored snapshot; flags client-supplied values that differ;
    returns (graph_dict, graph_features_source, flag_notes, client_mismatch).
    Source is strictly one of 'server', 'client', or 'default'.
    When server snapshot and client values differ, the server value must be used
    and the response must set graph_features_source = 'server' with client_mismatch = True.
    """
    fields_set: set = getattr(req, "model_fields_set", set())
    client_supplied = [k for k in GRAPH_FEATURE_KEYS if k in fields_set]

    snapshot = getattr(store, "graph_snapshot", None)
    b_data = snapshot.get("buyer_features", {}).get(req.buyer_id) if snapshot and req.buyer_id else None
    s_data = snapshot.get("seller_features", {}).get(req.seller_id) if snapshot and req.seller_id else None

    tx_hist = getattr(store, "transaction_history", None)
    req_date = getattr(req, "order_date", None)
    pit_weight: Optional[float] = None

    if tx_hist is not None and req.buyer_id and req.seller_id:
        pit_weight, pit_source, pit_note = tx_hist.get_prior_edge_weight(
            req.buyer_id, req.seller_id, order_date=req_date, order_id=req.order_id
        )
    else:
        pit_weight = 0.0
        pit_source = "fallback_history_unavailable"
        pit_note = "Point-in-time history store unavailable; defaulted to 0.0 (static snapshot NOT substituted)"

    server_computed = {}
    if b_data is not None:
        server_computed["share_degree"] = float(b_data.get("share_degree", 0.0))
        server_computed["share_component_size"] = float(b_data.get("share_component_size", 1.0))
        server_computed["buyer_seller_degree"] = float(b_data.get("buyer_seller_degree", 0.0))
        server_computed["buyer_pagerank"] = float(b_data.get("buyer_pagerank", 0.0))
    if s_data is not None:
        server_computed["seller_buyer_degree"] = float(s_data.get("seller_buyer_degree", 0.0))
        server_computed["seller_pagerank"] = float(s_data.get("seller_pagerank", 0.0))
        server_computed["seller_buyer_concentration_hhi"] = float(s_data.get("seller_buyer_concentration_hhi", 0.0))
    if b_data is not None or s_data is not None:
        server_computed["buyer_seller_edge_weight_before"] = float(pit_weight if pit_weight is not None else 0.0)

    resolved = {}
    flag_notes = ""
    client_mismatch = False

    if server_computed:
        mismatches = []
        if client_supplied:
            for k in client_supplied:
                if k in server_computed:
                    c_val = getattr(req, k, 0.0) or 0.0
                    s_val = server_computed[k]
                    if abs(float(c_val) - float(s_val)) > 1e-4:
                        mismatches.append(f"{k}: client={c_val} vs server={s_val}")
        if mismatches:
            client_mismatch = True
            flag_notes = f"Flagged client-supplied graph feature mismatch ({', '.join(mismatches)}). Server snapshot value enforced."
            logger.warning("Graph feature mismatch for order %s: %s", req.order_id, flag_notes)

        for k in GRAPH_FEATURE_KEYS:
            if k in server_computed:
                resolved[k] = float(server_computed[k])
            else:
                resolved[k] = 1.0 if k == "share_component_size" else 0.0
        graph_source = "server"

    elif client_supplied:
        for k in GRAPH_FEATURE_KEYS:
            if k in client_supplied:
                resolved[k] = float(getattr(req, k, 0.0) or 0.0)
            else:
                resolved[k] = 1.0 if k == "share_component_size" else 0.0
        graph_source = "client"

    else:
        for k in GRAPH_FEATURE_KEYS:
            resolved[k] = 1.0 if k == "share_component_size" else 0.0
        graph_source = "default"
        flag_notes = "Flagged: Graph features defaulted to 0.0 (entity not in snapshot and no client values provided)"

    return resolved, graph_source, flag_notes, client_mismatch

