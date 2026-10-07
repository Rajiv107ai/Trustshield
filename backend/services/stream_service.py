"""
TrustShield AI — Server-Sent Events (SSE) Real-Time Transaction Streaming Service.

Provides:
- High-throughput asynchronous event streaming (GET /stream/transactions)
- Real model scoring pipeline execution for every streamed transaction
- Heartbeat keepalive comments (every 15s)
- Client disconnect detection and graceful resource cleanup
- Bounded concurrency to protect server resources
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List

from fastapi import Request

from backend.schemas import StreamTransactionEvent, TransactionScoreRequest

logger = logging.getLogger(__name__)

# Bounded concurrency guard
MAX_CONCURRENT_SSE_CLIENTS = 50
_active_sse_clients = 0

# High-fidelity realistic stream event templates for transaction generator
STREAM_TEMPLATES: List[Dict[str, Any]] = [
    {
        "buyer_id": "BUYER_RING_MEMBER_04",
        "seller_id": "SELLER_RING_LEADER_01",
        "amount": 890.0,
        "base_price": 400.0,
        "category_median_price": 380.0,
        "buyer_orders_before": 1,
        "buyer_returns_before": 0,
        "buyer_age_days": 2.0,
        "seller_age_days": 35.0,
        "seller_total_listings_before": 12,
        "device_shared_buyer_count": 6.0,
        "share_degree": 5.0,
        "share_component_size": 8.0,
    },
    {
        "buyer_id": "BUYER_REFUND_ABUSER",
        "seller_id": "SELLER_ELECTRONICS_09",
        "amount": 320.0,
        "base_price": 320.0,
        "category_median_price": 280.0,
        "buyer_orders_before": 6,
        "buyer_returns_before": 5,
        "buyer_return_rate_before": 0.833,
        "buyer_age_days": 90.0,
        "seller_age_days": 200.0,
        "seller_total_listings_before": 45,
        "device_shared_buyer_count": 1.0,
        "share_degree": 0.0,
        "share_component_size": 1.0,
    },
    {
        "buyer_id": "BUYER_NEWBIE_99",
        "seller_id": "SELLER_UNKNOWN_44",
        "amount": 450.0,
        "base_price": 100.0,
        "category_median_price": 95.0,
        "buyer_orders_before": 0,
        "buyer_returns_before": 0,
        "buyer_age_days": 1.0,
        "seller_age_days": 5.0,
        "seller_total_listings_before": 2,
        "device_shared_buyer_count": 1.0,
        "share_degree": 0.0,
        "share_component_size": 1.0,
    },
    {
        "buyer_id": "BUYER_VERIFIED_77",
        "seller_id": "SELLER_REPUTABLE_12",
        "amount": 65.5,
        "base_price": 70.0,
        "category_median_price": 68.0,
        "buyer_orders_before": 15,
        "buyer_returns_before": 0,
        "buyer_return_rate_before": 0.0,
        "buyer_age_days": 180.0,
        "seller_age_days": 450.0,
        "seller_total_listings_before": 120,
        "device_shared_buyer_count": 1.0,
        "share_degree": 0.0,
        "share_component_size": 1.0,
    },
    {
        "buyer_id": "BUYER_MOBILE_USER_11",
        "seller_id": "SELLER_SHOES_55",
        "amount": 110.0,
        "base_price": 115.0,
        "category_median_price": 105.0,
        "buyer_orders_before": 8,
        "buyer_returns_before": 1,
        "buyer_return_rate_before": 0.125,
        "buyer_age_days": 120.0,
        "seller_age_days": 310.0,
        "seller_total_listings_before": 50,
        "device_shared_buyer_count": 1.0,
        "share_degree": 0.0,
        "share_component_size": 1.0,
    },
    {
        "buyer_id": "BUYER_RING_MEMBER_02",
        "seller_id": "SELLER_RING_LEADER_01",
        "amount": 940.0,
        "base_price": 420.0,
        "category_median_price": 380.0,
        "buyer_orders_before": 2,
        "buyer_returns_before": 0,
        "buyer_age_days": 3.0,
        "seller_age_days": 35.0,
        "seller_total_listings_before": 12,
        "device_shared_buyer_count": 5.0,
        "share_degree": 4.0,
        "share_component_size": 8.0,
    },
]


def get_active_sse_clients() -> int:
    """Return count of currently connected SSE streaming clients."""
    return _active_sse_clients


async def transaction_event_generator(
    request: Request,
    scoring_fn: Any,
    interval_seconds: float = 2.0,
    metrics_tracker: Any = None,
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events for incoming transactions scored dynamically
    through the actual model and Trust Engine.
    """
    global _active_sse_clients

    if _active_sse_clients >= MAX_CONCURRENT_SSE_CLIENTS:
        yield f"event: error\ndata: {json.dumps({'error': 'Max concurrent SSE connections reached'})}\n\n"
        return

    _active_sse_clients += 1
    if metrics_tracker and hasattr(metrics_tracker, "sse_clients_gauge"):
        metrics_tracker.sse_clients_gauge.inc()

    last_heartbeat = asyncio.get_event_loop().time()
    order_counter = random.randint(100000, 999999)

    try:
        # Initial greeting event
        yield (
            f"event: connected\n"
            f"data: {json.dumps({'status': 'stream_active', 'connected_at': datetime.now(timezone.utc).isoformat()})}\n\n"
        )

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info("SSE client disconnected.")
                break

            now = asyncio.get_event_loop().time()

            # Heartbeat keepalive every 15s
            if now - last_heartbeat >= 15.0:
                last_heartbeat = now
                yield ": keep-alive\n\n"

            # Select and mutate a transaction template
            template = random.choice(STREAM_TEMPLATES).copy()
            order_counter += 1
            order_id = f"ORD_{order_counter}"
            template["order_id"] = order_id
            template["amount"] = round(template["amount"] * random.uniform(0.9, 1.1), 2)
            template["order_amount"] = template["amount"]

            # Score through REAL scoring engine
            score_req = TransactionScoreRequest(**template)
            score_res = scoring_fn(score_req)

            event = StreamTransactionEvent(
                event_id=f"EVT_{order_counter}",
                timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S"),
                order_id=order_id,
                buyer_id=template["buyer_id"],
                seller_id=template["seller_id"],
                amount=template["amount"],
                risk_score=score_res.raw_fraud_probability or score_res.overall_fraud_probability,
                calibrated_risk=score_res.overall_fraud_probability,
                risk_level=score_res.risk_label,
                decision=score_res.decision,
                trust_score=score_res.trust_score,
                reason_codes=score_res.reason_codes,
                source="simulation_stream",
            )

            if metrics_tracker and hasattr(metrics_tracker, "sse_events_counter"):
                metrics_tracker.sse_events_counter.inc()

            payload = event.model_dump_json()
            yield f"event: transaction\ndata: {payload}\n\n"

            await asyncio.sleep(interval_seconds)

    except asyncio.CancelledError:
        logger.info("SSE transaction stream task cancelled.")
    finally:
        _active_sse_clients = max(0, _active_sse_clients - 1)
        if metrics_tracker and hasattr(metrics_tracker, "sse_clients_gauge"):
            metrics_tracker.sse_clients_gauge.dec()
