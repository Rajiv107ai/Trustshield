# TrustShield Real-Time Streaming Architecture

## 1. Overview & Protocol Selection

TrustShield ingests real-time transactions and streams decision telemetry directly to operations analysts and dashboard consoles.

### Why Server-Sent Events (SSE)?
TrustShield uses **Server-Sent Events (SSE)** via HTTP/1.1 (`text/event-stream`) over WebSockets for transaction event delivery:
1. **Unidirectional Simplicity:** The audit and risk ingest stream is primarily **Server &rarr; Client**. SSE provides clean unidirectional push over standard HTTP without WebSocket handshake overhead or binary protocol framing.
2. **Native Reconnection:** HTTP EventSource provides built-in reconnect semantics, `Last-Event-ID` tracking, and automatic reconnection backoff.
3. **Proxy & Corporate Firewall Friendly:** Works natively over standard HTTP/HTTPS ports (80/443) without custom WebSocket proxying or gateway protocol upgrades.
4. **Transport Independence:** When bi-directional input is needed, analysts trigger discrete operations via standard RESTful endpoints (`POST /transaction/score`, `POST /investigation/generate-dossier`).

---

## 2. API Endpoint Specification

- **Route:** `GET /stream/transactions`
- **Query Parameters:**
  - `interval` (float, default: `2.0`, range: `0.5` to `10.0` seconds): Controls event emission frequency.
- **Headers Returned:**
  - `Content-Type: text/event-stream`
  - `Cache-Control: no-cache, no-transform`
  - `Connection: keep-alive`
  - `X-Accel-Buffering: no`

### Event Schema (`event: transaction`)
```json
{
  "event_id": "EVT_1772999480_0001",
  "timestamp": "2026-10-08T02:15:00.123456+00:00",
  "order_id": "ORD_491823",
  "buyer_id": "BUYER_000491",
  "seller_id": "SELLER_000042",
  "amount": 284.50,
  "risk_score": 0.842,
  "calibrated_risk": 0.835,
  "risk_level": "high",
  "decision": "BLOCK",
  "trust_score": 16.5,
  "reason_codes": [
    "HIGH_RETURN_RATE",
    "SHARED_DEVICE_COLLISION"
  ],
  "source": "simulation_stream"
}
```

---

## 3. Real-Time Engine Scoring Pipeline

TrustShield does **NOT** emit static mock transactions or fake random numbers. Every event is dynamically evaluated through the complete ML pipeline:
1. **Transaction Synthesis:** Realistic orders drawn from historical test sets or parameterized test vectors.
2. **Context Resolution:** Feature lookup against Redis embedding cache or disk joblib tables.
3. **Model Inference:** Scored through Phase 5 Hybrid XGBoost model or Phase 3 Combined Graph model.
4. **Uncertainty & Calibration:** Isotonic probability calibrator produces calibrated probabilities; Shannon entropy derives confidence.
5. **Operational Decision Routing:** Canonical thresholds (`ALLOW` &le; 0.25, `REVIEW` &le; 0.55, `HOLD` &le; 0.70, `BLOCK` &gt; 0.70).
6. **Provenance Labeling:** The payload explicitly sets `"source": "simulation_stream"` to ensure simulated traffic is never misrepresented as production traffic.

---

## 4. Connection Management & Resource Bounding

To prevent resource exhaustion in production:
- **Client Disconnect Detection:** The streaming generator monitors `await request.is_disconnected()`. When the client closes the connection, the generator terminates immediately, cleaning up background tasks.
- **Keep-Alive Heartbeats:** Emits `: keep-alive\n\n` comments every interval to prevent intermediate load balancers (e.g. AWS ALB, Nginx) from closing idle connections.
- **Bounded Concurrency:** Capped at `MAX_SSE_CLIENTS = 50` concurrent connections using an atomic gauge tracker (`metrics_service.sse_clients`). When capacity is exceeded, incoming requests receive `503 Service Unavailable`.
- **Zero Memory Leaks:** Emitted events are streamed directly without maintaining unbounded in-memory history lists.

---

## 5. Frontend Reconnection & UX State Machine

The frontend client in `frontend/src/lib/api/client.ts` implements a resilient reconnection state machine:
- **`LIVE`:** EventSource connected and actively receiving events (pulsing green indicator).
- **`CONNECTING`:** Initial connection establishing.
- **`RECONNECTING`:** Connection dropped; exponential backoff active with automatic retry after 3 seconds.
- **`OFFLINE REPLAY`:** Backend gateway unreachable; UI gracefully displays frozen snapshot and indicates replay mode without crashing.
