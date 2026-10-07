# TrustShield Production Observability & Telemetry

## 1. Overview

TrustShield includes full production observability via **Prometheus metrics exposition** and **Grafana visualization dashboards**.

The metrics pipeline follows standard cloud-native architecture:
```
FastAPI Gateway (:8000)
    │
    ▼  Scraped every 15s
Prometheus Server (:9090)
    │
    ▼  Provisioned Data Source
Grafana Dashboards (:3001)
```

---

## 2. Prometheus Endpoint & Cardinality Rules

- **Exposition Route:** `GET /metrics`
- **Format:** Prometheus text format (`text/plain; version=0.0.4; charset=utf-8`)

### Strict Low-Cardinality Enforcement
Per Rule #23 and Rule #35:
- **Forbidden as Metric Labels:** `transaction_id`, `order_id`, `buyer_id`, `seller_id`, IP addresses, or device fingerprints. High-cardinality values cause metric store memory blowup.
- **Allowed Labels:**
  - `method`, `endpoint`, `status_code` for HTTP requests.
  - `decision` (`ALLOW`, `REVIEW`, `HOLD`, `BLOCK`) for fraud routing.
  - `risk_label` (`low`, `medium`, `high`) for risk severity.
  - `model_version` (`phase5-hybrid`, `phase3-combined`) for inference tracking.
  - `entity_type` (`transaction`, `order`, `buyer`, `seller`, `ring`) for investigations.
  - `service` (`redis`, `neo4j`, `models`, `investigation`) for fallback tracking.

---

## 3. Metrics Catalog

### Counters
| Metric Name | Labels | Description |
|---|---|---|
| `trustshield_http_requests_total` | `method, endpoint, status_code` | Total HTTP requests handled |
| `trustshield_decisions_total` | `decision` | Total fraud decisions rendered by Trust Engine |
| `trustshield_high_risk_decisions_total` | *None* | Count of high-risk transactions flagged (BLOCK/HOLD) |
| `trustshield_model_errors_total` | `model_version` | Model inference exception count |
| `trustshield_dossier_total` | `entity_type, status` | Forensic dossiers requested and generated |
| `trustshield_sse_events_total` | *None* | Transaction events emitted across all SSE clients |
| `trustshield_fallback_total` | `service, fallback_type` | Observable fallback invocations (e.g. disk artifacts) |

### Histograms
All histograms collect exponential bucket observations. p50, p95, and p99 are derived dynamically using `histogram_quantile()`:
| Metric Name | Labels | Buckets (seconds) |
|---|---|---|
| `trustshield_http_request_duration_seconds` | `endpoint` | 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0 |
| `trustshield_model_inference_seconds` | `model_version` | 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2 |
| `trustshield_trust_engine_seconds` | *None* | 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1 |
| `trustshield_redis_duration_seconds` | `operation` | 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05 |
| `trustshield_neo4j_duration_seconds` | `query_type` | 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5 |
| `trustshield_dossier_duration_seconds` | *None* | 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0 |

### Gauges
| Metric Name | Description |
|---|---|
| `trustshield_sse_connected_clients` | Number of currently active SSE stream connections |
| `trustshield_service_up` | Binary indicator (1=up, 0=down) for dependent infrastructure |

---

## 4. Empirical Benchmarks (`GET /system/benchmark`)

TrustShield provides a dedicated benchmark route (`GET /system/benchmark`) that executes high-resolution runtime tests against local hardware.

### Measured Hardware Baseline
- **Scoring Pipeline (Full Hybrid Inference):**
  - **p50:** ~7.69 ms
  - **p95:** ~8.85 ms
  - **p99:** ~21.37 ms
- **Redis Driver Overhead (Serialization + Vector Lookup):**
  - **p50:** 0.001 ms
  - **p95:** 0.003 ms
  - **p99:** 0.014 ms
- **Offline Fallback Overhead:** 0.00 ms (direct zero-copy in-memory joblib lookup).

---

## 5. Grafana Dashboard Layout

Located in `docker/grafana/dashboards/trustshield_overview.json`:
1. **API Gateway Ingest Rate (RPS):** `rate(trustshield_http_requests_total[1m])`
2. **HTTP Latency Percentiles (p50 / p95 / p99):** Derived via `histogram_quantile`
3. **HTTP 5xx Error Rate:** `rate(trustshield_http_requests_total{status_code=~"5.."}[1m])`
4. **Fraud Decision Routing:** Breakdown of `ALLOW`, `REVIEW`, `HOLD`, `BLOCK`
5. **High-Risk Block Rate Ratio:** High risk events vs total decisions
6. **Model Inference Latency:** `histogram_quantile(0.95, rate(trustshield_model_inference_seconds_bucket[1m]))`
7. **Trust Engine Evaluation Duration:** Quantiles for canonical risk scoring
8. **Redis Feature Store Latency:** Get embedding vs Cache set
9. **Neo4j Graph Traversal Latency:** Neighborhood vs Ego network
10. **Forensic Dossier Generation:** Success rate and latency quantiles
11. **Active Real-Time SSE Clients:** Live gauge of connected frontend clients
12. **Observable Infrastructure Fallbacks:** Count of disk artifact fallbacks invoked
