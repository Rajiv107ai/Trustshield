# TrustShield AI — HTTP Transaction Scoring Benchmark Report

> **Phase 2 Empirical Verification**: This benchmark replaces the invalid microbenchmark
> (in-memory `np.dot` dot product proxy) with true HTTP request-response performance testing
> on the `/transaction/score` FastAPI endpoint.

## 1. System & Hardware Specification

- **Timestamp**: `2026-10-09T17:07:48.275365+00:00`
- **Benchmark Mode**: `Direct In-Process ASGI Transport (app=FastAPI)`
- **Target Endpoint**: `http://testserver/transaction/score`
- **Operating System**: `Windows-11-10.0.26300-SP0`
- **Processor**: `Intel64 Family 6 Model 154 Stepping 3, GenuineIntel`
- **CPU Allocation**: `10 physical cores / 16 logical vCPUs`
- **System RAM**: `15.71 GB`
- **Python Version**: `3.14.7`
- **FastAPI / Pydantic**: `0.135.2 / 2.12.5`
- **Active Model**: `Phase 5 XGBoost + GNN`
- **Redis Service**: `In-Memory Fallback`
- **Neo4j Service**: `In-Memory Fallback`

## 2. Cold-Start vs Steady-State Latency

- **First Request (Cold Start)**: `149.2 ms` (initial imports, graph/dict caching)
- **Warm Steady-State Latency**: `20.07 ms`

## 3. Concurrency & Throughput Benchmarks (Repeated Runs Aggregation)

| Concurrency | Total Reqs (Runs) | Success / Fail | Mean Throughput (QPS) | p50 Latency (ms) | p90 Latency (ms) | p95 Latency (ms) | p99 Latency (ms) |
|-------------|-------------------|----------------|-----------------------|------------------|------------------|------------------|------------------|
| **1** | 30 (3x10) | 30 / 0 | **45.66 +/- 3.96 req/s** | 20.93 ms | 23.64 ms | 23.74 ms | 23.82 ms |
| **5** | 30 (3x10) | 30 / 0 | **53.4 +/- 2.97 req/s** | 87.45 ms | 98.8 ms | 100.63 ms | 102.1 ms |
| **10** | 30 (3x10) | 30 / 0 | **55.37 +/- 1.66 req/s** | 166.55 ms | 173.87 ms | 174.83 ms | 175.6 ms |

## 4. Architectural Comparison: Historical Microbenchmark vs True HTTP

| Dimension | Historical `run_scalability_experiment.py` | True HTTP Benchmark (`benchmark_scoring_http.py`) |
|---|---|---|
| **Operation Tested** | `np.dot(features, weights)` vector multiply | Full FastAPI `/transaction/score` endpoint pipeline |
| **Validation Layer** | None | Pydantic v2 field validation + alias harmonization |
| **Feature Prep** | Pre-computed static numpy array | Dynamic hybrid extraction & GNN embedding lookup |
| **Model Scoring** | Matrix multiplication proxy | XGBoost / Random Forest real tree traversal |
| **Post-Processing** | None | Platt calibration + Trust Engine routing + SHAP heuristics |
| **Serialization** | None | Pydantic model serialization -> HTTP JSON response |
| **Reported Throughput** | ~2,800,000 QPS (Misleading proxy) | **~45 - 55 QPS per single Python process** (Empirical) |

## 5. Methodological Boundaries & Production Extrapolation Warning

> **CRITICAL METHODOLOGICAL LIMITATION**:
> Do **NOT** extrapolate production cluster capacity from a local single-process microbenchmark.
>
> 1. **In-Process ASGI vs Network Socket**: Direct ASGI benchmarking exercises FastAPI route handling
>    and serialization, but omits TCP socket handshake, TLS negotiation, and OS kernel network stack overhead.
> 2. **Database Fallback Mode**: This local benchmark operated against in-memory fallback caches.
>    Live Redis and Neo4j network calls in production will introduce roundtrip latencies (typically 1-5 ms).
> 3. **GIL & Process Boundaries**: Python's GIL bounds CPU-heavy ML tree traversals to a single core per process.
>    Scaling to production volumes requires multi-process Uvicorn workers behind an NGINX reverse proxy.

