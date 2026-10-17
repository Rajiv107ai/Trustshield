# TrustShield AI — HTTP Transaction Scoring Benchmark Report

<!-- BEGIN GENERATED -->
- Timestamp: `2026-10-17T09:32:25.299699+00:00`
- Platform: `Windows-11-10.0.26300-SP0` (16 vCPUs, 15.71 GB RAM)
- Active Model: `Phase 5 XGBoost + GNN`
- First Request (Cold Start): `119.37 ms`
- Warm Steady-State Latency: `12.62 ms`

| Concurrency | Requests | Success/Fail | QPS (Mean +/- Std) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|---|---|
| **1** | 15 | 15/0 | 84.51 +/- 1.35 | 11.77 | 11.93 | 11.97 | 12.0 |
| **5** | 15 | 15/0 | 102.42 +/- 3.03 | 45.28 | 47.27 | 47.69 | 48.03 |
| **10** | 15 | 15/0 | 106.55 +/- 1.14 | 43.28 | 44.97 | 45.27 | 45.61 |
<!-- END GENERATED -->
