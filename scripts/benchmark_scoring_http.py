"""
TrustShield AI — HTTP Transaction Scoring Benchmark.
Empirical HTTP-level benchmarking of the real FastAPI /transaction/score endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
import numpy as np

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.main import app, store


def _p(name: str, oid: str, bid: str, sid: str, lid: str, amt: float, base: float, ord_b: int, ret_b: int, list_b: int, b_age: float, s_age: float, **extra) -> Dict[str, Any]:
    p = {
        "order_id": oid, "buyer_id": bid, "seller_id": sid, "listing_id": lid,
        "amount": amt, "base_price": base, "buyer_orders_before": ord_b, "buyer_returns_before": ret_b,
        "seller_total_listings_before": list_b, "buyer_age_days": b_age, "seller_age_days": s_age,
    }
    p.update(extra)
    return {"name": name, "payload": p}


BENCHMARK_PAYLOADS = [
    _p("known_buyer_low_risk", "ORD_BENCH_001", "BUY_000001", "SEL_000001", "LST_000001", 49.99, 50.00, 12, 0, 45, 180.0, 365.0),
    _p("high_amount_discrepancy", "ORD_BENCH_002", "BUY_000002", "SEL_000002", "LST_000002", 1299.99, 200.00, 1, 1, 2, 1.5, 3.0, category_median_price=180.00, price_vs_base_price_ratio=6.5),
    _p("cold_start_new_entities", "ORD_BENCH_003", "BUY_UNKNOWN_999", "SEL_UNKNOWN_999", "LST_000003", 89.50, 90.00, 0, 0, 0, 0.5, 0.5),
    _p("frequent_buyer_standard", "ORD_BENCH_004", "BUY_000005", "SEL_000003", "LST_000004", 120.00, 115.00, 25, 2, 80, 240.0, 400.0),
]


def collect_environment_metadata() -> Dict[str, Any]:
    """Capture runtime environment, hardware specs, and service states."""
    import psutil
    from backend.services.neo4j_service import neo4j_service
    from backend.services.redis_service import redis_service

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": sys.version.split()[0],
        "fastapi_version": "0.135.2",
        "pydantic_version": "2.12.5",
        "redis_live": redis_service.is_available,
        "neo4j_live": neo4j_service.is_available,
        "scoring_model_loaded": store.is_loaded,
        "phase5_model_loaded": store.phase5_loaded,
        "classifier_name": "Phase 5 XGBoost + GNN" if store.phase5_loaded else "Phase 3 Random Forest",
    }


async def run_single_request(client: httpx.AsyncClient, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=10.0)
        ms = (time.perf_counter() - t0) * 1000.0
        ok = resp.status_code == 200
        data = resp.json() if ok else {}
        return {
            "status_code": resp.status_code, "latency_ms": ms, "success": ok,
            "scoring_mode": data.get("scoring_mode"), "risk_score": data.get("risk_score"),
            "routing_action": data.get("routing_action"), "error": None if ok else resp.text[:200],
        }
    except Exception as exc:
        return {"status_code": 0, "latency_ms": (time.perf_counter() - t0) * 1000.0, "success": False, "scoring_mode": None, "risk_score": None, "routing_action": None, "error": str(exc)}


async def execute_concurrency_tier(client: httpx.AsyncClient, url: str, headers: Dict[str, str], concurrency: int, total_requests: int) -> Dict[str, Any]:
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
    for i in range(total_requests):
        queue.put_nowait(BENCHMARK_PAYLOADS[i % len(BENCHMARK_PAYLOADS)]["payload"])

    results: List[Dict[str, Any]] = []

    async def worker():
        while not queue.empty():
            try:
                p = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            results.append(await run_single_request(client, url, p, headers))
            queue.task_done()

    t0 = time.perf_counter()
    await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(concurrency)])
    duration = time.perf_counter() - t0

    lat = np.array([r["latency_ms"] for r in results]) if results else np.array([0.0])
    succ = sum(1 for r in results if r["success"])
    err = total_requests - succ

    return {
        "concurrency": concurrency, "total_requests": total_requests, "successful_requests": succ, "failed_requests": err,
        "error_rate_pct": round((err / total_requests) * 100.0, 2), "duration_seconds": round(duration, 3),
        "throughput_qps": round(succ / max(duration, 1e-6), 2),
        "latency_min_ms": round(float(np.min(lat)), 2), "latency_p50_ms": round(float(np.percentile(lat, 50)), 2),
        "latency_p90_ms": round(float(np.percentile(lat, 90)), 2), "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(lat, 99)), 2), "latency_max_ms": round(float(np.max(lat)), 2),
        "latency_mean_ms": round(float(np.mean(lat)), 2),
    }


async def run_benchmark(
    base_url: str | None = None,
    api_key: str = "dev-operator-api-key-32chars-long",
    concurrency_tiers: List[int] = [1, 5, 10],
    requests_per_tier: int = 100,
    warmup_count: int = 10,
    runs: int = 3,
) -> Dict[str, Any]:
    if not store.is_loaded:
        store.load()
    env_meta = collect_environment_metadata()
    headers = {"X-API-Key": api_key}

    if base_url:
        endpoint = f"{base_url.rstrip('/')}/transaction/score"
        mode = f"Live HTTP Socket ({base_url})"
        client = httpx.AsyncClient(headers=headers)
    else:
        endpoint = "http://testserver/transaction/score"
        mode = "Direct In-Process ASGI Transport (app=FastAPI)"
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers=headers)

    async with client:
        warm_lat = []
        for i in range(warmup_count):
            res = await run_single_request(client, endpoint, BENCHMARK_PAYLOADS[i % len(BENCHMARK_PAYLOADS)]["payload"], headers)
            warm_lat.append(res["latency_ms"])

        cold_start_ms = round(warm_lat[0], 2)
        warm_avg_ms = round(float(np.mean(warm_lat[1:]) if len(warm_lat) > 1 else warm_lat[0]), 2)

        tier_results: List[Dict[str, Any]] = []
        for c in concurrency_tiers:
            run_results = [await execute_concurrency_tier(client, endpoint, headers, c, requests_per_tier) for _ in range(runs)]
            total_reqs = sum(r["total_requests"] for r in run_results)
            total_failed = sum(r["failed_requests"] for r in run_results)
            tier_results.append({
                "concurrency": c, "runs": runs, "requests_per_run": requests_per_tier,
                "total_requests": total_reqs, "failed_requests": total_failed,
                "error_rate_pct": round((total_failed / max(1, total_reqs)) * 100.0, 2),
                "throughput_qps_mean": round(float(np.mean([r["throughput_qps"] for r in run_results])), 2),
                "throughput_qps_std": round(float(np.std([r["throughput_qps"] for r in run_results])), 2),
                "latency_p50_ms": round(float(np.median([r["latency_p50_ms"] for r in run_results])), 2),
                "latency_p90_ms": round(float(np.median([r["latency_p90_ms"] for r in run_results])), 2),
                "latency_p95_ms": round(float(np.median([r["latency_p95_ms"] for r in run_results])), 2),
                "latency_p99_ms": round(float(np.median([r["latency_p99_ms"] for r in run_results])), 2),
                "run_details": run_results,
            })

    return {
        "metadata": env_meta, "mode": mode, "endpoint": endpoint,
        "runs_per_tier": runs, "requests_per_tier": requests_per_tier,
        "cold_start": {"first_request_latency_ms": cold_start_ms, "warm_avg_latency_ms": warm_avg_ms, "warmup_requests": warmup_count},
        "concurrency_tiers": tier_results,
    }


def generate_markdown_report(data: Dict[str, Any], output_path: str):
    """Generate documentation artifact wrapped in generated blocks."""
    meta, cold, tiers = data["metadata"], data["cold_start"], data["concurrency_tiers"]
    tier_rows = "\n".join(
        f"| **{t['concurrency']}** | {t['total_requests']} | {t['total_requests'] - t['failed_requests']}/{t['failed_requests']} "
        f"| {t['throughput_qps_mean']} +/- {t['throughput_qps_std']} | {t['latency_p50_ms']} | {t['latency_p90_ms']} | {t['latency_p95_ms']} | {t['latency_p99_ms']} |"
        for t in tiers
    )
    content = f"""# TrustShield AI — HTTP Transaction Scoring Benchmark Report

<!-- BEGIN GENERATED -->
- Timestamp: `{meta['timestamp']}`
- Platform: `{meta['platform']}` ({meta['cpu_count_logical']} vCPUs, {meta['ram_total_gb']} GB RAM)
- Active Model: `{meta['classifier_name']}`
- First Request (Cold Start): `{cold['first_request_latency_ms']} ms`
- Warm Steady-State Latency: `{cold['warm_avg_latency_ms']} ms`

| Concurrency | Requests | Success/Fail | QPS (Mean +/- Std) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|---|---|
{tier_rows}
<!-- END GENERATED -->
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="TrustShield AI HTTP Transaction Scoring Benchmark")
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=os.getenv("TRUSTSHIELD_API_KEY", "dev-operator-api-key-32chars-long"))
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tiers", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-json", type=str, default="models/http_benchmark_results.json")
    parser.add_argument("--output-md", type=str, default="docs/HTTP_BENCHMARK_REPORT.md")
    args = parser.parse_args()

    results = asyncio.run(run_benchmark(args.url, args.api_key, args.tiers, args.requests, args.warmup, args.runs))
    for p, fn in [(os.path.join(_PROJECT_ROOT, args.output_json), lambda f: json.dump(results, f, indent=2)), (os.path.join(_PROJECT_ROOT, args.output_md), lambda f: generate_markdown_report(results, os.path.join(_PROJECT_ROOT, args.output_md)))]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if p.endswith(".json"):
            with open(p, "w", encoding="utf-8") as f:
                fn(f)
        else:
            generate_markdown_report(results, p)


if __name__ == "__main__":
    main()
