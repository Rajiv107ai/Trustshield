"""
TrustShield AI — Live HTTP Server Verification Test Suite.

Validates core REST endpoints against a running backend server (or in-process TestClient fallback):
1. GET  /health                      -- Server liveness & artifact status
2. GET  /ready                       -- Service readiness state
3. POST /transaction/score           -- Normal low-risk transaction scoring
4. POST /transaction/score           -- High-risk anomalous transaction scoring
5. POST /transaction/explain         -- Local TreeSHAP explainability attributions
6. GET  /fraud-rings                 -- Pre-computed graph fraud rings
7. POST /listing/analyze             -- Multimodal catalog listing verification
8. POST /return/analyze              -- Specialized return fraud detection
9. GET  /results                     -- Canonical scientific benchmark payload
10. GET /metrics                     -- Prometheus observability stream

Usage:
    python scripts/smoke_test_api.py
    python scripts/smoke_test_api.py --port 8000 --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, Optional
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")

DEFAULT_API_KEY = "dev-operator-api-key-32chars-long"


def make_http_client(base_url: str, api_key: str):
    """Simple HTTP client using standard library urllib."""
    def request(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> tuple[int, Dict[str, Any]]:
        url = f"{base_url.rstrip('/')}{path}"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        }
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
                raw = resp.read().decode("utf-8")
                try:
                    body = json.loads(raw)
                except Exception:
                    body = {"raw": raw}
                return status, body
        except urllib.error.HTTPError as err:
            raw_err = err.read().decode("utf-8")
            try:
                err_body = json.loads(raw_err)
            except Exception:
                err_body = {"error": str(err), "raw": raw_err}
            return err.code, err_body
        except Exception as exc:
            return 0, {"error": str(exc)}

    return request


def make_test_client():
    """In-process TestClient fallback if external server is not currently running."""
    from fastapi.testclient import TestClient
    from backend.main import app, store

    if not store.is_loaded:
        store.load()

    client = TestClient(app, headers={"X-API-Key": DEFAULT_API_KEY})

    def request(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> tuple[int, Dict[str, Any]]:
        resp = client.request(method, path, json=payload)
        try:
            body = resp.json()
        except Exception:
            body = {"text": resp.text}
        return resp.status_code, body

    return request


def run_smoke_tests(base_url: str, port: int, api_key: str) -> bool:
    print("=" * 70)
    print("TRUSTSHIELD AI — LIVE API HTTP ENDPOINT VERIFICATION")
    print("=" * 70)

    target_url = f"http://{base_url.replace('http://', '').replace('https://', '')}:{port}"
    print(f"Target backend URL: {target_url}")

    client_fn = make_http_client(target_url, api_key)
    status, health_body = client_fn("GET", "/health")

    if status == 0:
        print(f"\n[INFO] Backend not actively listening at {target_url} ({health_body.get('error')}).")
        print("[INFO] Initializing in-process TestClient to verify API contracts...")
        client_fn = make_test_client()
        status, health_body = client_fn("GET", "/health")

    all_passed = True

    # 1. /health
    print("\n[Test 1/10] Checking GET /health...")
    if status == 200 and health_body.get("status") == "ok":
        print(f"  PASS: status=ok, models_loaded={health_body.get('models_loaded')}, rings={health_body.get('n_rings')}")
    else:
        print(f"  FAIL: Expected 200 ok, got HTTP {status}: {health_body}")
        all_passed = False

    # 2. /ready
    print("\n[Test 2/10] Checking GET /ready...")
    status, ready_body = client_fn("GET", "/ready")
    if status == 200 and ready_body.get("status") in ("ready", "degraded"):
        print(f"  PASS: status={ready_body.get('status')}, models_ready={ready_body.get('models_ready')}")
    else:
        print(f"  FAIL: Expected 200 ready/degraded, got HTTP {status}: {ready_body}")
        all_passed = False

    # 3. /transaction/score (Normal)
    print("\n[Test 3/10] Checking POST /transaction/score (Normal Safe Transaction)...")
    normal_req = {
        "order_id": "SMOKE_ORD_SAFE_01",
        "buyer_id": "BUYER_VERIFIED_77",
        "seller_id": "SELLER_REPUTABLE_12",
        "amount": 65.50,
        "base_price": 70.0,
        "category_median_price": 68.0,
        "buyer_orders_before": 15,
        "buyer_returns_before": 0,
        "buyer_age_days": 180.0,
        "seller_age_days": 450.0,
        "seller_total_listings_before": 120,
    }
    status, score_body = client_fn("POST", "/transaction/score", normal_req)
    if status == 200 and score_body.get("decision") == "ALLOW":
        print(f"  PASS: risk={score_body.get('overall_fraud_probability')}, decision={score_body.get('decision')}, trust_score={score_body.get('trust_score')}")
    else:
        print(f"  FAIL: Expected 200 ALLOW, got HTTP {status}: {score_body}")
        all_passed = False

    # 4. /transaction/score (High Risk)
    print("\n[Test 4/10] Checking POST /transaction/score (High-Risk Collusive Order)...")
    risk_req = {
        "order_id": "SMOKE_ORD_RISK_01",
        "buyer_id": "BUYER_RING_MEMBER_01",
        "seller_id": "SELLER_RING_LEADER_01",
        "amount": 890.0,
        "base_price": 400.0,
        "category_median_price": 350.0,
        "buyer_orders_before": 1,
        "buyer_returns_before": 0,
        "buyer_age_days": 2.0,
        "seller_age_days": 35.0,
        "seller_total_listings_before": 12,
        "device_shared_buyer_count": 6.0,
        "share_degree": 5.0,
        "share_component_size": 8.0,
    }
    status, risk_body = client_fn("POST", "/transaction/score", risk_req)
    if status == 200 and risk_body.get("decision") in ("REVIEW", "HOLD", "BLOCK"):
        print(f"  PASS: risk={risk_body.get('overall_fraud_probability')}, decision={risk_body.get('decision')}, disagreement={risk_body.get('model_disagreement')}")
    else:
        print(f"  FAIL: Expected 200 REVIEW/HOLD/BLOCK, got HTTP {status}: {risk_body}")
        all_passed = False

    # 5. /transaction/explain (TreeSHAP)
    print("\n[Test 5/10] Checking POST /transaction/explain (TreeSHAP Attributions)...")
    status, explain_body = client_fn("POST", "/transaction/explain", {"transaction_data": risk_req})
    if status == 200 and "all_attributions" in explain_body:
        n_drivers = len(explain_body.get("top_positive_drivers", []))
        narrative = str(explain_body.get("investigator_narrative") or "")
        preview = narrative[:65] + ("..." if len(narrative) > 65 else "")
        print(f"  PASS: Top drivers returned ({n_drivers}), narrative preview: {preview}")
    else:
        print(f"  FAIL: Expected 200 with attributions, got HTTP {status}: {explain_body}")
        all_passed = False

    # 6. /fraud-rings
    print("\n[Test 6/10] Checking GET /fraud-rings (Precomputed Clusters)...")
    status, rings_body = client_fn("GET", "/fraud-rings?limit=5")
    if status == 200 and "rings" in rings_body:
        print(f"  PASS: total_rings={rings_body.get('total_rings')}, high_risk={rings_body.get('high_risk_rings')}, sample_count={len(rings_body.get('rings', []))}")
    else:
        print(f"  FAIL: Expected 200 with rings, got HTTP {status}: {rings_body}")
        all_passed = False

    # 7. /listing/analyze (Multimodal)
    print("\n[Test 7/10] Checking POST /listing/analyze (Multimodal Listing Detector)...")
    listing_req = {
        "listing_id": "SMOKE_LIST_01",
        "product_id": "PROD_TEST_01",
        "price": 85.0,
        "base_price": 200.0,
        "category_median_price": 190.0,
        "seller_age_days_at_listing": 4.0,
        "seller_listings_before": 1,
        "multimodal_similarity_score": 0.25,
    }
    status, listing_body = client_fn("POST", "/listing/analyze", listing_req)
    if status == 200 and "fake_listing_probability" in listing_body:
        print(f"  PASS: fake_prob={listing_body.get('fake_listing_probability')}, model={listing_body.get('model_used')}")
    else:
        print(f"  FAIL: Expected 200 with fake_listing_probability, got HTTP {status}: {listing_body}")
        all_passed = False

    # 8. /return/analyze
    print("\n[Test 8/10] Checking POST /return/analyze (Return Fraud Scorer)...")
    return_req = {
        "return_id": "SMOKE_RET_01",
        "order_amount": 320.0,
        "days_to_return": 1.0,
        "buyer_prior_returns": 5,
        "buyer_orders_before_return": 6,
        "buyer_return_rate_before": 0.833,
        "buyer_age_days_at_return": 90.0,
        "seller_age_days_at_return": 200.0,
        "seller_prior_returns": 2,
        "seller_orders_before_return": 45,
        "seller_return_rate_before": 0.044,
        "reason": "wrong_item_received",
    }
    status, ret_body = client_fn("POST", "/return/analyze", return_req)
    if status == 200 and "return_fraud_probability" in ret_body:
        print(f"  PASS: return_prob={ret_body.get('return_fraud_probability')}, decision={ret_body.get('decision')}, reasons={ret_body.get('reason_codes')}")
    else:
        print(f"  FAIL: Expected 200 with return_fraud_probability, got HTTP {status}: {ret_body}")
        all_passed = False

    # 9. /results
    print("\n[Test 9/10] Checking GET /results (Canonical Benchmark Results JSON)...")
    status, res_body = client_fn("GET", "/results")
    if status == 200 and "pre_registered_primary_analysis_20_seeds" in res_body:
        print("  PASS: Canonical 20-seed benchmark results payload verified.")
    else:
        print(f"  FAIL: Expected 200 with benchmark results, got HTTP {status}")
        all_passed = False

    # 10. /metrics (Prometheus)
    print("\n[Test 10/10] Checking GET /metrics (Prometheus Telemetry Stream)...")
    status, m_body = client_fn("GET", "/metrics")
    if status == 200:
        print("  PASS: Prometheus OpenMetrics stream reachable (HTTP 200).")
    else:
        print(f"  FAIL: Expected 200 for /metrics, got HTTP {status}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL 10 API ENDPOINTS VERIFIED SUCCESSFULLY!")
    else:
        print("SOME ENDPOINT CHECKS FAILED — REVIEW LOGS ABOVE")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrustShield AI API Smoke Test Harness")
    parser.add_argument("--host", default="127.0.0.1", help="Target hostname/IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Target port (default: 8000)")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API Key for authentication")
    args = parser.parse_args()

    success = run_smoke_tests(base_url=args.host, port=args.port, api_key=args.api_key)
    sys.exit(0 if success else 1)
