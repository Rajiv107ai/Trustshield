"""Phase 20 End-to-End Validation Suite for TrustShield AI.

Tests 9 canonical operational cases against the FastAPI application:
1. Normal transaction (warm buyer, warm seller, low risk)
2. High-risk transaction (price anomaly, shared device, ring connection)
3. New seller (cold start flag + confidence attenuation)
4. New buyer (cold start flag + confidence attenuation)
5. Missing image listing (/listing/analyze without visual embeddings)
6. Missing graph history (zero graph edges, unlinked entity)
7. Invalid input (conflicting amount vs order_amount returns HTTP 400)
8. Future-dated event / temporal boundary validation
9. Known suspicious ring (/fraud-rings returns precomputed rings)
"""

import sys
import os
import json
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "trustshield_project"))
sys.path.insert(0, os.path.join(ROOT_DIR, "backend"))

from backend.main import app
from backend.model_loader import store


def run_e2e_suite():
    print("=" * 60)
    print("TRUSTSHIELD AI - END-TO-END VALIDATION SUITE (9 CASES)")
    print("=" * 60)

    with TestClient(app) as client:
        # Case 1: Normal transaction
        print("\n[Case 1/9] Testing Normal Transaction...")
        c1_req = {
            "order_id": "NORM_001",
            "buyer_id": "BUYER_000001",
            "seller_id": "SELLER_000001",
            "amount": 50.0,
            "base_price": 50.0,
            "category_median_price": 50.0,
            "buyer_orders_before": 10,
            "seller_total_listings_before": 20,
            "buyer_age_days": 180.0,
            "seller_age_days": 365.0,
            "device_shared_buyer_count": 1.0,
            "share_degree": 0.0,
            "share_component_size": 1.0,
        }
        r1 = client.post("/transaction/score", json=c1_req)
        assert r1.status_code == 200, f"Case 1 failed: {r1.text}"
        d1 = r1.json()
        assert d1["decision"] in ("ALLOW", "REVIEW"), f"Unexpected decision {d1['decision']}"
        assert d1["cold_start"] is False, "Normal transaction should not be cold start"
        assert d1["overall_fraud_probability"] < 0.60, f"Normal risk should be low/medium: {d1['overall_fraud_probability']}"
        print(f"  PASS: Probability={d1['overall_fraud_probability']}, Decision={d1['decision']}, TrustScore={d1['trust_score']}")

        # Case 2: High-risk transaction
        print("\n[Case 2/9] Testing High-Risk Transaction...")
        c2_req = {
            "order_id": "RISK_002",
            "buyer_id": "BUYER_000099",
            "seller_id": "SELLER_000099",
            "amount": 2500.0,
            "base_price": 50.0,
            "category_median_price": 50.0,
            "buyer_orders_before": 25,
            "seller_total_listings_before": 5,
            "buyer_age_days": 10.0,
            "seller_age_days": 15.0,
            "device_shared_buyer_count": 8.0,
            "share_degree": 12.0,
            "share_component_size": 15.0,
            "buyer_pagerank": 0.05,
            "buyer_return_rate_before": 0.85,
        }
        r2 = client.post("/transaction/score", json=c2_req)
        assert r2.status_code == 200, f"Case 2 failed: {r2.text}"
        d2 = r2.json()
        assert d2["decision"] in ("REVIEW", "HOLD", "BLOCK"), f"Unexpected decision {d2['decision']}"
        assert d2["overall_fraud_probability"] > d1["overall_fraud_probability"], "Risk must be higher than Case 1"
        print(f"  PASS: Probability={d2['overall_fraud_probability']}, Decision={d2['decision']}, Disagreement={d2['model_disagreement']}")

        # Case 3: New seller (cold start)
        print("\n[Case 3/9] Testing New Seller (Cold Start)...")
        c3_req = {
            "order_id": "COLD_SELLER_003",
            "buyer_id": "BUYER_000001",
            "seller_id": "SELLER_BRAND_NEW",
            "amount": 60.0,
            "base_price": 60.0,
            "category_median_price": 60.0,
            "buyer_orders_before": 15,
            "seller_total_listings_before": 1,  # < 3
            "buyer_age_days": 120.0,
            "seller_age_days": 2.0,            # < 7
        }
        r3 = client.post("/transaction/score", json=c3_req)
        assert r3.status_code == 200, f"Case 3 failed: {r3.text}"
        d3 = r3.json()
        assert d3["cold_start"] is True, "Must flag cold_start=True for new seller"
        assert "COLD_START_INSUFFICIENT_HISTORY" in d3["reason_codes"], "Reason codes must include cold start notice"
        assert d3["confidence"] <= 0.75, f"Confidence should be attenuated, got {d3['confidence']}"
        print(f"  PASS: cold_start={d3['cold_start']}, confidence={d3['confidence']}, reasons={d3['reason_codes']}")

        # Case 4: New buyer (cold start)
        print("\n[Case 4/9] Testing New Buyer (Cold Start)...")
        c4_req = {
            "order_id": "COLD_BUYER_004",
            "buyer_id": "BUYER_BRAND_NEW",
            "seller_id": "SELLER_000001",
            "amount": 45.0,
            "base_price": 45.0,
            "category_median_price": 45.0,
            "buyer_orders_before": 0,           # < 3
            "seller_total_listings_before": 50,
            "buyer_age_days": 1.0,             # < 7
            "seller_age_days": 300.0,
        }
        r4 = client.post("/transaction/score", json=c4_req)
        assert r4.status_code == 200, f"Case 4 failed: {r4.text}"
        d4 = r4.json()
        assert d4["cold_start"] is True, "Must flag cold_start=True for new buyer"
        assert "COLD_START_INSUFFICIENT_HISTORY" in d4["reason_codes"]
        print(f"  PASS: cold_start={d4['cold_start']}, confidence={d4['confidence']}, reasons={d4['reason_codes']}")

        # Case 5: Missing image listing
        print("\n[Case 5/9] Testing Missing Image Listing...")
        c5_req = {
            "listing_id": "LISTING_NO_IMAGE_005",
            "seller_id": "SELLER_000001",
            "product_id": "UNKNOWN_PRODUCT",
            "price": 100.0,
            "price_difference_pct": 0.05,
            # Caller does not supply multimodal similarity score; image absent
        }
        r5 = client.post("/listing/analyze", json=c5_req)
        assert r5.status_code == 200, f"Case 5 failed: {r5.text}"
        d5 = r5.json()
        assert "fake_listing_probability" in d5
        assert "risk_label" in d5
        assert d5["listing_id"] == "LISTING_NO_IMAGE_005"
        print(f"  PASS: fake_listing_probability={d5['fake_listing_probability']}, risk_label={d5['risk_label']}, model={d5['model_used']}")

        # Case 6: Missing graph history
        print("\n[Case 6/9] Testing Missing Graph History (Isolated Node)...")
        c6_req = {
            "order_id": "ISOLATED_006",
            "buyer_id": "BUYER_ISOLATED",
            "seller_id": "SELLER_ISOLATED",
            "amount": 75.0,
            "base_price": 75.0,
            "category_median_price": 75.0,
            "buyer_orders_before": 5,
            "seller_total_listings_before": 5,
            "buyer_age_days": 30.0,
            "seller_age_days": 30.0,
            "share_degree": 0.0,
            "share_component_size": 1.0,
            "buyer_seller_degree": 0.0,
            "buyer_pagerank": 0.0,
        }
        r6 = client.post("/transaction/score", json=c6_req)
        assert r6.status_code == 200, f"Case 6 failed: {r6.text}"
        d6 = r6.json()
        assert 0.0 <= d6["overall_fraud_probability"] <= 1.0
        print(f"  PASS: Isolated entity scored successfully, risk={d6['overall_fraud_probability']}, decision={d6['decision']}")

        # Case 7: Invalid input (conflicting amount)
        print("\n[Case 7/9] Testing Invalid Input (Conflicting Amounts)...")
        c7_req = {
            "order_id": "INVALID_007",
            "amount": 100.0,
            "order_amount": 250.0,  # Conflicting with amount
        }
        r7 = client.post("/transaction/score", json=c7_req)
        assert r7.status_code == 400, f"Case 7 should return 400, got {r7.status_code}"
        print(f"  PASS: Conflicting inputs correctly rejected with HTTP 400: {r7.json()['detail']}")

        # Case 8: Future-dated event / temporal boundary validation
        print("\n[Case 8/9] Testing Temporal Boundary Validation...")
        from temporal_utils import is_strictly_before, filter_historical_events
        import pandas as pd
        t_decision = "2025-06-01 12:00:00"
        t_past = "2025-06-01 11:59:59"
        t_exact = "2025-06-01 12:00:00"
        t_future = "2025-06-01 12:00:01"

        assert is_strictly_before(t_past, t_decision) is True
        assert is_strictly_before(t_exact, t_decision) is False
        assert is_strictly_before(t_future, t_decision) is False

        df = pd.DataFrame({"ts": [t_past, t_exact, t_future], "val": [1, 2, 3]})
        filtered = filter_historical_events(df, "ts", t_decision)
        assert len(filtered) == 1
        assert filtered.iloc[0]["val"] == 1
        print("  PASS: Temporal strict historical invariant (< decision_time) validated")

        # Case 9: Known suspicious ring
        print("\n[Case 9/9] Testing Precomputed Fraud Rings Query...")
        r9 = client.get("/fraud-rings?min_risk=0.3&limit=5")
        assert r9.status_code == 200, f"Case 9 failed: {r9.text}"
        d9 = r9.json()
        assert d9["total_rings"] > 0, "Should have found rings"
        assert d9["high_risk_rings"] > 0, "Should have found high-risk rings"
        assert len(d9["rings"]) <= 5
        top_ring = d9["rings"][0]
        assert top_ring["avg_risk_score"] >= 0.3
        print(f"  PASS: Retrieved {d9['total_rings']} total rings ({d9['high_risk_rings']} high risk); Top Ring: {top_ring['ring_id']} (size={top_ring['size']}, risk={top_ring['avg_risk_score']})")

    print("\n" + "=" * 60)
    print("ALL 9 OPERATIONAL CASES PASSED CLEANLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_e2e_suite()
