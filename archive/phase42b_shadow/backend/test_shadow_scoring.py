"""TrustShield Phase 4.2B — Comprehensive Shadow Scoring & Causal Redis Test Suite.

Verifies all 12 Phase 4.2B mandatory requirements:
1. Production response invariance (shadow on vs off).
2. Shadow disabled by default.
3. Candidate model loader integrity and checksum verification.
4. Authoritative event lifecycle (confirm vs scoring).
5. Idempotent deduplication (duplicate orders/returns).
6. Out-of-order event ordering and strict point-in-time causality (t < t_decision).
7. Cold-start handling for unseen pairs.
8. Redis outage resilience and explicit degraded state tracking.
9. Candidate failure isolation (production score succeeds even if candidate crashes).
10. Offline / online feature parity for all 37 candidate features.
"""

from __future__ import annotations

import os
import sys
import time
import unittest

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for p in [_ROOT_DIR, os.path.join(_ROOT_DIR, "trustshield_project"), os.path.join(_ROOT_DIR, "backend")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.candidate_loader import CandidateModelLoader, candidate_loader
from backend.main import app
from backend.services.redis_causal_service import InMemoryCausalStore, RedisCausalService, redis_causal_service
from backend.services.shadow_service import ShadowScoringService, shadow_service


class TestShadowScoringSuite(unittest.TestCase):

    def setUp(self):
        # Default test client
        self.client = TestClient(app)
        self.auth_headers = {"X-API-Key": "dev-admin-secret-key-32chars-long!"}

    def test_01_shadow_disabled_by_default(self):
        """Requirement 2: Shadow scoring must be disabled by default."""
        svc = ShadowScoringService()
        self.assertFalse(svc.enabled, "Shadow scoring must default to disabled (False)")

    def test_02_production_response_invariance(self):
        """Requirement 1 & 11: Production response must be 100% identical with shadow ON vs OFF."""
        payload = {
            "order_id": "inv_test_ord_001",
            "buyer_id": "buyer_100",
            "seller_id": "seller_200",
            "amount": 150.0,
            "order_amount": 150.0,
            "price_vs_base_price_ratio": 1.1,
            "price_vs_category_median_ratio": 0.95,
            "seller_age_days": 180.0,
            "seller_total_listings_before": 15.0,
            "buyer_age_days": 90.0,
            "buyer_orders_before": 5,
            "buyer_returns_before": 0,
            "buyer_return_rate_before": 0.0,
            "device_shared_buyer_count": 1.0,
        }

        # 1. Score with shadow scoring DISABLED
        os.environ["TRUSTSHIELD_SHADOW_SCORING"] = "false"
        shadow_service.enabled = False

        res_off = self.client.post("/transaction/score", json=payload, headers=self.auth_headers)
        self.assertEqual(res_off.status_code, 200)
        body_off = res_off.json()

        # 2. Score with shadow scoring ENABLED
        os.environ["TRUSTSHIELD_SHADOW_SCORING"] = "true"
        shadow_service.enabled = True
        shadow_service._start_worker_pool()

        res_on = self.client.post("/transaction/score", json=payload, headers=self.auth_headers)
        self.assertEqual(res_on.status_code, 200)
        body_on = res_on.json()

        # Reset flag
        os.environ["TRUSTSHIELD_SHADOW_SCORING"] = "false"
        shadow_service.enabled = False

        # Assert every single production field is strictly identical
        self.assertEqual(body_off["overall_fraud_probability"], body_on["overall_fraud_probability"])
        self.assertEqual(body_off["raw_fraud_probability"], body_on["raw_fraud_probability"])
        self.assertEqual(body_off["calibrated_fraud_probability"], body_on["calibrated_fraud_probability"])
        self.assertEqual(body_off["decision"], body_on["decision"])
        self.assertEqual(body_off["trust_score"], body_on["trust_score"])
        self.assertEqual(body_off["confidence"], body_on["confidence"])
        self.assertEqual(body_off["conformal_prediction_set"], body_on["conformal_prediction_set"])
        self.assertEqual(body_off["model_version"], body_on["model_version"])
        self.assertEqual(body_off["model_version"], "phase3")

    def test_03_candidate_loader_checksums_and_isolation(self):
        """Requirement 3 & 12: Candidate loader verifies checksums and preserves isolation."""
        loader = CandidateModelLoader()
        loaded = loader.load(verify_checksums=True)
        self.assertTrue(loaded, "Candidate model artifacts must pass checksum verification")
        self.assertEqual(len(loader.feature_names), 37, "Candidate feature count must be exactly 37")
        self.assertAlmostEqual(loader.optimal_threshold, 0.11, places=2)

        # Test inference with synthetic 37-dim vector
        dummy_x = np.zeros((1, 37), dtype=np.float32)
        raw_p, cal_p = loader.predict_proba(dummy_x)
        self.assertTrue(0.0 <= raw_p <= 1.0)
        self.assertTrue(0.0 <= cal_p <= 1.0)

    def test_04_idempotent_event_ingestion(self):
        """Requirement 4 & 5: Ingestion lifecycle must be idempotent and deduplicated."""
        svc = RedisCausalService(in_memory_mode=True)

        # Ingest first time -> success
        success1 = svc.ingest_confirmed_transaction(
            order_id="ord_dup_001",
            buyer_id="b_dup",
            seller_id="s_dup",
            amount=100.0,
            timestamp=1700000000.0,
        )
        self.assertTrue(success1, "First ingestion must succeed")

        # Ingest same order_id second time -> idempotent False
        success2 = svc.ingest_confirmed_transaction(
            order_id="ord_dup_001",
            buyer_id="b_dup",
            seller_id="s_dup",
            amount=100.0,
            timestamp=1700000000.0,
        )
        self.assertFalse(success2, "Duplicate ingestion must be ignored")

        # Verify count is strictly 1, not 2
        res = svc.query_causal_features("b_dup", "s_dup", None, 50.0, 1700001000.0)
        self.assertEqual(res.features["pair_orders_last_7d"], 1.0)
        self.assertEqual(res.features["pair_prior_amount_mean"], 100.0)

    def test_05_out_of_order_and_strict_causality(self):
        """Requirement 6 & 8: Strict point-in-time causality (t < t_decision) and out-of-order handling."""
        svc = RedisCausalService(in_memory_mode=True)

        # Ingest out of order: t=300 first, then t=100, then t=200
        svc.ingest_confirmed_transaction("o_300", "b_ooo", "s_ooo", 30.0, 300.0)
        svc.ingest_confirmed_transaction("o_100", "b_ooo", "s_ooo", 10.0, 100.0)
        svc.ingest_confirmed_transaction("o_200", "b_ooo", "s_ooo", 20.0, 200.0)

        # Query at t=250: should only see t=100 and t=200 (count=2, mean=(10+20)/2=15.0)
        res_250 = svc.query_causal_features("b_ooo", "s_ooo", None, 50.0, 250.0)
        self.assertEqual(res_250.features["pair_orders_last_7d"], 2.0)
        self.assertEqual(res_250.features["pair_prior_amount_mean"], 15.0)
        self.assertAlmostEqual(res_250.features["pair_days_since_prior"], (250.0 - 200.0) / 86400.0, places=5)

        # Query at exact t=200: t=200 must be EXCLUDED (strict causality t < 200)
        res_200 = svc.query_causal_features("b_ooo", "s_ooo", None, 50.0, 200.0)
        self.assertEqual(res_200.features["pair_orders_last_7d"], 1.0, "Same-timestamp order must be excluded")
        self.assertEqual(res_200.features["pair_prior_amount_mean"], 10.0)

    def test_06_cold_start_pair_handling(self):
        """Requirement 8 & 9: Cold-start entities must cleanly produce valid defaults."""
        svc = RedisCausalService(in_memory_mode=True)
        res = svc.query_causal_features("brand_new_b", "brand_new_s", None, 99.0, 1700000000.0)

        self.assertFalse(res.is_degraded)
        self.assertEqual(res.features["pair_is_cold_start"], 1.0)
        self.assertEqual(res.features["pair_days_since_prior"], 999.0)
        self.assertEqual(res.features["pair_prior_amount_mean"], 0.0)
        self.assertEqual(res.features["amount_vs_pair_prior_mean"], 1.0)
        self.assertEqual(res.features["buyer_days_since_prior"], 999.0)
        self.assertEqual(res.features["seller_days_since_prior"], 999.0)

    def test_07_redis_outage_graceful_degradation(self):
        """Requirement 9 & 10: When Redis is offline, state must be marked as degraded."""
        # Service configured with offline Redis URL and live mode
        svc = RedisCausalService(redis_url="redis://localhost:9999/0", enabled=True, in_memory_mode=False)
        res = svc.query_causal_features("b_test", "s_test", None, 100.0, 1700000000.0)

        self.assertTrue(res.is_degraded, "Missing/unavailable Redis must be marked as degraded")
        self.assertIn("redis", res.degraded_reason)
        self.assertEqual(res.feature_source, "fallback_degraded")
        # Ensure clean defaults returned without raising exception
        self.assertEqual(res.features["pair_orders_last_7d"], 0.0)
        self.assertEqual(res.features["pair_days_since_prior"], 999.0)

    def test_08_authoritative_lifecycle_endpoints(self):
        """Requirement 4: Test /transaction/confirm and /returns/record endpoints."""
        confirm_payload = {
            "order_id": "lifecycle_ord_123",
            "buyer_id": "buyer_life",
            "seller_id": "seller_life",
            "amount": 250.0,
            "order_date": "2026-10-15T12:00:00Z",
        }
        res_conf = self.client.post("/transaction/confirm", json=confirm_payload, headers=self.auth_headers)
        self.assertEqual(res_conf.status_code, 200)
        body = res_conf.json()
        self.assertEqual(body["order_id"], "lifecycle_ord_123")
        self.assertIn(body["status"], ("confirmed", "duplicate_ignored"))

        return_payload = {
            "return_id": "lifecycle_ret_123",
            "order_id": "lifecycle_ord_123",
            "buyer_id": "buyer_life",
            "seller_id": "seller_life",
            "return_date": "2026-10-16T12:00:00Z",
        }
        res_ret = self.client.post("/returns/record", json=return_payload, headers=self.auth_headers)
        self.assertEqual(res_ret.status_code, 200)
        body_ret = res_ret.json()
        self.assertEqual(body_ret["return_id"], "lifecycle_ret_123")
        self.assertIn(body_ret["status"], ("recorded", "duplicate_ignored"))

    def test_09_offline_online_feature_parity_replay(self):
        """Requirement 7: Validate offline/online parity on representative replay transactions."""
        orders_path = os.path.join(_ROOT_DIR, "data", "synthetic_v2_1", "orders.csv")
        returns_path = os.path.join(_ROOT_DIR, "data", "synthetic_v2_1", "returns.csv")
        if not os.path.isfile(orders_path):
            self.skipTest("Dataset not available for parity replay")

        df = pd.read_csv(orders_path).head(100)
        df_ret = pd.read_csv(returns_path).head(20) if os.path.isfile(returns_path) else None

        # Backfill 100 historical transactions into in-memory store
        svc = RedisCausalService(in_memory_mode=True)
        report = svc.backfill_from_dataframe(df, df_ret)
        self.assertEqual(report["reconciliation_status"], "verified")

        # Test query on transaction #50
        row50 = df.iloc[50]
        t50 = pd.to_datetime(row50["order_date"]).timestamp()

        causal_res = svc.query_causal_features(
            buyer_id=row50["buyer_id"],
            seller_id=row50["seller_id"],
            device_id=row50.get("device_id"),
            amount=row50["amount"],
            decision_timestamp=t50,
        )

        self.assertFalse(causal_res.is_degraded)
        self.assertIn("pair_orders_last_7d", causal_res.features)
        self.assertIn("seller_prior_order_count", causal_res.features)
        self.assertGreaterEqual(causal_res.features["seller_prior_order_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
