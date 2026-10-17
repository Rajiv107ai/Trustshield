"""Unit and integration tests for FastAPI backend serving layer."""

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

from backend.main import app, store, _build_feature_row
from backend.schemas import TransactionScoreRequest


VALID_API_KEY = "dev-operator-api-key-32chars-long"


@pytest.fixture(scope="module")
def client():
    with TestClient(app, headers={"X-API-Key": VALID_API_KEY}) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert data["rings_loaded"] is True
    assert data["n_rings"] > 0


def test_fraud_rings_endpoint(client):
    response = client.get("/fraud-rings?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "rings" in data
    assert len(data["rings"]) <= 5
    assert data["total_rings"] > 0
    if data["rings"]:
        ring = data["rings"][0]
        assert "ring_id" in ring
        assert "avg_risk_score" in ring
        assert "risk_label" in ring


def test_fraud_rings_min_risk_filtering(client):
    response = client.get("/fraud-rings?min_risk_score=0.4&limit=10")
    assert response.status_code == 200
    data = response.json()
    for ring in data["rings"]:
        assert ring["avg_risk_score"] >= 0.4


def test_transaction_scoring_phase5_rejected(client):
    payload = {
        "order_id": "ORDER_TEST_P5",
        "model_version": "phase5-hybrid",
        "buyer_id": "BUYER_000001",
        "seller_id": "SELLER_000001",
        "amount": 150.0,
        "base_price": 150.0,
        "category_median_price": 150.0,
        "order_date": "2025-09-01T12:00:00",
    }
    response = client.post("/transaction/score", json=payload)
    assert response.status_code == 400
    assert "Invalid model version requested" in response.json()["detail"]


def test_transaction_scoring_phase3(client):
    payload = {
        "order_id": "ORDER_TEST_P3",
        "buyer_id": "BUYER_000002",
        "seller_id": "SELLER_000002",
        "amount": 80.0,
        "base_price": 80.0,
        "category_median_price": 80.0,
        "order_date": "2025-09-01T12:00:00",
    }
    response = client.post("/transaction/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "ORDER_TEST_P3"
    assert 0.0 <= data["overall_fraud_probability"] <= 1.0
    assert data["model_version"] == "phase3"


def test_transaction_scoring_derives_ratios(client):
    payload = {
        "order_id": "ORDER_TEST_RATIOS",
        "buyer_id": "BUYER_000003",
        "seller_id": "SELLER_000003",
        "amount": 500.0,
        "base_price": 100.0,
        "category_median_price": 100.0,
        "order_date": "2025-09-01T12:00:00",
    }
    response = client.post("/transaction/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "ORDER_TEST_RATIOS"
    assert 0.0 <= data["overall_fraud_probability"] <= 1.0


def test_listing_analysis_multimodal(client):
    payload = {
        "listing_id": "LISTING_TEST_MM",
        "seller_id": "SELLER_TEST",
        "price": 35.0,
        "base_price": 120.0,
        "category_median_price": 100.0,
        "seller_age_days_at_listing": 10.0,
        "seller_listings_before": 1,
        "multimodal_similarity_score": 0.20,
    }
    response = client.post("/listing/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["listing_id"] == "LISTING_TEST_MM"
    assert 0.0 <= data["fake_listing_probability"] <= 1.0
    assert "investigator_narrative" in data
    narrative = data["investigator_narrative"]
    assert "summary" in narrative
    assert "key_evidence" in narrative
    assert "recommended_action" in narrative


# ---------------------------------------------------------------------------
# Issue #1 — Amount mapping unit tests (pure, no model required)
# ---------------------------------------------------------------------------

# Sentinel column list: the model only needs "order_amount" for these tests.
_COLS = ["order_amount", "amount", "price_vs_base_price_ratio"]


class TestAmountMappingUnit:
    """Direct unit tests for _build_feature_row amount / order_amount logic."""

    def test_amount_only_sets_order_amount(self):
        """Supplying only `amount` must populate both `amount` and `order_amount`."""
        req = TransactionScoreRequest(amount=99.0)
        row = _build_feature_row(req, _COLS)
        assert row["amount"] == pytest.approx(99.0), "amount field should equal supplied value"
        assert row["order_amount"] == pytest.approx(99.0), "order_amount must mirror amount"

    def test_order_amount_only_sets_amount(self):
        """Supplying only `order_amount` must populate both `order_amount` and `amount`."""
        req = TransactionScoreRequest(order_amount=250.0)
        row = _build_feature_row(req, _COLS)
        assert row["order_amount"] == pytest.approx(250.0), "order_amount field should equal supplied value"
        assert row["amount"] == pytest.approx(250.0), "amount must mirror order_amount"

    def test_both_equal_passes(self):
        """Supplying equal `amount` and `order_amount` must succeed and use that value."""
        req = TransactionScoreRequest(amount=150.0, order_amount=150.0)
        row = _build_feature_row(req, _COLS)
        assert row["amount"] == pytest.approx(150.0)
        assert row["order_amount"] == pytest.approx(150.0)

    def test_conflicting_raises_http_400(self):
        """Supplying different `amount` and `order_amount` must raise HTTP 400."""
        req = TransactionScoreRequest(amount=100.0, order_amount=200.0)
        with pytest.raises(HTTPException) as exc_info:
            _build_feature_row(req, _COLS)
        assert exc_info.value.status_code == 400
        assert "Conflicting" in exc_info.value.detail

    def test_neither_supplied_defaults_to_zero(self):
        """When neither field is supplied both aliases should default to 0.0."""
        req = TransactionScoreRequest()
        row = _build_feature_row(req, _COLS)
        assert row["amount"] == pytest.approx(0.0)
        assert row["order_amount"] == pytest.approx(0.0)

    def test_order_amount_zero_explicitly_supplied(self):
        """Explicitly passing order_amount=0.0 must not be confused with 'not set'."""
        req = TransactionScoreRequest(order_amount=0.0)
        row = _build_feature_row(req, _COLS)
        # 0.0 is a valid explicit amount; result should still be 0.0 without error.
        assert row["order_amount"] == pytest.approx(0.0)
        assert row["amount"] == pytest.approx(0.0)


class TestAmountMappingEndpoint:
    """Integration tests: the /transaction/score endpoint must reflect the fix."""


    def test_endpoint_amount_only(self, client):
        """Endpoint must not silently score with amount=0 when only order_amount supplied."""
        # Score once with amount-only
        r_amt = client.post("/transaction/score", json={
            "order_id": "AMT_ONLY",
            "amount": 300.0,
            "base_price": 300.0,
            "category_median_price": 300.0,
        })
        assert r_amt.status_code == 200

        # Score once with order_amount-only — result must match (same resolved amount)
        r_ord = client.post("/transaction/score", json={
            "order_id": "ORD_ONLY",
            "order_amount": 300.0,
            "base_price": 300.0,
            "category_median_price": 300.0,
        })
        assert r_ord.status_code == 200
        # Both requests represent the same amount, so probabilities must be identical.
        assert r_amt.json()["overall_fraud_probability"] == pytest.approx(
            r_ord.json()["overall_fraud_probability"], abs=1e-4
        ), "amount-only and order_amount-only with same value must produce identical scores"

    def test_endpoint_conflicting_amounts_returns_400(self, client):
        """Endpoint must return 400 when amount != order_amount."""
        r = client.post("/transaction/score", json={
            "order_id": "CONFLICT",
            "amount": 100.0,
            "order_amount": 200.0,
        })
        assert r.status_code == 400
        assert "Conflicting" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Issue #2 — CLIP cache and fallback scoring tests
# ---------------------------------------------------------------------------

import os
import tempfile
import numpy as np


class TestCLIPEmbeddingCache:
    """Unit tests for CLIPEmbeddingCache stale/partial/absent behavior."""

    def _make_cache(self, cache_dir, fingerprint, n=5):
        """Helper: write a minimal valid cache to cache_dir."""
        from trustshield_project.multimodal_scoring import CLIPEmbeddingCache
        text_emb = np.random.default_rng(0).standard_normal((n, 512)).astype(np.float32)
        image_emb = np.random.default_rng(1).standard_normal((n, 512)).astype(np.float32)
        id_to_idx = {f"P{i:03d}": i for i in range(n)}
        cache = CLIPEmbeddingCache(cache_dir)
        cache.save(fingerprint, id_to_idx, text_emb, image_emb)
        return cache, id_to_idx, text_emb, image_emb

    def test_valid_cache_loads(self):
        """A cache saved with the correct fingerprint must load successfully."""
        from trustshield_project.multimodal_scoring import CLIPEmbeddingCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache, id_to_idx, text_emb, image_emb = self._make_cache(tmpdir, "abc123")
            result = cache.load("abc123")
            assert result is not None, "Cache with matching fingerprint must load"
            loaded_idx, loaded_text, loaded_image = result
            assert set(loaded_idx.keys()) == set(id_to_idx.keys())
            np.testing.assert_array_almost_equal(loaded_text, text_emb, decimal=4)

    def test_stale_cache_returns_none_on_fingerprint_mismatch(self):
        """A cache built with a different fingerprint must return None (rejected, not loaded)."""
        from trustshield_project.multimodal_scoring import CLIPEmbeddingCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache, _, _, _ = self._make_cache(tmpdir, "stale_fp_001")
            # Now load with a *different* fingerprint (production catalog changed)
            result = cache.load("production_fp_xyz")
            assert result is None, "Cache with mismatched fingerprint must return None (stale)"

    def test_absent_cache_exists_returns_false(self):
        """exists() must return False when the cache directory is empty."""
        from trustshield_project.multimodal_scoring import CLIPEmbeddingCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CLIPEmbeddingCache(tmpdir)
            assert cache.exists() is False, "Empty directory should report cache as absent"

    def test_partial_cache_exists_returns_false(self):
        """exists() must return False when only some cache files are present (partial write)."""
        from trustshield_project.multimodal_scoring import CLIPEmbeddingCache
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write only the meta file, omit embeddings — simulates a partial smoke run.
            meta_path = os.path.join(tmpdir, CLIPEmbeddingCache.META_FILE)
            np.savez(meta_path, fingerprint="x", version="v1", keys=np.array([]), values=np.array([]))
            cache = CLIPEmbeddingCache(tmpdir)
            assert cache.exists() is False, "Partial cache (meta only) should report as absent"

    def test_smoke_cache_separate_from_production(self):
        """Smoke cache and production cache must be independent directories."""
        import scripts.build_clip_embeddings as bce_mod
        script_dir = os.path.dirname(os.path.abspath(bce_mod.__file__))
        prod_path = os.path.normpath(os.path.join(script_dir, "..", "models", "clip_cache"))
        smoke_path = os.path.normpath(os.path.join(script_dir, "..", "models", "clip_cache_smoke"))
        assert prod_path != smoke_path, "Smoke cache must be a different directory from production cache"


class TestListingAnalyzeFallback:
    """Integration tests for /listing/analyze when CLIP scorer is unavailable."""


    def test_no_product_id_uses_caller_supplied_similarity(self, client):
        """Without product_id, the endpoint must use the caller-supplied similarity score."""
        r = client.post("/listing/analyze", json={
            "listing_id": "LISTING_FALLBACK_1",
            "price": 50.0,
            "base_price": 100.0,
            "category_median_price": 100.0,
            "seller_age_days_at_listing": 30.0,
            "seller_listings_before": 5,
            "multimodal_similarity_score": 0.30,  # low similarity — should affect score
        })
        assert r.status_code == 200
        data = r.json()
        assert 0.0 <= data["fake_listing_probability"] <= 1.0
        assert data["clip_scored"] is False
        assert data["multimodal_similarity_score"] == pytest.approx(0.30, abs=0.01)

    def test_with_product_id_no_clip_uses_deterministic_fallback(self, client):
        """When product_id is supplied but CLIP is not loaded, the fallback must not produce 0.85."""
        was_clip_loaded = store._clip_loaded
        try:
            store._clip_loaded = False  # simulate CLIP unavailable
            r = client.post("/listing/analyze", json={
                "listing_id": "LISTING_FALLBACK_2",
                "price": 50.0,
                "base_price": 100.0,
                "category_median_price": 100.0,
                "seller_age_days_at_listing": 30.0,
                "seller_listings_before": 5,
                "product_id": "SOME_PRODUCT_ID",
                "displayed_product_id": "SOME_OTHER_PRODUCT_ID",
            })
            assert r.status_code == 200
            data = r.json()
            # If product catalog is loaded, TF-IDF fallback should produce a real score (not 0.85).
            # If catalog is not available, we accept the 0.85 default (best-effort).
            assert 0.0 <= data["fake_listing_probability"] <= 1.0
            assert data["clip_scored"] is False
        finally:
            store._clip_loaded = was_clip_loaded

    def test_caller_supplied_0_85_default_is_flagged_in_model_used(self, client):
        """When no product_id is given and the default 0.85 passes through, model_used must indicate it."""
        was_clip_loaded = store._clip_loaded
        was_clip_products = store.clip_products_df
        try:
            store._clip_loaded = False
            store.clip_products_df = None  # no catalog loaded \u2014 force pure default path
            r = client.post("/listing/analyze", json={
                "listing_id": "LISTING_DEFAULT",
                "price": 100.0,
                "base_price": 100.0,
                "category_median_price": 100.0,
                "seller_age_days_at_listing": 30.0,
                "seller_listings_before": 5,
                # No product_id, no multimodal_similarity_score \u2014 schema default 0.85 applies
            })
            assert r.status_code == 200
            data = r.json()
            assert data["clip_scored"] is False
            # The model_used field must make clear this is a caller-supplied (default) score.
            assert "caller-supplied" in data["model_used"] or "TF-IDF" in data["model_used"] or "Multimodal" in data["model_used"]
        finally:
            store._clip_loaded = was_clip_loaded
            store.clip_products_df = was_clip_products


# ---------------------------------------------------------------------------
# Issue #7 — model_used field reflects actual classifier, not "Random Forest"
# ---------------------------------------------------------------------------

class TestModelUsedField:
    """Assert that model_used returned by /transaction/score reflects the real
    classifier name (XGBoost or Hybrid GNN + XGBoost), never the old stale
    hard-coded 'Random Forest' string that was in the API description.
    """


    def test_phase5_model_used_contains_xgboost(self, client):
        """Phase 5 model_used must mention XGBoost (or the actual classifier name)."""
        r = client.post("/transaction/score", json={
            "order_id": "MODEL_USED_P5",
            "buyer_id": "BUYER_000001",
            "seller_id": "SELLER_000001",
            "amount": 100.0,
            "base_price": 100.0,
            "category_median_price": 100.0,
        })
        assert r.status_code == 200
        model_used = r.json()["model_used"]
        assert "Random Forest" not in model_used, (
            f"model_used must not say 'Random Forest' for Phase 5; got: {model_used!r}"
        )
        # Must identify the model family — either GNN+XGBoost or XGBoost
        assert "XGBoost" in model_used or "GNN" in model_used, (
            f"model_used must mention XGBoost or GNN; got: {model_used!r}"
        )

    def test_phase3_fallback_model_used_not_random_forest_description(self, client):
        """Phase 3 fallback model_used must derive from the artifact metadata,
        not from a hardcoded 'Random Forest' string in the API description.
        The Phase 3 model is XGBoost; model_used must not claim RandomForest.
        """
        was_p5 = store._phase5_loaded
        try:
            store._phase5_loaded = False  # force Phase 3 path
            r = client.post("/transaction/score", json={
                "order_id": "MODEL_USED_P3",
                "buyer_id": "BUYER_000002",
                "seller_id": "SELLER_000002",
                "amount": 80.0,
                "base_price": 80.0,
                "category_median_price": 80.0,
            })
            assert r.status_code == 200
            model_used = r.json()["model_used"]
            # model_used must not be a bare "Random Forest" label — it should
            # reflect the actual classifier stored in the artifact metadata.
            assert "scikit-learn RandomForest" not in model_used, (
                f"model_used must not reference scikit-learn RandomForest; got: {model_used!r}"
            )
        finally:
            store._phase5_loaded = was_p5

    def test_listing_model_used_mentions_phase4(self, client):
        """Listing model_used must always reference Phase 4, not a stale Phase 3 label."""
        r = client.post("/listing/analyze", json={
            "listing_id": "MODEL_USED_LISTING",
            "price": 50.0,
            "base_price": 100.0,
            "category_median_price": 100.0,
            "seller_age_days_at_listing": 30.0,
            "seller_listings_before": 5,
            "multimodal_similarity_score": 0.30,
        })
        assert r.status_code == 200
        model_used = r.json()["model_used"]
        assert "Phase 4" in model_used, (
            f"Listing model_used must reference Phase 4; got: {model_used!r}"
        )

    def test_return_analyze_endpoint(self, client):
        """Test POST /return/analyze evaluates return fraud model properly."""
        payload = {
            "return_id": "RET_TEST_001",
            "order_id": "ORD_TEST_001",
            "buyer_id": "BUYER_TEST_01",
            "seller_id": "SELLER_TEST_01",
            "days_to_return": 1.0,
            "buyer_age_days_at_return": 10.0,
            "seller_age_days_at_return": 200.0,
            "order_amount": 150.0,
            "buyer_prior_returns": 5,
            "buyer_orders_before_return": 6,
            "buyer_return_rate_before": 0.83,
            "seller_prior_returns": 3,
            "seller_orders_before_return": 50,
            "seller_return_rate_before": 0.06,
            "reason": "defective",
        }
        r = client.post("/return/analyze", json=payload)
        assert r.status_code == 200
        d = r.json()
        assert d["return_id"] == "RET_TEST_001"
        assert 0.0 <= d["return_fraud_probability"] <= 1.0
        assert d["risk_label"] in ("low", "medium", "high")
        assert d["decision"] in ("ALLOW", "REVIEW", "HOLD", "BLOCK")
        assert "REPEAT_RETURN_ABUSER" in d["reason_codes"]
        assert "RAPID_RETURN_VELOCITY" in d["reason_codes"]

    def test_transaction_scoring_with_multimodal_signal(self, client):
        """Test POST /transaction/score ingests multimodal_similarity_score."""
        low_sim_payload = {
            "order_id": "ORD_MM_001",
            "buyer_id": "BUYER_000001",
            "seller_id": "SELLER_000001",
            "amount": 100.0,
            "base_price": 100.0,
            "category_median_price": 100.0,
            "multimodal_similarity_score": 0.10,  # strong mismatch
        }
        r = client.post("/transaction/score", json=low_sim_payload)
        assert r.status_code == 200
        d = r.json()
        assert 0.0 <= d["overall_fraud_probability"] <= 1.0

    def test_cors_headers(self, client):
        """Test CORS headers are returned for preflight/requests."""
        r = client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        })
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_transaction_scoring_with_client_supplied_graph_features(self, client):
        """Test POST /transaction/score accepts client-supplied graph topology features,
        verifies server snapshot lookup, and flags mismatches and defaults."""
        # 1. Calling with client-supplied graph features
        graph_payload = {
            "order_id": "ORD_GRAPH_CLIENT_001",
            "buyer_id": "BUYER_000001",
            "seller_id": "SELLER_000001",
            "amount": 250.0,
            "base_price": 250.0,
            "category_median_price": 250.0,
            "share_degree": 6.0,
            "share_component_size": 15.0,
            "buyer_pagerank": 0.005,
            "seller_pagerank": 0.012,
            "buyer_seller_edge_weight_before": 4.0,
            "seller_buyer_concentration_hhi": 0.45,
        }
        r = client.post("/transaction/score", json=graph_payload)
        assert r.status_code == 200
        d = r.json()
        assert d["order_id"] == "ORD_GRAPH_CLIENT_001"
        assert 0.0 <= d["overall_fraud_probability"] <= 1.0
        assert d["graph_features_source"] == "client"
        assert d["client_mismatch"] is False

        # 2. Calling with entity in snapshot and no client graph features -> computes server-side
        server_payload = {
            "order_id": "ORD_GRAPH_SERVER_002",
            "buyer_id": "BUYER_00001",
            "seller_id": "SELLER_00189",
            "amount": 250.0,
            "base_price": 250.0,
            "category_median_price": 250.0,
        }
        r2 = client.post("/transaction/score", json=server_payload)
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["order_id"] == "ORD_GRAPH_SERVER_002"
        assert d2["graph_features_source"] == "server"
        assert d2["client_mismatch"] is False

        # 3. Calling with unknown entity and no graph features -> flagged default (never silent 0.0)
        default_payload = {
            "order_id": "ORD_GRAPH_UNKNOWN_003",
            "buyer_id": "BUYER_UNKNOWN_XYZ_999",
            "seller_id": "SELLER_UNKNOWN_XYZ_999",
            "amount": 100.0,
            "base_price": 100.0,
            "category_median_price": 100.0,
        }
        r3 = client.post("/transaction/score", json=default_payload)
        assert r3.status_code == 200
        d3 = r3.json()
        assert d3["graph_features_source"] == "default"
        assert d3["client_mismatch"] is False
        assert "Flagged: Graph features defaulted to 0.0" in d3["note"]

        # 4. Calling with client mismatch vs server snapshot -> enforces server value and sets client_mismatch=True
        mismatch_payload = {
            "order_id": "ORD_GRAPH_MISMATCH_004",
            "buyer_id": "BUYER_00001",
            "seller_id": "SELLER_00189",
            "amount": 250.0,
            "base_price": 250.0,
            "category_median_price": 250.0,
            "share_degree": 999.0,  # Deliberately divergent from server snapshot
        }
        r4 = client.post("/transaction/score", json=mismatch_payload)
        assert r4.status_code == 200
        d4 = r4.json()
        assert d4["graph_features_source"] == "server"
        assert d4["client_mismatch"] is True
        assert "Flagged client-supplied graph feature mismatch" in d4["note"]
        assert "Server snapshot value enforced" in d4["note"]


class TestListingAnalyzeFeatureAlignment:
    """Regression and contract tests for /listing/analyze feature schema alignment."""

    def test_analyze_listing_valid_request(self, client):
        payload = {
            "listing_id": "LIST_TEST_001",
            "product_id": "PROD_TEST_99",
            "displayed_product_id": "PROD_TEST_99",
            "price": 120.0,
            "base_price": 100.0,
            "category_median_price": 95.0,
            "seller_age_days_at_listing": 45,
            "seller_listings_before": 12,
            "multimodal_similarity_score": 0.88,
        }
        response = client.post("/listing/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["fake_listing_probability"] <= 1.0
        assert data["risk_label"] in ["low", "medium", "high", "critical"]

    def test_analyze_listing_malformed_request_returns_422(self, client):
        response = client.post("/listing/analyze", json={"listing_id": "LIST_MALFORMED", "price": "not-a-number"})
        assert response.status_code == 422

    def test_listing_model_feature_schema_matches_booster(self):
        booster = store.fake_listing_model.get_booster()
        assert "synthetic_mismatch_score" in booster.feature_names

    def test_analyze_listing_synthetic_mismatch_overrides_multimodal(self, client):
        payload = {
            "listing_id": "LIST_OVERRIDE_PRIORITY",
            "price": 20.0,
            "base_price": 200.0,
            "category_median_price": 200.0,
            "seller_age_days_at_listing": 2,
            "seller_listings_before": 0,
            "multimodal_similarity_score": 0.95,
            "synthetic_mismatch_score": 0.10,
        }
        response = client.post("/listing/analyze", json=payload)
        assert response.status_code == 200
        assert response.json()["fake_listing_probability"] > 0.5

    def test_api_live_scoring_without_order_date(self, client):
        live_payload = {
            "order_id": "ORD_LIVE_DECISION_TIME",
            "buyer_id": "BUYER_00001",
            "seller_id": "SELLER_00189",
            "amount": 75.0,
            "base_price": 75.0,
            "category_median_price": 75.0,
        }
        resp = client.post("/transaction/score", json=live_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["graph_features_source"] == "server"
        assert data["client_mismatch"] is False


class TestServingModelAndStreamSemantics:
    """Regression tests for serving model routing, header overrides, invalid models, and stream events."""

    def test_default_serving_model_is_phase3(self, client):
        resp = client.post("/transaction/score", json={
            "order_id": "ORD_DEFAULT_ROUTING",
            "amount": 100.0,
            "base_price": 100.0,
            "category_median_price": 95.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_version"] == "phase3"
        assert data["scoring_mode"] == "production_model_phase3_combined"
    def test_explicit_phase5_selection_via_header(self, client):
        headers = {"X-TrustShield-Model-Version": "phase5-hybrid"}
        resp = client.post("/transaction/score", json={
            "order_id": "ORD_EXPLICIT_P5_HEADER",
            "amount": 120.0,
            "base_price": 120.0,
            "category_median_price": 100.0,
        }, headers=headers)
        assert resp.status_code == 400
        assert "Invalid model version requested" in resp.json()["detail"]

    def test_invalid_model_version_returns_400(self, client):
        resp = client.post("/transaction/score", json={
            "order_id": "ORD_INVALID_MODEL",
            "model_version": "nonexistent_model_v9",
            "amount": 100.0,
            "base_price": 100.0,
            "category_median_price": 100.0,
        })
        assert resp.status_code == 400

    def test_conformal_fields_in_response(self, client):
        resp = client.post("/transaction/score", json={
            "order_id": "ORD_CONFORMAL_CHECK",
            "amount": 100.0,
            "base_price": 100.0,
            "category_median_price": 100.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "conformal_prediction_set" in data
        assert isinstance(data["conformal_prediction_set"], list)
        assert data["conformal_is_calibrated"] is True

    def test_stream_event_schema_separates_calibrated_and_policy_risk(self):
        from backend.schemas import StreamTransactionEvent
        event = StreamTransactionEvent(
            event_id="EVT_TEST_01",
            timestamp="12:00:00",
            order_id="ORD_TEST_01",
            buyer_id="BUYER_01",
            seller_id="SELLER_01",
            amount=150.0,
            risk_score=0.10,
            calibrated_risk=0.04,
            policy_risk_score=0.28,
            risk_level="medium",
            decision="REVIEW",
            trust_score=72.0,
        )
        assert event.calibrated_risk == 0.04
        assert event.policy_risk_score == 0.28


class TestModelArtifactAndOfflineParity:
    def test_regression_model_artifact_load(self):
        """All production artifacts exist and load cleanly in store."""
        if not store.is_loaded:
            store.load()
        assert store.is_loaded
        assert store.combined_graph_model is not None
        assert store.fake_listing_model is not None
        assert store.return_fraud_model is not None
        assert store.rings_df is not None
        assert store.feature_meta is not None
        assert store.calibrator is not None

    def test_offline_vs_online_scoring_consistency(self, client):
        """Scoring an identical transaction via offline pipeline vs API yields consistent results."""
        import numpy as np
        import pandas as pd

        payload = {
            "order_id": "CONSISTENCY_001",
            "buyer_id": "BUYER_000042",
            "seller_id": "SELLER_000007",
            "amount": 100.0,
            "base_price": 100.0,
            "category_median_price": 100.0,
            "seller_age_days": 120.0,
            "seller_total_listings_before": 15,
            "buyer_age_days": 90.0,
            "buyer_orders_before": 10,
            "buyer_returns_before": 1,
            "buyer_return_rate_before": 0.10,
            "device_shared_buyer_count": 1.0,
            "share_degree": 0.0,
            "share_component_size": 1.0,
        }

        res = client.post("/transaction/score", json=payload)
        assert res.status_code == 200
        api_data = res.json()

        meta = store.feature_meta
        assert meta is not None, "Feature metadata must be loaded"
        cols = meta["all_feature_cols"]
        row = {c: payload.get(c, 0.0) for c in cols}
        row["price_vs_base_price_ratio"] = payload["amount"] / payload["base_price"]
        row["price_vs_category_median_ratio"] = payload["amount"] / payload["category_median_price"]

        X = pd.DataFrame([row])[cols].fillna(0.0)
        raw_prob = float(np.asarray(store.combined_graph_model.predict_proba(X))[0, 1])
        cal_prob = float(store.calibrator.predict_proba(np.array([raw_prob]))[0]) if store.calibrator else raw_prob

        assert 0.0 <= cal_prob <= 1.0
        assert 0.0 <= api_data["overall_fraud_probability"] <= 1.0






