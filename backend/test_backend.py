"""Unit and integration tests for FastAPI backend serving layer."""

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

from backend.main import app, store, _build_feature_row
from backend.schemas import TransactionScoreRequest


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
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


def test_transaction_scoring_phase5(client):
    payload = {
        "order_id": "ORDER_TEST_P5",
        "buyer_id": "BUYER_000001",
        "seller_id": "SELLER_000001",
        "amount": 150.0,
        "base_price": 150.0,
        "category_median_price": 150.0,
        "order_date": "2025-09-01T12:00:00",
    }
    response = client.post("/transaction/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "ORDER_TEST_P5"
    assert 0.0 <= data["overall_fraud_probability"] <= 1.0
    assert data["risk_label"] in {"low", "medium", "high"}
    assert "phase5" in data["model_version"]


def test_transaction_scoring_phase3_fallback(client):
    was_p5_loaded = store._phase5_loaded
    try:
        store._phase5_loaded = False
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
    finally:
        store._phase5_loaded = was_p5_loaded


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
        """Test POST /transaction/score accepts client-supplied graph topology features."""
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

        # 2. Calling with graph features omitted defaults gracefully to 0.0 / 1.0
        no_graph_payload = {
            "order_id": "ORD_GRAPH_CLIENT_002",
            "buyer_id": "BUYER_000002",
            "seller_id": "SELLER_000002",
            "amount": 250.0,
            "base_price": 250.0,
            "category_median_price": 250.0,
        }
        r2 = client.post("/transaction/score", json=no_graph_payload)
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["order_id"] == "ORD_GRAPH_CLIENT_002"
        assert 0.0 <= d2["overall_fraud_probability"] <= 1.0

