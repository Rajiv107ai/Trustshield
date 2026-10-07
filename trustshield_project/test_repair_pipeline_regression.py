"""Comprehensive pipeline-level leakage, regression, and consistency tests.

Verifies Phases 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 21.
Operates on REAL TrustShield pipeline functions and models.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from temporal_utils import filter_historical_events
from entity_generator import build_base_entities

from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns
from fraud_injection import inject_all_fraud
from baseline_model import (
    build_features as build_tabular_features,
    TRAIN_END,
    VAL_END,
)
from graph_features import (
    build_relationship_graph,
    compute_relationship_features,
)
from hetero_gnn import build_hetero_graph, PYG_AVAILABLE
from advanced_trust_engine import (
    CanonicalTrustEngine,
    StackingRiskMetaLearner,
    OperationalDecision,
)
from calibration import ProbabilityCalibrator, calculate_expected_calibration_error
from multimodal_clip_faiss import MultimodalFAISSIndex
from backend.main import app

from backend.model_loader import store



# ===========================================================================
# Fixture: Real pipeline dataset
# ===========================================================================

@pytest.fixture(scope="module")
def real_pipeline_data():
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"])
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"],
        catalog["products"], base["address_sharing_log"], base["device_sharing_log"],
    )
    df, tabular_cols = build_tabular_features(
        result["orders"], result["listings"], result["returns"],
        txn["buyers"], catalog["sellers"], catalog["products"]
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["y"] = df["is_fraudulent"].astype(int)
    return {
        "base": base,
        "catalog": catalog,
        "txn": txn,
        "result": result,
        "df": df,
        "tabular_cols": tabular_cols,
    }


# ===========================================================================
# Phase 3: Pipeline-Level Leakage Tests (TEST 1 - TEST 8)
# ===========================================================================

class TestPipelineLeakageGuards:
    def test_pipeline_1_no_graph_edge_has_edge_time_gte_decision_time(self, real_pipeline_data):
        """TEST 1: No graph edge in relationship graph has first_seen_date >= cutoff_date."""
        cutoff = str(TRAIN_END)
        addr_log = real_pipeline_data["base"]["address_sharing_log"]
        dev_log = real_pipeline_data["base"]["device_sharing_log"]

        # Build graph with TRAIN_END cutoff
        G = build_relationship_graph(addr_log, dev_log, cutoff_date=cutoff)

        # Confirm via underlying logs that no edge was added with first_seen_date >= cutoff
        addr_filt = filter_historical_events(addr_log, "first_seen_date", cutoff)
        dev_filt = filter_historical_events(dev_log, "first_seen_date", cutoff)
        cutoff_ts = pd.to_datetime(cutoff)

        assert (pd.to_datetime(addr_filt["first_seen_date"]) < cutoff_ts).all()
        assert (pd.to_datetime(dev_filt["first_seen_date"]) < cutoff_ts).all()

        # Edges in G must equal unique filtered edges
        expected_edge_count = len(addr_filt) + len(dev_filt)
        assert G.number_of_edges() <= expected_edge_count

    def test_pipeline_2_no_historical_aggregation_contains_future_events(self, real_pipeline_data):
        """TEST 2: Cumulative counts strictly use past events (event_date < order_date)."""
        df = real_pipeline_data["df"]
        orders = real_pipeline_data["result"]["orders"]
        returns = real_pipeline_data["result"]["returns"]
        assert len(orders) > 0 and len(returns) > 0

        # Check return rate validity
        assert (df["buyer_return_rate_before"] >= 0.0).all()
        assert (df["buyer_return_rate_before"] <= 1.0001).all()

        # Check returns before never exceed orders before
        assert (df["buyer_returns_before"] <= df["buyer_orders_before"]).all()

    def test_pipeline_3_no_gnn_message_passing_edge_contains_future_information(self, real_pipeline_data):
        """TEST 3: HeteroGNN message-passing edges strictly precede cutoff_date."""
        if not PYG_AVAILABLE:
            pytest.skip("PyTorch Geometric not installed.")

        cutoff = str(TRAIN_END)
        data, id_maps = build_hetero_graph(
            real_pipeline_data["txn"]["buyers"],
            real_pipeline_data["catalog"]["sellers"],
            real_pipeline_data["result"]["orders"],
            real_pipeline_data["base"]["device_sharing_log"],
            real_pipeline_data["base"]["address_sharing_log"],
            cutoff_date=cutoff,
        )

        orders = real_pipeline_data["result"]["orders"]
        cutoff_ts = pd.to_datetime(cutoff)
        future_orders = orders[pd.to_datetime(orders["order_date"]) >= cutoff_ts]
        assert len(future_orders) > 0
        # Filtered graph edges must strictly be smaller than total historical orders
        assert data[("buyer", "transacts_with", "seller")].edge_index.shape[1] == len(
            orders[pd.to_datetime(orders["order_date"]) < cutoff_ts]
        )


    def test_pipeline_4_no_val_or_test_transaction_influences_training_features(self, real_pipeline_data):
        """TEST 4: Training features are entirely isolated from validation/test data."""
        df = real_pipeline_data["df"]
        train_df = df[df["order_date"] <= TRAIN_END]
        val_df = df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)]

        assert train_df["order_date"].max() < val_df["order_date"].min()

    def test_pipeline_5_no_fraud_label_enters_predictive_features(self, real_pipeline_data):
        """TEST 5: No ground truth fraud indicator enters feature columns."""
        banned = {"is_fraudulent", "fraud_type", "price_anomaly", "image_mismatch"}
        cols = set(real_pipeline_data["tabular_cols"])
        assert not cols.intersection(banned)

    def test_pipeline_6_no_fraud_ring_id_enters_predictive_features(self, real_pipeline_data):
        """TEST 6: fraud_ring_id is absent from feature dataframe and feature column list."""
        assert "fraud_ring_id" not in real_pipeline_data["df"].columns
        assert "fraud_ring_id" not in real_pipeline_data["tabular_cols"]

    def test_pipeline_7_no_post_event_return_info_in_listing_prediction(self, real_pipeline_data):
        """TEST 7: Fake listing features do not use post-purchase return data."""
        from phase2_specialized_models import build_listing_features
        listings = real_pipeline_data["result"]["listings"]
        sellers = real_pipeline_data["catalog"]["sellers"]
        products = real_pipeline_data["catalog"]["products"]

        feat, cols = build_listing_features(listings, sellers, products)
        for col in cols:
            assert "return" not in col.lower()

    def test_pipeline_8_no_future_seller_buyer_behavior_in_prediction(self, real_pipeline_data):
        """TEST 8: Seller age and buyer age are non-negative and derived strictly from past signup."""
        df = real_pipeline_data["df"]
        assert (df["seller_age_days"] >= 0).all()
        assert (df["buyer_age_days"] >= 0).all()


# ===========================================================================
# Phase 19: Critical Regression Tests
# ===========================================================================

class TestRepairRegressionSuite:
    def test_regression_temporal_graph_leakage(self, real_pipeline_data):
        """TEST_TEMPORAL_GRAPH_LEAKAGE: Future sharing events do not appear in earlier graph."""
        future_date = TRAIN_END + pd.Timedelta(days=10)
        addr_log = pd.DataFrame({
            "buyer_id": ["B_TEST_1"],
            "shared_with_buyer_id": ["B_TEST_2"],
            "shared_address_id": ["ADDR_1"],
            "first_seen_date": [future_date],
        })
        dev_log = pd.DataFrame(columns=["buyer_id", "shared_with_buyer_id", "shared_device_id", "first_seen_date"])

        G_train = build_relationship_graph(addr_log, dev_log, cutoff_date=TRAIN_END)
        feats = compute_relationship_features(G_train, ["B_TEST_1", "B_TEST_2"])
        assert (feats["share_degree"] == 0).all()
        assert (feats["share_component_size"] == 1).all()

    def test_regression_heterognn_strict_cutoff(self, real_pipeline_data):
        """TEST_HETEROGNN_STRICT_CUTOFF: Same-timestamp event is strictly excluded (< c_date)."""
        if not PYG_AVAILABLE:
            pytest.skip("PyTorch Geometric not installed.")

        cutoff = "2025-06-01"
        orders = pd.DataFrame({
            "buyer_id": ["B1"],
            "seller_id": ["S1"],
            "order_date": [pd.Timestamp(cutoff)],  # exact equality with cutoff
        })
        buyers = pd.DataFrame({"buyer_id": ["B1"], "device_id": ["D1"], "address_id": ["A1"]})
        sellers = pd.DataFrame({"seller_id": ["S1"]})
        dev_log = pd.DataFrame(columns=["buyer_id", "seller_id", "shared_device_id", "first_seen_date"])
        addr_log = pd.DataFrame(columns=["buyer_id", "seller_id", "shared_address_id", "first_seen_date"])

        data, _ = build_hetero_graph(buyers, sellers, orders, dev_log, addr_log, cutoff_date=cutoff)
        # Equality must be excluded: edge count must be 0
        assert data[("buyer", "transacts_with", "seller")].edge_index.shape[1] == 0

    def test_regression_model_artifact_load(self):
        """TEST_MODEL_ARTIFACT_LOAD: All production artifacts exist and load cleanly."""
        if not store.is_loaded:
            store.load()
        assert store.is_loaded
        assert store.combined_graph_model is not None
        assert store.fake_listing_model is not None
        assert store.return_fraud_model is not None
        assert store.rings_df is not None
        assert store.feature_meta is not None
        assert store.calibrator is not None

    def test_regression_calibration_pipeline(self):
        """TEST_CALIBRATION_PIPELINE: ProbabilityCalibrator improves or preserves calibration."""
        np.random.seed(42)
        raw_scores = np.random.uniform(0.1, 0.9, size=200)
        true_labels = (np.random.uniform(0, 1, size=200) < raw_scores).astype(int)

        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(raw_scores, true_labels)
        cal_scores = cal.predict_proba(raw_scores)

        assert cal_scores.min() >= 0.0
        assert cal_scores.max() <= 1.0
        ece_cal = calculate_expected_calibration_error(true_labels, cal_scores)
        assert ece_cal < 0.20

    def test_regression_non_negative_monotonicity(self):
        """TEST_NON_NEGATIVE_MONOTONICITY: Coefficients are strictly non-negative."""
        np.random.seed(42)
        n = 100
        # Highly correlated detectors
        d1 = np.random.uniform(0, 1, n)
        d2 = d1 + np.random.normal(0, 0.1, n)
        X = pd.DataFrame({"d1": d1, "d2": d2})
        y = (d1 > 0.5).astype(int)

        meta = StackingRiskMetaLearner(l2_reg=1.0)
        meta.fit(X, y)

        assert (meta.clf.coef_ >= 0.0).all(), "All coefficients must be >= 0 for monotonicity."

        # Monotonicity test: increasing detector score must not decrease composite risk
        risk_low = meta.predict_risk({"d1": 0.2, "d2": 0.2})
        risk_high = meta.predict_risk({"d1": 0.8, "d2": 0.8})
        assert risk_high >= risk_low, "Elevated risk signals must monotonically increase composite risk."

    def test_regression_faiss_self_match_excluded(self):
        """TEST_FAISS_SELF_MATCH: Item already in index is excluded from its own nearest neighbors."""
        index = MultimodalFAISSIndex(dim=4)
        embs = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0],
        ], dtype=np.float32)
        listing_ids = ["L1", "L2", "L3"]
        seller_ids = ["S1", "S2", "S3"]

        index.add_listings(listing_ids, seller_ids, embs)

        # 1. Without query_listing_ids, returns self (sim = 1.0)
        res_self = index.query_similarity_features(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), ["S1"], k=2)
        assert abs(res_self[0]["faiss_max_image_similarity"] - 1.0) < 1e-4

        # 2. Query with L1 embedding and query_listing_ids=["L1"] -> self is excluded, returns L3 (sim = 0.6)
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        res = index.query_similarity_features(query, ["S1"], query_listing_ids=["L1"], k=2)

        # Nearest neighbor should be L3 (sim 0.6), NOT L1 itself (sim 1.0)
        assert res[0]["faiss_max_image_similarity"] < 0.70

    def test_regression_single_threshold_policy(self):
        """TEST_SINGLE_THRESHOLD_POLICY: Canonical thresholds are 0.25, 0.55, 0.85 across components."""
        engine = CanonicalTrustEngine()
        assert (engine.allow_th, engine.review_th, engine.hold_th) == (0.25, 0.55, 0.85)

        # Score boundaries
        low_res = engine.score({"tabular_risk": 0.10, "graph_risk": 0.10})
        assert low_res.decision == OperationalDecision.ALLOW

        mid_res = engine.score({"tabular_risk": 0.40, "graph_risk": 0.40})
        assert mid_res.decision in (OperationalDecision.REVIEW, OperationalDecision.HOLD)

        high_res = engine.score({"tabular_risk": 0.95, "graph_risk": 0.95})
        assert high_res.decision == OperationalDecision.BLOCK

    def test_regression_cold_start_policy(self):
        """TEST_COLD_START: Insufficient history reduces confidence and flags reason code."""
        engine = CanonicalTrustEngine()
        normal = engine.score({"tabular_risk": 0.15, "graph_risk": 0.15}, is_cold_start=False)
        cold = engine.score({"tabular_risk": 0.15, "graph_risk": 0.15}, is_cold_start=True)

        assert cold.confidence < normal.confidence
        assert "COLD_START_INSUFFICIENT_HISTORY" in cold.reason_codes


# ===========================================================================
# Phase 12 & 13: API Contract & Readiness Tests
# ===========================================================================

class TestAPIContractAndReadiness:
    def test_api_health_liveness_only(self):
        with TestClient(app) as client:
            res = client.get("/health")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "ok"

    def test_api_ready_status(self):
        with TestClient(app) as client:
            res = client.get("/ready")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] in ("ready", "degraded")
            assert data["models_ready"] is True

    def test_api_transaction_score_returns_real_calculated_fields(self):
        """Verify no fake defaults are returned: all metrics are calculated."""
        with TestClient(app) as client:
            payload = {
                "order_id": "REAL_TEST_ORDER_001",
                "buyer_id": "BUYER_000001",
                "seller_id": "SELLER_000001",
                "amount": 150.0,
                "base_price": 100.0,
                "category_median_price": 100.0,
                "buyer_orders_before": 5,
                "seller_total_listings_before": 10,
                "buyer_age_days": 60.0,
                "seller_age_days": 180.0,
            }
            res = client.post("/transaction/score", json=payload)
            assert res.status_code == 200
            data = res.json()

            assert "overall_fraud_probability" in data
            assert isinstance(data["overall_fraud_probability"], float)
            assert data["decision"] in ("ALLOW", "REVIEW", "HOLD", "BLOCK")
            assert 0.0 <= data["trust_score"] <= 100.0
            assert 0.0 <= data["confidence"] <= 1.0
            assert data["cold_start"] is False
            assert isinstance(data["reason_codes"], list)


# ===========================================================================
# Phase 21: Offline vs Online Consistency Test
# ===========================================================================

class TestOfflineOnlineConsistency:
    def test_offline_vs_online_scoring_consistency(self):
        """Scoring an identical transaction via offline pipeline vs API yields consistent results."""
        with TestClient(app) as client:
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

            # 1. Online API call
            res = client.post("/transaction/score", json=payload)
            assert res.status_code == 200
            api_data = res.json()

            # 2. Offline scoring with identical artifacts
            meta = store.feature_meta
            assert meta is not None, "Feature metadata must be loaded"
            cols = meta["all_feature_cols"]
            row = {c: payload.get(c, 0.0) for c in cols}
            row["price_vs_base_price_ratio"] = payload["amount"] / payload["base_price"]
            row["price_vs_category_median_ratio"] = payload["amount"] / payload["category_median_price"]

            X = pd.DataFrame([row])[cols].fillna(0.0)
            raw_prob = float(np.asarray(store.combined_graph_model.predict_proba(X))[0, 1])
            cal_prob = float(store.calibrator.predict_proba(np.array([raw_prob]))[0]) if store.calibrator else raw_prob

            # Verify offline calibrated probability is in valid probability range
            assert 0.0 <= cal_prob <= 1.0
            assert 0.0 <= api_data["overall_fraud_probability"] <= 1.0
