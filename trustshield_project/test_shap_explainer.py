"""
TrustShield AI — Unit & Integration Tests for TreeSHAP Explainability.

Tests:
1. TrustShieldSHAPExplainer initialization and attribution computations.
2. Positive risk drivers and negative risk dampeners separation.
3. Feature friendly name formatting and GNN latent dimension handling.
4. Plain-English investigator narrative synthesis.
5. FastAPI /transaction/explain endpoint contract and responses.
6. Integration of SHAP attributions into /transaction/score and /investigation/generate-dossier.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add project root and trustshield_project to path
_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from trustshield_project.shap_explainer import (
    TrustShieldSHAPExplainer,
    FEATURE_FRIENDLY_NAMES,
)
from backend.main import app, store
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def synthetic_tree_model():
    """Build a fast 2-tree RandomForest classifier for deterministic testing."""
    X = pd.DataFrame({
        "amount": [10.0, 500.0, 20.0, 800.0, 15.0, 950.0],
        "share_degree": [0.0, 5.0, 0.0, 8.0, 1.0, 12.0],
        "seller_age_days": [300.0, 2.0, 500.0, 1.0, 180.0, 3.0],
        "gnn_buyer_emb_0": [0.1, -0.5, 0.2, -0.8, 0.05, -0.9],
    })
    y = np.array([0, 1, 0, 1, 0, 1])
    clf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
    clf.fit(X, y)
    return clf, list(X.columns)


def test_shap_explainer_initialization(synthetic_tree_model):
    clf, feature_names = synthetic_tree_model
    explainer = TrustShieldSHAPExplainer(
        model=clf,
        feature_names=feature_names,
        model_name="Synthetic Test Detector",
    )
    assert explainer.is_available is True
    assert explainer.model_name == "Synthetic Test Detector"
    assert explainer.feature_names == feature_names


def test_shap_explain_instance_dict(synthetic_tree_model):
    clf, feature_names = synthetic_tree_model
    explainer = TrustShieldSHAPExplainer(model=clf, feature_names=feature_names)
    
    # High risk instance
    high_risk_tx = {
        "amount": 900.0,
        "share_degree": 10.0,
        "seller_age_days": 1.0,
        "gnn_buyer_emb_0": -0.85,
    }
    result = explainer.explain_instance(high_risk_tx, top_k=2)
    
    assert result["available"] is True
    assert isinstance(result["base_value"], float)
    assert len(result["attributions"]) == len(feature_names)
    assert "amount" in result["attributions"]
    assert "share_degree" in result["attributions"]
    
    # Check drivers structure
    assert isinstance(result["top_positive_drivers"], list)
    assert isinstance(result["top_negative_dampeners"], list)
    assert isinstance(result["narrative"], str)
    assert len(result["narrative"]) > 0


def test_shap_friendly_names_and_gnn_formatting(synthetic_tree_model):
    clf, feature_names = synthetic_tree_model
    explainer = TrustShieldSHAPExplainer(model=clf, feature_names=feature_names)
    
    tx = {"amount": 750.0, "share_degree": 4.0, "seller_age_days": 2.0, "gnn_buyer_emb_0": -0.5}
    result = explainer.explain_instance(tx)
    
    # Check that friendly names are populated
    for driver in result["top_positive_drivers"] + result["top_negative_dampeners"]:
        if driver["feature_name"] == "share_degree":
            assert driver["friendly_name"] == FEATURE_FRIENDLY_NAMES["share_degree"]
        elif driver["feature_name"] == "gnn_buyer_emb_0":
            assert "Buyer GNN Latent Topology" in driver["friendly_name"]


def test_shap_explain_instance_pandas_formats(synthetic_tree_model):
    clf, feature_names = synthetic_tree_model
    explainer = TrustShieldSHAPExplainer(model=clf, feature_names=feature_names)
    
    # Test pd.Series input
    s = pd.Series({"amount": 100.0, "share_degree": 2.0, "seller_age_days": 50.0, "gnn_buyer_emb_0": 0.0})
    res_s = explainer.explain_instance(s)
    assert res_s["available"] is True
    assert len(res_s["attributions"]) == 4

    # Test pd.DataFrame input
    df = pd.DataFrame([{"amount": 100.0, "share_degree": 2.0, "seller_age_days": 50.0, "gnn_buyer_emb_0": 0.0}])
    res_df = explainer.explain_instance(df)
    assert res_df["available"] is True
    assert len(res_df["attributions"]) == 4


def test_backend_health_and_ready_shap_status(client):
    # Verify /health probe reports shap_loaded
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    hdata = health_resp.json()
    assert "shap_loaded" in hdata
    assert hdata["shap_loaded"] is True

    # Verify /ready probe reports shap_ready
    ready_resp = client.get("/ready")
    assert ready_resp.status_code == 200
    rdata = ready_resp.json()
    assert "shap_ready" in rdata
    assert rdata["shap_ready"] is True
    assert rdata["components"]["explainability"] == "available"


def test_transaction_scoring_includes_shap_attributions(client):
    payload = {
        "order_id": "ORDER_SHAP_TEST_01",
        "buyer_id": "BUYER_000001",
        "seller_id": "SELLER_000001",
        "amount": 420.0,
        "base_price": 100.0,
        "category_median_price": 110.0,
        "share_degree": 3.0,
        "share_component_size": 4.0,
    }
    response = client.post("/transaction/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "shap_attributions" in data
    assert data["shap_attributions"] is not None
    assert isinstance(data["shap_attributions"], dict)
    assert len(data["shap_attributions"]) > 0

    assert "top_risk_drivers" in data
    assert data["top_risk_drivers"] is not None
    assert isinstance(data["top_risk_drivers"], list)
    if data["top_risk_drivers"]:
        top = data["top_risk_drivers"][0]
        assert "feature_name" in top
        assert "friendly_name" in top
        assert "shap_value" in top


def test_transaction_explain_endpoint(client):
    payload = {
        "order_id": "ORDER_EXPLAIN_TEST_99",
        "amount": 890.0,
        "base_price": 120.0,
        "category_median_price": 130.0,
        "share_degree": 6.0,
        "share_component_size": 7.0,
        "seller_age_days": 4.0,
        "buyer_return_rate_before": 0.75,
    }
    response = client.post("/transaction/explain", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["order_id"] == "ORDER_EXPLAIN_TEST_99"
    assert "base_value" in data
    assert "overall_fraud_probability" in data
    assert "decision" in data
    assert "top_positive_drivers" in data
    assert "top_negative_dampeners" in data
    assert "all_attributions" in data
    assert "investigator_narrative" in data
    assert len(data["investigator_narrative"]) > 0
    assert len(data["all_attributions"]) > 0


def test_dossier_incorporates_shap_evidence(client):
    payload = {
        "entity_type": "transaction",
        "entity_id": "ORD_SHAP_DOSSIER_42",
        "transaction_data": {
            "order_id": "ORD_SHAP_DOSSIER_42",
            "buyer_id": "BUYER_SUSPECT_01",
            "seller_id": "SELLER_SUSPECT_02",
            "amount": 650.0,
            "base_price": 150.0,
            "category_median_price": 160.0,
            "share_degree": 5.0,
            "share_component_size": 6.0,
            "buyer_return_rate_before": 0.8,
            "device_shared_buyer_count": 4.0,
        },
    }
    response = client.post("/investigation/generate-dossier", json=payload)
    assert response.status_code == 200
    data = response.json()

    model_inf = data["model_inference"]
    assert "shap_attributions" in model_inf
    assert model_inf["shap_attributions"] is not None
    assert "top_risk_drivers" in model_inf
    assert model_inf["top_risk_drivers"] is not None
    assert "shap_narrative" in model_inf
    assert model_inf["shap_narrative"] is not None

    # Check verified facts or executive summary
    exec_summary = data["executive_summary"]
    assert "TreeSHAP risk drivers" in exec_summary or "TreeSHAP" in model_inf["shap_narrative"]
