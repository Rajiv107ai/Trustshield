import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fastapi.testclient import TestClient

from trustshield_project.shap_explainer import TrustShieldSHAPExplainer, FEATURE_FRIENDLY_NAMES
from backend.main import app

VALID_API_KEY = "dev-operator-api-key-32chars-long"


@pytest.fixture(scope="module")
def client():
    with TestClient(app, headers={"X-API-Key": VALID_API_KEY}) as c:
        yield c


@pytest.fixture
def synthetic_tree_model():
    X = pd.DataFrame({
        "amount": [10.0, 500.0, 20.0, 800.0, 15.0, 950.0],
        "share_degree": [0.0, 5.0, 0.0, 8.0, 1.0, 12.0],
        "seller_age_days": [300.0, 2.0, 500.0, 1.0, 180.0, 3.0],
        "gnn_buyer_emb_0": [0.1, -0.5, 0.2, -0.8, 0.05, -0.9],
    })
    clf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
    clf.fit(X, np.array([0, 1, 0, 1, 0, 1]))
    return clf, list(X.columns)


def test_shap_explainer_initialization(synthetic_tree_model):
    clf, fn = synthetic_tree_model
    exp = TrustShieldSHAPExplainer(model=clf, feature_names=fn, model_name="Synthetic Test Detector")
    assert exp.is_available and exp.model_name == "Synthetic Test Detector" and exp.feature_names == fn


def test_shap_explain_instance_dict(synthetic_tree_model):
    clf, fn = synthetic_tree_model
    res = TrustShieldSHAPExplainer(model=clf, feature_names=fn).explain_instance(
        {"amount": 900.0, "share_degree": 10.0, "seller_age_days": 1.0, "gnn_buyer_emb_0": -0.85}, top_k=2
    )
    assert res["available"] and isinstance(res["base_value"], float) and len(res["attributions"]) == len(fn)
    assert "amount" in res["attributions"] and "share_degree" in res["attributions"]
    assert isinstance(res["top_positive_drivers"], list) and isinstance(res["top_negative_dampeners"], list) and len(res["narrative"]) > 0


def test_shap_friendly_names_and_gnn_formatting(synthetic_tree_model):
    clf, fn = synthetic_tree_model
    res = TrustShieldSHAPExplainer(model=clf, feature_names=fn).explain_instance(
        {"amount": 750.0, "share_degree": 4.0, "seller_age_days": 2.0, "gnn_buyer_emb_0": -0.5}
    )
    for d in res["top_positive_drivers"] + res["top_negative_dampeners"]:
        if d["feature_name"] == "share_degree":
            assert d["friendly_name"] == FEATURE_FRIENDLY_NAMES["share_degree"]
        elif d["feature_name"] == "gnn_buyer_emb_0":
            assert "Buyer GNN Latent Topology" in d["friendly_name"]


def test_shap_explain_instance_pandas_formats(synthetic_tree_model):
    clf, fn = synthetic_tree_model
    exp = TrustShieldSHAPExplainer(model=clf, feature_names=fn)
    d = {"amount": 100.0, "share_degree": 2.0, "seller_age_days": 50.0, "gnn_buyer_emb_0": 0.0}
    assert exp.explain_instance(pd.Series(d))["available"]
    assert len(exp.explain_instance(pd.DataFrame([d]))["attributions"]) == 4


def test_backend_health_and_ready_shap_status(client):
    assert client.get("/health").json()["shap_loaded"] is True
    r = client.get("/ready").json()
    assert r["shap_ready"] is True and r["components"]["explainability"] == "available"


def test_transaction_scoring_includes_shap_attributions(client):
    res = client.post("/transaction/score", json={
        "order_id": "O1", "buyer_id": "B1", "seller_id": "S1", "amount": 420.0,
        "base_price": 100.0, "category_median_price": 110.0, "share_degree": 3.0, "share_component_size": 4.0
    }).json()
    assert isinstance(res["shap_attributions"], dict) and len(res["shap_attributions"]) > 0
    assert isinstance(res["top_risk_drivers"], list)
    if res["top_risk_drivers"]:
        assert {"feature_name", "friendly_name", "shap_value"}.issubset(res["top_risk_drivers"][0].keys())


def test_transaction_explain_endpoint(client):
    res = client.post("/transaction/explain", json={
        "order_id": "O_EXPLAIN", "amount": 890.0, "base_price": 120.0, "category_median_price": 130.0,
        "share_degree": 6.0, "share_component_size": 7.0, "seller_age_days": 4.0, "buyer_return_rate_before": 0.75
    }).json()
    assert res["order_id"] == "O_EXPLAIN" and "base_value" in res and "overall_fraud_probability" in res
    assert len(res["investigator_narrative"]) > 0 and len(res["all_attributions"]) > 0


def test_dossier_incorporates_shap_evidence(client):
    payload = {
        "entity_type": "transaction", "entity_id": "O_DOSSIER",
        "transaction_data": {
            "order_id": "O_DOSSIER", "buyer_id": "B1", "seller_id": "S1", "amount": 650.0,
            "base_price": 150.0, "category_median_price": 160.0, "share_degree": 5.0, "share_component_size": 6.0,
            "buyer_return_rate_before": 0.8, "device_shared_buyer_count": 4.0
        }
    }
    res = client.post("/investigation/generate-dossier", json=payload).json()
    inf = res["model_inference"]
    assert inf["shap_attributions"] is not None and inf["top_risk_drivers"] is not None and inf["shap_narrative"] is not None
    assert "TreeSHAP" in res["executive_summary"] or "TreeSHAP" in inf["shap_narrative"]
