"""
TrustShield Phase 4.3B — Golden Baseline Fixture Generator.

Generates the authoritative post-Phase-4.3B frozen baseline fixture
at backend/fixtures/phase3_golden_baseline.json covering:
1. Low-risk standard transaction (warm accounts, normal behavior)
2. High-risk anomalous transaction (extreme price ratio, return rate, ring/device sharing)
3. Historical point-in-time transaction (timestamped order_date replay)
4. Cold-start transaction (new buyer/seller with 0 history)

Records exact environment, artifact hashes, 18 model features in exact order,
and complete 29 response fields per scenario without secrets or credentials.

Execution:
    python scripts/generate_phase3_golden_baseline.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import fastapi
import joblib
import numpy as np
import sklearn
import xgboost

# ---------------------------------------------------------------------------
# Path & Import Setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from backend.main import app, store
from backend.schemas import TransactionScoreResponse


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Synthetic Deterministic Scenarios
# ---------------------------------------------------------------------------
GOLDEN_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "low_risk_standard": {
        "description": "Established warm buyer and seller, low order amount, normal price ratios, 0 returns, legitimate trust score.",
        "payload": {
            "order_id": "golden_p3_001_low",
            "buyer_id": "buyer_golden_low",
            "seller_id": "seller_golden_low",
            "amount": 45.0,
            "order_amount": 45.0,
            "price_vs_base_price_ratio": 1.0,
            "price_vs_category_median_ratio": 1.0,
            "seller_age_days": 365.0,
            "seller_total_listings_before": 50.0,
            "buyer_age_days": 180.0,
            "buyer_orders_before": 12,
            "buyer_returns_before": 0,
            "buyer_return_rate_before": 0.0,
            "device_shared_buyer_count": 1.0,
            "multimodal_similarity_score": 0.95,
        },
    },
    "high_risk_anomalous": {
        "description": "High order amount, severe price inflation, 100% return rate, collusive device sharing, high ring component size, multimodal mismatch.",
        "payload": {
            "order_id": "golden_p3_002_high",
            "buyer_id": "buyer_golden_high",
            "seller_id": "seller_golden_high",
            "amount": 8500.0,
            "order_amount": 8500.0,
            "price_vs_base_price_ratio": 8.0,
            "price_vs_category_median_ratio": 9.0,
            "seller_age_days": 100.0,
            "seller_total_listings_before": 10.0,
            "buyer_age_days": 60.0,
            "buyer_orders_before": 10,
            "buyer_returns_before": 10,
            "buyer_return_rate_before": 1.0,
            "device_shared_buyer_count": 25.0,
            "share_degree": 10.0,
            "share_component_size": 20.0,
            "buyer_pagerank": 0.15,
            "multimodal_similarity_score": 0.02,
        },
    },
    "historical_point_in_time": {
        "description": "Point-in-time replay order with explicit order_date timestamp ensuring deterministic prior-state isolation.",
        "payload": {
            "order_id": "golden_p3_003_historical",
            "buyer_id": "buyer_golden_hist",
            "seller_id": "seller_golden_hist",
            "order_date": "2026-10-15T12:00:00Z",
            "amount": 120.0,
            "order_amount": 120.0,
            "price_vs_base_price_ratio": 1.15,
            "price_vs_category_median_ratio": 1.10,
            "seller_age_days": 120.0,
            "seller_total_listings_before": 25.0,
            "buyer_age_days": 60.0,
            "buyer_orders_before": 3,
            "buyer_returns_before": 1,
            "buyer_return_rate_before": 0.333,
            "device_shared_buyer_count": 1.0,
            "multimodal_similarity_score": 0.88,
        },
    },
    "cold_start_new_entities": {
        "description": "Brand-new buyer and seller with 0 previous orders and 0 listings triggering multi-signal cold_start penalty.",
        "payload": {
            "order_id": "golden_p3_004_cold_start",
            "buyer_id": "buyer_golden_cold_new",
            "seller_id": "seller_golden_cold_new",
            "amount": 65.0,
            "order_amount": 65.0,
            "price_vs_base_price_ratio": 1.0,
            "price_vs_category_median_ratio": 1.0,
            "seller_age_days": 2.0,
            "seller_total_listings_before": 1.0,
            "buyer_age_days": 1.0,
            "buyer_orders_before": 0,
            "buyer_returns_before": 0,
            "buyer_return_rate_before": 0.0,
            "device_shared_buyer_count": 1.0,
            "multimodal_similarity_score": 0.90,
        },
    },
}


def generate_golden_fixture() -> Path:
    # 1. Inspect response schema
    schema_fields = list(TransactionScoreResponse.model_fields.keys())
    assert len(schema_fields) == 29, f"Expected 29 schema fields, found {len(schema_fields)}"

    # 2. Inspect feature meta
    feature_meta_path = REPO_ROOT / "models" / "feature_meta.joblib"
    feature_meta = joblib.load(feature_meta_path)
    all_features = feature_meta["all_feature_cols"]
    assert len(all_features) == 18, f"Expected 18 feature cols, found {len(all_features)}"

    # 3. Compute artifact hashes
    artifacts = {
        "combined_graph_model.joblib": _compute_sha256(REPO_ROOT / "models" / "combined_graph_model.joblib"),
        "calibrator.joblib": _compute_sha256(REPO_ROOT / "models" / "calibrator.joblib"),
        "graph_snapshot.joblib": _compute_sha256(REPO_ROOT / "models" / "graph_snapshot.joblib"),
        "feature_meta.joblib": _compute_sha256(REPO_ROOT / "models" / "feature_meta.joblib"),
        "phase3_val_predictions.npy": _compute_sha256(REPO_ROOT / "models" / "phase3_val_predictions.npy"),
    }

    # 4. Initialize client and execute scenarios
    client = TestClient(app)
    headers = {"X-API-Key": "dev-admin-secret-key-32chars-long!"}

    # Reset any live history before deterministic run
    if getattr(store, "transaction_history", None) is not None:
        store.transaction_history.reset_live_history()

    scenario_results = {}
    for scenario_name, cfg in GOLDEN_SCENARIOS.items():
        resp = client.post("/transaction/score", json=cfg["payload"], headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to score scenario '{scenario_name}': HTTP {resp.status_code} - {resp.text}")

        data = resp.json()

        # Validate that all 29 fields are present in response
        missing_fields = [f for f in schema_fields if f not in data]
        if missing_fields:
            raise ValueError(f"Scenario '{scenario_name}' response is missing schema fields: {missing_fields}")

        # Ensure model_version is Phase 3
        assert data["model_version"] == "phase3", f"Expected model_version='phase3', got {data['model_version']}"

        scenario_results[scenario_name] = {
            "description": cfg["description"],
            "request_payload": cfg["payload"],
            "expected_response": data,
        }

    # Reset again after generation
    if getattr(store, "transaction_history", None) is not None:
        store.transaction_history.reset_live_history()

    fixture_payload = {
        "metadata": {
            "baseline_id": "phase3_production_baseline_post_phase43b",
            "baseline_version": "1.0.0",
            "baseline_label": "Current post-Phase-4.3B frozen baseline",
            "historical_invariance_note": (
                "Authoritative frozen regression baseline established post-Phase-4.3B simplification. "
                "Because Phase 4.2B online components were never committed before removal, historical "
                "before/after invariance cannot be claimed against Git history. This fixture serves as "
                "the authoritative post-simplification baseline for all regression protection."
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator_command": "python scripts/generate_phase3_golden_baseline.py",
            "model_version": "phase3",
            "scoring_mode": "production_model_phase3_combined",
            "response_field_count": len(schema_fields),
            "response_fields": schema_fields,
            "feature_count": len(all_features),
            "feature_order": all_features,
            "runtime_environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "scikit_learn_version": sklearn.__version__,
                "xgboost_version": xgboost.__version__,
                "fastapi_version": fastapi.__version__,
                "numpy_version": np.__version__,
            },
            "model_artifact_hashes": artifacts,
        },
        "scenarios": scenario_results,
    }

    fixtures_dir = REPO_ROOT / "backend" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    out_file = fixtures_dir / "phase3_golden_baseline.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(fixture_payload, f, indent=2)

    print(f"Successfully generated golden baseline fixture at: {out_file}")
    print(f"  Scenarios covered: {list(scenario_results.keys())}")
    print(f"  Response fields verified: {len(schema_fields)}")
    print(f"  Artifacts hashed: {len(artifacts)}")
    return out_file


if __name__ == "__main__":
    generate_golden_fixture()
