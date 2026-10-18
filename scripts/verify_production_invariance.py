"""
TrustShield AI — Production Baseline Contract Invariance Verification.

Verifies complete regression protection against the frozen post-Phase-4.3B
golden baseline fixture (29 schema fields across all 4 production scenarios).

Usage:
    python scripts/verify_production_invariance.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "backend" / "fixtures" / "phase3_golden_baseline.json"
AUTH_HEADERS = {"X-API-Key": "dev-admin-secret-key-32chars-long!"}


def verify_invariance() -> bool:
    print("=" * 70)
    print("TRUSTSHIELD AI — PRODUCTION INVARIANCE REGRESSION CHECK")
    print(f"Fixture: {FIXTURE_PATH}")
    print("=" * 70)

    if not FIXTURE_PATH.is_file():
        print(f"ERROR: Golden fixture not found at {FIXTURE_PATH}")
        return False

    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        golden_data: Dict[str, Any] = json.load(f)

    from fastapi.testclient import TestClient
    from backend.main import app, store
    from backend.test_production_baseline_regression import compare_response_against_golden

    if not store.is_loaded:
        store.load()

    client = TestClient(app, headers=AUTH_HEADERS)

    scenarios = [
        "low_risk_standard",
        "high_risk_anomalous",
        "historical_point_in_time",
        "cold_start_new_entities",
    ]

    all_passed = True
    for sc in scenarios:
        entry = golden_data["scenarios"][sc]
        req_payload = entry["request_payload"]
        expected_resp = entry["expected_response"]

        resp = client.post("/transaction/score", json=req_payload)
        if resp.status_code != 200:
            print(f"FAIL [{sc}]: Expected HTTP 200, got {resp.status_code}: {resp.text}")
            all_passed = False
            continue

        actual_resp = resp.json()
        try:
            compare_response_against_golden(actual_resp, expected_resp)
            print(f"PASS [{sc}]: All 29 fields matched golden baseline.")
        except AssertionError as err:
            print(f"FAIL [{sc}]: Invariance violation: {err}")
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL 4 PRODUCTION SCENARIOS PASSED 29-FIELD INVARIANCE VERIFICATION!")
    else:
        print("PRODUCTION INVARIANCE REGRESSION DETECTED!")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = verify_invariance()
    sys.exit(0 if success else 1)
