#!/usr/bin/env python3
"""Check the critical backend/OpenAPI/frontend contract surface.

This is intentionally a small guard rather than a generated client. It catches
renamed or removed fields on the health, readiness, version, score, and
operational response paths while keeping the synthetic research API stable.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app


FRONTEND_TYPES = ROOT / "frontend" / "src" / "lib" / "types" / "api.ts"

REQUIRED_PATHS = {
    "/health": "get",
    "/ready": "get",
    "/version": "get",
    "/transaction/score": "post",
    "/transaction/explain": "post",
    "/fraud-rings": "get",
    "/listing/analyze": "post",
    "/return/analyze": "post",
    "/contracts/events/validate": "post",
    "/contracts/labels/validate": "post",
}

REQUIRED_SCHEMA_FIELDS = {
    "HealthResponse": {"status", "models_loaded", "rings_loaded", "n_rings"},
    "ReadyResponse": {"status", "models_ready", "phase3_ready", "pit_history_ready", "details"},
    "VersionResponse": {"version", "manifest_present", "manifest_valid", "manifest_required"},
    "TransactionScoreResponse": {"overall_fraud_probability", "risk_label", "decision", "trust_score"},
    "FraudRingsResponse": {"total_rings", "high_risk_rings", "rings"},
}

FRONTEND_INTERFACE_NAMES = {
    "HealthResponse": "HealthResponse",
    "ReadyResponse": "ReadyResponse",
    "VersionResponse": "VersionResponse",
    "TransactionScoreResponse": "TransactionScoreResponse",
    "FraudRingsResponse": "FraudRingsResponse",
}


def _frontend_interface_fields(text: str, name: str) -> set[str]:
    match = re.search(rf"interface\s+{re.escape(name)}\s*\{{(?P<body>.*?)\n\}}", text, re.DOTALL)
    if not match:
        raise AssertionError(f"frontend interface {name} is missing")
    return set(re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\??\s*:", match.group("body"), re.MULTILINE))


def check_contract() -> list[str]:
    openapi = app.openapi()
    errors: list[str] = []
    paths = openapi.get("paths", {})
    for path, method in REQUIRED_PATHS.items():
        if path not in paths or method not in paths[path]:
            errors.append(f"missing OpenAPI operation: {method.upper()} {path}")

    schemas = openapi.get("components", {}).get("schemas", {})
    frontend_text = FRONTEND_TYPES.read_text(encoding="utf-8")
    for schema_name, expected_fields in REQUIRED_SCHEMA_FIELDS.items():
        schema = schemas.get(schema_name, {})
        properties = set(schema.get("properties", {}))
        missing_backend = expected_fields - properties
        if missing_backend:
            errors.append(f"{schema_name} missing backend fields: {sorted(missing_backend)}")
        frontend_fields = _frontend_interface_fields(frontend_text, FRONTEND_INTERFACE_NAMES[schema_name])
        missing_frontend = expected_fields - frontend_fields
        if missing_frontend:
            errors.append(f"{schema_name} missing frontend fields: {sorted(missing_frontend)}")
    return errors


def main() -> int:
    errors = check_contract()
    if errors:
        for error in errors:
            print(f"CONTRACT ERROR: {error}")
        return 1
    print("OpenAPI/frontend critical contract check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
