"""Pytest entry point for the OpenAPI/frontend contract guard."""

from scripts.check_openapi_contract import check_contract


def test_critical_openapi_frontend_contract_is_aligned():
    assert check_contract() == []
