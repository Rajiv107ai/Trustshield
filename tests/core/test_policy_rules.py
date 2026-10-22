"""Regression tests for centralized policy thresholds."""

from trustshield_project.policy_config import RISK_POLICY


def test_policy_thresholds_are_ordered_and_versioned():
    assert 0.0 <= RISK_POLICY.review <= RISK_POLICY.hold <= RISK_POLICY.block <= 1.0
    assert RISK_POLICY.version.startswith("policy.")
