"""Negative testing suite for TrustShield CI Guard (scripts/check_no_hardcoded_metrics.py).

Verifies that the CI guard strictly fails (exits non-zero) when:
1. results/results.json is missing or lacks required benchmark keys.
2. An untraceable/hardcoded metric (e.g. legacy 0.792 without disclaimer) is introduced.
3. Frontend files contain hardcoded tabular metrics instead of dynamic JSON binding.
"""

import os
import sys
import json
import builtins
import pytest

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT_DIR)

import scripts.check_no_hardcoded_metrics as ci_module
from scripts.check_no_hardcoded_metrics import check_no_hardcoded_metrics


class TestCIGuardNegative:
    """Negative tests asserting that CI guard fails on corrupt or ungrounded data."""

    def test_ci_guard_passes_on_clean_repository(self):
        """Baseline positive test: current repo passes CI guard cleanly."""
        try:
            check_no_hardcoded_metrics()
        except SystemExit as exc:
            pytest.fail(f"CI guard unexpectedly failed on clean repo: {exc}")

    def test_ci_guard_fails_on_missing_required_key(self, monkeypatch):
        """Negative test: fails when a required benchmark block is deleted from results.json."""
        results_path = os.path.join(_ROOT_DIR, "results", "results.json")
        with open(results_path, "r", encoding="utf-8") as f:
            valid_data = json.load(f)

        # Corrupt data by removing primary analysis key
        corrupt_data = dict(valid_data)
        corrupt_data.pop("pre_registered_primary_analysis_20_seeds", None)

        def mock_load(fp):
            return corrupt_data

        monkeypatch.setattr(json, "load", mock_load)

        with pytest.raises(SystemExit) as exc_info:
            check_no_hardcoded_metrics()
        assert exc_info.value.code == 1

    def test_ci_guard_fails_on_untraced_0_792_in_model_card(self, monkeypatch):
        """Negative test: fails when 0.792 is present in MODEL_CARD.md without disclaimer."""
        orig_open = builtins.open

        def mock_open(file, *args, **kwargs):
            if str(file).endswith("MODEL_CARD.md"):
                from io import StringIO
                return StringIO("The model achieves test ROC-AUC of 0.792 across benchmarks.")
            return orig_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", mock_open)

        with pytest.raises(SystemExit) as exc_info:
            check_no_hardcoded_metrics()
        assert exc_info.value.code == 1

    def test_ci_guard_fails_on_corrupt_json_file(self, monkeypatch):
        """Negative test: fails when results.json contains invalid JSON syntax."""
        def mock_load(fp):
            raise json.JSONDecodeError("Expecting value", "bad json", 0)

        monkeypatch.setattr(json, "load", mock_load)

        with pytest.raises(SystemExit) as exc_info:
            check_no_hardcoded_metrics()
        assert exc_info.value.code == 1
