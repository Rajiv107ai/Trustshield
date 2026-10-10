"""Negative testing suite for TrustShield CI Guard (scripts/check_no_hardcoded_metrics.py).

Verifies that the generic metric regex strictly catches and fails (exits non-zero) when:
1. "ROC-AUC 0.912" is inserted in README.md.
2. "ROC-AUC 0.912" is inserted in a docs/*.md file (e.g. docs/MODEL_CARD.md).
3. "ROC-AUC 0.912" is inserted in a frontend .tsx file (e.g. frontend/src/app/models/page.tsx).
4. "94.2%" is inserted in a frontend page (e.g. frontend/src/app/evaluation/page.tsx).
5. results/results.json is corrupted or lacks required benchmark keys.
"""

import os
import sys
import json
import builtins
import pytest
from io import StringIO

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT_DIR)

from scripts.check_no_hardcoded_metrics import check_no_hardcoded_metrics


class TestCIGuardNegative:
    """Negative tests asserting that CI guard fails on ungrounded metrics using generic regexes."""

    def test_ci_guard_passes_on_clean_repository(self):
        """Baseline positive test: current repo passes CI guard cleanly."""
        try:
            check_no_hardcoded_metrics()
        except SystemExit as exc:
            pytest.fail(f"CI guard unexpectedly failed on clean repo: {exc}")

    def test_ci_guard_fails_on_roc_0_912_in_readme(self, monkeypatch):
        """Negative test: fails when 'ROC-AUC 0.912' is inserted in README.md."""
        orig_open = builtins.open

        def mock_open(file, *args, **kwargs):
            if str(file).replace("\\", "/").endswith("README.md"):
                return StringIO("Performance claim: Our new model achieves ROC-AUC 0.912 on test data.")
            return orig_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", mock_open)

        with pytest.raises(SystemExit) as exc_info:
            check_no_hardcoded_metrics()
        assert exc_info.value.code == 1

    def test_ci_guard_fails_on_roc_0_912_in_docs_model_card(self, monkeypatch):
        """Negative test: fails when 'ROC-AUC 0.912' is inserted in docs/MODEL_CARD.md."""
        orig_open = builtins.open

        def mock_open(file, *args, **kwargs):
            if str(file).replace("\\", "/").endswith("docs/MODEL_CARD.md"):
                return StringIO("Model Specification: Audited ROC-AUC 0.912 reported on holdout.")
            return orig_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", mock_open)

        with pytest.raises(SystemExit) as exc_info:
            check_no_hardcoded_metrics()
        assert exc_info.value.code == 1

    def test_ci_guard_fails_on_roc_0_912_in_frontend_tsx(self, monkeypatch):
        """Negative test: fails when 'ROC-AUC 0.912' is inserted in a frontend .tsx file."""
        orig_open = builtins.open

        def mock_open(file, *args, **kwargs):
            if str(file).replace("\\", "/").endswith("frontend/src/app/models/page.tsx"):
                return StringIO('export default function Page() { return <div>ROC-AUC 0.912</div>; }')
            return orig_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", mock_open)

        with pytest.raises(SystemExit) as exc_info:
            check_no_hardcoded_metrics()
        assert exc_info.value.code == 1

    def test_ci_guard_fails_on_94_2_pct_in_frontend_page(self, monkeypatch):
        """Negative test: fails when '94.2%' is inserted in a frontend page."""
        orig_open = builtins.open

        def mock_open(file, *args, **kwargs):
            if str(file).replace("\\", "/").endswith("frontend/src/app/evaluation/page.tsx"):
                return StringIO('export default function Page() { return <div>Fraud Rate: 94.2%</div>; }')
            return orig_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", mock_open)

        with pytest.raises(SystemExit) as exc_info:
            check_no_hardcoded_metrics()
        assert exc_info.value.code == 1

    def test_ci_guard_fails_on_missing_required_key(self, monkeypatch):
        """Negative test: fails when a required benchmark block is deleted from results.json."""
        results_path = os.path.join(_ROOT_DIR, "results", "results.json")
        with open(results_path, "r", encoding="utf-8") as f:
            valid_data = json.load(f)

        corrupt_data = dict(valid_data)
        corrupt_data.pop("pre_registered_primary_analysis_20_seeds", None)

        def mock_load(fp):
            return corrupt_data

        monkeypatch.setattr(json, "load", mock_load)

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
