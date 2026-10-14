"""Negative testing suite for TrustShield CI Guard (scripts/check_no_hardcoded_metrics.py).

Verifies that the allowlist CI Guard strictly catches and fails (exits non-zero) when:
1. Any unapproved metric string is inserted outside a generated block across all 4 target file types:
   - README.md
   - docs/*.md (e.g. docs/MODEL_CARD.md)
   - results/RESULTS.md
   - frontend/src/**/*.tsx (e.g. frontend/src/app/models/page.tsx)
2. Parametrized across the 10 required audit strings:
   - "roc_auc: 0.912"
   - "'roc_auc': 0.912"
   - "ROC-AUC of 0.912"
   - "<td>ROC-AUC</td><td>0.912</td>"
   - "F-1 0.91"
   - "ECE 0.021"
   - "Brier 0.058"
   - "lift +0.0115"
   - "TPR 0.88"
   - "94 %"
3. frontend/public/results.json differs from results/results.json.
4. results/results.json is corrupted or missing required schema keys.
"""

from __future__ import annotations
import os
import sys
import json
import builtins
import pytest
from io import StringIO, BytesIO

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT_DIR)

from scripts.check_no_hardcoded_metrics import check_no_hardcoded_metrics, scan_file_content

REQUIRED_NEGATIVE_STRINGS = [
    "roc_auc: 0.912",
    "'roc_auc': 0.912",
    "ROC-AUC of 0.912",
    "<td>ROC-AUC</td><td>0.912</td>",
    "F-1 0.91",
    "ECE 0.021",
    "Brier 0.058",
    "lift +0.0115",
    "TPR 0.88",
    "94 %",
]

TARGET_FILE_TYPES = [
    ("README.md", "README.md"),
    ("docs/MODEL_CARD.md", "docs/MODEL_CARD.md"),
    ("results/RESULTS.md", "results/RESULTS.md"),
    ("frontend/src/app/models/page.tsx", "frontend/src/app/models/page.tsx"),
]


class TestCIGuardNegative:
    """Negative testing suite ensuring zero tolerance for unapproved metric assertions."""

    def test_ci_guard_passes_on_clean_repository(self):
        """Baseline positive test: current repo passes CI guard cleanly."""
        try:
            check_no_hardcoded_metrics()
        except SystemExit as exc:
            pytest.fail(f"CI guard unexpectedly failed on clean repo: {exc}")

    @pytest.mark.parametrize("test_str", REQUIRED_NEGATIVE_STRINGS)
    @pytest.mark.parametrize("file_relpath, file_key", TARGET_FILE_TYPES)
    def test_ci_guard_fails_on_string_in_file_type(self, monkeypatch, test_str, file_relpath, file_key):
        """Parametrized negative test: inserting test_str into file outside generated block fails CI."""
        target_abs = os.path.normpath(os.path.join(_ROOT_DIR, file_relpath)).replace("\\", "/")
        orig_open = builtins.open

        def mock_open(file, *args, **kwargs):
            norm_file = os.path.normpath(str(file)).replace("\\", "/")
            if norm_file == target_abs:
                # Read genuine content and append test string outside any generated block
                with orig_open(file, *args, **kwargs) as real_f:
                    real_content = real_f.read()
                corrupted_content = real_content + f"\n\n<!-- Unapproved Claim -->\nClaim: {test_str}\n"
                return StringIO(corrupted_content)
            return orig_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", mock_open)

        with pytest.raises(SystemExit) as exc_info:
            check_no_hardcoded_metrics()
        assert exc_info.value.code == 1, f"Expected CI guard to exit with code 1 for '{test_str}' in {file_relpath}"

    @pytest.mark.parametrize("test_str", REQUIRED_NEGATIVE_STRINGS)
    def test_scan_file_content_flags_each_required_string(self, test_str):
        """Direct unit test verifying scan_file_content flags each required negative string."""
        errs = scan_file_content(f"Performance result: {test_str}", "test_snippet")
        assert len(errs) > 0, f"Expected scan_file_content to flag '{test_str}', but got 0 errors."

    def test_ci_guard_fails_when_public_results_differs(self, monkeypatch):
        """Negative test: fails when frontend/public/results.json differs from results/results.json."""
        orig_open = builtins.open
        public_path = os.path.normpath(os.path.join(_ROOT_DIR, "frontend", "public", "results.json")).replace("\\", "/")

        def mock_open(file, *args, **kwargs):
            norm_file = os.path.normpath(str(file)).replace("\\", "/")
            if norm_file == public_path and "rb" in args:
                return BytesIO(b'{"corrupted": true}')
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

    def test_ci_guard_fails_on_corrupt_json_syntax(self, monkeypatch):
        """Negative test: fails when results.json contains invalid JSON syntax."""
        def mock_load(fp):
            raise json.JSONDecodeError("Expecting value", "bad json", 0)

        monkeypatch.setattr(json, "load", mock_load)

        with pytest.raises(SystemExit) as exc_info:
            check_no_hardcoded_metrics()
        assert exc_info.value.code == 1

    def test_verify_results_fails_when_metric_tampered_in_readme(self, monkeypatch):
        """Negative test: verify_results.py exits non-zero if 0.7410 is tampered to 0.7510 in README.md."""
        from scripts.verify_results import verify
        orig_open = builtins.open
        readme_norm = os.path.normpath(os.path.join(_ROOT_DIR, "README.md")).replace("\\", "/")

        def mock_open(file, *args, **kwargs):
            norm_file = os.path.normpath(str(file)).replace("\\", "/")
            if norm_file == readme_norm:
                real_content = orig_open(file, *args, **kwargs).read()
                tampered = real_content.replace("0.7410", "0.7510")
                return StringIO(tampered)
            return orig_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", mock_open)
        with pytest.raises(SystemExit) as exc_info:
            verify()
        assert exc_info.value.code != 0

