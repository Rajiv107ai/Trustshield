"""Focused tests for Stage 1 release provenance."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.release_manifest import (
    build_release_manifest,
    get_release_manifest_status,
    validate_release_manifest_configuration,
    write_release_manifest,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "release-repo"
    root.mkdir()
    (root / "models").mkdir()
    (root / "frontend").mkdir()
    (root / "artifact.bin").write_bytes(b"artifact-v1")
    (root / "requirements.txt").write_text("fastapi==1.0\n", encoding="utf-8")
    (root / "requirements-dev.txt").write_text("pytest==1.0\n", encoding="utf-8")
    (root / "frontend" / "package-lock.json").write_text("{\"lockfileVersion\": 3}\n", encoding="utf-8")
    (root / "feature-schema.json").write_text("{\"features\": [\"amount\"]}\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "release-tests@example.invalid")
    _git(root, "config", "user.name", "Release Tests")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "initial")
    return root


def _manifest(root: Path, output: Path) -> dict:
    return build_release_manifest(
        root,
        release_id="test-release",
        output_path=output,
        required_artifact_paths=["artifact.bin"],
        optional_artifact_paths=[],
        feature_schema_path="feature-schema.json",
    )


def test_manifest_records_dirty_tree_and_validation_rejects_it(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "artifact.bin").write_bytes(b"dirty-artifact")
    output = root / "release-manifest.json"
    manifest = _manifest(root, output)

    assert manifest["working_tree_clean"] is False
    assert manifest["git"]["working_tree_clean"] is False
    write_release_manifest(manifest, output)
    status = get_release_manifest_status(root_dir=root, manifest_path=output)
    assert status.valid is False
    assert any("dirty" in error for error in status.errors)


def test_manifest_hash_and_size_mismatch_are_detected(tmp_path: Path):
    root = _repo(tmp_path)
    output = root / "release-manifest.json"
    manifest = _manifest(root, output)
    write_release_manifest(manifest, output)
    assert get_release_manifest_status(root_dir=root, manifest_path=output).valid is True

    (root / "artifact.bin").write_bytes(b"changed-after-build")
    status = get_release_manifest_status(root_dir=root, manifest_path=output)
    assert status.valid is False
    assert any("sha256 mismatch" in error for error in status.errors)
    assert any("size mismatch" in error for error in status.errors)


def test_required_manifest_configuration_fails_closed_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _repo(tmp_path)
    monkeypatch.setenv("TRUSTSHIELD_REQUIRE_RELEASE_MANIFEST", "true")
    monkeypatch.setenv("TRUSTSHIELD_RELEASE_ROOT", str(root))
    monkeypatch.setenv("TRUSTSHIELD_RELEASE_MANIFEST_PATH", "missing.json")
    with pytest.raises(RuntimeError, match="CRITICAL RELEASE PROVENANCE"):
        validate_release_manifest_configuration()


def test_application_startup_fails_closed_when_manifest_is_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TRUSTSHIELD_REQUIRE_RELEASE_MANIFEST", "true")
    monkeypatch.setenv("TRUSTSHIELD_RELEASE_ROOT", str(tmp_path))
    monkeypatch.setenv("TRUSTSHIELD_RELEASE_MANIFEST_PATH", "missing.json")
    with pytest.raises(RuntimeError, match="CRITICAL RELEASE PROVENANCE"):
        with TestClient(app):
            pass


def test_version_endpoint_reports_status_without_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"release_id": "safe-release", "secret": "do-not-return"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRUSTSHIELD_RELEASE_ROOT", str(tmp_path))
    monkeypatch.setenv("TRUSTSHIELD_RELEASE_MANIFEST_PATH", str(manifest_path))
    monkeypatch.delenv("TRUSTSHIELD_REQUIRE_RELEASE_MANIFEST", raising=False)
    with TestClient(app) as client:
        response = client.get("/version")
    assert response.status_code == 200
    payload = response.json()
    assert payload["manifest_present"] is True
    assert payload["manifest_valid"] is False
    assert payload["release_id"] == "safe-release"
    assert "do-not-return" not in response.text
