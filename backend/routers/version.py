"""Public release/version endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from backend.release_manifest import (
    get_release_manifest_status,
    public_release_manifest,
    release_manifest_required,
)
from backend.schemas import VersionResponse


router = APIRouter()


@router.get("/version", response_model=VersionResponse, tags=["system"])
def version() -> VersionResponse:
    """Return non-secret application and release-provenance metadata."""

    status = get_release_manifest_status()
    manifest = status.manifest
    git_info = manifest.get("git", {}) if isinstance(manifest, dict) else {}
    if not isinstance(git_info, dict):
        git_info = {}
    commit = git_info.get("commit") or (manifest or {}).get("git_commit")
    clean = git_info.get("working_tree_clean")
    schema_version = manifest.get("schema_version") if isinstance(manifest, dict) else None
    return VersionResponse(
        service="trustshield",
        version="0.2.0",
        app_version="0.2.0",
        release_id=(manifest or {}).get("release_id") if isinstance(manifest, dict) else None,
        manifest_present=status.present,
        manifest_valid=status.valid,
        manifest_required=release_manifest_required(),
        manifest_schema_version=schema_version if isinstance(schema_version, int) else None,
        git_commit=commit if isinstance(commit, str) else None,
        working_tree_clean=clean if isinstance(clean, bool) else None,
        validation_errors=list(status.errors),
        manifest=public_release_manifest(manifest),
    )


__all__ = ["router", "version"]
