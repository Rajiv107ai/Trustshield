"""Release-manifest generation, loading, and validation.

Stage 1 release provenance is deliberately metadata-only.  The generator hashes
the serving artifacts and lock files that are already part of the repository;
it does not enumerate, open, or hash the customer/synthetic data directories.

The module is usable from both the build script and the running API.  Runtime
validation checks the current checkout, lock files, and artifact fingerprints so
that a manifest cannot silently describe a different build.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

try:
    from trustshield_project.policy_config import RISK_POLICY
except ImportError:  # pragma: no cover - manifest tooling should still import
    RISK_POLICY = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_NAME = "models/release_manifest.json"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / DEFAULT_MANIFEST_NAME
MANIFEST_SCHEMA_VERSION = 1

# These are the artifacts loaded by backend/model_loader.py.  Keep this list
# explicit rather than walking models/; model directories can contain datasets,
# caches, or other files that do not belong in a release provenance record.
DEFAULT_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "models/combined_graph_model.joblib",
    "models/fake_listing_model.joblib",
    "models/return_fraud_model.joblib",
    "models/fraud_rings.joblib",
    "models/feature_meta.joblib",
)
DEFAULT_OPTIONAL_ARTIFACTS: tuple[str, ...] = (
    "models/calibrator.joblib",
    "models/graph_snapshot.joblib",
    "models/product_catalog.parquet",
)
DEFAULT_PYTHON_LOCK = "requirements.txt"
DEFAULT_PYTHON_DEV_LOCK = "requirements-dev.txt"
DEFAULT_FRONTEND_LOCK = "frontend/package-lock.json"
DEFAULT_FEATURE_SCHEMA = "models/feature_meta.joblib"

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:secret|password|token|api[_-]?key|authorization|credential|private[_-]?key)",
    re.IGNORECASE,
)


class ReleaseManifestError(ValueError):
    """Raised when a release manifest cannot be decoded or addressed safely."""


@dataclass(frozen=True)
class FileFingerprint:
    """A release-safe fingerprint for one repository file."""

    path: str
    sha256: Optional[str]
    size: Optional[int]
    available: bool = True
    required: bool = True

    def as_dict(self) -> dict[str, Any]:
        # Keep both spellings for consumers that use the natural ``size`` name
        # and consumers that prefer the explicit ``size_bytes`` name.
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "size_bytes": self.size,
            "available": self.available,
            "required": self.required,
        }


@dataclass(frozen=True)
class ManifestValidationResult:
    """Result returned by the reusable loader/validator."""

    present: bool
    valid: bool
    errors: tuple[str, ...] = ()
    manifest: Optional[dict[str, Any]] = None

    def __bool__(self) -> bool:
        return self.valid

    @property
    def error_messages(self) -> list[str]:
        return list(self.errors)


def _project_path(root_dir: Optional[os.PathLike[str] | str]) -> Path:
    return Path(root_dir or PROJECT_ROOT).expanduser().resolve()


def _path_inside(root_dir: Path, path: os.PathLike[str] | str) -> Path:
    """Resolve a manifest path and reject path traversal outside the project."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root_dir / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root_dir)
    except ValueError as exc:
        raise ReleaseManifestError(f"Manifest path is outside the release root: {path}") from exc
    return candidate


def _relative_path(root_dir: Path, path: os.PathLike[str] | str) -> str:
    resolved = _path_inside(root_dir, path)
    return resolved.relative_to(root_dir).as_posix()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_manifest_digest(
    root_dir: Optional[os.PathLike[str] | str] = None,
    manifest_path: Optional[os.PathLike[str] | str] = None,
) -> Optional[str]:
    """Return the SHA-256 of the release manifest file, if present."""

    _root, path = _manifest_path_from_environment(root_dir, manifest_path)
    return _hash_file(path) if path.is_file() else None


def fingerprint_file(
    root_dir: os.PathLike[str] | str,
    path: os.PathLike[str] | str,
    *,
    required: bool = True,
) -> FileFingerprint:
    """Hash one file relative to ``root_dir`` without reading unrelated files."""

    root = _project_path(root_dir)
    relative = _relative_path(root, path)
    resolved = root / relative
    if not resolved.is_file():
        return FileFingerprint(
            path=relative,
            sha256=None,
            size=None,
            available=False,
            required=required,
        )
    return FileFingerprint(
        path=relative,
        sha256=_hash_file(resolved),
        size=resolved.stat().st_size,
        available=True,
        required=required,
    )


def _run_git(root_dir: Path, *args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # An empty successful result is meaningful for commands such as
    # ``git status --porcelain`` (it means a clean tree), so preserve ``""``
    # here. Callers that require a non-empty value normalize it explicitly.
    return completed.stdout.strip()


def git_commit(root_dir: os.PathLike[str] | str) -> Optional[str]:
    """Return the checkout's full commit, or ``None`` outside a Git checkout."""

    value = _run_git(_project_path(root_dir), "rev-parse", "HEAD")
    return value or None


def _normalise_status_path(raw_path: str) -> str:
    # ``git status --porcelain`` uses ``old -> new`` for renames.  For the
    # purpose of excluding the manifest itself, either side is sufficient.
    return raw_path.strip().replace("\\", "/").lstrip("./")


def git_worktree_clean(
    root_dir: os.PathLike[str] | str,
    *,
    ignore_path: Optional[os.PathLike[str] | str] = None,
) -> Optional[bool]:
    """Return whether tracked checkout state is clean.

    The generated manifest is normally an untracked release artifact.  It is
    excluded from this check so writing the manifest does not make a previously
    clean checkout claim to be dirty during later validation.
    """

    root = _project_path(root_dir)
    output = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if output is None:
        return None
    ignored: set[str] = set()
    if ignore_path is not None:
        try:
            ignored.add(_relative_path(root, ignore_path))
        except ReleaseManifestError:
            pass

    for line in output.splitlines():
        if not line:
            continue
        status_code = line[:2]
        raw_path = line[3:] if len(line) >= 4 else line
        candidates = raw_path.split(" -> ")
        # Only an untracked output file is ignored.  If a repository tracks
        # the manifest and it is modified, that modification is part of the
        # working-tree state and must make the release dirty.
        if status_code == "??" and any(_normalise_status_path(item) in ignored for item in candidates):
            continue
        return False
    return True


def _git_release_id(root_dir: Path, commit: Optional[str]) -> str:
    described = _run_git(root_dir, "describe", "--tags", "--always", "--dirty")
    if described:
        return described
    if commit:
        return commit[:12]
    return "unreleased"


def _file_record(
    root_dir: Path,
    path: os.PathLike[str] | str,
    *,
    required: bool,
) -> dict[str, Any]:
    return fingerprint_file(root_dir, path, required=required).as_dict()


def _load_feature_metadata(root_dir: Path, feature_path: str) -> dict[str, Any]:
    """Read only the known feature metadata artifact, never customer data."""

    metadata: dict[str, Any] = {}
    path = root_dir / feature_path
    if not path.is_file():
        return metadata
    try:
        import joblib  # type: ignore[import-not-found]

        loaded = joblib.load(path)
        if isinstance(loaded, Mapping):
            for key in ("classifier", "calibrated", "calibrator_method", "thresholds"):
                if key in loaded:
                    value = loaded[key]
                    if key == "thresholds" and isinstance(value, (tuple, list)):
                        value = [float(item) for item in value]
                    elif key == "calibrated":
                        value = bool(value)
                    elif not isinstance(value, (str, int, float, bool, type(None))):
                        continue
                    metadata[key] = value
    except Exception:
        # A hash is still useful when the optional joblib dependency is not
        # installed in a release-building environment.
        metadata["metadata_available"] = False
    return metadata


def _placeholder_tests() -> dict[str, Any]:
    return {
        "status": "not_run",
        "passed": None,
        "failed": None,
        "skipped": None,
        "command": None,
        "results": [],
    }


def build_release_manifest(
    repo_root: os.PathLike[str] | str = PROJECT_ROOT,
    *,
    release_id: Optional[str] = None,
    output_path: Optional[os.PathLike[str] | str] = None,
    required_artifact_paths: Optional[Sequence[os.PathLike[str] | str]] = None,
    optional_artifact_paths: Optional[Sequence[os.PathLike[str] | str]] = None,
    python_lock_path: os.PathLike[str] | str = DEFAULT_PYTHON_LOCK,
    python_dev_lock_path: os.PathLike[str] | str = DEFAULT_PYTHON_DEV_LOCK,
    frontend_lock_path: os.PathLike[str] | str = DEFAULT_FRONTEND_LOCK,
    feature_schema_path: os.PathLike[str] | str = DEFAULT_FEATURE_SCHEMA,
) -> dict[str, Any]:
    """Build a metadata-only manifest dictionary.

    ``git_worktree_clean`` is evaluated before the output is written, so a
    dirty source checkout is represented honestly as ``false``.  No test,
    dataset, or customer-data directories are traversed.
    """

    root = _project_path(repo_root)
    output = (
        (root / DEFAULT_MANIFEST_NAME).resolve()
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    commit = git_commit(root)
    working_tree_clean = git_worktree_clean(root, ignore_path=output)
    clean_value = working_tree_clean is True
    commit_value = commit or "unknown"

    requested_release_id = release_id or os.getenv("TRUSTSHIELD_RELEASE_ID")
    release_value = requested_release_id or _git_release_id(root, commit)

    runtime = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(aliased=True),
    }
    runtime_hash = hashlib.sha256(
        json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    required_paths = tuple(required_artifact_paths or DEFAULT_REQUIRED_ARTIFACTS)
    optional_paths = tuple(optional_artifact_paths or DEFAULT_OPTIONAL_ARTIFACTS)
    artifact_records: dict[str, dict[str, Any]] = {}
    for path in required_paths:
        record = _file_record(root, path, required=True)
        artifact_records[record["path"]] = record
    for path in optional_paths:
        record = _file_record(root, path, required=False)
        artifact_records[record["path"]] = record

    python_record = _file_record(root, python_lock_path, required=True)
    python_dev_record = _file_record(root, python_dev_lock_path, required=False)
    frontend_record = _file_record(root, frontend_lock_path, required=True)
    feature_record = _file_record(root, feature_schema_path, required=False)
    feature_metadata = _load_feature_metadata(root, feature_record["path"])

    thresholds = feature_metadata.get(
        "thresholds",
        [RISK_POLICY.review, RISK_POLICY.hold, RISK_POLICY.block] if RISK_POLICY else [0.25, 0.55, 0.85],
    )
    if not isinstance(thresholds, list) or len(thresholds) != 3:
        thresholds = [RISK_POLICY.review, RISK_POLICY.hold, RISK_POLICY.block] if RISK_POLICY else [0.25, 0.55, 0.85]
    classifier = str(feature_metadata.get("classifier", "unknown"))
    calibrated = bool(feature_metadata.get("calibrated", False))
    calibration_method = str(feature_metadata.get("calibrator_method", "unknown"))
    tests = _placeholder_tests()

    manifest: dict[str, Any] = {
        "manifest_type": "trustshield.release_manifest",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "release_id": str(release_value),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git": {
            "commit": commit_value,
            "working_tree_clean": clean_value,
            "status": "clean" if clean_value else "dirty",
        },
        # Top-level aliases make the JSON convenient for shell/CI consumers.
        "git_commit": commit_value,
        "commit_sha": commit_value,
        "working_tree_clean": clean_value,
        "working_tree_status": "clean" if clean_value else "dirty",
        "runtime_hash": runtime_hash,
        "python_runtime_hash": runtime_hash,
        "runtime": {
            **runtime,
            "runtime_hash": runtime_hash,
            "python_runtime_hash": runtime_hash,
        },
        "python": runtime,
        "dependencies": {
            "python": python_record,
            "python_dev": python_dev_record,
            "python_lock_sha256": python_record["sha256"],
            "python_lock_hash": python_record["sha256"],
            "requirements_sha256": python_record["sha256"],
            "python_dev_lock_sha256": python_dev_record["sha256"],
            "python_dev_lock_hash": python_dev_record["sha256"],
            "requirements_dev_sha256": python_dev_record["sha256"],
        },
        "dependency_lock_hashes": {
            "python": python_record["sha256"],
            "python_dev": python_dev_record["sha256"],
            "frontend": frontend_record["sha256"],
        },
        "python_lock_hash": python_record["sha256"],
        "frontend": {
            "lock_file": frontend_record["path"],
            "lock": frontend_record,
            "lock_sha256": frontend_record["sha256"],
            "lock_hash": frontend_record["sha256"],
            "package_lock_sha256": frontend_record["sha256"],
        },
        "frontend_lock_hash": frontend_record["sha256"],
        "frontend_package_lock_sha256": frontend_record["sha256"],
        "artifacts": artifact_records,
        "required_artifacts": {
            path: record
            for path, record in artifact_records.items()
            if record.get("required") is True
        },
        "artifact_hashes": {
            path: record["sha256"] for path, record in artifact_records.items()
        },
        "artifact_sizes": {
            path: record["size_bytes"] for path, record in artifact_records.items()
        },
        "feature_schema": feature_record,
        "feature_schema_hash": feature_record["sha256"],
        "policy": {
            "name": "CanonicalTrustEngine",
            "decision_order": ["ALLOW", "REVIEW", "HOLD", "BLOCK"],
            "version": RISK_POLICY.version if RISK_POLICY else "policy.v1",
            "thresholds": thresholds,
        },
        "model": {
            "serving_model_version": "phase3",
            "model_version": "phase3",
            "classifier": classifier,
            "artifact": "models/combined_graph_model.joblib",
        },
        "calibration": {
            "enabled": calibrated,
            "method": calibration_method,
            "artifact": "models/calibrator.joblib",
            "artifact_sha256": artifact_records.get("models/calibrator.joblib", {}).get("sha256"),
        },
        "tests": tests,
        "test_results": dict(tests),
    }
    return manifest


def write_release_manifest(
    manifest: Mapping[str, Any],
    output_path: os.PathLike[str] | str,
) -> Path:
    """Write a manifest atomically enough for local/CI release generation."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def generate_release_manifest(
    repo_root: os.PathLike[str] | str = PROJECT_ROOT,
    *,
    output_path: Optional[os.PathLike[str] | str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build and write a manifest; useful to callers outside the CLI."""

    root = _project_path(repo_root)
    output = (
        (root / DEFAULT_MANIFEST_NAME).resolve()
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    manifest = build_release_manifest(root, output_path=output, **kwargs)
    write_release_manifest(manifest, output)
    return manifest


def _manifest_path_from_environment(
    root_dir: Optional[os.PathLike[str] | str] = None,
    manifest_path: Optional[os.PathLike[str] | str] = None,
) -> tuple[Path, Path]:
    root = _project_path(root_dir or os.getenv("TRUSTSHIELD_RELEASE_ROOT") or PROJECT_ROOT)
    configured = manifest_path or os.getenv("TRUSTSHIELD_RELEASE_MANIFEST_PATH")
    if configured:
        configured_path = Path(configured).expanduser()
        path = (configured_path if configured_path.is_absolute() else root / configured_path).resolve()
    else:
        path = root / DEFAULT_MANIFEST_NAME
    return root, path


def release_manifest_required() -> bool:
    """Whether startup must reject a missing or invalid release manifest."""

    for variable in ("TRUSTSHIELD_REQUIRE_RELEASE_MANIFEST", "TRUSTSHIELD_RELEASE_MANIFEST_REQUIRED"):
        value = os.getenv(variable)
        if value is not None:
            return value.strip().lower() in {"1", "true", "yes", "on"}
    return os.getenv("TRUSTSHIELD_ENV", os.getenv("ENV", "development")).strip().lower() in {
        "production",
        "prod",
    }


def load_release_manifest(
    manifest_path: Optional[os.PathLike[str] | str] = None,
    root_dir: Optional[os.PathLike[str] | str] = None,
) -> Optional[dict[str, Any]]:
    """Load a JSON manifest, returning ``None`` when it is absent.

    Malformed JSON raises :class:`ReleaseManifestError`; callers that need a
    non-throwing health/status result should use ``get_release_manifest_status``.
    """

    root, path = _manifest_path_from_environment(root_dir, manifest_path)
    _ = root  # root resolution also enforces the path boundary.
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseManifestError(f"Unable to read release manifest: {path.name}") from exc
    if not isinstance(loaded, dict):
        raise ReleaseManifestError("Release manifest root must be a JSON object")
    return loaded


def _manifest_file_record(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    return None


def _record_path(record: Mapping[str, Any]) -> Optional[str]:
    path = record.get("path")
    return str(path) if isinstance(path, str) and path else None


def _record_hash(record: Mapping[str, Any]) -> Optional[str]:
    value = record.get("sha256", record.get("hash"))
    return str(value) if isinstance(value, str) else None


def _record_size(record: Mapping[str, Any]) -> Optional[int]:
    value = record.get("size_bytes", record.get("size"))
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _validate_file_record(
    errors: list[str],
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
    required_by_schema: bool = False,
) -> None:
    path_value = _record_path(record)
    required = required_by_schema or bool(record.get("required", False))
    if path_value is None:
        errors.append(f"{label}: missing path")
        return
    try:
        path = _path_inside(root, path_value)
    except ReleaseManifestError:
        errors.append(f"{label}: path outside release root")
        return
    if not path.is_file():
        if required:
            errors.append(f"{label}: required file is missing")
        return

    expected_hash = _record_hash(record)
    expected_size = _record_size(record)
    if expected_hash is None or not _SHA256_RE.fullmatch(expected_hash):
        errors.append(f"{label}: invalid sha256")
    elif _hash_file(path).lower() != expected_hash.lower():
        errors.append(f"{label}: sha256 mismatch")
    if expected_size is None or expected_size < 0:
        errors.append(f"{label}: invalid size")
    elif path.stat().st_size != expected_size:
        errors.append(f"{label}: size mismatch")


def _git_status_for_validation(root: Path, manifest_path: Optional[Path]) -> Optional[bool]:
    return git_worktree_clean(root, ignore_path=manifest_path)


def _validate_hash_alias(
    errors: list[str],
    container: Mapping[str, Any],
    key: str,
    expected: Optional[str],
    *,
    label: str,
) -> None:
    """Ensure convenience hash aliases cannot disagree with file records."""

    if key not in container or expected is None:
        return
    value = container.get(key)
    if value != expected:
        errors.append(f"{label}: hash alias mismatch")


def validate_release_manifest(
    manifest: Mapping[str, Any],
    root_dir: os.PathLike[str] | str = PROJECT_ROOT,
    *,
    manifest_path: Optional[os.PathLike[str] | str] = None,
) -> ManifestValidationResult:
    """Validate manifest structure, checkout identity, and file fingerprints."""

    root = _project_path(root_dir)
    errors: list[str] = []
    if not isinstance(manifest, Mapping):
        return ManifestValidationResult(False, False, ("manifest root must be an object",), None)

    schema_version = manifest.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        errors.append("unsupported manifest schema_version")
    release_id = manifest.get("release_id")
    if not isinstance(release_id, str) or not release_id.strip():
        errors.append("release_id is required")

    git_value = manifest.get("git")
    git_info: Mapping[str, Any] = git_value if isinstance(git_value, Mapping) else {}
    commit = git_info.get("commit") or manifest.get("git_commit")
    clean = git_info.get("working_tree_clean")
    if clean is None:
        clean = manifest.get("working_tree_clean")
    top_level_commit = manifest.get("git_commit")
    if isinstance(git_info.get("commit"), str) and isinstance(top_level_commit, str):
        if git_info.get("commit") != top_level_commit:
            errors.append("git commit alias mismatch")
    top_level_clean = manifest.get("working_tree_clean")
    if isinstance(git_info.get("working_tree_clean"), bool) and isinstance(top_level_clean, bool):
        if git_info.get("working_tree_clean") != top_level_clean:
            errors.append("working-tree status alias mismatch")
    if not isinstance(commit, str) or not _GIT_COMMIT_RE.fullmatch(commit):
        errors.append("git commit is missing or invalid")
    if isinstance(commit, str) and isinstance(manifest.get("commit_sha"), str):
        if manifest.get("commit_sha") != commit:
            errors.append("git commit alias mismatch")
    if not isinstance(clean, bool):
        errors.append("working_tree_clean must be boolean")
    elif not clean:
        errors.append("manifest records a dirty working tree")

    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping) or not isinstance(runtime.get("python_version"), str):
        errors.append("runtime.python_version is required")
    elif runtime.get("python_version") != platform.python_version():
        errors.append("python runtime mismatch")
    if isinstance(runtime, Mapping):
        runtime_for_hash = {
            key: runtime.get(key)
            for key in ("python_version", "python_implementation", "platform")
            if runtime.get(key) is not None
        }
        runtime_hash = hashlib.sha256(
            json.dumps(runtime_for_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        recorded_runtime_hash = runtime.get("runtime_hash") or runtime.get("python_runtime_hash")
        if recorded_runtime_hash is not None and recorded_runtime_hash != runtime_hash:
            errors.append("runtime hash mismatch")
        _validate_hash_alias(errors, manifest, "runtime_hash", runtime_hash, label="runtime")
        _validate_hash_alias(errors, manifest, "python_runtime_hash", runtime_hash, label="runtime")

    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, Mapping):
        errors.append("dependencies metadata is required")
    else:
        python_record = _manifest_file_record(dependencies.get("python"))
        if python_record is None:
            errors.append("dependencies.python lock record is required")
        else:
            _validate_file_record(errors, root, python_record, label="python dependency lock", required_by_schema=True)
            python_hash = _record_hash(python_record)
            _validate_hash_alias(errors, dependencies, "python_lock_sha256", python_hash, label="python lock")
            _validate_hash_alias(errors, dependencies, "python_lock_hash", python_hash, label="python lock")
            _validate_hash_alias(errors, dependencies, "requirements_sha256", python_hash, label="python lock")
            _validate_hash_alias(errors, manifest, "python_lock_hash", python_hash, label="python lock")
            lock_aliases = manifest.get("dependency_lock_hashes")
            if isinstance(lock_aliases, Mapping):
                _validate_hash_alias(errors, lock_aliases, "python", python_hash, label="python lock")
        python_dev_record = _manifest_file_record(dependencies.get("python_dev"))
        if python_dev_record is not None:
            _validate_file_record(errors, root, python_dev_record, label="python dev dependency lock")
            python_dev_hash = _record_hash(python_dev_record)
            _validate_hash_alias(errors, dependencies, "python_dev_lock_sha256", python_dev_hash, label="python dev lock")
            _validate_hash_alias(errors, dependencies, "python_dev_lock_hash", python_dev_hash, label="python dev lock")
            _validate_hash_alias(errors, dependencies, "requirements_dev_sha256", python_dev_hash, label="python dev lock")
            lock_aliases = manifest.get("dependency_lock_hashes")
            if isinstance(lock_aliases, Mapping):
                _validate_hash_alias(errors, lock_aliases, "python_dev", python_dev_hash, label="python dev lock")

    frontend = manifest.get("frontend")
    frontend_info: Mapping[str, Any] = frontend if isinstance(frontend, Mapping) else {}
    frontend_record = None
    if frontend_info:
        frontend_record = _manifest_file_record(frontend_info.get("lock"))
    if frontend_record is None:
        errors.append("frontend lock record is required")
    else:
        _validate_file_record(errors, root, frontend_record, label="frontend dependency lock", required_by_schema=True)
        frontend_hash = _record_hash(frontend_record)
        _validate_hash_alias(errors, frontend_info, "lock_sha256", frontend_hash, label="frontend lock")
        _validate_hash_alias(errors, frontend_info, "lock_hash", frontend_hash, label="frontend lock")
        _validate_hash_alias(errors, frontend_info, "package_lock_sha256", frontend_hash, label="frontend lock")
        _validate_hash_alias(errors, manifest, "frontend_lock_hash", frontend_hash, label="frontend lock")
        _validate_hash_alias(errors, manifest, "frontend_package_lock_sha256", frontend_hash, label="frontend lock")
        lock_aliases = manifest.get("dependency_lock_hashes")
        if isinstance(lock_aliases, Mapping):
            _validate_hash_alias(errors, lock_aliases, "frontend", frontend_hash, label="frontend lock")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        errors.append("artifacts metadata is required")
    else:
        for name, value in artifacts.items():
            record = _manifest_file_record(value)
            if record is None:
                errors.append(f"artifact {name}: invalid record")
                continue
            _validate_file_record(
                errors,
                root,
                record,
                label=f"artifact {name}",
                required_by_schema=bool(record.get("required", False)),
            )
            artifact_hashes = manifest.get("artifact_hashes")
            if isinstance(artifact_hashes, Mapping) and name in artifact_hashes:
                expected_hash = _record_hash(record)
                if expected_hash is not None and artifact_hashes.get(name) != expected_hash:
                    errors.append(f"artifact {name}: hash alias mismatch")
            artifact_sizes = manifest.get("artifact_sizes")
            if isinstance(artifact_sizes, Mapping) and name in artifact_sizes:
                expected_size = _record_size(record)
                if expected_size is not None and artifact_sizes.get(name) != expected_size:
                    errors.append(f"artifact {name}: size alias mismatch")

    feature_schema = manifest.get("feature_schema")
    if isinstance(feature_schema, Mapping) and bool(feature_schema.get("available", True)):
        _validate_file_record(errors, root, feature_schema, label="feature schema")
        feature_hash = _record_hash(feature_schema)
        recorded_feature_hash = manifest.get("feature_schema_hash")
        if feature_hash is not None and recorded_feature_hash is not None and recorded_feature_hash != feature_hash:
            errors.append("feature schema: hash alias mismatch")

    for metadata_name in ("policy", "model", "calibration", "tests"):
        if not isinstance(manifest.get(metadata_name), Mapping):
            errors.append(f"{metadata_name} metadata is required")

    current_commit = git_commit(root)
    if current_commit is None:
        supplied_commit = os.getenv("TRUSTSHIELD_RELEASE_COMMIT")
        if not (
            isinstance(supplied_commit, str)
            and isinstance(commit, str)
            and supplied_commit == commit
        ):
            errors.append("current git commit is unavailable; TRUSTSHIELD_RELEASE_COMMIT must match the manifest")
    elif isinstance(commit, str) and _GIT_COMMIT_RE.fullmatch(commit) and current_commit != commit:
        errors.append("git commit mismatch")

    resolved_manifest_path: Optional[Path] = None
    if manifest_path is not None:
        try:
            resolved_manifest_path = _path_inside(root, manifest_path)
        except ReleaseManifestError:
            # A manifest may be mounted outside the checkout.  It cannot
            # affect the checkout status in that case, so there is no path to
            # exclude from ``git status``.
            resolved_manifest_path = None
    current_clean = _git_status_for_validation(root, resolved_manifest_path)
    if current_clean is None:
        supplied_clean = os.getenv("TRUSTSHIELD_RELEASE_WORKTREE_CLEAN", "").strip().lower()
        if supplied_clean not in {"1:true", "true", "yes", "on"}:
            errors.append("current git working-tree status is unavailable; TRUSTSHIELD_RELEASE_WORKTREE_CLEAN=true is required")
    elif isinstance(clean, bool) and current_clean != clean:
        errors.append("working-tree status mismatch")

    return ManifestValidationResult(True, not errors, tuple(errors), dict(manifest))


class ReleaseManifestLoader:
    """Reusable loader/validator used by startup checks and ``/version``."""

    def __init__(
        self,
        root_dir: Optional[os.PathLike[str] | str] = None,
        manifest_path: Optional[os.PathLike[str] | str] = None,
    ) -> None:
        self.root_dir, self.manifest_path = _manifest_path_from_environment(root_dir, manifest_path)

    def load(self) -> Optional[dict[str, Any]]:
        return load_release_manifest(self.manifest_path, root_dir=self.root_dir)

    def validate(self, manifest: Optional[Mapping[str, Any]] = None) -> ManifestValidationResult:
        if manifest is None:
            try:
                manifest = self.load()
            except ReleaseManifestError as exc:
                return ManifestValidationResult(True, False, (str(exc),), None)
        if manifest is None:
            return ManifestValidationResult(False, False, ("release manifest is missing",), None)
        return validate_release_manifest(
            manifest,
            root_dir=self.root_dir,
            manifest_path=self.manifest_path,
        )

    def status(self) -> ManifestValidationResult:
        return self.validate()


def get_release_manifest_status(
    root_dir: Optional[os.PathLike[str] | str] = None,
    manifest_path: Optional[os.PathLike[str] | str] = None,
) -> ManifestValidationResult:
    """Return a non-throwing manifest status for health/version surfaces."""

    try:
        return ReleaseManifestLoader(root_dir=root_dir, manifest_path=manifest_path).status()
    except ReleaseManifestError as exc:
        return ManifestValidationResult(False, False, (str(exc),), None)


def load_and_validate_release_manifest(
    root_dir: Optional[os.PathLike[str] | str] = None,
    manifest_path: Optional[os.PathLike[str] | str] = None,
) -> ManifestValidationResult:
    """Convenience alias for callers that want one load-and-validate operation."""

    return get_release_manifest_status(root_dir=root_dir, manifest_path=manifest_path)


def validate_release_manifest_configuration() -> ManifestValidationResult:
    """Fail closed at startup when release provenance is explicitly required."""

    status = get_release_manifest_status()
    if release_manifest_required() and not status.valid:
        details = "; ".join(status.errors) or "manifest is invalid"
        raise RuntimeError(f"CRITICAL RELEASE PROVENANCE: {details}")
    return status


def _safe_string(value: Any, *, max_length: int = 256) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value[:max_length]


def _safe_record(value: Any) -> Optional[dict[str, Any]]:
    record = _manifest_file_record(value)
    if record is None:
        return None
    result: dict[str, Any] = {}
    for key in ("path", "sha256", "size", "size_bytes", "available", "required"):
        if key in record:
            item = record[key]
            if key in {"path", "sha256"}:
                item = _safe_string(item)
            elif key in {"size", "size_bytes"}:
                item = item if isinstance(item, int) and not isinstance(item, bool) else None
            elif key in {"available", "required"}:
                item = bool(item)
            result[key] = item
    return result


def public_release_manifest(manifest: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    """Return an allowlisted, secret-free manifest view for ``GET /version``."""

    if not isinstance(manifest, Mapping):
        return None
    public: dict[str, Any] = {}
    for key in ("manifest_type", "schema_version", "release_id", "created_at"):
        value = manifest.get(key)
        if key in {"release_id", "created_at"}:
            value = _safe_string(value)
        if value is not None:
            public[key] = value

    git_value = manifest.get("git")
    git_info: Mapping[str, Any] = git_value if isinstance(git_value, Mapping) else {}
    commit = git_info.get("commit") or manifest.get("git_commit")
    clean = git_info.get("working_tree_clean")
    public["git"] = {
        "commit": _safe_string(commit, max_length=64),
        "working_tree_clean": clean if isinstance(clean, bool) else None,
        "status": _safe_string(git_info.get("status")),
    }

    runtime = manifest.get("runtime")
    if isinstance(runtime, Mapping):
        public["runtime"] = {
            key: _safe_string(runtime.get(key))
            for key in ("python_version", "python_implementation", "platform", "runtime_hash")
            if runtime.get(key) is not None
        }

    dependencies = manifest.get("dependencies")
    if isinstance(dependencies, Mapping):
        public["dependencies"] = {
            key: _safe_record(dependencies.get(key))
            for key in ("python", "python_dev")
            if _safe_record(dependencies.get(key)) is not None
        }
        for key in ("python_lock_sha256", "python_dev_lock_sha256"):
            if isinstance(dependencies.get(key), str):
                public["dependencies"][key] = dependencies[key][:64]

    frontend = manifest.get("frontend")
    if isinstance(frontend, Mapping):
        lock = _safe_record(frontend.get("lock"))
        if lock is not None:
            public["frontend"] = {"lock": lock}
        for key in ("lock_sha256", "lock_hash"):
            if isinstance(frontend.get(key), str):
                public.setdefault("frontend", {})[key] = frontend[key][:64]

    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, Mapping):
        public["artifacts"] = {
            str(name): _safe_record(value)
            for name, value in artifacts.items()
            if _safe_record(value) is not None
        }

    feature_schema = _safe_record(manifest.get("feature_schema"))
    if feature_schema is not None:
        public["feature_schema"] = feature_schema
    if isinstance(manifest.get("feature_schema_hash"), str):
        public["feature_schema_hash"] = manifest["feature_schema_hash"][:64]

    for metadata_name in ("policy", "model", "calibration"):
        metadata = manifest.get(metadata_name)
        if isinstance(metadata, Mapping):
            # Values are explicitly allowlisted; arbitrary manifest keys (which
            # could contain credentials or customer data) never reach the API.
            allowed = {
                key: metadata[key]
                for key in (
                    "name",
                    "decision_order",
                    "thresholds",
                    "serving_model_version",
                    "model_version",
                    "classifier",
                    "artifact",
                    "enabled",
                    "method",
                    "artifact_sha256",
                )
                if key in metadata
            }
            public[metadata_name] = allowed

    tests = manifest.get("tests")
    if isinstance(tests, Mapping):
        public["tests"] = {
            key: tests[key]
            for key in ("status", "passed", "failed", "skipped")
            if key in tests and (tests[key] is None or isinstance(tests[key], (str, int, float, bool)))
        }
    return public


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_MANIFEST_NAME",
    "DEFAULT_REQUIRED_ARTIFACTS",
    "DEFAULT_OPTIONAL_ARTIFACTS",
    "FileFingerprint",
    "ManifestValidationResult",
    "ReleaseManifestError",
    "ReleaseManifestLoader",
    "build_release_manifest",
    "fingerprint_file",
    "generate_release_manifest",
    "get_release_manifest_status",
    "git_commit",
    "git_worktree_clean",
    "load_and_validate_release_manifest",
    "load_release_manifest",
    "public_release_manifest",
    "release_manifest_required",
    "release_manifest_digest",
    "validate_release_manifest",
    "validate_release_manifest_configuration",
    "write_release_manifest",
]
