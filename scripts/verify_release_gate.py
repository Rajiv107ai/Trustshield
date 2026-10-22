#!/usr/bin/env python3
"""Run metadata-only release gates that are safe to execute in CI.

The gate intentionally does not start Docker, call a provider, or require a
real API credential.  It verifies the release manifest against the checkout,
the critical OpenAPI/frontend contract, required serving artifacts, frontend
public-key exposure, and optional benchmark provenance.

Typical CI usage::

    python scripts/build_release_manifest.py --output "$RUNNER_TEMP/release.json"
    python scripts/verify_release_gate.py --manifest "$RUNNER_TEMP/release.json"

The benchmark check is informational when the configured benchmark exists but
has no commit field.  If commit metadata is present, it must identify the
same checkout (short or full SHA prefixes are accepted).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.release_manifest import (  # noqa: E402
    DEFAULT_REQUIRED_ARTIFACTS,
    get_release_manifest_status,
)


_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
_PUBLIC_KEY_MARKER = "NEXT_PUBLIC_API_KEY"
_DEFAULT_BENCHMARK = Path("models/http_benchmark_results.json")
_PUBLIC_FRONTEND_FILES = {
    "next.config.js",
    "next.config.mjs",
    "next.config.ts",
    "package.json",
}


def _iter_public_frontend_files(root: Path) -> Iterable[Path]:
    frontend = root / "frontend"
    for directory_name in ("src", "public"):
        directory = frontend / directory_name
        if directory.is_dir():
            yield from (path for path in directory.rglob("*") if path.is_file())
    for name in _PUBLIC_FRONTEND_FILES:
        path = frontend / name
        if path.is_file():
            yield path
    # A developer may have a local frontend env template.  Check it if it is
    # present, while never requiring one to exist in CI.
    if frontend.is_dir():
        yield from (
            path
            for path in frontend.iterdir()
            if path.is_file() and path.name.startswith(".env")
        )


def check_required_artifacts(root: Path) -> list[str]:
    """Return errors for missing serving artifacts, without opening them."""

    return [
        f"required release artifact is missing: {relative}"
        for relative in DEFAULT_REQUIRED_ARTIFACTS
        if not (root / relative).is_file()
    ]


def check_manifest_required_artifacts(manifest: Optional[Mapping[str, Any]]) -> list[str]:
    """Ensure every serving artifact is represented as required in the manifest."""

    if not isinstance(manifest, Mapping):
        return []
    records = manifest.get("artifacts")
    if not isinstance(records, Mapping):
        return ["release manifest does not contain artifact records"]
    errors: list[str] = []
    for relative in DEFAULT_REQUIRED_ARTIFACTS:
        record = records.get(relative)
        if not isinstance(record, Mapping):
            errors.append(f"release manifest is missing required artifact record: {relative}")
            continue
        if record.get("required") is not True:
            errors.append(f"release manifest artifact is not marked required: {relative}")
        if record.get("available") is not True:
            errors.append(f"release manifest required artifact is unavailable: {relative}")
    return errors


def check_public_api_key_exposure(root: Path) -> list[str]:
    """Reject a browser-visible API-key environment variable name/value."""

    errors: list[str] = []
    for path in _iter_public_frontend_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors.append(f"could not inspect public frontend file {path}: {exc}")
            continue
        if _PUBLIC_KEY_MARKER in text:
            errors.append(f"public frontend file exposes {_PUBLIC_KEY_MARKER}: {path}")
    return errors


def check_openapi_contract() -> list[str]:
    """Run the existing contract guard without starting an HTTP server."""

    try:
        from scripts.check_openapi_contract import check_contract

        return list(check_contract())
    except Exception as exc:  # pragma: no cover - defensive CI diagnostics
        return [f"OpenAPI contract guard could not run: {type(exc).__name__}: {exc}"]


def _commit_from_benchmark(payload: Mapping[str, Any]) -> Optional[str]:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("git_commit", "commit_sha", "commit"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("git_commit", "commit_sha", "commit"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def check_benchmark_commit(
    root: Path,
    manifest: Optional[Mapping[str, Any]],
    benchmark_path: Optional[Path] = None,
) -> tuple[list[str], str]:
    """Check benchmark provenance when a generated benchmark has it."""

    path = benchmark_path or (root / _DEFAULT_BENCHMARK)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        return [], f"benchmark metadata not present ({path})"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"benchmark metadata is unreadable: {path}: {exc}"], ""
    if not isinstance(payload, Mapping):
        return [f"benchmark metadata root must be an object: {path}"], ""

    benchmark_commit = _commit_from_benchmark(payload)
    if benchmark_commit is None:
        return [], f"benchmark commit metadata not available ({path})"
    if not _COMMIT_RE.fullmatch(benchmark_commit):
        return [f"benchmark commit metadata is not a valid SHA: {path}"], ""

    manifest_commit: Any = None
    if isinstance(manifest, Mapping):
        git = manifest.get("git")
        if isinstance(git, Mapping):
            manifest_commit = git.get("commit")
        manifest_commit = manifest_commit or manifest.get("git_commit")
    if not isinstance(manifest_commit, str) or not _COMMIT_RE.fullmatch(manifest_commit):
        return ["benchmark commit metadata cannot be compared because the manifest commit is unavailable"], ""
    if not (
        manifest_commit.lower().startswith(benchmark_commit.lower())
        or benchmark_commit.lower().startswith(manifest_commit.lower())
    ):
        return [
            "benchmark commit metadata does not match the release manifest "
            f"({benchmark_commit} != {manifest_commit})"
        ], ""
    return [], f"benchmark commit metadata matches ({benchmark_commit})"


def _manifest_path(root: Path, value: Optional[Path]) -> Path:
    if value is None:
        return root / "models" / "release_manifest.json"
    return value if value.is_absolute() else root / value


def run_gate(
    *,
    root: Path,
    manifest_path: Path,
    benchmark_path: Optional[Path] = None,
    allow_missing_manifest: bool = False,
) -> tuple[list[str], list[str]]:
    """Return ``(errors, notes)`` for the release gate."""

    errors: list[str] = []
    notes: list[str] = []

    status = get_release_manifest_status(root_dir=root, manifest_path=manifest_path)
    if not status.present:
        message = f"release manifest is missing: {manifest_path}"
        if allow_missing_manifest:
            notes.append(message + " (allowed by --allow-missing-manifest)")
        else:
            errors.append(message)
    elif not status.valid:
        errors.extend(f"release manifest: {error}" for error in status.errors)
    else:
        notes.append(f"release manifest is valid: {manifest_path}")

    errors.extend(check_required_artifacts(root))
    errors.extend(check_manifest_required_artifacts(status.manifest))
    errors.extend(check_public_api_key_exposure(root))

    contract_errors = check_openapi_contract()
    errors.extend(f"OpenAPI contract: {error}" for error in contract_errors)
    notes.append("OpenAPI/frontend contract guard passed" if not contract_errors else "")

    manifest_payload = status.manifest
    benchmark_errors, benchmark_note = check_benchmark_commit(
        root,
        manifest_payload,
        benchmark_path=benchmark_path,
    )
    errors.extend(benchmark_errors)
    if benchmark_note:
        notes.append(benchmark_note)
    return errors, [note for note in notes if note]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Release manifest path (defaults to <root>/models/release_manifest.json)",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="Optional generated benchmark JSON to provenance-check",
    )
    parser.add_argument(
        "--allow-missing-manifest",
        action="store_true",
        help="Allow local checks when a release manifest has not been generated",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    manifest = _manifest_path(root, args.manifest.expanduser() if args.manifest else None)
    benchmark = args.benchmark.expanduser() if args.benchmark else None
    if benchmark is not None and not benchmark.is_absolute():
        benchmark = root / benchmark
    errors, notes = run_gate(
        root=root,
        manifest_path=manifest,
        benchmark_path=benchmark,
        allow_missing_manifest=args.allow_missing_manifest,
    )
    for note in notes:
        print(f"RELEASE GATE: {note}")
    for error in errors:
        print(f"RELEASE GATE ERROR: {error}", file=sys.stderr)
    if errors:
        print(f"Release gate failed with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("Release gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
