#!/usr/bin/env python3
"""Build a metadata-only TrustShield release manifest.

The script hashes known serving artifacts and dependency lock files.  It does
not read or traverse ``data/`` or any other customer-data directory.

Examples:
    python scripts/build_release_manifest.py --release-id 2026.10.20
    python scripts/build_release_manifest.py --output dist/release-manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.release_manifest import (  # noqa: E402
    DEFAULT_MANIFEST_NAME,
    build_release_manifest,
    write_release_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_MANIFEST_NAME))
    parser.add_argument("--release-id", default=None)
    parser.add_argument(
        "--required-artifact",
        dest="required_artifacts",
        action="append",
        default=None,
        help="Repository-relative serving artifact; may be repeated.",
    )
    parser.add_argument(
        "--optional-artifact",
        dest="optional_artifacts",
        action="append",
        default=None,
        help="Repository-relative optional artifact; may be repeated.",
    )
    parser.add_argument("--python-lock", default="requirements.txt")
    parser.add_argument("--python-dev-lock", default="requirements-dev.txt")
    parser.add_argument("--frontend-lock", default="frontend/package-lock.json")
    parser.add_argument("--feature-schema", default="models/feature_meta.joblib")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.expanduser().resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    manifest = build_release_manifest(
        root,
        release_id=args.release_id,
        output_path=output,
        required_artifact_paths=args.required_artifacts,
        optional_artifact_paths=args.optional_artifacts,
        python_lock_path=args.python_lock,
        python_dev_lock_path=args.python_dev_lock,
        frontend_lock_path=args.frontend_lock,
        feature_schema_path=args.feature_schema,
    )
    written = write_release_manifest(manifest, output)
    required_missing = [
        path
        for path, record in manifest.get("artifacts", {}).items()
        if record.get("required") and not record.get("available")
    ]
    print(json.dumps({"manifest": str(written), "release_id": manifest["release_id"], "working_tree_clean": manifest["working_tree_clean"]}, sort_keys=True))
    if required_missing:
        print(
            "Required release artifacts are missing: " + ", ".join(required_missing),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
