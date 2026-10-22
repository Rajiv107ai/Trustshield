# Stage 1 Release Provenance

TrustShield can ship a metadata-only `models/release_manifest.json` alongside a
backend release.  The manifest records the release identifier, Git commit and
working-tree state, Python/runtime information, dependency lock hashes,
frontend lock hash, serving-artifact SHA-256/size fingerprints, feature-schema
hash when available, policy/model/calibration metadata, and empty test-result
placeholders for a later release stage.

The generator hashes only the explicit serving artifacts and lock files.  It
does not read or enumerate `data/`, customer records, or raw transaction data.

```bash
python scripts/build_release_manifest.py --release-id 2026.10.20
```

The default output is `models/release_manifest.json`. Use
`--output` for a different release artifact location.  A dirty checkout is
recorded as `working_tree_clean: false`; it is never represented as clean.

## Production enforcement

Development and tests do not require a manifest. For a production process,
manifest validation is enabled automatically; it can be made explicit with:

```bash
TRUSTSHIELD_REQUIRE_RELEASE_MANIFEST=true
```

Startup then fails closed if the manifest is absent, malformed, dirty,
different from the current Git commit, or has changed dependency-lock or
serving-artifact hashes.  `TRUSTSHIELD_RELEASE_MANIFEST_REQUIRED=true` is an
accepted compatibility alias.  The optional path overrides are:

- `TRUSTSHIELD_RELEASE_ROOT` — release checkout root.
- `TRUSTSHIELD_RELEASE_MANIFEST_PATH` — manifest path relative to that root or
  an absolute mounted path.

The production image intentionally excludes `.git`. Pass
`TRUSTSHIELD_RELEASE_COMMIT` and `TRUSTSHIELD_RELEASE_WORKTREE_CLEAN=true`
from the generated manifest into the container so startup can validate the
immutable image against its recorded source provenance.

`GET /version` is public like the liveness probes and returns typed status
fields (`manifest_present`, `manifest_valid`, and `manifest_required`) plus an
allowlisted provenance view.  Secrets and arbitrary manifest keys are not
returned.
