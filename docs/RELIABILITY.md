# Reliability and Release Operations

This document defines the reliability contract for TrustShield's serving and
operator-action paths. Values marked **TBD** are deployment-owned SLO inputs;
they are intentionally not presented as measured capacity or a universal
service guarantee. Record approved values in the deployment change and pass
the same values to the load scripts through environment variables.

## SLO placeholders and signals

| Signal | Initial target | Measurement and owner |
| --- | --- | --- |
| Scoring request availability | **TBD** (for example, `99.x%`) | `http_req_failed` and API status metrics; service owner |
| Scoring p95/p99 latency | **TBD** ms / **TBD** ms | `tests/load/scoring.js`; platform owner |
| Listing scoring p95/p99 latency | **TBD** ms / **TBD** ms | `tests/load/listing.js`; platform owner |
| Stream connection/event availability | **TBD** | `tests/load/stream.js` and SSE status/event checks |
| Durable action enqueue availability | **TBD** | action route status and audit/outbox metrics |
| Action completion age | **TBD** | pending age, attempts, and provider receipts |
| Readiness convergence after restart | **TBD** | `/ready` probes; deployment owner |

Configure a deployment gate explicitly rather than relying on the script
defaults. For example:

```text
BASE_URL=https://staging.example \
API_KEY="$STAGING_API_KEY" \
TARGET_RPS=5 TARGET_P95_MS=250 TARGET_P99_MS=500 \
MAX_ERROR_RATE=1e-2 DURATION=60s \
k6 run tests/load/scoring.js
```

The load scripts use a configurable arrival rate. They do not claim a fixed
throughput level; an approved target is meaningful only with the deployment's
worker count, model artifacts, instance type, and data dependencies recorded.

## Failure policies

### Release artifacts and manifest

The five Phase 3 serving artifacts are required. A missing artifact, corrupt
artifact, hash/size mismatch, dirty checkout recorded in the release manifest,
or commit mismatch is a release failure. The model loader must not mark the
store loaded after a failed load. Optional CLIP/Phase 5 artifacts may degrade
their feature surface only when the deployment policy allows that degradation.

`scripts/verify_release_gate.py` checks the manifest, required artifact
presence, the OpenAPI/frontend contract, and public frontend configuration.
The gate does not contact a provider or start Docker. A generated benchmark's
commit metadata is checked when that metadata is present; an older benchmark
without a commit field is reported as informational rather than silently
treated as provenance.

### Authentication and contract failures

Authentication failures are fail-closed: missing, invalid, expired, or revoked
credentials receive `401` and do not reach a protected operation. CORS is not
an authentication mechanism. Action payloads are strict and allowlisted;
invalid fields, non-boolean confirmation, target mismatches, and unsupported
actions receive `422` before anything is enqueued.

### Point-in-time history

When `TRUSTSHIELD_REQUIRE_PIT_HISTORY=true` (and by default in production), a
missing point-in-time history store makes `/ready` return `503` with a
`not_ready` status. Development may report a degraded state only when the
deployment explicitly permits it. The service must not silently represent a
required historical feature as current or complete.

Production also requires `TRUSTSHIELD_FEATURE_SERVICE_READY=true`. This is a
separate gate from having an orders file: it confirms that the full governed
point-in-time feature service, not caller-provided/default tabular features,
is connected. Until then `/ready` is `not_ready` and scoring returns `503`.

### Durable actions, duplicates, and provider outcomes

Action HTTP requests persist a pending command and return its durable receipt;
they do not imply a provider-side success. Idempotency keys are unique:

* retrying the same key and equivalent request returns the original receipt;
* reusing the key for a different request returns `409`;
* provider success persists a provider receipt and becomes `succeeded`;
* a provider rejection becomes `failed` with a safe diagnostic;
* a timeout has an unknown provider outcome and remains `pending` for a
  worker retry with the same idempotency key.

On worker restart, pending commands remain in the durable outbox and are
eligible for bounded retry. A worker must not acknowledge a pending command as
successful without a provider receipt. In production, the action outbox must
use its configured durable PostgreSQL backend and durable audit is required;
the deterministic adapters are for local tests/development only.

Transaction scoring also writes an allowlisted decision snapshot when
`TRUSTSHIELD_AUDIT_DECISIONS=true` (the production default). If the required
audit store cannot append the record, scoring returns `503` rather than
claiming a decision that cannot be reproduced later.

## Canary, rollback, and release gates

1. Build a release manifest from the exact checkout and retain its commit,
   dependency-lock hashes, artifact hashes, runtime metadata, and test receipt.
2. Run the metadata-only release gate and the focused failure-injection tests
   in CI. CI does not need provider credentials or a Docker daemon for these
   checks.
3. Deploy the immutable image/artifact set to a small canary. Keep the prior
   manifest and image available. Run smoke checks for `/health`, `/ready`,
   `/version`, scoring, listing scoring, and the action enqueue/status path.
4. Observe the approved latency/error/readiness/action-pending SLOs for the
   agreed observation window (**TBD**). Do not promote on a green process check
   alone when a user-facing SLO is failing.
5. Promote only after the canary gate passes. If the gate fails, stop rollout
   and route traffic to the last known-good manifest/image. Preserve pending
   action commands and investigate their provider receipts before replaying
   anything manually.

Rollback is an artifact/image rollback, not an in-place model overwrite. The
old release manifest identifies the exact dependency and model hashes to
restore. If an action provider has an unknown outcome, reconcile by
idempotency key/provider receipt before retrying; rollback must not duplicate a
side effect.

## RPO and RTO placeholders

| Data or capability | RPO | RTO | Required restore evidence |
| --- | --- | --- | --- |
| Action outbox and audit records | **TBD** | **TBD** | durable database restore plus receipt reconciliation |
| Point-in-time history | **TBD** | **TBD** | snapshot/load validation and effective cutoff check |
| Release/model artifacts | **TBD** | **TBD** | manifest hash validation and `/version` check |
| Optional monitoring/reporting state | **TBD** | **TBD** | deployment-specific replay or accepted loss |

RPO/RTO values must be approved per environment. A backup existing is not
evidence that a restore is usable: restoration, artifact loading, PIT cutoff
validation, action-outbox integrity, and readiness checks must be recorded.

For the local development audit store, a repeatable integrity restore check is:

```bash
python scripts/restore_drill.py \
  --source .runtime/audit.sqlite3 \
  --backup /tmp/trustshield-audit-restore.sqlite3
```

This does not replace a PostgreSQL backup/restore drill. Production evidence
must include the managed database restore, action-receipt reconciliation, and
measured RPO/RTO values.

## External restore and load execution is a deployment gate

This repository provides deterministic unit/failure-injection checks and
configurable k6 scenarios. It does not claim that an external database restore,
provider reconciliation, staging load test, or canary observation has been
executed from CI. Those activities require the deployment environment,
approved credentials, representative data, and an operator-owned change
window. External restore/load execution is therefore a **deployment gate**;
release promotion remains blocked until its evidence is attached to the
release record.
