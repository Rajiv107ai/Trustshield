# TrustShield AI — Production Deployment Guide

<!-- BEGIN GENERATED -->

## 1. Architectural Overview

TrustShield AI operates as a **zero-daemon, single-container inference service** powered by FastAPI, backed by high-performance in-memory graph representations, XGBoost tabular models, TreeSHAP explainability, and conformal uncertainty prediction.

```
                  ┌───────────────────────────────────────────────┐
                  │           Reverse Proxy / Gateway             │
                  │              (Nginx / Cloudflare)             │
                  └───────────────────────┬───────────────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        │                                   │
                 HTTP (Port 3000)                    HTTP (Port 8000)
                        │                                   │
              ┌─────────▼─────────┐               ┌─────────▼─────────┐
              │  Next.js 16 UI    │               │  FastAPI Backend  │
              │  (Turbopack SSR)  │               │ (Single Container)│
              └───────────────────┘               └─────────┬─────────┘
                                                            │
                                                ┌───────────┴───────────┐
                                                │  In-Memory Components │
                                                │  • NetworkX Graph     │
                                                │  • Calibrator (Iso)   │
                                                │  • Phase 3 XGBoost    │
                                                │  • Conformal Pred.    │
                                                │  • TreeSHAP Explainer │
                                                │  • Precomputed Rings  │
                                                └───────────────────────┘
```

### Key Architectural Characteristics
- **Zero External Daemons**: No external dependencies on Redis, Neo4j, or PyG runtimes at serving time.
- **Sub-50ms Latency**: In-memory temporal graph edge lookups and local tree traversals eliminate network hops.
- **Calibrated Decision Routing**: Cost-loss thresholding scales dynamically with transaction value to minimize business risk ($C_{\text{review}} < C_{\text{fraud}} \cdot \text{amount}$).
- **Guaranteed Conformal Bounds**: Conformal coverage set $C(X)$ guarantees $1 - \alpha = 0.95$ ground truth coverage.

---

## 2. System & Hardware Specifications

### Minimum Requirements (Evaluation & Low Traffic)
- **CPU**: 2 vCPUs (x86_64 or ARM64)
- **RAM**: 2 GB
- **Disk**: 5 GB SSD storage
- **Operating System**: Linux (Ubuntu 22.04+ recommended), macOS, or Windows 10/11 with WSL2

### Recommended Production Specifications (Up to 1,000 req/sec)
- **CPU**: 4 to 8 vCPUs
- **RAM**: 4 GB to 8 GB (ensures comfortable headroom for 500k+ edge in-memory graphs)
- **Network**: 1 Gbps NIC
- **Disk**: 20 GB NVMe SSD

---

## 3. Environment Variables Reference

Copy `.env.example` to `.env` or set these in your container orchestrator:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PORT` | `8000` | HTTP port the FastAPI backend listens on |
| `HOST` | `0.0.0.0` | Host interface binding (`0.0.0.0` for containers) |
| `LOG_LEVEL` | `info` | Logging verbosity (`debug`, `info`, `warning`, `error`) |
| `TRUSTSHIELD_ENV` | `development` | Deployment environment (`production` or `development`). Must be set to `production` in production. |
| `TRUSTSHIELD_ADMIN_KEY` | *(dev default)* | Admin API key (min 16 chars required in production) |
| `TRUSTSHIELD_API_KEY` | *(dev default)* | Operator API key (min 16 chars required in production) |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3000` | Strict comma-separated origin whitelist (wildcards prohibited in production) |
| `PROMETHEUS_MULTIPROC_DIR` | `/tmp/prometheus_multiproc` | Directory for multi-worker Prometheus telemetry |
| `MODEL_DIR` | `models` | Path containing serialized `.joblib` model artifacts |
| `TRUSTSHIELD_REQUIRE_PIT_HISTORY` | `true` in production | Require the packaged or mounted point-in-time history before readiness succeeds |
| `TRUSTSHIELD_REQUIRE_RELEASE_MANIFEST` | `true` in production | Require release-manifest validation before startup |
| `TRUSTSHIELD_STRICT_SERVER_FEATURES` | `true` in production | Reject caller-supplied derived fraud/security features |
| `TRUSTSHIELD_FEATURE_SERVICE_READY` | `false` until configured | Explicitly enable the governed real-data point-in-time feature service; production scoring remains paused otherwise |
| `TRUSTSHIELD_AUDIT_DECISIONS` | `true` in production | Append allowlisted decision snapshots to the audit store |
| `TRUSTSHIELD_REQUIRE_AUDIT_DURABILITY` | `true` in production | Require PostgreSQL-backed audit persistence |
| `TRUSTSHIELD_AUDIT_POSTGRES_DSN` | *(required in production)* | PostgreSQL DSN for decision/operator audit records |
| `TRUSTSHIELD_ACTION_POSTGRES_DSN` | *(required in production)* | PostgreSQL DSN for action outbox commands and receipts |

> [!IMPORTANT]
> **Production Fail-Closed Enforcement**:
> In production (`TRUSTSHIELD_ENV=production`), the application strictly validates security configuration at startup and fails fast if:
> - `TRUSTSHIELD_ADMIN_KEY` is not provided or shorter than 16 characters.
> - `TRUSTSHIELD_API_KEY` is not provided or shorter than 16 characters.
> - `CORS_ALLOWED_ORIGINS` is not set or contains wildcard `*`.
> - Authentication bypass flags are set.
> - Point-in-time history is required but unavailable.

The production image includes only the minimal `data/synthetic_v2_1/orders.csv`
history needed for causal serving. Larger research datasets remain excluded by
`.dockerignore`. If an externally managed history store is used instead, mount
it read-only and keep `TRUSTSHIELD_REQUIRE_PIT_HISTORY=true`.

Before building the production image, generate the release manifest from the
exact checkout:

```bash
python scripts/build_release_manifest.py --output models/release_manifest.json
docker compose build --no-cache
```

The manifest is ignored by Git and is supplied as release build input; the
image contains it under `/app/models/release_manifest.json`.

### Example Production `.env`
```env
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=info
TRUSTSHIELD_ENV=production
TRUSTSHIELD_AUTH_ENABLED=true
TRUSTSHIELD_ADMIN_KEY=prod-admin-secret-key-32chars-long-example!
TRUSTSHIELD_API_KEY=prod-operator-api-key-32chars-long-example
CORS_ALLOWED_ORIGINS=https://trustshield.yourdomain.com
```

---

## 4. Container Deployment (Docker & Compose)

### A. Single Command Launch via Docker Compose

```bash
# Build and run the single-container backend (reads .env automatically)
docker compose up -d --build

# Inspect container logs
docker compose logs -f trustshield-api
```

### B. Standalone Docker Build & Run

```bash
# 1. Build the multi-stage minimal runtime image
docker build -t trustshield-api:latest .

# 2. Run container with automated healthcheck
docker run -d \
  --name trustshield-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -e PORT=8000 \
  -e LOG_LEVEL=info \
  -e TRUSTSHIELD_ENV=production \
  -e TRUSTSHIELD_ADMIN_KEY="prod-admin-secret-key-32chars-long-example!" \
  -e TRUSTSHIELD_API_KEY="prod-operator-api-key-32chars-long-example" \
  -e CORS_ALLOWED_ORIGINS="https://trustshield.yourdomain.com" \
  trustshield-api:latest
```

### C. Verify Container Health

```bash
# Liveness probe
curl -f http://localhost:8000/health

# Readiness probe
curl -f http://localhost:8000/ready
```

---

## 5. Next.js Frontend Deployment

### Production Build & Launch

```bash
cd frontend

# Install exact locked dependencies
npm ci

# Compile Next.js 16 production build with Turbopack
npm run build

# Start production server on port 3000
npm start -- -p 3000
```

### Frontend Environment Variables (`frontend/.env.production`)

```env
NEXT_PUBLIC_API_URL=https://api.trustshield.yourdomain.com
```

> `X-API-Key` header, including for the transaction SSE stream.

### Frontend Toolchain & Security Audit Note

- **Production Runtime Security**: All production frontend runtime dependencies (`dependencies`) are clean of vulnerabilities and audited.
- **Dev Toolchain Advisory Context**: In `devDependencies`, `npm audit` reports advisories related to `braces` via the Next.js linter path (`eslint-config-next` -> `@next/eslint-plugin-next` -> `fast-glob` -> `micromatch` -> `braces`).
  - These packages are strictly build-time development linting tools and are completely excluded from the production Next.js runtime artifact.
  - Running `npm audit fix --force` is strictly prohibited because it forces an unsafe major downgrade to `eslint-config-next@14`, which breaks Next.js 16 and React 19 compatibility.
  - `package.json` overrides pin `braces` to `^3.0.3` and `micromatch` to `^4.0.8` to ensure no legacy 2.x versions can be resolved.


---

## 6. Production Reverse Proxy Configuration (Nginx)

```nginx
# Nginx upstream configuration for TrustShield AI
server {
    listen 80;
    server_name trustshield.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name trustshield.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/trustshield.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/trustshield.yourdomain.com/privkey.pem;

    # Next.js Frontend UI
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # FastAPI Backend Routing
    location /api/ {
        rewrite ^/api/(.*) /$1 break;
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

<!-- END GENERATED -->
