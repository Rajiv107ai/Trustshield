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
| `API_KEYS` | *(Generated 32-char key)* | Comma-separated list of authorized API keys |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3000` | Strict comma-separated origin whitelist |
| `PROMETHEUS_MULTIPROC_DIR` | `/tmp/prometheus_multiproc` | Directory for multi-worker Prometheus telemetry |
| `MODEL_DIR` | `models` | Path containing serialized `.joblib` model artifacts |

---

## 4. Container Deployment (Docker & Compose)

### A. Single Command Launch via Docker Compose

```bash
# Build and run the single-container backend
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
  -e API_KEYS="prod-operator-api-key-32chars-long" \
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
NEXT_PUBLIC_API_KEY=prod-operator-api-key-32chars-long
```

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
