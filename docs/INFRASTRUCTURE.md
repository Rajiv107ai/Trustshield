# TrustShield Production Infrastructure & Service Mesh

## 1. Overview & Service Mesh Topology

TrustShield operates as a coordinated 5-container service mesh orchestrated via Docker Compose:
- **`trustshield-api`:** FastAPI backend gateway hosting model inference, Trust Engine, SSE stream, and dossier endpoints.
- **`trustshield-redis`:** In-memory key-value store hosting 16-dimensional GNN entity vectors and feature contexts.
- **`trustshield-neo4j`:** Labeled property graph database running Community Edition for collusion ring and bipartite graph queries.
- **`prometheus`:** Time-series metrics engine scraping backend latency and decision counters.
- **`grafana`:** Visual operational dashboard pre-configured with the TrustShield Overview dashboard.

---

## 2. Redis Integration & 16D Embedding Cache

Implemented in `backend/services/redis_service.py`:
- **Client:** Official `redis` Python package.
- **Embedding Format:** Packed 16-dimensional 32-bit floats (`np.float32.tobytes()`).
- **Key Convention:** `trustshield:embedding:{entity_type}:{entity_id}`
- **TTL Support:** Feature context keys expire automatically after configurable TTL (`REDIS_FEATURE_TTL_SECONDS = 3600`).
- **Connection Cooldown Guard:** Implements a 30-second cooldown on offline connection attempts to prevent reconnection storms from degrading HTTP request latency.
- **Graceful Fallback:** If Redis is offline, requests seamlessly pull vectors from disk joblib artifacts (`buyer_embeddings.joblib`, `seller_embeddings.joblib`) with observable counter increments (`trustshield_fallback_total{service="redis"}`).

---

## 3. Neo4j Integration & Temporal Cypher Safety

Implemented in `backend/services/neo4j_service.py`:
- **Client:** Official `neo4j` Python driver.
- **100% Parameterized Queries:** String interpolation is strictly prohibited. All user inputs are passed via `$entity_id`, `$cutoff_date`, etc.
- **Strict Temporal Invariant:** To prevent historical data leakage, all Cypher queries enforce `event_time < decision_time` using ISO-8601 timestamps.
- **Ego Network Traversal:** Discovers shared devices, IP addresses, shipping destinations, and collusion clusters up to 2 hops.
- **Graceful Fallback:** If Neo4j is offline, requests fall back to disk artifacts (`rings_df` in `models/fraud_rings.joblib`) and return `"graph_source": "disk_artifact"` in metadata.

---

## 4. Docker Architecture & Container Networking

All inter-container communication uses Docker internal DNS service names—never `localhost` or hardcoded IP addresses:
- Redis URI inside Docker: `redis://trustshield-redis:6379/0`
- Neo4j URI inside Docker: `bolt://trustshield-neo4j:7687`
- Prometheus Scrape Target: `trustshield-api:8000`

### Artifact Strategy
- **Baked into Image:** Core source code, lightweight schemas, and configuration templates.
- **Mounted via Volumes:** Large ML model artifacts (`models/`) are volume-mounted from the host (`./models:/app/models:ro`) into containers. This prevents multi-gigabyte Docker build contexts while ensuring models are immediately accessible and read-only.
- **Persistent Data:**
  - Redis: `redis_data` volume.
  - Neo4j: `neo4j_data` and `neo4j_import` volumes.
  - Prometheus: `prometheus_data` volume.
  - Grafana: `grafana_data` volume.

---

## 5. Environment Configuration

All settings are configured via `.env` (derived from `.env.example`):
```bash
# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_FEATURE_TTL_SECONDS=3600

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=trustshield_dev_secret

# Observability
PROMETHEUS_ENABLED=true
GRAFANA_PORT=3001

# Real-Time & Investigations
SSE_ENABLED=true
INVESTIGATION_ENABLED=true
```

---

## 6. Offline Degradation Contract

TrustShield is guaranteed to boot and score transactions even when all external infrastructure is offline:
| Component | Offline Status | Operational Behavior | Observability |
|---|---|---|---|
| **Redis** | Offline | Reads 16D embeddings from `models/*_embeddings.joblib` | `redis_status: "unavailable"`, counter inc |
| **Neo4j** | Offline | Reads community rings from `models/fraud_rings.joblib` | `neo4j_status: "unavailable"`, counter inc |
| **Prometheus** | Offline | Metrics accumulate in memory until scraped | Export available via `GET /metrics` |
| **Core Models**| Missing | System halts at startup with descriptive error | `GET /ready` returns HTTP 503 |
